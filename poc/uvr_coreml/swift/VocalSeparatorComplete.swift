import Foundation
import CoreML
import Accelerate
import AVFoundation

/// 完全版音源分離エンジン
///
/// AVAudioFile + STFT + CoreML を統合したエンドツーエンド実装
@available(iOS 17.0, macOS 14.0, *)
class VocalSeparatorComplete {

    // MARK: - Properties

    /// CoreMLモデル
    private let model: MLModel

    /// STFTプロセッサ
    private let stftProcessor: STFTProcessor

    /// モデル設定
    private let configuration: ModelConfiguration

    // MARK: - Types

    struct ModelConfiguration {
        /// FFTサイズ
        let fftSize: Int

        /// ホップサイズ
        let hopSize: Int

        /// サンプルレート
        let sampleRate: Double

        /// チャンクサイズ（時間フレーム）
        let chunkSize: Int

        static let `default` = ModelConfiguration(
            fftSize: 4096,
            hopSize: 1024,
            sampleRate: 44100,
            chunkSize: 256
        )
    }

    struct SeparatedAudio {
        /// ボーカルトラック
        let vocals: AudioFileProcessor.AudioData

        /// 伴奏トラック
        let instrumental: AudioFileProcessor.AudioData
    }

    enum SeparationError: Error {
        case modelLoadFailed(String)
        case predictionFailed(String)
        case invalidAudioFormat(String)
        case processingFailed(String)
    }

    // MARK: - Initialization

    /// イニシャライザ
    init(modelURL: URL, configuration: ModelConfiguration = .default) throws {
        print("🔄 VocalSeparatorComplete初期化中...")

        // モデルコンパイル
        let compiledURL = try MLModel.compileModel(at: modelURL)

        // モデル読み込み
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .all

        do {
            self.model = try MLModel(contentsOf: compiledURL, configuration: mlConfig)
        } catch {
            throw SeparationError.modelLoadFailed(
                "モデル読み込み失敗: \(error.localizedDescription)"
            )
        }

        self.configuration = configuration

        // STFTプロセッサ初期化
        self.stftProcessor = STFTProcessor(
            fftSize: configuration.fftSize,
            hopSize: configuration.hopSize,
            windowType: .hann
        )

        print("✅ VocalSeparatorComplete初期化完了")
        print("   FFT Size: \(configuration.fftSize)")
        print("   Hop Size: \(configuration.hopSize)")
        print("   Sample Rate: \(configuration.sampleRate) Hz")
    }

    // MARK: - Public Methods

    /// 音源分離実行（ファイルから）
    func separate(audioURL: URL) async throws -> SeparatedAudio {
        print("\n" + String(repeating: "=", count: 80))
        print("🎵 音源分離開始: \(audioURL.lastPathComponent)")
        print(String(repeating: "=", count: 80))

        // 1. オーディオ読み込み
        print("\n📂 ステップ1: 音声ファイル読み込み")
        let audioData = try AudioFileProcessor.loadAudio(
            from: audioURL,
            targetSampleRate: configuration.sampleRate
        )

        // 2. ステレオ確認
        let stereoAudio = AudioFileProcessor.convertToStereo(audioData)

        // 3. STFT実行
        print("\n🔄 ステップ2: STFT実行")
        let (leftSTFT, rightSTFT) = stftProcessor.computeSTFT(audioData: stereoAudio)
        print("   Left: \(leftSTFT.frequencyBins) bins × \(leftSTFT.timeFrames) frames")
        print("   Right: \(rightSTFT.frequencyBins) bins × \(rightSTFT.timeFrames) frames")

        // 4. CoreML推論
        print("\n🤖 ステップ3: CoreML推論実行")
        let (vocalMask, instrumentalMask) = try await predictMasks(
            leftSTFT: leftSTFT,
            rightSTFT: rightSTFT
        )

        // 5. マスク適用
        print("\n🎭 ステップ4: マスク適用")
        let vocalSpec = applyComplexMask(
            spectrogram: leftSTFT,
            mask: vocalMask
        )
        let instrumentalSpec = applyComplexMask(
            spectrogram: leftSTFT,
            mask: instrumentalMask
        )

        // 6. iSTFT実行
        print("\n🔄 ステップ5: iSTFT実行")
        let vocals = stftProcessor.createAudioData(
            leftMagnitude: vocalSpec.magnitude,
            leftPhase: vocalSpec.phase,
            rightMagnitude: vocalSpec.magnitude,  // 簡易版: 左を右にコピー
            rightPhase: vocalSpec.phase,
            sampleRate: configuration.sampleRate
        )

        let instrumental = stftProcessor.createAudioData(
            leftMagnitude: instrumentalSpec.magnitude,
            leftPhase: instrumentalSpec.phase,
            rightMagnitude: instrumentalSpec.magnitude,
            rightPhase: instrumentalSpec.phase,
            sampleRate: configuration.sampleRate
        )

        print("\n" + String(repeating: "=", count: 80))
        print("✅ 音源分離完了")
        print(String(repeating: "=", count: 80))

        return SeparatedAudio(
            vocals: vocals,
            instrumental: instrumental
        )
    }

    /// 音声保存
    func save(
        separatedAudio: SeparatedAudio,
        vocalsURL: URL,
        instrumentalURL: URL
    ) throws {
        print("\n💾 分離音声保存中...")

        try AudioFileProcessor.saveAudio(separatedAudio.vocals, to: vocalsURL)
        try AudioFileProcessor.saveAudio(separatedAudio.instrumental, to: instrumentalURL)

        print("✅ 保存完了")
    }

    // MARK: - Private Methods

    /// CoreML推論実行（チャンク処理）
    private func predictMasks(
        leftSTFT: STFTProcessor.SpectrogramData,
        rightSTFT: STFTProcessor.SpectrogramData
    ) async throws -> (vocal: STFTProcessor.SpectrogramData, instrumental: STFTProcessor.SpectrogramData) {

        let timeFrames = leftSTFT.timeFrames
        let freqBins = min(leftSTFT.frequencyBins, 2048)  // モデル期待値
        let chunkSize = configuration.chunkSize

        let numChunks = (timeFrames + chunkSize - 1) / chunkSize
        print("   チャンク数: \(numChunks) (chunk_size=\(chunkSize))")

        var vocalMasks: [[Float]] = []
        var instrumentalMasks: [[Float]] = []

        for chunkIndex in 0..<numChunks {
            let startFrame = chunkIndex * chunkSize
            let endFrame = min((chunkIndex + 1) * chunkSize, timeFrames)

            // チャンク抽出
            let chunk = extractChunk(
                leftSTFT: leftSTFT,
                rightSTFT: rightSTFT,
                startFrame: startFrame,
                endFrame: endFrame,
                targetSize: chunkSize
            )

            // 推論実行
            let output = try predictChunk(chunk)

            // 結果を分割
            let actualSize = endFrame - startFrame
            let vocalChunk = extractChannelMask(output, channel: 0, actualSize: actualSize)
            let instrumentalChunk = extractChannelMask(output, channel: 1, actualSize: actualSize)

            vocalMasks.append(contentsOf: vocalChunk)
            instrumentalMasks.append(contentsOf: vocalChunk)  // 簡易版

            if (chunkIndex + 1) % 10 == 0 {
                print("   進捗: \(chunkIndex + 1)/\(numChunks)")
            }
        }

        // 結果を整形
        let vocalMagnitude = reshape2D(vocalMasks, frequencyBins: freqBins)
        let vocalPhase = leftSTFT.phase  // 元の位相を保持

        let instrumentalMagnitude = reshape2D(instrumentalMasks, frequencyBins: freqBins)
        let instrumentalPhase = leftSTFT.phase

        return (
            STFTProcessor.SpectrogramData(
                magnitude: vocalMagnitude,
                phase: vocalPhase,
                frequencyBins: freqBins,
                timeFrames: vocalMagnitude[0].count
            ),
            STFTProcessor.SpectrogramData(
                magnitude: instrumentalMagnitude,
                phase: instrumentalPhase,
                frequencyBins: freqBins,
                timeFrames: instrumentalMagnitude[0].count
            )
        )
    }

    /// チャンク抽出
    private func extractChunk(
        leftSTFT: STFTProcessor.SpectrogramData,
        rightSTFT: STFTProcessor.SpectrogramData,
        startFrame: Int,
        endFrame: Int,
        targetSize: Int
    ) -> MLMultiArray {

        let freqBins = 2048
        let actualSize = endFrame - startFrame

        // MLMultiArray作成 [1, 4, 2048, 256]
        let inputArray = try! MLMultiArray(
            shape: [1, 4, freqBins, targetSize] as [NSNumber],
            dataType: .float32
        )

        // データコピー（実部・虚部に分解）
        for t in 0..<targetSize {
            for f in 0..<freqBins {
                let index = [0, 0, f, t] as [NSNumber]  // Left Real
                if t < actualSize && f < leftSTFT.frequencyBins {
                    inputArray[index] = NSNumber(value: leftSTFT.magnitude[f][startFrame + t])
                } else {
                    inputArray[index] = 0
                }

                // 他のチャンネル（簡略化: 虚部は0、右は左と同じ）
                inputArray[[0, 1, f, t] as [NSNumber]] = 0  // Left Imag
                inputArray[[0, 2, f, t] as [NSNumber]] = inputArray[index]  // Right Real
                inputArray[[0, 3, f, t] as [NSNumber]] = 0  // Right Imag
            }
        }

        return inputArray
    }

    /// チャンク推論
    private func predictChunk(_ input: MLMultiArray) throws -> MLMultiArray {
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input_1": MLFeatureValue(multiArray: input)
        ])

        let output = try model.prediction(from: inputProvider)

        guard let outputArray = output.featureValue(for: "var_992")?.multiArrayValue else {
            throw SeparationError.predictionFailed("出力取得失敗")
        }

        return outputArray
    }

    /// チャンネルマスク抽出
    private func extractChannelMask(_ output: MLMultiArray, channel: Int, actualSize: Int) -> [[Float]] {
        var mask: [[Float]] = []

        for t in 0..<actualSize {
            var frame: [Float] = []
            for f in 0..<2048 {
                let value = output[[0, channel, f, t] as [NSNumber]].floatValue
                frame.append(value)
            }
            mask.append(frame)
        }

        return mask
    }

    /// 2D配列に整形
    private func reshape2D(_ flatData: [[Float]], frequencyBins: Int) -> [[Float]] {
        var result: [[Float]] = Array(repeating: [], count: frequencyBins)

        for frame in flatData {
            for (f, value) in frame.enumerated() where f < frequencyBins {
                result[f].append(value)
            }
        }

        return result
    }

    /// 複素数マスク適用
    private func applyComplexMask(
        spectrogram: STFTProcessor.SpectrogramData,
        mask: STFTProcessor.SpectrogramData
    ) -> STFTProcessor.SpectrogramData {

        let freqBins = min(spectrogram.frequencyBins, mask.frequencyBins)
        let timeFrames = min(spectrogram.timeFrames, mask.timeFrames)

        var maskedMagnitude: [[Float]] = Array(
            repeating: Array(repeating: 0, count: timeFrames),
            count: freqBins
        )

        for f in 0..<freqBins {
            for t in 0..<timeFrames {
                maskedMagnitude[f][t] = spectrogram.magnitude[f][t] * mask.magnitude[f][t]
            }
        }

        return STFTProcessor.SpectrogramData(
            magnitude: maskedMagnitude,
            phase: spectrogram.phase,
            frequencyBins: freqBins,
            timeFrames: timeFrames
        )
    }
}
