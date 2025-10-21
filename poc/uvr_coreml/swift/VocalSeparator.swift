import Foundation
import CoreML
import Accelerate

/// 音源分離エンジン
///
/// UVR MDX-NetモデルとSTFT処理を統合した高レベルAPI
@available(iOS 17.0, macOS 14.0, *)
class VocalSeparator {

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
        let sampleRate: Int

        /// モデル入力形状 [batch, channel, frequency, time]
        let inputShape: (batch: Int, channel: Int, frequency: Int, time: Int)

        static let `default` = ModelConfiguration(
            fftSize: 4096,
            hopSize: 1024,
            sampleRate: 44100,
            inputShape: (batch: 1, channel: 2, frequency: 2049, time: 256)
        )
    }

    struct SeparatedAudio {
        /// ボーカルトラック
        let vocals: [Float]

        /// 伴奏トラック
        let instrumental: [Float]

        /// サンプルレート
        let sampleRate: Int
    }

    enum SeparationError: Error {
        case modelLoadFailed(String)
        case predictionFailed(String)
        case invalidAudioFormat(String)
        case processingFailed(String)
    }

    // MARK: - Initialization

    /// イニシャライザ
    ///
    /// - Parameters:
    ///   - modelURL: CoreMLモデルのURL
    ///   - configuration: モデル設定
    init(modelURL: URL, configuration: ModelConfiguration = .default) throws {
        // モデルロード
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .all  // CPU + GPU + Neural Engine

        do {
            self.model = try MLModel(contentsOf: modelURL, configuration: mlConfig)
        } catch {
            throw SeparationError.modelLoadFailed(
                "Failed to load model from \(modelURL): \(error.localizedDescription)"
            )
        }

        self.configuration = configuration

        // STFTプロセッサ初期化
        self.stftProcessor = STFTProcessor(
            fftSize: configuration.fftSize,
            hopSize: configuration.hopSize,
            windowType: .hann
        )

        print("✅ VocalSeparator initialized")
        print("   Model: \(modelURL.lastPathComponent)")
        print("   FFT Size: \(configuration.fftSize)")
        print("   Hop Size: \(configuration.hopSize)")
        print("   Sample Rate: \(configuration.sampleRate) Hz")
    }

    // MARK: - Public Methods

    /// 音源分離実行
    ///
    /// - Parameter audioURL: 入力音声ファイルURL
    /// - Returns: 分離された音声（ボーカル・伴奏）
    func separate(audioURL: URL) async throws -> SeparatedAudio {
        print("\n🎵 音源分離開始: \(audioURL.lastPathComponent)")

        // 1. オーディオ読み込み
        let (audioData, sampleRate) = try loadAudio(from: audioURL)
        print("   オーディオ読み込み完了: \(audioData.count) サンプル")

        // 2. STFT実行
        print("   STFT実行中...")
        let (leftSTFT, rightSTFT) = stftProcessor.computeSTFTStereo(audio: audioData)
        print("   STFT完了: \(leftSTFT.timeFrames) フレーム")

        // 3. CoreML推論
        print("   CoreML推論実行中...")
        let (vocalMask, instrumentalMask) = try await predict(
            leftSTFT: leftSTFT,
            rightSTFT: rightSTFT
        )
        print("   推論完了")

        // 4. マスク適用
        print("   マスク適用中...")
        let vocalSpectrogram = applyMask(
            magnitude: leftSTFT.magnitude,
            mask: vocalMask
        )
        let instrumentalSpectrogram = applyMask(
            magnitude: leftSTFT.magnitude,
            mask: instrumentalMask
        )

        // 5. iSTFT実行
        print("   iSTFT実行中...")
        let vocals = stftProcessor.computeISTFT(
            magnitude: vocalSpectrogram,
            phase: leftSTFT.phase
        )
        let instrumental = stftProcessor.computeISTFT(
            magnitude: instrumentalSpectrogram,
            phase: leftSTFT.phase
        )

        print("✅ 音源分離完了")

        return SeparatedAudio(
            vocals: vocals,
            instrumental: instrumental,
            sampleRate: sampleRate
        )
    }

    // MARK: - Private Methods

    /// オーディオファイル読み込み
    private func loadAudio(from url: URL) throws -> ([Float], Int) {
        // TODO: AVAudioFile を使用した実装
        // 簡易版：固定サンプルレート、ステレオ
        let sampleRate = configuration.sampleRate

        // ダミーデータ（実装時はAVAudioFileで置き換え）
        let dummyData = [Float](repeating: 0, count: sampleRate * 10 * 2)  // 10秒ステレオ

        return (dummyData, sampleRate)
    }

    /// CoreML推論
    private func predict(
        leftSTFT: STFTProcessor.SpectrogramData,
        rightSTFT: STFTProcessor.SpectrogramData
    ) async throws -> (vocalMask: [[Float]], instrumentalMask: [[Float]]) {

        // スペクトログラムを CoreML 入力形式に変換
        let inputArray = prepareInput(leftSTFT: leftSTFT, rightSTFT: rightSTFT)

        // MLMultiArray作成
        let shape = [
            configuration.inputShape.batch,
            configuration.inputShape.channel,
            configuration.inputShape.frequency,
            configuration.inputShape.time
        ] as [NSNumber]

        guard let multiArray = try? MLMultiArray(shape: shape, dataType: .float32) else {
            throw SeparationError.processingFailed("Failed to create MLMultiArray")
        }

        // データコピー
        for (index, value) in inputArray.enumerated() {
            multiArray[index] = NSNumber(value: value)
        }

        // 推論実行
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "spectrogram": MLFeatureValue(multiArray: multiArray)
        ])

        let output = try model.prediction(from: input)

        // 出力解析
        guard let outputMultiArray = output.featureValue(for: "mask")?.multiArrayValue else {
            throw SeparationError.predictionFailed("Invalid model output")
        }

        // マスク抽出
        let vocalMask = extractMask(from: outputMultiArray, channel: 0)
        let instrumentalMask = extractMask(from: outputMultiArray, channel: 1)

        return (vocalMask, instrumentalMask)
    }

    /// 入力データ準備
    private func prepareInput(
        leftSTFT: STFTProcessor.SpectrogramData,
        rightSTFT: STFTProcessor.SpectrogramData
    ) -> [Float] {
        // [batch, channel, frequency, time] 形式に変換
        var inputArray = [Float]()

        let freqBins = configuration.inputShape.frequency
        let timeFrames = configuration.inputShape.time

        // Leftチャンネル
        for t in 0..<timeFrames {
            for f in 0..<freqBins {
                if t < leftSTFT.timeFrames && f < leftSTFT.frequencyBins {
                    inputArray.append(leftSTFT.magnitude[f][t])
                } else {
                    inputArray.append(0)  // パディング
                }
            }
        }

        // Rightチャンネル
        for t in 0..<timeFrames {
            for f in 0..<freqBins {
                if t < rightSTFT.timeFrames && f < rightSTFT.frequencyBins {
                    inputArray.append(rightSTFT.magnitude[f][t])
                } else {
                    inputArray.append(0)  // パディング
                }
            }
        }

        return inputArray
    }

    /// マスク抽出
    private func extractMask(from multiArray: MLMultiArray, channel: Int) -> [[Float]] {
        let freqBins = configuration.inputShape.frequency
        let timeFrames = configuration.inputShape.time

        var mask = Array(
            repeating: Array(repeating: Float(0), count: timeFrames),
            count: freqBins
        )

        for t in 0..<timeFrames {
            for f in 0..<freqBins {
                let index = [0, channel, f, t] as [NSNumber]
                mask[f][t] = multiArray[index].floatValue
            }
        }

        return mask
    }

    /// マスク適用
    private func applyMask(magnitude: [[Float]], mask: [[Float]]) -> [[Float]] {
        var masked = magnitude

        for f in 0..<magnitude.count {
            for t in 0..<magnitude[f].count {
                if t < mask[f].count {
                    masked[f][t] *= mask[f][t]
                }
            }
        }

        return masked
    }

    /// 音声保存
    func save(audio: [Float], sampleRate: Int, to url: URL) throws {
        // TODO: AVAudioFile を使用した実装
        print("💾 音声保存: \(url.lastPathComponent)")
    }
}

// MARK: - Convenience Methods

@available(iOS 17.0, macOS 14.0, *)
extension VocalSeparator {
    /// 簡易版: ボーカルのみ抽出
    func extractVocals(from audioURL: URL) async throws -> [Float] {
        let separated = try await separate(audioURL: audioURL)
        return separated.vocals
    }

    /// 簡易版: 伴奏のみ抽出
    func extractInstrumental(from audioURL: URL) async throws -> [Float] {
        let separated = try await separate(audioURL: audioURL)
        return separated.instrumental
    }
}
