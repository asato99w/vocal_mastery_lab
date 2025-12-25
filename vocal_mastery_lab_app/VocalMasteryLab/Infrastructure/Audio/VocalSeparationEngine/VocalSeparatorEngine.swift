import Foundation
import CoreML
import Accelerate
import AVFoundation
import os.log

private let logger = Logger(subsystem: "com.kazuasato.VocalMasteryLab", category: "VocalSeparatorEngine")

/// Vocal separation engine using CoreML and STFT
/// Based on UVR-MDX-NET implementation (Python compatible)
final class VocalSeparatorEngine {

    // MARK: - Constants (Voc_FTモデル用パラメータ)

    private let nFFT: Int = 6144
    private let dimF: Int = 3072
    private let dimT: Int = 256  // 2^8
    private let hop: Int = 1024
    private let targetSampleRate: Int = 44100
    private let dimC: Int = 4  // ステレオ x (実部+虚部)

    private var nBins: Int { nFFT / 2 + 1 }  // 3841
    private var chunkSize: Int { hop * (dimT - 1) }  // 261120

    // MARK: - Properties

    private let model: MLModel
    private let window: [Float]
    private var dftSetupForward: OpaquePointer?
    private var dftSetupInverse: OpaquePointer?

    // MARK: - Types

    struct ModelConfiguration {
        let fftSize: Int
        let hopSize: Int
        let sampleRate: Double
        let chunkSize: Int

        static let `default` = ModelConfiguration(
            fftSize: 6144,
            hopSize: 1024,
            sampleRate: 44100,
            chunkSize: 256
        )
    }

    struct SeparationResult {
        let vocals: AudioProcessor.AudioData
        let instrumental: AudioProcessor.AudioData
    }

    enum SeparationError: Error, LocalizedError {
        case modelLoadFailed(String)
        case predictionFailed(String)
        case invalidAudioFormat(String)
        case processingFailed(String)

        var errorDescription: String? {
            switch self {
            case .modelLoadFailed(let msg): return "Model load failed: \(msg)"
            case .predictionFailed(let msg): return "Prediction failed: \(msg)"
            case .invalidAudioFormat(let msg): return "Invalid audio format: \(msg)"
            case .processingFailed(let msg): return "Processing failed: \(msg)"
            }
        }
    }

    /// Progress callback type
    typealias ProgressHandler = (Double, String) -> Void

    // MARK: - Initialization

    init(modelURL: URL, configuration: ModelConfiguration = .default) throws {
        logger.info("🔄 VocalSeparatorEngine 初期化中...")

        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
            logger.info("📦 [MODEL] Using pre-compiled model: \(modelURL.lastPathComponent)")
        } else {
            logger.info("🔨 [MODEL] Compiling model: \(modelURL.lastPathComponent)")
            do {
                compiledURL = try MLModel.compileModel(at: modelURL)
            } catch {
                throw SeparationError.modelLoadFailed(error.localizedDescription)
            }
        }

        let mlConfig = MLModelConfiguration()
        // シミュレータではANE/GPUが使えないためCPUOnlyを使用
        #if targetEnvironment(simulator)
        mlConfig.computeUnits = .cpuOnly
        #else
        mlConfig.computeUnits = .all
        #endif

        do {
            self.model = try MLModel(contentsOf: compiledURL, configuration: mlConfig)
            logger.info("✅ [MODEL] Model loaded successfully")
        } catch {
            throw SeparationError.modelLoadFailed(error.localizedDescription)
        }

        // Hann window (periodic=True, PyTorch compatible)
        var w = [Float](repeating: 0, count: nFFT)
        for i in 0..<nFFT {
            w[i] = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(i) / Float(nFFT))
        }
        self.window = w

        // DFT setup
        self.dftSetupForward = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .FORWARD)
        self.dftSetupInverse = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .INVERSE)

        guard dftSetupForward != nil, dftSetupInverse != nil else {
            throw SeparationError.processingFailed("DFT setup failed")
        }

        logger.info("✅ VocalSeparatorEngine 初期化完了")
        logger.info("   n_fft=\(self.nFFT), dim_f=\(self.dimF), dim_t=\(self.dimT), hop=\(self.hop)")
    }

    deinit {
        if let setup = dftSetupForward { vDSP_DFT_DestroySetup(setup) }
        if let setup = dftSetupInverse { vDSP_DFT_DestroySetup(setup) }
    }

    // MARK: - Public Methods

    /// Separate vocals from audio file
    func separate(
        audioURL: URL,
        progressHandler: ProgressHandler? = nil
    ) throws -> SeparationResult {
        logger.info("🎵 [SEPARATION_START] Starting vocal separation")
        logger.info("📂 [INPUT_FILE] \(audioURL.lastPathComponent)")

        // 1. Load audio
        progressHandler?(0.1, "音声を読み込み中...")
        let (left, right) = try loadAudio(url: audioURL)
        logger.info("📥 [AUDIO_LOADED] frames=\(left.count)")

        // 2. Demix (separate) - Voc_FTモデルはボーカルを直接出力
        // Note: denoise=false でPython/PoC互換（単一推論）
        progressHandler?(0.2, "ボーカルを抽出中...")
        let vocalsExtracted = try demix(left: left, right: right, denoise: false, progressHandler: progressHandler)

        // 3. Voc_FTはボーカルモデル → モデル出力がボーカル (ステレオ)
        // demix returns [L samples, R samples] concatenated
        let halfLen = vocalsExtracted.count / 2
        let vocalsLeft = Array(vocalsExtracted[0..<halfLen])
        let vocalsRight = Array(vocalsExtracted[halfLen..<vocalsExtracted.count])

        let vocals = AudioProcessor.AudioData(
            samples: [vocalsLeft, vocalsRight],
            sampleRate: Double(targetSampleRate),
            frameCount: vocalsLeft.count
        )

        // 4. 伴奏を計算: instrumental = original - vocals
        progressHandler?(0.95, "伴奏を計算中...")
        let frameCount = min(left.count, vocalsLeft.count)
        var instrumentalLeft = [Float](repeating: 0, count: frameCount)
        var instrumentalRight = [Float](repeating: 0, count: frameCount)

        // vDSP_vsub: C = B - A (第1引数を第2引数から引く)
        vDSP_vsub(vocalsLeft, 1, left, 1, &instrumentalLeft, 1, vDSP_Length(frameCount))
        vDSP_vsub(vocalsRight, 1, right, 1, &instrumentalRight, 1, vDSP_Length(frameCount))

        let instrumental = AudioProcessor.AudioData(
            samples: [instrumentalLeft, instrumentalRight],
            sampleRate: Double(targetSampleRate),
            frameCount: frameCount
        )

        // Log output stats
        let vocalsRms = sqrt(vocalsLeft.reduce(0) { $0 + $1 * $1 } / Float(vocalsLeft.count))
        let instRms = sqrt(instrumentalLeft.reduce(0) { $0 + $1 * $1 } / Float(instrumentalLeft.count))
        logger.info("🎤 [VOCALS_STATS] rms=\(vocalsRms), frames=\(vocals.frameCount)")
        logger.info("🎸 [INSTRUMENTAL_STATS] rms=\(instRms), frames=\(instrumental.frameCount)")
        logger.info("✅ [SEPARATION_COMPLETE]")

        progressHandler?(1.0, "完了")

        return SeparationResult(vocals: vocals, instrumental: instrumental)
    }

    /// Save separated audio to files
    /// - Parameters:
    ///   - result: The separation result containing vocals and instrumental
    ///   - vocalsURL: URL for saving the vocals track
    ///   - instrumentalURL: Optional URL for saving the instrumental track
    ///   - normalize: Whether to normalize audio to peak 1.0 (default: false for model accuracy)
    func save(result: SeparationResult, vocalsURL: URL, instrumentalURL: URL? = nil, normalize: Bool = false) throws {
        // Save vocals
        let vocalsToSave = normalize ? AudioProcessor.normalize(result.vocals) : result.vocals
        try AudioProcessor.saveAudio(vocalsToSave, to: vocalsURL)
        logger.info("🎤 [SAVED] Vocals: \(vocalsURL.lastPathComponent)")

        // Save instrumental if URL provided
        if let instURL = instrumentalURL {
            let instToSave = normalize ? AudioProcessor.normalize(result.instrumental) : result.instrumental
            try AudioProcessor.saveAudio(instToSave, to: instURL)
            logger.info("🎸 [SAVED] Instrumental: \(instURL.lastPathComponent)")
        }
    }

    /// Save separated vocals to file (backward compatibility)
    func save(result: SeparationResult, to url: URL) throws {
        try save(result: result, vocalsURL: url, instrumentalURL: nil, normalize: false)
    }

    // MARK: - Private Methods

    private func loadAudio(url: URL) throws -> (left: [Float], right: [Float]) {
        let file = try AVAudioFile(forReading: url)
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(targetSampleRate),
            channels: 2,
            interleaved: false
        )!

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: UInt32(file.length)) else {
            throw SeparationError.invalidAudioFormat("Buffer creation failed")
        }

        try file.read(into: buffer)

        guard let channelData = buffer.floatChannelData else {
            throw SeparationError.invalidAudioFormat("No channel data")
        }

        let frameCount = Int(buffer.frameLength)
        let left = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        let right = buffer.format.channelCount > 1 ?
            Array(UnsafeBufferPointer(start: channelData[1], count: frameCount)) : left

        return (left, right)
    }

    /// demix: PoCと同じシンプルな処理方式
    private func demix(
        left: [Float],
        right: [Float],
        denoise: Bool,
        margin: Int = 44100,
        chunks: Int = 15,
        progressHandler: ProgressHandler? = nil
    ) throws -> [Float] {
        let nSample = left.count
        let trim = nFFT / 2
        let genSize = chunkSize - 2 * trim
        let pad = genSize - nSample % genSize

        // パディング（PoCと同じ方法）
        let lPadded = [Float](repeating: 0, count: trim) + left +
                      [Float](repeating: 0, count: pad) +
                      [Float](repeating: 0, count: trim)
        let rPadded = [Float](repeating: 0, count: trim) + right +
                      [Float](repeating: 0, count: pad) +
                      [Float](repeating: 0, count: trim)

        // チャンクに分割
        var mixChunks: [([Float], [Float])] = []
        var i = 0
        while i < nSample + pad {
            let chunkL = Array(lPadded[i..<(i + chunkSize)])
            let chunkR = Array(rPadded[i..<(i + chunkSize)])
            mixChunks.append((chunkL, chunkR))
            i += genSize
        }

        logger.info("   チャンク数: \(mixChunks.count)")

        // 各チャンクを処理
        var outputChunks: [[Float]] = []
        for (idx, (leftChunk, rightChunk)) in mixChunks.enumerated() {
            let spek = stft(left: leftChunk, right: rightChunk)

            let specPred: MLMultiArray
            if denoise {
                let predPos = try predict(spek)
                let negSpek = negateMultiArray(spek)
                let predNeg = try predict(negSpek)
                specPred = averagePredictions(predPos, negPredNeg: predNeg)
            } else {
                specPred = try predict(spek)
            }

            let waves = istft(specPred)

            // Trim
            let trimmedL = Array(waves[trim..<(chunkSize - trim)])
            let trimmedR = Array(waves[(chunkSize + trim)..<(2 * chunkSize - trim)])
            outputChunks.append(trimmedL + trimmedR)

            // Progress update
            let progress = 0.2 + (Double(idx + 1) / Double(mixChunks.count)) * 0.7
            progressHandler?(progress, "ボーカルを抽出中... (\(idx + 1)/\(mixChunks.count))")
        }

        logger.info("   進捗: \(mixChunks.count)/\(mixChunks.count)")

        // 結合してパディングを除去（PoCと同じ方法）
        var vocalsL = [Float](repeating: 0, count: nSample)
        var vocalsR = [Float](repeating: 0, count: nSample)
        var offset = 0
        for chunk in outputChunks {
            let copyLen = min(chunk.count / 2, nSample - offset)
            if copyLen <= 0 { break }

            for j in 0..<copyLen {
                vocalsL[j + offset] = chunk[j]
                vocalsR[j + offset] = chunk[chunk.count / 2 + j]
            }
            offset += copyLen
        }

        return vocalsL + vocalsR
    }

    /// STFT: PyTorch compatible (center=True)
    /// PoCと同じフラットインデックス方式で実装
    private func stft(left: [Float], right: [Float]) -> MLMultiArray {
        // PoCと同じFloat32で作成（CoreMLが自動的にFloat16に変換）
        let inputArray = try! MLMultiArray(shape: [1, dimC, dimF, dimT] as [NSNumber], dataType: .float32)

        // 一時的なフラット配列 (PoCと同じ方式)
        var result = [Float](repeating: 0, count: dimC * dimF * dimT)

        let numFrames = dimT
        let pad = nFFT / 2  // center=True パディング

        // center=True: 前後に nFFT/2 のパディングを追加
        let leftPadded = [Float](repeating: 0, count: pad) + left + [Float](repeating: 0, count: pad)
        let rightPadded = [Float](repeating: 0, count: pad) + right + [Float](repeating: 0, count: pad)

        // チャンネルごとに処理 (PoCと同じ構造)
        for ch in 0..<2 {
            let paddedAudio = ch == 0 ? leftPadded : rightPadded

            for frame in 0..<numFrames {
                let start = frame * hop

                // Windowedセグメントを作成
                var windowedReal = [Float](repeating: 0, count: nFFT)
                var windowedImag = [Float](repeating: 0, count: nFFT)

                for i in 0..<nFFT {
                    windowedReal[i] = paddedAudio[start + i] * window[i]
                }

                // DFT実行
                var outputReal = [Float](repeating: 0, count: nFFT)
                var outputImag = [Float](repeating: 0, count: nFFT)

                vDSP_DFT_Execute(dftSetupForward!, windowedReal, windowedImag, &outputReal, &outputImag)

                // Store in [4, dimF, dimT] format: [L_real, L_imag, R_real, R_imag]
                // PoCと同じインデックス計算
                for f in 0..<dimF {
                    let realIdx = (ch * 2) * dimF * dimT + f * dimT + frame
                    let imagIdx = (ch * 2 + 1) * dimF * dimT + f * dimT + frame
                    result[realIdx] = outputReal[f]
                    result[imagIdx] = outputImag[f]
                }
            }
        }

        // フラットインデックスでMLMultiArrayに転送 (PoCと同じ方式)
        for i in 0..<result.count {
            inputArray[i] = NSNumber(value: result[i])
        }

        return inputArray
    }

    /// CoreML prediction
    private func predict(_ input: MLMultiArray) throws -> MLMultiArray {
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input": MLFeatureValue(multiArray: input)
        ])

        let output = try model.prediction(from: inputProvider)

        guard let result = output.featureValue(for: "var_992")?.multiArrayValue else {
            throw SeparationError.predictionFailed("Output 'var_992' not found")
        }

        return result
    }

    /// iSTFT: PyTorch compatible (center=True, stereo)
    /// Returns: [Left channel (chunkSize samples), Right channel (chunkSize samples)]
    /// PoCと同じフラットインデックス方式で実装
    private func istft(_ specPred: MLMultiArray) -> [Float] {
        var audio = [Float](repeating: 0, count: 2 * chunkSize)
        let pad = nFFT / 2  // center=True パディング
        let paddedLen = chunkSize + nFFT  // パディング後の長さ

        for ch in 0..<2 {
            var paddedAudio = [Float](repeating: 0, count: paddedLen)
            var windowSum = [Float](repeating: 0, count: paddedLen)

            for frame in 0..<dimT {
                var realFull = [Float](repeating: 0, count: nFFT)
                var imagFull = [Float](repeating: 0, count: nFFT)

                // モデル出力からスペクトルを取得 (0 to dimF-1)
                // PoCと同じフラットインデックス計算
                for f in 0..<dimF {
                    let realIdx = (ch * 2) * dimF * dimT + f * dimT + frame
                    let imagIdx = (ch * 2 + 1) * dimF * dimT + f * dimT + frame
                    realFull[f] = specPred[realIdx].floatValue
                    imagFull[f] = specPred[imagIdx].floatValue
                }

                // 共役対称性を利用して後半を埋める
                // X[k] = conj(X[N-k]) for k > N/2
                for f in (nFFT / 2 + 1)..<nFFT {
                    realFull[f] = realFull[nFFT - f]
                    imagFull[f] = -imagFull[nFFT - f]
                }

                // IDFT
                var realOut = [Float](repeating: 0, count: nFFT)
                var imagOut = [Float](repeating: 0, count: nFFT)
                vDSP_DFT_Execute(dftSetupInverse!, &realFull, &imagFull, &realOut, &imagOut)

                // スケーリング (1/N) と窓関数適用
                let scale = 1.0 / Float(nFFT)
                for i in 0..<nFFT {
                    realOut[i] = realOut[i] * scale * window[i]
                }

                // Overlap-add (パディング込みの配列に)
                let start = frame * hop
                for i in 0..<nFFT {
                    paddedAudio[start + i] += realOut[i]
                    windowSum[start + i] += window[i] * window[i]
                }
            }

            // Normalize
            for i in 0..<paddedLen {
                if windowSum[i] > 1e-8 {
                    paddedAudio[i] /= windowSum[i]
                }
            }

            // center=True のパディングを除去して出力
            for i in 0..<chunkSize {
                audio[ch * chunkSize + i] = paddedAudio[pad + i]
            }
        }

        return audio
    }

    private func negateMultiArray(_ array: MLMultiArray) -> MLMultiArray {
        let result = try! MLMultiArray(shape: array.shape, dataType: .float32)
        let count = array.count
        for i in 0..<count {
            result[i] = NSNumber(value: -array[i].floatValue)
        }
        return result
    }

    private func averagePredictions(_ pos: MLMultiArray, negPredNeg: MLMultiArray) -> MLMultiArray {
        let result = try! MLMultiArray(shape: pos.shape, dataType: .float32)
        let count = pos.count
        for i in 0..<count {
            let posVal = pos[i].floatValue
            let negVal = -negPredNeg[i].floatValue
            result[i] = NSNumber(value: (posVal + negVal) * 0.5)
        }
        return result
    }
}
