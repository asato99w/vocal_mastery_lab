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
        mlConfig.computeUnits = .all

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

    private func demix(
        left: [Float],
        right: [Float],
        denoise: Bool,
        margin: Int = 44100,
        chunks: Int = 15,
        progressHandler: ProgressHandler? = nil
    ) throws -> [Float] {
        let samples = left.count
        let segmentSize = chunks * targetSampleRate

        var actualMargin = margin
        if actualMargin > segmentSize {
            actualMargin = segmentSize
        }

        // Segment division
        var segments: [(start: Int, data: ([Float], [Float]))] = []
        var counter = -1
        var skip = 0

        while skip < samples {
            counter += 1
            let sMargin = counter == 0 ? 0 : actualMargin
            let end = min(skip + segmentSize + actualMargin, samples)
            let start = skip - sMargin

            let leftSeg = Array(left[max(0, start)..<end])
            let rightSeg = Array(right[max(0, start)..<end])

            if start < 0 {
                let padSize = -start
                let leftPadded = [Float](repeating: 0, count: padSize) + leftSeg
                let rightPadded = [Float](repeating: 0, count: padSize) + rightSeg
                segments.append((skip, (leftPadded, rightPadded)))
            } else {
                segments.append((skip, (leftSeg, rightSeg)))
            }

            if end >= samples { break }
            skip += segmentSize
        }

        // Process each segment
        var chunkedSources: [[Float]] = []
        let trim = nFFT / 2
        let genSize = chunkSize - 2 * trim

        for (segIdx, (_, (leftSeg, rightSeg))) in segments.enumerated() {
            let nSample = leftSeg.count
            let pad = genSize - nSample % genSize

            // Padding
            let leftPadded = [Float](repeating: 0, count: trim) + leftSeg + [Float](repeating: 0, count: pad + trim)
            let rightPadded = [Float](repeating: 0, count: trim) + rightSeg + [Float](repeating: 0, count: pad + trim)

            // Split into chunks
            var mixWaves: [([Float], [Float])] = []
            var i = 0
            while i < nSample + pad {
                let leftChunk = Array(leftPadded[i..<(i + chunkSize)])
                let rightChunk = Array(rightPadded[i..<(i + chunkSize)])
                mixWaves.append((leftChunk, rightChunk))
                i += genSize
            }

            // Inference
            var tarWaves: [[Float]] = []

            for (leftWave, rightWave) in mixWaves {
                let spek = stft(left: leftWave, right: rightWave)

                let specPred: MLMultiArray
                if denoise {
                    let predPos = try predict(spek)
                    let negSpek = negateMultiArray(spek)
                    let predNeg = try predict(negSpek)
                    specPred = averagePredictions(predPos, negPredNeg: predNeg)
                } else {
                    specPred = try predict(spek)
                }

                // iSTFT returns [L (chunkSize), R (chunkSize)]
                let waves = istft(specPred)

                // Trim both channels
                let trimmedL = Array(waves[trim..<(chunkSize - trim)])
                let trimmedR = Array(waves[(chunkSize + trim)..<(2 * chunkSize - trim)])
                tarWaves.append(trimmedL + trimmedR)
            }

            // Combine (stereo)
            var tarSignalL = [Float](repeating: 0, count: (nSample + pad))
            var tarSignalR = [Float](repeating: 0, count: (nSample + pad))
            for (waveIdx, wave) in tarWaves.enumerated() {
                let startIdx = waveIdx * genSize
                let halfLen = wave.count / 2
                for i in 0..<halfLen {
                    if startIdx + i < tarSignalL.count {
                        tarSignalL[startIdx + i] = wave[i]
                        tarSignalR[startIdx + i] = wave[halfLen + i]
                    }
                }
            }
            // Margin processing (stereo)
            let cutStart = segIdx == 0 ? 0 : actualMargin
            let cutEnd = segIdx == segments.count - 1 ? nSample : nSample - actualMargin

            if cutEnd > cutStart {
                let cutL = Array(tarSignalL[cutStart..<cutEnd])
                let cutR = Array(tarSignalR[cutStart..<cutEnd])
                chunkedSources.append(cutL + cutR)
            }

            // Progress update
            let progress = 0.2 + (Double(segIdx + 1) / Double(segments.count)) * 0.7
            progressHandler?(progress, "ボーカルを抽出中... (\(segIdx + 1)/\(segments.count))")
            logger.info("   進捗: \(segIdx + 1)/\(segments.count)")
        }

        // chunkedSources: [[L1...Ln, R1...Rn], [L1'...Ln', R1'...Rn'], ...]
        // Return: [all L samples, all R samples]
        var resultL = [Float]()
        var resultR = [Float]()
        for chunk in chunkedSources {
            let halfLen = chunk.count / 2
            resultL.append(contentsOf: chunk[0..<halfLen])
            resultR.append(contentsOf: chunk[halfLen..<chunk.count])
        }
        return resultL + resultR
    }

    /// STFT: PyTorch compatible (center=True)
    private func stft(left: [Float], right: [Float]) -> MLMultiArray {
        let inputArray = try! MLMultiArray(shape: [1, 4, dimF, dimT] as [NSNumber], dataType: .float32)

        let numFrames = dimT
        let pad = nFFT / 2  // center=True パディング

        // center=True: 前後に nFFT/2 のパディングを追加
        let leftPadded = [Float](repeating: 0, count: pad) + left + [Float](repeating: 0, count: pad)
        let rightPadded = [Float](repeating: 0, count: pad) + right + [Float](repeating: 0, count: pad)

        for t in 0..<numFrames {
            let start = t * hop

            // Windowed segments
            var leftWindowed = [Float](repeating: 0, count: nFFT)
            var rightWindowed = [Float](repeating: 0, count: nFFT)

            for i in 0..<nFFT {
                leftWindowed[i] = leftPadded[start + i] * window[i]
                rightWindowed[i] = rightPadded[start + i] * window[i]
            }

            // DFT
            var leftReal = [Float](repeating: 0, count: nFFT)
            var leftImag = [Float](repeating: 0, count: nFFT)
            var rightReal = [Float](repeating: 0, count: nFFT)
            var rightImag = [Float](repeating: 0, count: nFFT)
            var zeroImag = [Float](repeating: 0, count: nFFT)

            vDSP_DFT_Execute(dftSetupForward!, &leftWindowed, &zeroImag, &leftReal, &leftImag)
            vDSP_DFT_Execute(dftSetupForward!, &rightWindowed, &zeroImag, &rightReal, &rightImag)

            // Store in MLMultiArray (up to dim_f)
            for f in 0..<dimF {
                inputArray[[0, 0, f, t] as [NSNumber]] = NSNumber(value: leftReal[f])
                inputArray[[0, 1, f, t] as [NSNumber]] = NSNumber(value: leftImag[f])
                inputArray[[0, 2, f, t] as [NSNumber]] = NSNumber(value: rightReal[f])
                inputArray[[0, 3, f, t] as [NSNumber]] = NSNumber(value: rightImag[f])
            }
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
    private func istft(_ specPred: MLMultiArray) -> [Float] {
        var audio = [Float](repeating: 0, count: 2 * chunkSize)
        let pad = nFFT / 2  // center=True パディング
        let paddedLen = chunkSize + nFFT  // パディング後の長さ

        for ch in 0..<2 {
            var paddedAudio = [Float](repeating: 0, count: paddedLen)
            var windowSum = [Float](repeating: 0, count: paddedLen)

            for t in 0..<dimT {
                var realFull = [Float](repeating: 0, count: nFFT)
                var imagFull = [Float](repeating: 0, count: nFFT)

                // モデル出力からスペクトルを取得 (0 to dimF-1)
                for f in 0..<dimF {
                    let realIdx = (ch * 2)  // ch=0: 0(L_real), ch=1: 2(R_real)
                    let imagIdx = (ch * 2 + 1)  // ch=0: 1(L_imag), ch=1: 3(R_imag)
                    realFull[f] = specPred[[0, realIdx, f, t] as [NSNumber]].floatValue
                    imagFull[f] = specPred[[0, imagIdx, f, t] as [NSNumber]].floatValue
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
                let start = t * hop
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
