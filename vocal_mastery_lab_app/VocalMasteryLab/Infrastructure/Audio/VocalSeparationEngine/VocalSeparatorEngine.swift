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

    private let nFFT: Int = 7680
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
            fftSize: 7680,
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
        progressHandler?(0.2, "ボーカルを抽出中...")
        let vocalsExtracted = try demix(left: left, right: right, denoise: true, progressHandler: progressHandler)

        // 3. Voc_FTはボーカルモデル → モデル出力がボーカル
        // 左右チャンネルは同じ処理結果を使用（モノラル処理）
        let vocalsLeft = vocalsExtracted
        let vocalsRight = vocalsExtracted

        let vocals = AudioProcessor.AudioData(
            samples: [vocalsLeft, vocalsRight],
            sampleRate: Double(targetSampleRate),
            frameCount: vocalsLeft.count
        )

        // 4. 伴奏を計算: instrumental = original - vocals
        progressHandler?(0.95, "伴奏を計算中...")
        let frameCount = min(left.count, vocalsExtracted.count)
        var instrumentalLeft = [Float](repeating: 0, count: frameCount)
        var instrumentalRight = [Float](repeating: 0, count: frameCount)

        // vDSP_vsub: C = B - A (第1引数を第2引数から引く)
        vDSP_vsub(vocalsExtracted, 1, left, 1, &instrumentalLeft, 1, vDSP_Length(frameCount))
        vDSP_vsub(vocalsExtracted, 1, right, 1, &instrumentalRight, 1, vDSP_Length(frameCount))

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
    func save(result: SeparationResult, vocalsURL: URL, instrumentalURL: URL? = nil) throws {
        // Save vocals
        let normalizedVocals = AudioProcessor.normalize(result.vocals)
        try AudioProcessor.saveAudio(normalizedVocals, to: vocalsURL)
        logger.info("🎤 [SAVED] Vocals: \(vocalsURL.lastPathComponent)")

        // Save instrumental if URL provided
        if let instURL = instrumentalURL {
            let normalizedInstrumental = AudioProcessor.normalize(result.instrumental)
            try AudioProcessor.saveAudio(normalizedInstrumental, to: instURL)
            logger.info("🎸 [SAVED] Instrumental: \(instURL.lastPathComponent)")
        }
    }

    /// Save separated vocals to file (backward compatibility)
    func save(result: SeparationResult, to url: URL) throws {
        try save(result: result, vocalsURL: url, instrumentalURL: nil)
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

                let waves = istft(specPred)
                tarWaves.append(waves)
            }

            // Combine
            var tarSignal = [Float](repeating: 0, count: (nSample + pad))
            for (waveIdx, wave) in tarWaves.enumerated() {
                let startIdx = waveIdx * genSize
                let trimmedWave = Array(wave[trim..<(wave.count - trim)])
                for (i, val) in trimmedWave.enumerated() {
                    if startIdx + i < tarSignal.count {
                        tarSignal[startIdx + i] = val
                    }
                }
            }
            tarSignal = Array(tarSignal[0..<nSample])

            // Margin processing
            let cutStart = segIdx == 0 ? 0 : actualMargin
            let cutEnd = segIdx == segments.count - 1 ? tarSignal.count : tarSignal.count - actualMargin

            if cutEnd > cutStart {
                chunkedSources.append(Array(tarSignal[cutStart..<cutEnd]))
            }

            // Progress update
            let progress = 0.2 + (Double(segIdx + 1) / Double(segments.count)) * 0.7
            progressHandler?(progress, "ボーカルを抽出中... (\(segIdx + 1)/\(segments.count))")
            logger.info("   進捗: \(segIdx + 1)/\(segments.count)")
        }

        return chunkedSources.flatMap { $0 }
    }

    /// STFT: PyTorch compatible
    private func stft(left: [Float], right: [Float]) -> MLMultiArray {
        let inputArray = try! MLMultiArray(shape: [1, 4, dimF, dimT] as [NSNumber], dataType: .float32)

        let numFrames = dimT

        for t in 0..<numFrames {
            let startIdx = t * hop
            let endIdx = min(startIdx + nFFT, left.count)

            // Left channel
            var leftFrame = [Float](repeating: 0, count: nFFT)
            for i in 0..<min(nFFT, endIdx - startIdx) {
                leftFrame[i] = left[startIdx + i]
            }

            // Right channel
            var rightFrame = [Float](repeating: 0, count: nFFT)
            for i in 0..<min(nFFT, endIdx - startIdx) {
                rightFrame[i] = right[startIdx + i]
            }

            // Apply window
            var leftWindowed = [Float](repeating: 0, count: nFFT)
            var rightWindowed = [Float](repeating: 0, count: nFFT)
            vDSP_vmul(leftFrame, 1, window, 1, &leftWindowed, 1, vDSP_Length(nFFT))
            vDSP_vmul(rightFrame, 1, window, 1, &rightWindowed, 1, vDSP_Length(nFFT))

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

    /// iSTFT: PyTorch compatible
    private func istft(_ specPred: MLMultiArray) -> [Float] {
        var output = [Float](repeating: 0, count: chunkSize)
        var windowSum = [Float](repeating: 0, count: chunkSize)

        for t in 0..<dimT {
            var realFull = [Float](repeating: 0, count: nFFT)
            var imagFull = [Float](repeating: 0, count: nFFT)

            // Positive frequencies (left channel only)
            for f in 0..<dimF {
                realFull[f] = specPred[[0, 0, f, t] as [NSNumber]].floatValue
                imagFull[f] = specPred[[0, 1, f, t] as [NSNumber]].floatValue
            }

            // Frequency padding (dim_f to nBins)
            for f in dimF..<nBins {
                realFull[f] = 0
                imagFull[f] = 0
            }

            // Negative frequencies (conjugate symmetry)
            for f in 1..<(nBins - 1) {
                let mirrorIdx = nFFT - f
                realFull[mirrorIdx] = realFull[f]
                imagFull[mirrorIdx] = -imagFull[f]
            }

            // IDFT
            var realOut = [Float](repeating: 0, count: nFFT)
            var imagOut = [Float](repeating: 0, count: nFFT)
            vDSP_DFT_Execute(dftSetupInverse!, &realFull, &imagFull, &realOut, &imagOut)

            // Scaling (1/N)
            var scale = 1.0 / Float(nFFT)
            vDSP_vsmul(realOut, 1, &scale, &realOut, 1, vDSP_Length(nFFT))

            // OLA
            let startIdx = t * hop
            for i in 0..<nFFT {
                let outIdx = startIdx + i
                if outIdx < chunkSize {
                    output[outIdx] += realOut[i] * window[i]
                    windowSum[outIdx] += window[i] * window[i]
                }
            }
        }

        // Normalization
        for i in 0..<chunkSize {
            if windowSum[i] > 1e-8 {
                output[i] /= windowSum[i]
            }
        }

        return output
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
