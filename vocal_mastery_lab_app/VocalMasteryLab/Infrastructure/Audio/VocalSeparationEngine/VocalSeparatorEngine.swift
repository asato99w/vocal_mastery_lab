import Foundation
import CoreML
import Accelerate
import AVFoundation
import os.log

private let logger = Logger(subsystem: "com.kazuasato.VocalMasteryLab", category: "VocalSeparatorEngine")

/// Vocal separation engine using CoreML and STFT
///
/// Integrates AVAudioFile + STFT + CoreML for end-to-end vocal extraction
final class VocalSeparatorEngine {

    // MARK: - Properties

    private let model: MLModel
    private let stftProcessor: STFTProcessorV2
    private let configuration: ModelConfiguration

    // MARK: - Types

    struct ModelConfiguration {
        let fftSize: Int
        let hopSize: Int
        let sampleRate: Double
        let chunkSize: Int

        static let `default` = ModelConfiguration(
            fftSize: 4096,
            hopSize: 1024,
            sampleRate: 44100,
            chunkSize: 256
        )
    }

    struct SeparationResult {
        let vocals: AudioProcessor.AudioData
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
        let compiledURL: URL

        // Check if model is already compiled (.mlmodelc) or needs compilation (.mlpackage)
        if modelURL.pathExtension == "mlmodelc" {
            // Already compiled, use directly
            compiledURL = modelURL
            logger.info("📦 [MODEL] Using pre-compiled model: \(modelURL.lastPathComponent)")
        } else {
            // Needs compilation
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

        self.configuration = configuration
        self.stftProcessor = STFTProcessorV2(
            fftSize: configuration.fftSize,
            hopSize: configuration.hopSize
        )
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
        let audioData = try AudioProcessor.loadAudio(
            from: audioURL,
            targetSampleRate: configuration.sampleRate
        )
        logger.info("📥 [AUDIO_LOADED] channels=\(audioData.channelCount), frames=\(audioData.frameCount), sampleRate=\(audioData.sampleRate)")

        // 2. Convert to stereo if needed
        let stereoAudio = AudioProcessor.convertToStereo(audioData)

        // Log audio stats (POC compatible format)
        let leftStats = computeStats(stereoAudio.samples[0])
        logger.info("📊 [LEFT_AUDIO_STATS] min=\(leftStats.min), max=\(leftStats.max), mean=\(leftStats.mean), rms=\(leftStats.rms)")

        // Log first 10 samples for debugging
        let firstSamples = Array(stereoAudio.samples[0].prefix(10))
        logger.info("🔢 [FIRST_10_SAMPLES] \(firstSamples)")

        // 3. Compute STFT (magnitude + phase) - POC compatible format
        progressHandler?(0.2, "音声を解析中...")
        let (leftSTFT, rightSTFT) = stftProcessor.computeSTFT(audioData: stereoAudio)
        logger.info("📈 [STFT] freqBins=\(leftSTFT.frequencyBins), timeFrames=\(leftSTFT.timeFrames)")

        // 4. CoreML inference with magnitude input (POC compatible)
        let vocalMask = try predictVocalMask(
            leftSTFT: leftSTFT,
            rightSTFT: rightSTFT,
            progressHandler: progressHandler
        )

        // Log mask statistics
        let maskStats = computeSpectrogramStats(vocalMask.magnitude)
        logger.info("🎭 [VOCAL_MASK_STATS] min=\(maskStats.min), max=\(maskStats.max), mean=\(maskStats.mean)")

        // 5. Apply mask
        progressHandler?(0.9, "出力を生成中...")
        let vocalSpec = applyComplexMask(spectrogram: leftSTFT, mask: vocalMask)

        // Log vocal spectrogram stats
        let vocalSpecStats = computeSpectrogramStats(vocalSpec.magnitude)
        let origSpecStats = computeSpectrogramStats(leftSTFT.magnitude)
        logger.info("📊 [ORIG_SPEC_STATS] min=\(origSpecStats.min), max=\(origSpecStats.max), mean=\(origSpecStats.mean)")
        logger.info("🎤 [VOCAL_SPEC_STATS] min=\(vocalSpecStats.min), max=\(vocalSpecStats.max), mean=\(vocalSpecStats.mean)")

        // 6. Compute iSTFT
        let vocals = stftProcessor.createAudioData(
            leftMagnitude: vocalSpec.magnitude,
            leftPhase: vocalSpec.phase,
            rightMagnitude: vocalSpec.magnitude,
            rightPhase: vocalSpec.phase,
            sampleRate: configuration.sampleRate
        )

        // Log output audio stats
        let outputStats = computeStats(vocals.samples[0])
        logger.info("🎵 [OUTPUT_AUDIO_STATS] min=\(outputStats.min), max=\(outputStats.max), mean=\(outputStats.mean), rms=\(outputStats.rms)")
        logger.info("✅ [SEPARATION_COMPLETE] outputFrames=\(vocals.frameCount)")

        progressHandler?(1.0, "完了")

        return SeparationResult(vocals: vocals)
    }

    /// Save separated vocals to file
    func save(result: SeparationResult, to url: URL) throws {
        let normalizedVocals = AudioProcessor.normalize(result.vocals)
        try AudioProcessor.saveAudio(normalizedVocals, to: url)
    }

    // MARK: - Private Methods

    /// CoreML inference with magnitude input (POC compatible format)
    private func predictVocalMask(
        leftSTFT: STFTProcessorV2.SpectrogramData,
        rightSTFT: STFTProcessorV2.SpectrogramData,
        progressHandler: ProgressHandler?
    ) throws -> STFTProcessorV2.SpectrogramData {

        // Reset chunk counter for debugging
        chunkCounter = 0

        let timeFrames = leftSTFT.timeFrames
        let freqBins = min(leftSTFT.frequencyBins, 2048)
        let chunkSize = configuration.chunkSize

        let numChunks = (timeFrames + chunkSize - 1) / chunkSize
        logger.info("🤖 [COREML_START] timeFrames=\(timeFrames), freqBins=\(freqBins), chunkSize=\(chunkSize), numChunks=\(numChunks)")

        var vocalMasks: [[Float]] = []

        for chunkIndex in 0..<numChunks {
            let startFrame = chunkIndex * chunkSize
            let endFrame = min((chunkIndex + 1) * chunkSize, timeFrames)

            // Extract chunk with magnitude data (POC compatible)
            let chunk = extractChunk(
                leftSTFT: leftSTFT,
                rightSTFT: rightSTFT,
                startFrame: startFrame,
                endFrame: endFrame,
                targetSize: chunkSize
            )

            // Predict
            let output = try predictChunk(chunk)

            // Extract vocal mask (Channel 0 = vocals)
            let actualSize = endFrame - startFrame
            let vocalChunk = extractChannelMask(output, channel: 0, actualSize: actualSize)
            vocalMasks.append(contentsOf: vocalChunk)

            // Update progress (20% - 90% range for inference)
            let progress = 0.2 + (Double(chunkIndex + 1) / Double(numChunks)) * 0.7
            progressHandler?(progress, "ボーカルを抽出中... (\(chunkIndex + 1)/\(numChunks))")
        }

        // Reshape results
        let vocalMagnitude = reshape2D(vocalMasks, frequencyBins: freqBins)

        // Use original phase for iSTFT reconstruction
        let vocalPhase = trimPhase(leftSTFT.phase, targetBins: freqBins)

        return STFTProcessorV2.SpectrogramData(
            magnitude: vocalMagnitude,
            phase: vocalPhase
        )
    }

    /// Extract chunk for CoreML input (POC compatible: magnitude with zero imaginary)
    private func extractChunk(
        leftSTFT: STFTProcessorV2.SpectrogramData,
        rightSTFT: STFTProcessorV2.SpectrogramData,
        startFrame: Int,
        endFrame: Int,
        targetSize: Int
    ) -> MLMultiArray {

        let freqBins = 2048
        let actualSize = endFrame - startFrame

        // Create MLMultiArray [1, 4, 2048, 256]
        // POC compatible format: [Left Magnitude, 0, Right Magnitude, 0]
        let inputArray = try! MLMultiArray(
            shape: [1, 4, freqBins, targetSize] as [NSNumber],
            dataType: .float32
        )

        // Copy magnitude data (POC compatible: imaginary channels are zero)
        for t in 0..<targetSize {
            for f in 0..<freqBins {
                if t < actualSize && f < leftSTFT.frequencyBins {
                    // Channel 0: Left magnitude
                    inputArray[[0, 0, f, t] as [NSNumber]] = NSNumber(value: leftSTFT.magnitude[f][startFrame + t])
                    // Channel 1: Left imaginary = 0 (POC compatible)
                    inputArray[[0, 1, f, t] as [NSNumber]] = 0
                    // Channel 2: Right magnitude (same as left in POC)
                    inputArray[[0, 2, f, t] as [NSNumber]] = NSNumber(value: leftSTFT.magnitude[f][startFrame + t])
                    // Channel 3: Right imaginary = 0 (POC compatible)
                    inputArray[[0, 3, f, t] as [NSNumber]] = 0
                } else {
                    // Zero padding for out-of-bounds
                    inputArray[[0, 0, f, t] as [NSNumber]] = 0
                    inputArray[[0, 1, f, t] as [NSNumber]] = 0
                    inputArray[[0, 2, f, t] as [NSNumber]] = 0
                    inputArray[[0, 3, f, t] as [NSNumber]] = 0
                }
            }
        }

        return inputArray
    }

    private var chunkCounter = 0

    private func predictChunk(_ input: MLMultiArray) throws -> MLMultiArray {
        chunkCounter += 1

        // Log input stats (sample first few values) - POC compatible format
        let inputStats = computeMLArrayStats(input)
        if self.chunkCounter <= 3 {
            logger.info("🔢 [MODEL_INPUT] chunk=\(self.chunkCounter), shape=\(input.shape), min=\(inputStats.min), max=\(inputStats.max), mean=\(inputStats.mean)")

            // Sample first channel values for debugging
            var sampleValues: [Float] = []
            for f in 0..<min(5, 2048) {
                let value = input[[0, 0, f, 0] as [NSNumber]].floatValue
                sampleValues.append(value)
            }
            logger.info("🔢 [INPUT_CH0_SAMPLE] first 5 freq bins: \(sampleValues)")
        }

        let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input_1": MLFeatureValue(multiArray: input)
        ])

        let output = try model.prediction(from: inputProvider)

        guard let outputArray = output.featureValue(for: "var_992")?.multiArrayValue else {
            throw SeparationError.predictionFailed("Failed to get output")
        }

        // Log output stats - POC compatible format
        let outputStats = computeMLArrayStats(outputArray)
        if self.chunkCounter <= 3 {
            logger.info("🔢 [MODEL_OUTPUT] chunk=\(self.chunkCounter), shape=\(outputArray.shape), min=\(outputStats.min), max=\(outputStats.max), mean=\(outputStats.mean)")

            // Sample output channel values for debugging
            for channel in 0..<4 {
                var sampleValues: [Float] = []
                for f in 0..<min(5, 2048) {
                    let value = outputArray[[0, channel, f, 0] as [NSNumber]].floatValue
                    sampleValues.append(value)
                }
                logger.info("🔢 [OUTPUT_CH\(channel)_SAMPLE] first 5 freq bins: \(sampleValues)")
            }
        }

        return outputArray
    }

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

        // DEBUG: Log mask statistics (POC compatible)
        if mask.count > 0 && mask[0].count > 0 && chunkCounter <= 3 {
            let firstValue = mask[0][0]
            let avgValue = mask[0].reduce(0, +) / Float(mask[0].count)
            let maxValue = mask[0].max() ?? 0
            let minValue = mask[0].min() ?? 0
            logger.info("🎭 [MASK_CH\(channel)] first=\(firstValue), avg=\(avgValue), min=\(minValue), max=\(maxValue)")
        }

        return mask
    }

    private func reshape2D(_ flatData: [[Float]], frequencyBins: Int) -> [[Float]] {
        var result: [[Float]] = Array(repeating: [], count: frequencyBins)

        for frame in flatData {
            for (f, value) in frame.enumerated() where f < frequencyBins {
                result[f].append(value)
            }
        }

        return result
    }

    private func trimPhase(_ phase: [[Float]], targetBins: Int) -> [[Float]] {
        let timeFrames = phase[0].count
        var trimmed: [[Float]] = Array(
            repeating: Array(repeating: 0, count: timeFrames),
            count: targetBins
        )

        let actualBins = min(phase.count, targetBins)
        for f in 0..<actualBins {
            trimmed[f] = phase[f]
        }

        return trimmed
    }

    private func applyComplexMask(
        spectrogram: STFTProcessorV2.SpectrogramData,
        mask: STFTProcessorV2.SpectrogramData
    ) -> STFTProcessorV2.SpectrogramData {

        let freqBins = min(spectrogram.frequencyBins, mask.frequencyBins)
        let timeFrames = min(spectrogram.timeFrames, mask.timeFrames)

        var maskedMagnitude: [[Float]] = Array(
            repeating: Array(repeating: 0, count: timeFrames),
            count: freqBins
        )

        var maskedPhase: [[Float]] = Array(
            repeating: Array(repeating: 0, count: timeFrames),
            count: freqBins
        )

        for f in 0..<freqBins {
            for t in 0..<timeFrames {
                maskedMagnitude[f][t] = spectrogram.magnitude[f][t] * mask.magnitude[f][t]
                maskedPhase[f][t] = spectrogram.phase[f][t]
            }
        }

        return STFTProcessorV2.SpectrogramData(
            magnitude: maskedMagnitude,
            phase: maskedPhase
        )
    }

    // MARK: - Statistics Helpers

    private struct Stats {
        let min: Float
        let max: Float
        let mean: Float
        let rms: Float
    }

    private func computeStats(_ samples: [Float]) -> Stats {
        guard !samples.isEmpty else {
            return Stats(min: 0, max: 0, mean: 0, rms: 0)
        }
        let minVal = samples.min() ?? 0
        let maxVal = samples.max() ?? 0
        let sum = samples.reduce(0, +)
        let mean = sum / Float(samples.count)
        let sumSquared = samples.reduce(0) { $0 + $1 * $1 }
        let rms = sqrtf(sumSquared / Float(samples.count))
        return Stats(min: minVal, max: maxVal, mean: mean, rms: rms)
    }

    private func computeSpectrogramStats(_ spectrogram: [[Float]]) -> Stats {
        var allValues: [Float] = []
        for bin in spectrogram {
            allValues.append(contentsOf: bin)
        }
        return computeStats(allValues)
    }

    private func computeMLArrayStats(_ array: MLMultiArray) -> Stats {
        let count = array.count
        guard count > 0 else {
            return Stats(min: 0, max: 0, mean: 0, rms: 0)
        }

        var minVal: Float = .greatestFiniteMagnitude
        var maxVal: Float = -.greatestFiniteMagnitude
        var sum: Float = 0

        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            let val = ptr[i]
            minVal = Swift.min(minVal, val)
            maxVal = Swift.max(maxVal, val)
            sum += val
        }

        let mean = sum / Float(count)
        return Stats(min: minVal, max: maxVal, mean: mean, rms: 0)
    }
}
