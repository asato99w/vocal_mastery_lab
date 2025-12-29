import Foundation
import CoreML
import Accelerate
import VocalisDomain

/// FCPE (Fast Context-based Pitch Estimation) strategy using CoreML
/// Provides neural network based pitch detection with high accuracy
public final class FCPEStrategy: PitchDetectionStrategy {

    // MARK: - Constants

    private enum Constants {
        static let targetSampleRate: Double = 16000.0
        static let nMels = 128
        static let nFFT = 1024
        static let winSize = 1024
        static let hopLength = 160  // 10ms at 16kHz
        static let fmin: Float = 0
        static let fmax: Float = 8000
        static let clipVal: Float = 1e-5

        // FCPE decoder constants
        static let f0Min: Double = 32.7
        static let f0Max: Double = 1975.5
        static let outDims = 360
        static let voicedThreshold: Float = 0.006

        // Chunking constants for long audio processing
        // Process in 15-second chunks - CoreML model has issues with 30s (3000 frames)
        // Shorter chunks produce valid output while still maintaining efficiency
        static let maxChunkDurationSeconds: Double = 15.0
        static let chunkOverlapSeconds: Double = 0.5  // 500ms overlap for continuity
    }

    // MARK: - PitchDetectionStrategy

    public let name: String = "FCPE"

    /// FCPE uses neural network with good octave stability, no correction needed
    public let requiresOctaveCorrection: Bool = false

    // MARK: - Properties

    private var model: MLModel?
    private var melFilterbank: [[Float]]?
    private let centTable: [Float]

    // MARK: - Initialization

    public init() {
        // Pre-compute cent table for decoder
        let f0MelMin = 1200.0 * log2(Constants.f0Min / 10.0)
        let f0MelMax = 1200.0 * log2(Constants.f0Max / 10.0)
        self.centTable = (0..<Constants.outDims).map { i in
            Float(f0MelMin + (f0MelMax - f0MelMin) * Double(i) / Double(Constants.outDims - 1))
        }

        // Load filterbank first (needed for model warmup), then model
        loadMelFilterbankIfNeeded()
        loadModelIfNeeded()
    }

    // MARK: - Pitch Detection

    public func detectPitch(samples: [Float], sampleRate: Double) -> [PitchFrame] {
        FileLogger.shared.log(level: "INFO", category: "FCPE", message: "detectPitch called: \(samples.count) samples at \(sampleRate)Hz")

        guard !samples.isEmpty else {
            FileLogger.shared.log(level: "WARN", category: "FCPE", message: "Empty samples array")
            return []
        }

        guard let model = model, let melFilterbank = melFilterbank else {
            let modelStatus = model == nil ? "nil" : "loaded"
            let filterbankStatus = melFilterbank == nil ? "nil" : "loaded"
            FileLogger.shared.log(level: "ERROR", category: "FCPE", message: "Model or filterbank not loaded - model: \(modelStatus), filterbank: \(filterbankStatus)")
            print("[FCPE] Model or filterbank not loaded")
            return []
        }

        FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Model and filterbank ready")

        // 1. Resample to 16kHz if needed
        let resampled = resample(samples, from: sampleRate, to: Constants.targetSampleRate)
        FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Resampled: \(samples.count) -> \(resampled.count) samples")

        // Check if chunking is needed
        let audioDurationSeconds = Double(resampled.count) / Constants.targetSampleRate
        if audioDurationSeconds > Constants.maxChunkDurationSeconds {
            FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Long audio detected (\(String(format: "%.1f", audioDurationSeconds))s), using chunked processing")
            return detectPitchChunked(resampled: resampled, model: model, progress: nil)
        }

        // Short audio: process directly
        return detectPitchDirect(resampled: resampled, model: model)
    }

    /// Async version with progress reporting - overrides default extension for FCPE
    public func detectPitch(samples: [Float], sampleRate: Double, progress: PitchDetectionProgressCallback?) async -> [PitchFrame] {
        FileLogger.shared.log(level: "INFO", category: "FCPE", message: "detectPitch (async) called: \(samples.count) samples at \(sampleRate)Hz")

        await progress?(0.0)

        guard !samples.isEmpty else {
            FileLogger.shared.log(level: "WARN", category: "FCPE", message: "Empty samples array")
            await progress?(1.0)
            return []
        }

        guard let model = model, let melFilterbank = melFilterbank else {
            let modelStatus = model == nil ? "nil" : "loaded"
            let filterbankStatus = melFilterbank == nil ? "nil" : "loaded"
            FileLogger.shared.log(level: "ERROR", category: "FCPE", message: "Model or filterbank not loaded - model: \(modelStatus), filterbank: \(filterbankStatus)")
            print("[FCPE] Model or filterbank not loaded")
            await progress?(1.0)
            return []
        }

        FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Model and filterbank ready")

        // 1. Resample to 16kHz if needed
        let resampled = resample(samples, from: sampleRate, to: Constants.targetSampleRate)
        FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Resampled: \(samples.count) -> \(resampled.count) samples")

        // Check if chunking is needed
        let audioDurationSeconds = Double(resampled.count) / Constants.targetSampleRate
        if audioDurationSeconds > Constants.maxChunkDurationSeconds {
            FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Long audio detected (\(String(format: "%.1f", audioDurationSeconds))s), using chunked processing with progress")
            let result = await detectPitchChunkedAsync(resampled: resampled, model: model, progress: progress)
            await progress?(1.0)
            return result
        }

        // Short audio: process directly (report progress at start and end)
        await progress?(0.1)  // Small progress to show we started
        let result = detectPitchDirect(resampled: resampled, model: model)
        await progress?(1.0)
        return result
    }

    // MARK: - Direct Processing (for short audio)

    private func detectPitchDirect(resampled: [Float], model: MLModel) -> [PitchFrame] {
        do {
            // 1. Calculate per-frame amplitudes from resampled audio
            let frameAmplitudes = calculateFrameAmplitudes(resampled)
            FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Calculated \(frameAmplitudes.count) frame amplitudes")

            // 2. Compute Mel spectrogram
            let melSpec = computeMelSpectrogram(resampled)
            FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Mel spectrogram: \(melSpec.count) frames x \(melSpec.first?.count ?? 0) mels")

            guard !melSpec.isEmpty else {
                FileLogger.shared.log(level: "WARN", category: "FCPE", message: "Empty mel spectrogram")
                return []
            }

            // 3. Run CoreML inference
            let logits = try runInference(melSpec: melSpec, model: model)
            FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Inference complete: \(logits.count) frames x \(logits.first?.count ?? 0) bins")

            // Log sample logit values for debugging
            if let firstFrame = logits.first {
                let maxLogit = firstFrame.max() ?? 0
                let minLogit = firstFrame.min() ?? 0
                FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Logit range: min=\(minLogit), max=\(maxLogit), threshold=\(Constants.voicedThreshold)")
            }

            // 4. Decode logits to F0
            let f0Hz = decodeLogits(logits)
            let voicedCount = f0Hz.filter { $0 > 0 }.count
            FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Decoded: \(f0Hz.count) frames, \(voicedCount) voiced (\(String(format: "%.1f", Double(voicedCount) / Double(max(1, f0Hz.count)) * 100))%)")

            // Log frequency range if voiced frames exist
            let voicedFreqs = f0Hz.filter { $0 > 0 }
            if !voicedFreqs.isEmpty {
                let minFreq = voicedFreqs.min() ?? 0
                let maxFreq = voicedFreqs.max() ?? 0
                FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Frequency range: \(String(format: "%.1f", minFreq))Hz - \(String(format: "%.1f", maxFreq))Hz")
            }

            // 5. Convert to PitchFrames with actual amplitudes
            let frames = buildPitchFrames(f0Hz: f0Hz, amplitudes: frameAmplitudes)
            let voicedFrames = frames.filter { $0.frequency != nil }
            FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Result: \(frames.count) PitchFrames, \(voicedFrames.count) with pitch data")

            return frames
        } catch {
            FileLogger.shared.log(level: "ERROR", category: "FCPE", message: "Error: \(error)")
            print("[FCPE] Error: \(error)")
            return []
        }
    }

    // MARK: - Chunked Processing (for long audio)

    /// Synchronous chunked processing (for backward compatibility)
    private func detectPitchChunked(resampled: [Float], model: MLModel, progress: PitchDetectionProgressCallback?) -> [PitchFrame] {
        let chunkSamples = Int(Constants.maxChunkDurationSeconds * Constants.targetSampleRate)
        let overlapSamples = Int(Constants.chunkOverlapSeconds * Constants.targetSampleRate)
        let stepSamples = chunkSamples - overlapSamples

        // Calculate total number of chunks for progress
        let totalChunks = max(1, Int(ceil(Double(resampled.count - chunkSamples) / Double(stepSamples))) + 1)

        var allFrames: [PitchFrame] = []
        var chunkIndex = 0
        var sampleOffset = 0

        while sampleOffset < resampled.count {
            let chunkEnd = min(sampleOffset + chunkSamples, resampled.count)
            let chunk = Array(resampled[sampleOffset..<chunkEnd])

            FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Processing chunk \(chunkIndex): samples \(sampleOffset)-\(chunkEnd) (\(chunk.count) samples)")

            do {
                // Process this chunk
                let frameAmplitudes = calculateFrameAmplitudes(chunk)
                let melSpec = computeMelSpectrogram(chunk)

                guard !melSpec.isEmpty else {
                    FileLogger.shared.log(level: "WARN", category: "FCPE", message: "Empty mel spectrogram for chunk \(chunkIndex)")
                    sampleOffset += stepSamples
                    chunkIndex += 1
                    continue
                }

                let logits = try runInference(melSpec: melSpec, model: model)
                let f0Hz = decodeLogits(logits)

                // Log voiced/unvoiced statistics
                let voicedInChunk = f0Hz.filter { $0 > 0 }.count
                FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Chunk \(chunkIndex) decoded: \(f0Hz.count) frames, \(voicedInChunk) voiced (\(String(format: "%.1f", Double(voicedInChunk) / Double(max(1, f0Hz.count)) * 100))%)")

                // Calculate time offset for this chunk
                let timeOffset = Double(sampleOffset) / Constants.targetSampleRate

                // Build frames with correct timestamps
                let chunkFrames = buildPitchFrames(f0Hz: f0Hz, amplitudes: frameAmplitudes, timeOffset: timeOffset)

                // For first chunk, add all frames
                // For subsequent chunks, skip overlap region to avoid duplicates
                if chunkIndex == 0 {
                    allFrames.append(contentsOf: chunkFrames)
                } else {
                    // Skip frames in the overlap region (already processed in previous chunk)
                    let overlapFrames = Int(Constants.chunkOverlapSeconds / (Double(Constants.hopLength) / Constants.targetSampleRate))
                    let framesToSkip = min(overlapFrames, chunkFrames.count)
                    let newFrames = Array(chunkFrames.dropFirst(framesToSkip))
                    allFrames.append(contentsOf: newFrames)
                }

                FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Chunk \(chunkIndex) processed: \(chunkFrames.count) frames, total: \(allFrames.count)")

            } catch {
                FileLogger.shared.log(level: "ERROR", category: "FCPE", message: "Error processing chunk \(chunkIndex): \(error)")
            }

            sampleOffset += stepSamples
            chunkIndex += 1
        }

        let voicedFrames = allFrames.filter { $0.frequency != nil }
        FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Chunked processing complete: \(allFrames.count) total frames, \(voicedFrames.count) voiced (\(String(format: "%.1f", Double(voicedFrames.count) / Double(max(1, allFrames.count)) * 100))%)")

        return allFrames
    }

    /// Async chunked processing with progress reporting
    private func detectPitchChunkedAsync(resampled: [Float], model: MLModel, progress: PitchDetectionProgressCallback?) async -> [PitchFrame] {
        let chunkSamples = Int(Constants.maxChunkDurationSeconds * Constants.targetSampleRate)
        let overlapSamples = Int(Constants.chunkOverlapSeconds * Constants.targetSampleRate)
        let stepSamples = chunkSamples - overlapSamples

        // Calculate total number of chunks for progress
        let totalChunks = max(1, Int(ceil(Double(resampled.count - chunkSamples) / Double(stepSamples))) + 1)

        var allFrames: [PitchFrame] = []
        var chunkIndex = 0
        var sampleOffset = 0

        while sampleOffset < resampled.count {
            let chunkEnd = min(sampleOffset + chunkSamples, resampled.count)
            let chunk = Array(resampled[sampleOffset..<chunkEnd])

            FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Processing chunk \(chunkIndex)/\(totalChunks): samples \(sampleOffset)-\(chunkEnd) (\(chunk.count) samples)")

            do {
                // Process this chunk
                let frameAmplitudes = calculateFrameAmplitudes(chunk)
                let melSpec = computeMelSpectrogram(chunk)

                guard !melSpec.isEmpty else {
                    FileLogger.shared.log(level: "WARN", category: "FCPE", message: "Empty mel spectrogram for chunk \(chunkIndex)")
                    sampleOffset += stepSamples
                    chunkIndex += 1
                    continue
                }

                let logits = try runInference(melSpec: melSpec, model: model)
                let f0Hz = decodeLogits(logits)

                // Log voiced/unvoiced statistics
                let voicedInChunk = f0Hz.filter { $0 > 0 }.count
                FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Chunk \(chunkIndex) decoded: \(f0Hz.count) frames, \(voicedInChunk) voiced (\(String(format: "%.1f", Double(voicedInChunk) / Double(max(1, f0Hz.count)) * 100))%)")

                // Calculate time offset for this chunk
                let timeOffset = Double(sampleOffset) / Constants.targetSampleRate

                // Build frames with correct timestamps
                let chunkFrames = buildPitchFrames(f0Hz: f0Hz, amplitudes: frameAmplitudes, timeOffset: timeOffset)

                // For first chunk, add all frames
                // For subsequent chunks, skip overlap region to avoid duplicates
                if chunkIndex == 0 {
                    allFrames.append(contentsOf: chunkFrames)
                } else {
                    // Skip frames in the overlap region (already processed in previous chunk)
                    let overlapFrames = Int(Constants.chunkOverlapSeconds / (Double(Constants.hopLength) / Constants.targetSampleRate))
                    let framesToSkip = min(overlapFrames, chunkFrames.count)
                    let newFrames = Array(chunkFrames.dropFirst(framesToSkip))
                    allFrames.append(contentsOf: newFrames)
                }

                FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Chunk \(chunkIndex) processed: \(chunkFrames.count) frames, total: \(allFrames.count)")

                // Report progress after each chunk
                let currentProgress = Double(chunkIndex + 1) / Double(totalChunks)
                await progress?(currentProgress)
                FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Progress: \(String(format: "%.1f", currentProgress * 100))%")

            } catch {
                FileLogger.shared.log(level: "ERROR", category: "FCPE", message: "Error processing chunk \(chunkIndex): \(error)")
            }

            sampleOffset += stepSamples
            chunkIndex += 1
        }

        let voicedFrames = allFrames.filter { $0.frequency != nil }
        FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Chunked processing complete: \(allFrames.count) total frames, \(voicedFrames.count) voiced (\(String(format: "%.1f", Double(voicedFrames.count) / Double(max(1, allFrames.count)) * 100))%)")

        return allFrames
    }

    // MARK: - Model Loading

    private func loadModelIfNeeded() {
        guard model == nil else { return }

        FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Loading CoreML model...")

        // First try to load compiled model (.mlmodelc) - Xcode compiles .mlpackage to .mlmodelc at build time
        if let compiledModelURL = Bundle.main.url(
            forResource: "fcpe_core_fp32",
            withExtension: "mlmodelc"
        ) {
            FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Found compiled model: \(compiledModelURL.path)")
            do {
                model = try MLModel(contentsOf: compiledModelURL)
                FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Compiled model loaded successfully")
                print("[FCPE] Compiled model loaded successfully")
                // Perform warmup inference to ensure model is fully initialized
                warmupModel()
                return
            } catch {
                FileLogger.shared.log(level: "ERROR", category: "FCPE", message: "Failed to load compiled model: \(error)")
                print("[FCPE] Failed to load compiled model: \(error)")
            }
        }

        // Fallback: try to load source package and compile at runtime
        guard let modelURL = Bundle.main.url(
            forResource: "fcpe_core_fp32",
            withExtension: "mlpackage"
        ) else {
            FileLogger.shared.log(level: "ERROR", category: "FCPE", message: "Model not found in bundle (neither .mlmodelc nor .mlpackage)")
            print("[FCPE] Model not found in bundle")
            return
        }

        FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Found source model: \(modelURL.path)")

        do {
            let compiledURL = try MLModel.compileModel(at: modelURL)
            FileLogger.shared.log(level: "DEBUG", category: "FCPE", message: "Model compiled to: \(compiledURL.path)")
            model = try MLModel(contentsOf: compiledURL)
            FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Model loaded successfully")
            print("[FCPE] Model loaded successfully")
            // Perform warmup inference to ensure model is fully initialized
            warmupModel()
        } catch {
            FileLogger.shared.log(level: "ERROR", category: "FCPE", message: "Failed to load model: \(error)")
            print("[FCPE] Failed to load model: \(error)")
        }
    }

    /// Perform a warmup inference to ensure CoreML model is fully initialized
    /// This prevents the first real inference from returning all zeros
    private func warmupModel() {
        guard let model = model, let melFilterbank = melFilterbank else {
            FileLogger.shared.log(level: "WARN", category: "FCPE", message: "Cannot warmup: model or filterbank not loaded")
            return
        }

        FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Performing model warmup inference...")

        // Create a small dummy mel spectrogram (100 frames = 1 second at 10ms per frame)
        let warmupFrames = 100
        var dummyMelSpec = [[Float]]()
        for _ in 0..<warmupFrames {
            // Create a simple mel frame with some variation to avoid constant input
            var frame = [Float](repeating: 0.0, count: Constants.nMels)
            for m in 0..<Constants.nMels {
                frame[m] = Float.random(in: -10.0...0.0)  // Log-scale mel values
            }
            dummyMelSpec.append(frame)
        }

        do {
            // Run warmup inference to ensure CoreML model is fully initialized
            // This is necessary because the model can produce zeros on first long inference
            let _ = try runInference(melSpec: dummyMelSpec, model: model)
            FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Model warmup complete")
            print("[FCPE] Model warmup complete")
        } catch {
            FileLogger.shared.log(level: "ERROR", category: "FCPE", message: "Model warmup failed: \(error)")
            print("[FCPE] Model warmup failed: \(error)")
        }
    }

    private func loadMelFilterbankIfNeeded() {
        guard melFilterbank == nil else { return }

        FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Loading Mel filterbank...")

        guard let url = Bundle.main.url(
            forResource: "mel_filterbank_16k_128",
            withExtension: "bin"
        ) else {
            FileLogger.shared.log(level: "ERROR", category: "FCPE", message: "Mel filterbank not found in bundle")
            print("[FCPE] Mel filterbank not found in bundle")
            return
        }

        do {
            let data = try Data(contentsOf: url)
            let floatCount = data.count / MemoryLayout<Float>.size
            var floats = [Float](repeating: 0, count: floatCount)
            _ = floats.withUnsafeMutableBytes { data.copyBytes(to: $0) }

            // Reshape to 128 x 513
            let nBins = Constants.nFFT / 2 + 1  // 513
            var filterbank = [[Float]]()
            for i in 0..<Constants.nMels {
                let start = i * nBins
                let end = start + nBins
                filterbank.append(Array(floats[start..<end]))
            }
            melFilterbank = filterbank
            FileLogger.shared.log(level: "INFO", category: "FCPE", message: "Mel filterbank loaded: \(Constants.nMels) x \(nBins)")
            print("[FCPE] Mel filterbank loaded: \(Constants.nMels) x \(nBins)")
        } catch {
            FileLogger.shared.log(level: "ERROR", category: "FCPE", message: "Failed to load filterbank: \(error)")
            print("[FCPE] Failed to load filterbank: \(error)")
        }
    }

    // MARK: - Amplitude Calculation

    /// Calculate per-frame RMS amplitudes normalized to 0.0-1.0 range
    /// Uses the same hop size as FCPE to align with pitch frames
    private func calculateFrameAmplitudes(_ samples: [Float]) -> [Float] {
        let windowSize = Constants.winSize  // 1024 samples
        let hopSize = Constants.hopLength   // 160 samples (10ms at 16kHz)

        // First pass: calculate global max RMS for normalization
        var maxRms: Float = 0.0001  // Minimum to avoid division by zero
        var position = 0
        while position + windowSize <= samples.count {
            let chunk = Array(samples[position..<(position + windowSize)])
            let rms = sqrt(chunk.map { $0 * $0 }.reduce(0, +) / Float(chunk.count))
            maxRms = max(maxRms, rms)
            position += hopSize
        }

        // Second pass: calculate normalized amplitudes
        var amplitudes: [Float] = []
        position = 0
        while position + windowSize <= samples.count {
            let chunk = Array(samples[position..<(position + windowSize)])
            let rms = sqrt(chunk.map { $0 * $0 }.reduce(0, +) / Float(chunk.count))
            let normalizedAmplitude = min(1.0, rms / maxRms)
            amplitudes.append(normalizedAmplitude)
            position += hopSize
        }

        return amplitudes
    }

    // MARK: - Resampling

    private func resample(_ samples: [Float], from sourceSR: Double, to targetSR: Double) -> [Float] {
        guard sourceSR != targetSR else { return samples }

        let ratio = targetSR / sourceSR
        let outputLength = Int(Double(samples.count) * ratio)

        guard outputLength > 0 else { return [] }

        var output = [Float](repeating: 0, count: outputLength)

        // Simple linear interpolation resampling
        for i in 0..<outputLength {
            let srcIndex = Double(i) / ratio
            let srcIndexInt = Int(srcIndex)
            let frac = Float(srcIndex - Double(srcIndexInt))

            if srcIndexInt + 1 < samples.count {
                output[i] = samples[srcIndexInt] * (1 - frac) + samples[srcIndexInt + 1] * frac
            } else if srcIndexInt < samples.count {
                output[i] = samples[srcIndexInt]
            }
        }

        return output
    }

    // MARK: - Mel Spectrogram

    private func computeMelSpectrogram(_ samples: [Float]) -> [[Float]] {
        guard let melFilterbank = melFilterbank else { return [] }

        // torchfcpe's custom padding
        let padLeft = (Constants.winSize - Constants.hopLength) / 2  // 432
        let padRight = max((Constants.winSize - Constants.hopLength + 1) / 2,
                          Constants.winSize - samples.count - padLeft)

        // Reflect padding
        var padded = [Float](repeating: 0, count: padLeft + samples.count + padRight)

        // Left reflection
        for i in 0..<padLeft {
            let srcIndex = padLeft - i
            if srcIndex < samples.count {
                padded[i] = samples[srcIndex]
            }
        }

        // Copy original samples
        for i in 0..<samples.count {
            padded[padLeft + i] = samples[i]
        }

        // Right reflection
        for i in 0..<padRight {
            let srcIndex = samples.count - 2 - i
            if srcIndex >= 0 {
                padded[padLeft + samples.count + i] = samples[srcIndex]
            }
        }

        // STFT with center=False
        let numFrames = (padded.count - Constants.winSize) / Constants.hopLength + 1
        guard numFrames > 0 else { return [] }

        var melSpec = [[Float]]()

        // Create Hann window
        var window = [Float](repeating: 0, count: Constants.winSize)
        vDSP_hann_window(&window, vDSP_Length(Constants.winSize), Int32(vDSP_HANN_NORM))

        // FFT setup
        let log2n = vDSP_Length(log2(Float(Constants.nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return []
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        for frame in 0..<numFrames {
            let start = frame * Constants.hopLength

            // Extract and window frame
            var frameData = [Float](repeating: 0, count: Constants.nFFT)
            for i in 0..<Constants.winSize {
                if start + i < padded.count {
                    frameData[i] = padded[start + i] * window[i]
                }
            }

            // FFT
            var realPart = [Float](repeating: 0, count: Constants.nFFT / 2)
            var imagPart = [Float](repeating: 0, count: Constants.nFFT / 2)

            frameData.withUnsafeBufferPointer { bufferPtr in
                var splitComplex = DSPSplitComplex(realp: &realPart, imagp: &imagPart)
                bufferPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: Constants.nFFT / 2) { complexPtr in
                    vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(Constants.nFFT / 2))
                }
                vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
            }

            // Magnitude: sqrt(real² + imag² + 1e-9)
            var magnitude = [Float](repeating: 0, count: Constants.nFFT / 2 + 1)
            for i in 0..<(Constants.nFFT / 2) {
                magnitude[i] = sqrt(realPart[i] * realPart[i] + imagPart[i] * imagPart[i] + 1e-9)
            }
            // DC and Nyquist
            magnitude[0] = abs(realPart[0])
            magnitude[Constants.nFFT / 2] = abs(imagPart[0])

            // Mel transformation
            var melFrame = [Float](repeating: 0, count: Constants.nMels)
            for m in 0..<Constants.nMels {
                var sum: Float = 0
                for k in 0..<(Constants.nFFT / 2 + 1) {
                    sum += melFilterbank[m][k] * magnitude[k]
                }
                // Log compression
                melFrame[m] = log(max(sum, Constants.clipVal))
            }

            melSpec.append(melFrame)
        }

        return melSpec
    }

    // MARK: - CoreML Inference

    private func runInference(melSpec: [[Float]], model: MLModel) throws -> [[Float]] {
        let timeFrames = melSpec.count

        // Create MLMultiArray with shape (1, 128, T)
        let shape = [1, Constants.nMels, timeFrames] as [NSNumber]
        let melInput = try MLMultiArray(shape: shape, dataType: .float32)

        // Fill data: (batch, mels, time)
        for t in 0..<timeFrames {
            for m in 0..<Constants.nMels {
                let index = m * timeFrames + t
                melInput[index] = NSNumber(value: melSpec[t][m])
            }
        }

        // Run prediction
        let inputFeature = try MLDictionaryFeatureProvider(
            dictionary: ["mel_spectrogram": MLFeatureValue(multiArray: melInput)]
        )
        let output = try model.prediction(from: inputFeature)

        guard let logitsArray = output.featureValue(for: "f0_logits")?.multiArrayValue else {
            throw FCPEError.invalidOutput
        }

        // Extract logits: shape (1, T, 360)
        var logits = [[Float]]()
        for t in 0..<timeFrames {
            var frame = [Float](repeating: 0, count: Constants.outDims)
            for d in 0..<Constants.outDims {
                let index = t * Constants.outDims + d
                frame[d] = logitsArray[index].floatValue
            }
            logits.append(frame)
        }

        return logits
    }

    // MARK: - Decoding

    private func decodeLogits(_ logits: [[Float]]) -> [Float] {
        var f0Hz = [Float]()

        for t in 0..<logits.count {
            let frame = logits[t]

            // Find max confidence and index
            var maxConf: Float = -Float.infinity
            var maxIdx = 0
            for i in 0..<frame.count {
                if frame[i] > maxConf {
                    maxConf = frame[i]
                    maxIdx = i
                }
            }

            // Unvoiced check
            if maxConf <= Constants.voicedThreshold {
                f0Hz.append(0.0)
                continue
            }

            // local_argmax: 9-bin weighted average
            var localSum: Float = 0
            var weightSum: Float = 0

            for offset in -4...4 {
                let idx = max(0, min(Constants.outDims - 1, maxIdx + offset))
                let weight = frame[idx]
                localSum += centTable[idx] * weight
                weightSum += weight
            }

            let cents = weightSum > 0 ? localSum / weightSum : centTable[maxIdx]
            let frequency = Float(10.0 * pow(2.0, Double(cents) / 1200.0))

            f0Hz.append(frequency)
        }

        return f0Hz
    }

    // MARK: - PitchFrame Building

    private func buildPitchFrames(f0Hz: [Float], amplitudes: [Float], timeOffset: Double = 0.0) -> [PitchFrame] {
        let hopSeconds = Double(Constants.hopLength) / Constants.targetSampleRate

        return f0Hz.enumerated().map { index, frequency in
            let timestamp = timeOffset + Double(index) * hopSeconds
            let isVoiced = frequency > 0

            // Use actual amplitude from audio, or 0 if index out of bounds
            let amplitude = index < amplitudes.count ? amplitudes[index] : 0.0

            return PitchFrame(
                timestamp: timestamp,
                frequency: isVoiced ? frequency : nil,
                confidence: isVoiced ? 1.0 : 0.0,
                amplitude: amplitude
            )
        }
    }
}

// MARK: - Errors

enum FCPEError: Error {
    case modelNotFound
    case invalidOutput
    case preprocessingFailed
}
