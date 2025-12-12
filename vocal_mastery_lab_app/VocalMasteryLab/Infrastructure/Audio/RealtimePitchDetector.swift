import Foundation
import AVFoundation
import VocalisDomain
import Combine
import Accelerate
import OSLog

/// Errors that can occur during pitch detection
public enum PitchDetectorError: Error, LocalizedError {
    case invalidAudioFormat
    case engineStartFailed
    case audioSessionError(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAudioFormat:
            return "Invalid audio format detected (sample rate or channel count is zero)"
        case .engineStartFailed:
            return "Failed to start audio engine after multiple retries"
        case .audioSessionError(let message):
            return "Audio session error: \(message)"
        }
    }
}

/// Real-time pitch detector using FFT-based analysis
@MainActor
public class RealtimePitchDetector: ObservableObject, PitchDetectorProtocol {
    @Published public private(set) var detectedPitch: DetectedPitch?
    @Published public private(set) var isDetecting: Bool = false
    @Published public private(set) var spectrum: [Float]?
    @Published public private(set) var audioLevel: Float = -160.0  // dB value (-160 to 0)

    // Publisher for protocol conformance
    public var detectedPitchPublisher: AnyPublisher<DetectedPitch?, Never> {
        $detectedPitch.eraseToAnyPublisher()
    }

    private let audioEngine = AVAudioEngine()
    private var audioBuffer: [Float] = []
    private let bufferSize = 8192  // Phase 2D: Improved frequency resolution (5.39 Hz/bin vs 10.77 Hz/bin)
    private let hopSize = 2048

    // FFT setup
    private var fftSetup: vDSP_DFT_Setup?
    private let log2n: vDSP_Length

    // RMS silence threshold - configurable for different environments
    // Default 0.02 is for normal conditions (real device)
    // Lower values (e.g., 0.005) can be used in iOS Simulator where
    // AVAudioRecorder/AVAudioEngine competition causes reduced RMS
    private var rmsSilenceThreshold: Float

    // Confidence threshold for pitch detection
    // Default 0.4 - only report pitches with confidence above this threshold
    private var confidenceThreshold: Float

    // Actual sample rate from audio input (detected at runtime)
    // Default 44100.0 Hz, but may be 48000.0 Hz on some devices/simulators
    private var actualSampleRate: Double = 44100.0

    public init(rmsSilenceThreshold: Float = 0.02, confidenceThreshold: Float = 0.4) {
        self.rmsSilenceThreshold = rmsSilenceThreshold
        self.confidenceThreshold = confidenceThreshold
        log2n = vDSP_Length(log2(Double(bufferSize)))
        fftSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(bufferSize), vDSP_DFT_Direction.FORWARD)
    }

    /// Update audio detection settings
    public func updateSettings(_ settings: AudioDetectionSettings) {
        self.rmsSilenceThreshold = settings.rmsSilenceThreshold
        self.confidenceThreshold = settings.confidenceThreshold
    }

    deinit {
        if let setup = fftSetup {
            vDSP_DFT_DestroySetup(setup)
        }
    }

    // MARK: - Multi-Factor Confidence Calculation

    /// Calculate noise floor from bottom 10% of spectrum
    private func calculateNoiseFloor(magnitudes: [Float]) -> Float {
        let sortedMagnitudes = magnitudes.sorted()
        let bottomCount = max(1, magnitudes.count / 10)
        let bottom10Percent = sortedMagnitudes.prefix(bottomCount)
        return bottom10Percent.reduce(0, +) / Float(bottomCount)
    }

    /// Check if harmonics are present above noise threshold
    private func calculateHarmonicConsistency(
        frequency: Double,
        magnitudes: [Float],
        sampleRate: Double,
        bufferSize: Int,
        noiseFloor: Float
    ) -> Double {
        let fundamentalBin = Int(frequency * Double(bufferSize) / sampleRate)
        let threshold = noiseFloor * 2.0 // Harmonics should be above noise

        var presentHarmonics = 0
        let totalHarmonics = 4 // Check 2f0, 3f0, 4f0, 5f0

        for harmonic in 2...5 {
            let harmonicBin = fundamentalBin * harmonic
            guard harmonicBin < magnitudes.count else { continue }

            if magnitudes[harmonicBin] > threshold {
                presentHarmonics += 1
            }
        }

        return Double(presentHarmonics) / Double(totalHarmonics)
    }

    /// Calculate spectral clarity (peak vs noise ratio)
    private func calculateSpectralClarity(
        peakMagnitude: Float,
        noiseFloor: Float
    ) -> Double {
        let ratio = Double(peakMagnitude / (noiseFloor + 0.001))
        // Normalize: SNR of 10 → 1.0
        return min(1.0, ratio / 10.0)
    }

    /// Calculate multi-factor confidence combining peak prominence, harmonic consistency, and spectral clarity
    private func calculateMultiFactorConfidence(
        peakMagnitude: Float,
        avgMagnitude: Float,
        frequency: Double,
        magnitudes: [Float],
        sampleRate: Double,
        bufferSize: Int
    ) -> Double {
        // 1. Peak Prominence (existing metric)
        let peakProminence = min(1.0, Double(peakMagnitude / (avgMagnitude + 0.001)) / 10.0)

        // 2. Harmonic Consistency (most important for pitch accuracy)
        let noiseFloor = calculateNoiseFloor(magnitudes: magnitudes)
        let harmonicConsistency = calculateHarmonicConsistency(
            frequency: frequency,
            magnitudes: magnitudes,
            sampleRate: sampleRate,
            bufferSize: bufferSize,
            noiseFloor: noiseFloor
        )

        // 3. Spectral Clarity
        let spectralClarity = calculateSpectralClarity(
            peakMagnitude: peakMagnitude,
            noiseFloor: noiseFloor
        )

        // Weighted combination
        // w2 (harmonic) is most important as true fundamental should have harmonics
        let w1 = 0.3  // Peak prominence
        let w2 = 0.5  // Harmonic consistency (most important)
        let w3 = 0.2  // Spectral clarity

        let confidence = w1 * peakProminence + w2 * harmonicConsistency + w3 * spectralClarity

        return min(1.0, max(0.0, confidence))
    }

    /// Start real-time pitch detection from microphone
    public func startRealtimeDetection() async throws {
        FileLogger.shared.log(level: "INFO", category: "pitch", message: "startRealtimeDetection() called")

        guard !isDetecting else {
            FileLogger.shared.log(level: "DEBUG", category: "pitch", message: "Already detecting, returning early")
            return
        }

        // Try to start with retry logic for AudioSession invalidation recovery
        var lastError: Error?
        let maxRetries = 2

        for attempt in 1...maxRetries {
            do {
                try await startRealtimeDetectionInternal(attempt: attempt)
                return // Success
            } catch {
                lastError = error
                FileLogger.shared.log(level: "WARNING", category: "pitch", message: "⚠️ Attempt \(attempt)/\(maxRetries) failed: \(error.localizedDescription)")

                if attempt < maxRetries {
                    // Reset AVAudioEngine state and reconfigure AudioSession before retry
                    FileLogger.shared.log(level: "INFO", category: "pitch", message: "🔄 Resetting audio engine and reconfiguring session for retry...")
                    await resetAudioEngineForRetry()
                    try? await Task.sleep(nanoseconds: 300_000_000) // 300ms wait before retry
                }
            }
        }

        // All retries failed
        FileLogger.shared.log(level: "ERROR", category: "pitch", message: "❌ All \(maxRetries) attempts failed")
        throw lastError ?? PitchDetectorError.engineStartFailed
    }

    /// Reset audio engine state for retry after failure
    private func resetAudioEngineForRetry() async {
        // Stop and reset the engine
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        audioEngine.reset()

        // Reconfigure AudioSession to ensure valid format
        do {
            try AudioSessionManager.shared.configureForRecordingAndPlayback()
            try AudioSessionManager.shared.activate()
            FileLogger.shared.log(level: "INFO", category: "pitch", message: "✅ AudioSession reconfigured for retry")
        } catch {
            FileLogger.shared.log(level: "ERROR", category: "pitch", message: "❌ Failed to reconfigure AudioSession: \(error.localizedDescription)")
        }
    }

    /// Internal implementation of startRealtimeDetection
    private func startRealtimeDetectionInternal(attempt: Int) async throws {
        FileLogger.shared.log(level: "INFO", category: "pitch", message: "startRealtimeDetectionInternal() attempt \(attempt)")

        // Audio session should already be configured by AudioSessionManager
        // Don't reconfigure to avoid conflicts with recording and playback
        FileLogger.shared.log(level: "INFO", category: "pitch", message: "Using existing audio session configuration (managed by AudioSessionManager)")

        // Ensure audio session is active (safe to call multiple times)
        do {
            try AudioSessionManager.shared.activateIfNeeded()
            FileLogger.shared.log(level: "INFO", category: "pitch", message: "✅ Audio session activated")
        } catch {
            FileLogger.shared.log(level: "ERROR", category: "pitch", message: "❌ Failed to activate audio session: \(error.localizedDescription)")
            throw error
        }

        // Wait for audio hardware to stabilize after AudioSession configuration
        // This is especially important for Bluetooth devices which may need time to adjust sample rates
        Logger.pitchDetection.debug("Waiting for audio hardware to stabilize...")
        try? await Task.sleep(nanoseconds: 200_000_000) // 200ms
        FileLogger.shared.log(level: "INFO", category: "pitch", message: "Audio hardware stabilization delay completed")

        // Configure audio engine
        Logger.pitchDetection.debug("Configuring AVAudioEngine...")
        let inputNode = audioEngine.inputNode

        // CRITICAL: Always remove any existing tap to prevent "Tap already exists" crash
        // Tap can exist even if engine is not running, so always try to remove
        do {
            inputNode.removeTap(onBus: 0)
            Logger.pitchDetection.debug("Removed existing tap")
            FileLogger.shared.log(level: "INFO", category: "pitch", message: "Removed existing tap before installing new one")
        } catch {
            // Ignore error if no tap exists - this is expected on first run
            Logger.pitchDetection.debug("No existing tap to remove (expected on first run)")
        }

        // Stop engine if running
        if audioEngine.isRunning {
            Logger.pitchDetection.debug("Stopping existing audio engine...")
            audioEngine.stop()
        }

        // CRITICAL: Use hardware input format directly to avoid format mismatch
        // The hardware input format (e.g., 16kHz for Bluetooth HFP) must be used for the tap
        // Using nil or outputFormat causes error -10868 when formats don't match
        let hardwareInputFormat = inputNode.inputFormat(forBus: 0)
        let outputFormat = inputNode.outputFormat(forBus: 0)
        Logger.pitchDetection.debug("Hardware input format: \(String(describing: hardwareInputFormat))")
        FileLogger.shared.log(level: "INFO", category: "pitch", message: "Hardware input format: \(String(describing: hardwareInputFormat))")
        Logger.pitchDetection.debug("Node output format: \(String(describing: outputFormat))")
        FileLogger.shared.log(level: "INFO", category: "pitch", message: "Node output format: \(String(describing: outputFormat))")

        // Validate format before proceeding - detect invalid format early
        guard hardwareInputFormat.sampleRate > 0 && hardwareInputFormat.channelCount > 0 else {
            FileLogger.shared.log(level: "ERROR", category: "pitch", message: "❌ Invalid hardware format detected: sampleRate=\(hardwareInputFormat.sampleRate), channels=\(hardwareInputFormat.channelCount)")
            throw PitchDetectorError.invalidAudioFormat
        }

        // Note: Actual sample rate will be determined from the buffer in processAudioBuffer
        // The buffer will have the hardware's native sample rate (e.g., 16kHz for Bluetooth)

        // Install tap on input node using HARDWARE INPUT FORMAT
        // This ensures the tap matches the actual hardware format
        Logger.pitchDetection.debug("Installing tap with hardware input format...")
        FileLogger.shared.log(level: "INFO", category: "pitch", message: "Installing tap with hardware input format: \(String(describing: hardwareInputFormat))")
        inputNode.installTap(onBus: 0, bufferSize: UInt32(hopSize), format: hardwareInputFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            self.processAudioBuffer(buffer)
        }
        Logger.pitchDetection.debug("✅ Tap installed successfully")
        FileLogger.shared.log(level: "INFO", category: "pitch", message: "✅ Tap installed with hardware format")

        // Register as active audio component BEFORE starting engine
        // This prevents AudioSession from being deactivated while we're using it
        AudioSessionManager.shared.registerActiveComponent("RealtimePitchDetector")

        // Start engine
        Logger.pitchDetection.debug("Starting audio engine...")
        do {
            try audioEngine.start()
            FileLogger.shared.log(level: "INFO", category: "pitch", message: "✅ Audio engine started successfully")
        } catch {
            // Clean up tap and unregister before throwing
            inputNode.removeTap(onBus: 0)
            AudioSessionManager.shared.unregisterActiveComponent("RealtimePitchDetector")
            FileLogger.shared.log(level: "ERROR", category: "pitch", message: "❌ Failed to start audio engine: \(error.localizedDescription)")
            throw error
        }

        isDetecting = true
        FileLogger.shared.log(level: "INFO", category: "pitch", message: "✅ Pitch detection started successfully (isDetecting = true)")
    }

    /// Stop real-time pitch detection
    public func stopRealtimeDetection() {
        guard isDetecting else {
            // Even if not officially detecting, clean up engine if it's running
            if audioEngine.isRunning {
                Logger.pitchDetection.debug("Cleaning up orphaned audio engine")
                audioEngine.inputNode.removeTap(onBus: 0)
                audioEngine.stop()
                // Unregister component even in cleanup case
                AudioSessionManager.shared.unregisterActiveComponent("RealtimePitchDetector")
            }
            return
        }

        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        isDetecting = false
        detectedPitch = nil
        spectrum = nil
        audioLevel = -160.0

        // Unregister as active audio component
        // AudioSession will only deactivate if this was the last component
        AudioSessionManager.shared.unregisterActiveComponent("RealtimePitchDetector")
        FileLogger.shared.log(level: "INFO", category: "pitch", message: "✅ Pitch detection stopped, component unregistered")
    }

    /// Process audio buffer and detect pitch
    private var bufferProcessCount = 0
    private var successfulDetectionCount = 0

    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        bufferProcessCount += 1
        if bufferProcessCount == 1 {
            // Update actual sample rate from buffer format (important when format is nil in installTap)
            actualSampleRate = buffer.format.sampleRate
            FileLogger.shared.log(level: "INFO", category: "pitch", message: "processAudioBuffer called for the first time - audio input is working, actual sample rate: \(actualSampleRate) Hz")
        }
        if bufferProcessCount % 100 == 0 {
            Logger.pitchDetection.debug("processAudioBuffer called \(self.bufferProcessCount) times")
        }

        guard let channelData = buffer.floatChannelData else {
            FileLogger.shared.log(level: "ERROR", category: "pitch", message: "❌ No channel data in buffer")
            return
        }

        let frameLength = Int(buffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))

        // Append to buffer
        audioBuffer.append(contentsOf: samples)

        // Keep buffer size manageable
        if audioBuffer.count > bufferSize {
            audioBuffer.removeFirst(audioBuffer.count - bufferSize)
        }

        // Detect pitch if we have enough samples
        if audioBuffer.count >= bufferSize {
            let samplesToProcess = Array(audioBuffer.suffix(bufferSize))

            // 🔧 Execute pitch detection on background queue instead of MainActor
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self = self else { return }
                self.detectPitchFromSamples(samplesToProcess)
            }
        }
    }

    private var pitchDetectionCount = 0

    /// Detect pitch from audio samples using FFT-based analysis (runs on background queue)
    private func detectPitchFromSamples(_ samples: [Float]) {
        let detectionStartTime = Date()
        pitchDetectionCount += 1
        let count = pitchDetectionCount

        // Calculate RMS amplitude
        let rms = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count))

        // Convert RMS to dB and update audio level (clamp to -160 to 0 range)
        let dB = rms > 0 ? 20 * log10(rms) : -160.0
        let clampedDB = max(-160.0, min(0.0, dB))
        Task { @MainActor in
            self.audioLevel = clampedDB
        }

        if count == 1 {
            FileLogger.shared.log(level: "INFO", category: "pitch", message: "detectPitchFromSamples called for the first time, RMS: \(String(format: "%.4f", rms))")
        }
        if count % 100 == 0 {
            Logger.pitchDetection.debug("detectPitchFromSamples called \(count) times, RMS: \(String(format: "%.4f", rms))")
        }

        // Silence threshold check using configurable threshold
        guard rms > rmsSilenceThreshold else {
            if count <= 10 || count % 100 == 0 {
                FileLogger.shared.log(level: "DEBUG", category: "pitch", message: "⚠️ RMS (\(String(format: "%.4f", rms))) below silence threshold (\(String(format: "%.3f", rmsSilenceThreshold)))")
            }
            let beforeTaskSchedule = Date()
            Task { @MainActor in
                let taskExecutionTime = Date()
                let scheduleDelay = taskExecutionTime.timeIntervalSince(beforeTaskSchedule) * 1000
                if count <= 10 || count % 100 == 0 {
                    FileLogger.shared.log(level: "DEBUG", category: "pitch_timing", message: "⏱️ Task(nil) schedule delay: \(String(format: "%.1f", scheduleDelay))ms")
                }
                self.detectedPitch = nil
                self.spectrum = nil
            }
            return
        }

        guard let setup = fftSetup else {
            Task { @MainActor in
                self.detectedPitch = nil
                self.spectrum = nil
            }
            return
        }

        let sampleRate = actualSampleRate  // Use actual sample rate from audio input
        let minFreq = 100.0  // G2 - avoid low frequency noise
        let maxFreq = 800.0  // G5 - typical singing range for realtime

        // Prepare buffers for FFT
        var realPartIn = [Float](repeating: 0, count: bufferSize)
        var imagPartIn = [Float](repeating: 0, count: bufferSize)
        var realPartOut = [Float](repeating: 0, count: bufferSize)
        var imagPartOut = [Float](repeating: 0, count: bufferSize)

        // Apply Blackman-Harris window for better frequency resolution
        // Coefficients for Blackman-Harris window: a0 = 0.35875, a1 = 0.48829, a2 = 0.14128, a3 = 0.01168
        var window = [Float](repeating: 0, count: bufferSize)
        let a0: Float = 0.35875
        let a1: Float = 0.48829
        let a2: Float = 0.14128
        let a3: Float = 0.01168
        for i in 0..<bufferSize {
            let n = Float(i)
            let N = Float(bufferSize)
            window[i] = a0
                - a1 * cos(2.0 * .pi * n / N)
                + a2 * cos(4.0 * .pi * n / N)
                - a3 * cos(6.0 * .pi * n / N)
        }

        var windowedSamples = [Float](repeating: 0, count: bufferSize)
        vDSP_vmul(samples, 1, window, 1, &windowedSamples, 1, vDSP_Length(bufferSize))

        // Copy windowed samples to real part
        realPartIn = windowedSamples

        // Perform FFT
        vDSP_DFT_Execute(setup, &realPartIn, &imagPartIn, &realPartOut, &imagPartOut)

        // Calculate magnitude spectrum
        var magnitudes = [Float](repeating: 0, count: bufferSize / 2)
        realPartOut.withUnsafeMutableBufferPointer { realPtr in
            imagPartOut.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                vDSP_zvabs(&splitComplex, 1, &magnitudes, 1, vDSP_Length(bufferSize / 2))
            }
        }

        // Apply Harmonic Product Spectrum (HPS) to find fundamental frequency
        let minBin = Int(minFreq * Double(bufferSize) / sampleRate)
        let maxBin = Int(maxFreq * Double(bufferSize) / sampleRate)

        guard minBin < maxBin && maxBin < magnitudes.count else {
            Task { @MainActor in
                self.detectedPitch = nil
                self.spectrum = nil
            }
            return
        }

        // Publish spectrum data for visualization (100-800 Hz range)
        let spectrumData = Array(magnitudes[minBin..<maxBin])

        // Log raw spectrum data for debugging
        let binResolution = sampleRate / Double(bufferSize)  // Hz per bin
        Logger.pitchDetection.logToFile(level: "DEBUG", message: "🎵 RAW SPECTRUM - Count: \(spectrumData.count), Resolution: \(String(format: "%.2f", binResolution)) Hz/bin")

        // Log top 10 peaks in spectrum
        let indexedSpectrum = spectrumData.enumerated().map { ($0.offset, $0.element) }
        let topPeaks = indexedSpectrum.sorted { $0.1 > $1.1 }.prefix(10)
        for (index, magnitude) in topPeaks {
            let frequency = minFreq + Double(index) * binResolution
            Logger.pitchDetection.logToFile(level: "DEBUG", message: String(format: "  Peak[%3d]: %.1f Hz = %.4f", index, frequency, magnitude))
        }

        Task { @MainActor in
            self.spectrum = spectrumData
        }

        // Create HPS by multiplying harmonics (start from fundamental, then multiply by harmonics)
        // Increased from 5 to 7 to improve low-frequency accuracy and reduce octave errors
        let numHarmonics = 7 // Use harmonics 1, 2, 3, 4, 5, 6, 7
        var hps = [Float](repeating: 0.0, count: maxBin)

        // Initialize with fundamental (1st harmonic)
        for bin in minBin..<maxBin {
            hps[bin] = magnitudes[bin]
        }

        // Multiply by higher harmonics (2nd through 5th)
        for harmonic in 2...numHarmonics {
            for bin in minBin..<maxBin {
                let downsampledBin = bin * harmonic
                if downsampledBin < magnitudes.count {
                    hps[bin] *= magnitudes[downsampledBin]
                }
            }
        }

        // Find peak in HPS (this will be the fundamental frequency)
        var maxMagnitude: Float = 0
        var peakBin = 0

        for bin in minBin..<maxBin {
            if hps[bin] > maxMagnitude {
                maxMagnitude = hps[bin]
                peakBin = bin
            }
        }

        guard maxMagnitude > 0.01 else {
            Task { @MainActor in
                self.detectedPitch = nil
            }
            return
        }

        // Quinn's first estimator for improved sub-bin accuracy
        // More accurate than parabolic interpolation, especially for reducing high-frequency bias
        var interpolatedBin = Double(peakBin)
        if peakBin > 0 && peakBin < hps.count - 1 {
            let alpha = Double(hps[peakBin - 1])
            let beta = Double(hps[peakBin])
            let gamma = Double(hps[peakBin + 1])

            // Quinn's first estimator formula
            // tau(x) = (alpha - gamma) / (2 * (2*beta - alpha - gamma))
            let numerator = alpha - gamma
            let denominator = 2.0 * (2.0 * beta - alpha - gamma)

            if abs(denominator) > 0.0001 && abs(numerator / denominator) <= 1.0 {
                let tau = numerator / denominator
                interpolatedBin = Double(peakBin) + tau
            }
        }

        // Convert bin to frequency
        let frequency = interpolatedBin * sampleRate / Double(bufferSize)

        // Calculate multi-factor confidence
        let avgMagnitude = magnitudes[minBin..<maxBin].reduce(0, +) / Float(maxBin - minBin)
        let confidence = calculateMultiFactorConfidence(
            peakMagnitude: maxMagnitude,
            avgMagnitude: avgMagnitude,
            frequency: frequency,
            magnitudes: magnitudes,
            sampleRate: sampleRate,
            bufferSize: bufferSize
        )

        // Only report if confidence is high enough
        guard confidence > Double(confidenceThreshold) else {
            // Log rejected detections for first 20 times
            if count <= 20 {
                FileLogger.shared.log(
                    level: "DEBUG",
                    category: "pitch_timing",
                    message: "⏱️ REJECTED #\(count): freq=\(String(format: "%.1f", frequency))Hz, confidence=\(String(format: "%.2f", confidence)) < 0.4"
                )
            }
            Task { @MainActor in
                self.detectedPitch = nil
            }
            return
        }

        // Create detected pitch
        let pitch = DetectedPitch.fromFrequency(
            frequency,
            confidence: confidence
        )

        let fftCompleteTime = Date()
        let fftDuration = fftCompleteTime.timeIntervalSince(detectionStartTime) * 1000

        successfulDetectionCount += 1
        let successCount = successfulDetectionCount

        // Log FFT processing time for all successful detections (first 50) or every 10th
        if successCount <= 50 || successCount % 10 == 0 {
            FileLogger.shared.log(level: "INFO", category: "pitch_timing", message: "⏱️ SUCCESS #\(successCount) (total attempts: \(count)): FFT=\(String(format: "%.1f", fftDuration))ms, freq=\(String(format: "%.1f", frequency))Hz, conf=\(String(format: "%.2f", confidence))")
        }

        let beforeTaskSchedule = Date()
        Task { @MainActor in
            let taskExecutionTime = Date()
            let scheduleDelay = taskExecutionTime.timeIntervalSince(beforeTaskSchedule) * 1000
            let totalDelay = taskExecutionTime.timeIntervalSince(detectionStartTime) * 1000

            // Log timing for first 50 successful updates or every 10th
            if successCount <= 50 || successCount % 10 == 0 {
                FileLogger.shared.log(level: "INFO", category: "pitch_timing", message: "⏱️ Task execution #\(successCount): schedule=\(String(format: "%.1f", scheduleDelay))ms, total=\(String(format: "%.1f", totalDelay))ms")
            }

            self.detectedPitch = pitch
        }
    }

    /// Analyze pitch from audio file (for playback analysis)
    public func analyzePitchFromFile(
        _ url: URL,
        atTime time: TimeInterval,
        completion: @escaping (DetectedPitch?) -> Void
    ) {
        Task {
            do {
                let audioFile = try AVAudioFile(forReading: url)
                let format = audioFile.processingFormat
                let sampleRate = format.sampleRate

                // Calculate frame position
                let framePosition = AVAudioFramePosition(time * sampleRate)

                guard framePosition < audioFile.length else {
                    await MainActor.run { completion(nil) }
                    return
                }

                // Read samples around the target time
                audioFile.framePosition = max(0, framePosition - AVAudioFramePosition(bufferSize / 2))

                guard let buffer = AVAudioPCMBuffer(
                    pcmFormat: format,
                    frameCapacity: AVAudioFrameCount(bufferSize)
                ) else {
                    await MainActor.run { completion(nil) }
                    return
                }

                try audioFile.read(into: buffer)

                guard let floatChannelData = buffer.floatChannelData else {
                    await MainActor.run { completion(nil) }
                    return
                }

                let samples = Array(UnsafeBufferPointer(
                    start: floatChannelData[0],
                    count: Int(buffer.frameLength)
                ))

                // Detect pitch from samples
                let pitch = await detectPitchFromSamplesSync(samples, sampleRate: sampleRate)
                await MainActor.run { completion(pitch) }

            } catch {
                await MainActor.run { completion(nil) }
            }
        }
    }

    /// Synchronous pitch detection for file analysis using FFT
    private func detectPitchFromSamplesSync(_ samples: [Float], sampleRate: Double) async -> DetectedPitch? {
        // Calculate RMS amplitude
        let rms = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count))

        // Lower threshold for recorded playback (recorded files tend to have lower amplitude)
        if rms <= 0.001 {
            await MainActor.run { self.spectrum = nil }
            return nil
        }

        guard let setup = fftSetup else {
            await MainActor.run { self.spectrum = nil }
            return nil
        }

        // Focus on fundamental frequency range for singing voice
        let minFreq = 100.0  // G2 - avoid low frequency noise
        let maxFreq = 500.0  // B4 - typical singing range
        let size = min(samples.count, bufferSize)

        // Prepare buffers
        var realPartIn = [Float](repeating: 0, count: bufferSize)
        var imagPartIn = [Float](repeating: 0, count: bufferSize)
        var realPartOut = [Float](repeating: 0, count: bufferSize)
        var imagPartOut = [Float](repeating: 0, count: bufferSize)

        // Apply Blackman-Harris window for better frequency resolution
        var window = [Float](repeating: 0, count: size)
        let a0: Float = 0.35875
        let a1: Float = 0.48829
        let a2: Float = 0.14128
        let a3: Float = 0.01168
        for i in 0..<size {
            let n = Float(i)
            let N = Float(size)
            window[i] = a0
                - a1 * cos(2.0 * .pi * n / N)
                + a2 * cos(4.0 * .pi * n / N)
                - a3 * cos(6.0 * .pi * n / N)
        }

        var windowedSamples = [Float](repeating: 0, count: size)
        vDSP_vmul(Array(samples.prefix(size)), 1, window, 1, &windowedSamples, 1, vDSP_Length(size))

        realPartIn[0..<size] = windowedSamples[0..<size]

        // Perform FFT
        vDSP_DFT_Execute(setup, &realPartIn, &imagPartIn, &realPartOut, &imagPartOut)

        // Calculate magnitudes
        var magnitudes = [Float](repeating: 0, count: bufferSize / 2)
        realPartOut.withUnsafeMutableBufferPointer { realPtr in
            imagPartOut.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                vDSP_zvabs(&splitComplex, 1, &magnitudes, 1, vDSP_Length(bufferSize / 2))
            }
        }

        // Apply Harmonic Product Spectrum (HPS) to find fundamental frequency
        let minBin = Int(minFreq * Double(bufferSize) / sampleRate)
        let maxBin = Int(maxFreq * Double(bufferSize) / sampleRate)

        guard minBin < maxBin && maxBin < magnitudes.count else {
            await MainActor.run { self.spectrum = nil }
            return nil
        }

        // Publish spectrum data for visualization (100-500 Hz range for playback)
        await MainActor.run {
            self.spectrum = Array(magnitudes[minBin..<maxBin])
        }

        // Create HPS by multiplying harmonics (start from fundamental, then multiply by harmonics)
        // Increased from 5 to 7 to improve low-frequency accuracy and reduce octave errors
        let numHarmonics = 7 // Use harmonics 1, 2, 3, 4, 5, 6, 7
        var hps = [Float](repeating: 0.0, count: maxBin)

        // Initialize with fundamental (1st harmonic)
        for bin in minBin..<maxBin {
            hps[bin] = magnitudes[bin]
        }

        // Multiply by higher harmonics (2nd through 5th)
        for harmonic in 2...numHarmonics {
            for bin in minBin..<maxBin {
                let downsampledBin = bin * harmonic
                if downsampledBin < magnitudes.count {
                    hps[bin] *= magnitudes[downsampledBin]
                }
            }
        }

        // Find peak in HPS (this will be the fundamental frequency)
        var maxMagnitude: Float = 0
        var peakBin = 0

        for bin in minBin..<maxBin {
            if hps[bin] > maxMagnitude {
                maxMagnitude = hps[bin]
                peakBin = bin
            }
        }

        if maxMagnitude <= 0.001 {  // Lowered from 0.01
            return nil
        }

        // Quinn's first estimator for improved sub-bin accuracy
        // More accurate than parabolic interpolation, especially for reducing high-frequency bias
        var interpolatedBin = Double(peakBin)
        if peakBin > 0 && peakBin < hps.count - 1 {
            let alpha = Double(hps[peakBin - 1])
            let beta = Double(hps[peakBin])
            let gamma = Double(hps[peakBin + 1])

            // Quinn's first estimator formula
            // tau(x) = (alpha - gamma) / (2 * (2*beta - alpha - gamma))
            let numerator = alpha - gamma
            let denominator = 2.0 * (2.0 * beta - alpha - gamma)

            if abs(denominator) > 0.0001 && abs(numerator / denominator) <= 1.0 {
                let tau = numerator / denominator
                interpolatedBin = Double(peakBin) + tau
            }
        }

        let frequency = interpolatedBin * sampleRate / Double(bufferSize)

        // Validate frequency is in expected range
        guard frequency >= minFreq && frequency <= maxFreq else {
            return nil
        }

        // Calculate multi-factor confidence
        let avgMagnitude = magnitudes[minBin..<maxBin].reduce(0, +) / Float(maxBin - minBin)
        let confidence = calculateMultiFactorConfidence(
            peakMagnitude: maxMagnitude,
            avgMagnitude: avgMagnitude,
            frequency: frequency,
            magnitudes: magnitudes,
            sampleRate: sampleRate,
            bufferSize: bufferSize
        )

        // Lower confidence threshold for playback analysis (75% of realtime threshold)
        if confidence <= Double(confidenceThreshold * 0.75) {
            return nil
        }

        let pitch = DetectedPitch.fromFrequency(frequency, confidence: confidence)
        return pitch
    }
}
