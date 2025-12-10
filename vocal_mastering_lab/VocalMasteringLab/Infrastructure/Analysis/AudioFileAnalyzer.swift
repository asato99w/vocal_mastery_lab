import Foundation
import AVFoundation
import VocalisDomain
import Accelerate
import OSLog

/// Analyzes audio files for pitch and spectrogram data
/// Uses configurable pitch detection strategy (default: YIN) and FFT for spectrogram
public class AudioFileAnalyzer: AudioFileAnalyzerProtocol {
    private let logger = Logger(subsystem: "com.kazuasato.VocalMasteringLab", category: "AudioFileAnalyzer")

    // Pitch detection strategy (default: YIN)
    private let pitchStrategy: PitchDetectionStrategy

    // Standard sample rate
    private let sampleRate = 44100.0

    // Latency compensation: FFT window center offset
    // The detected pitch corresponds to the center of the FFT window, not the start
    // This offset compensates for that delay to align pitch with actual vocalization time
    private var pitchDetectionLatencyOffset: Double {
        // Use YIN buffer size (2048) for latency calculation
        Double(2048 / 2) / sampleRate  // ~23.2ms for 2048 buffer at 44.1kHz
    }

    // Analysis parameters
    private let spectrogramSamplingInterval = 0.05  // 50ms for spectrogram (Phase 2: 100ms→50ms for 2x time resolution)

    // Spectrogram parameters
    private let spectrogramFreqBins = 800  // Number of frequency bins for spectrogram (8000Hz / 10Hz = 800 bins)
    private let spectrogramMaxFreq = 8000.0  // Max frequency for spectrogram (extended to 8kHz for voice analysis features)

    // Spectrogram FFT buffer size
    private let spectrogramBufferSize = 4096  // Performance: optimized for balance (10.76Hz/bin theoretical resolution)

    /// Initialize with default YIN strategy
    public convenience init() {
        self.init(pitchStrategy: YINStrategy())
    }

    /// Initialize with custom pitch detection strategy
    public init(pitchStrategy: PitchDetectionStrategy) {
        self.pitchStrategy = pitchStrategy
    }

    public func analyze(fileURL: URL, progress: @escaping @MainActor (Double) async -> Void) async throws -> (pitchData: PitchAnalysisData, spectrogramData: SpectrogramData) {
        logger.info("Starting analysis for file: \(fileURL.path)")

        // Report initial progress
        await progress(0.0)

        // Load audio file
        let audioFile = try AVAudioFile(forReading: fileURL)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AnalysisError.bufferAllocationFailed
        }

        try audioFile.read(into: buffer)

        guard let channelData = buffer.floatChannelData else {
            throw AnalysisError.noChannelData
        }

        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))
        let duration = Double(samples.count) / sampleRate

        logger.info("File loaded: \(samples.count) samples, duration: \(String(format: "%.2f", duration))s")

        // Analyze pitch (0% → 50%)
        let pitchData = try await analyzePitch(samples: samples, duration: duration) { pitchProgress in
            await progress(pitchProgress * 0.5)  // Scale to 0.0 → 0.5
        }

        // Analyze spectrogram (50% → 100%)
        let spectrogramData = try await analyzeSpectrogram(samples: samples, duration: duration) { spectrogramProgress in
            await progress(0.5 + spectrogramProgress * 0.5)  // Scale to 0.5 → 1.0
        }

        // Report final progress
        await progress(1.0)

        logger.info("Analysis completed")

        return (pitchData, spectrogramData)
    }

    public func analyzeSpectrogramOnly(fileURL: URL, progress: @escaping @MainActor (Double) async -> Void) async throws -> SpectrogramData {
        logger.info("Starting spectrogram-only analysis for file: \(fileURL.path)")

        // Report initial progress
        await progress(0.0)

        // Load audio file
        let audioFile = try AVAudioFile(forReading: fileURL)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AnalysisError.bufferAllocationFailed
        }

        try audioFile.read(into: buffer)

        guard let channelData = buffer.floatChannelData else {
            throw AnalysisError.noChannelData
        }

        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))
        let duration = Double(samples.count) / sampleRate

        logger.info("File loaded for spectrogram: \(samples.count) samples, duration: \(String(format: "%.2f", duration))s")

        // Analyze spectrogram only (0% → 100%)
        let spectrogramData = try await analyzeSpectrogram(samples: samples, duration: duration, progress: progress)

        // Report final progress
        await progress(1.0)

        logger.info("Spectrogram-only analysis completed")

        return spectrogramData
    }

    // MARK: - Pitch Analysis

    private func analyzePitch(samples: [Float], duration: Double, progress: @escaping @MainActor (Double) async -> Void) async throws -> PitchAnalysisData {
        logger.info("Using pitch detection strategy: \(self.pitchStrategy.name)")

        // Use strategy to detect pitch
        let frames = pitchStrategy.detectPitch(samples: samples, sampleRate: sampleRate)

        // Apply latency compensation to timestamps
        var timeStamps: [Double] = []
        var frequencies: [Float] = []
        var confidences: [Float] = []
        var targetNotes: [MIDINote?] = []
        var amplitudes: [Float] = []

        // For detecting note transitions in logs
        var lastLoggedMidiNote: Int = -1

        for (index, frame) in frames.enumerated() {
            guard let frequency = frame.frequency else { continue }

            // Apply latency compensation: shift timestamp earlier to account for FFT window center
            let compensatedTimestamp = max(0, frame.timestamp - pitchDetectionLatencyOffset)

            timeStamps.append(compensatedTimestamp)
            frequencies.append(frequency)
            confidences.append(frame.confidence)
            targetNotes.append(nil)  // Target notes will be set by ViewModel based on scaleSettings
            amplitudes.append(frame.amplitude)

            // Log pitch detection data for timing analysis
            let midiNote = Int(round(69 + 12 * log2(frequency / 440.0)))
            let isNoteTransition = midiNote != lastLoggedMidiNote

            // Log on: first 5 detections, note transitions, or every 10th detection
            if timeStamps.count <= 5 || isNoteTransition || timeStamps.count % 10 == 1 {
                let transitionMarker = isNoteTransition && lastLoggedMidiNote != -1 ? " [NOTE_CHANGE]" : ""
                FileLogger.shared.log(
                    level: "DEBUG",
                    category: "pitch_detection",
                    message: "\(pitchStrategy.name) #\(timeStamps.count): time=\(String(format: "%.3f", compensatedTimestamp))s, freq=\(String(format: "%.1f", frequency))Hz, MIDI=\(midiNote), conf=\(String(format: "%.2f", frame.confidence))\(transitionMarker)"
                )

                if isNoteTransition {
                    lastLoggedMidiNote = midiNote
                }
            }

            // Report progress periodically
            if index % 10 == 0 {
                let currentProgress = Double(index) / Double(frames.count)
                await progress(currentProgress)
            }
        }

        // Report final progress
        await progress(1.0)

        let detectionRate = frames.isEmpty ? 0.0 : Double(timeStamps.count) / Double(frames.count) * 100.0

        logger.info("Pitch analysis (\(self.pitchStrategy.name)): \(timeStamps.count) voiced frames detected (\(String(format: "%.1f", detectionRate))%)")
        if timeStamps.isEmpty {
            logger.warning("No pitch data detected - audio might be too quiet or contains no vocal content")
        } else {
            let minFreq = frequencies.min() ?? 0
            let maxFreq = frequencies.max() ?? 0
            let avgConfidence = confidences.reduce(0, +) / Float(confidences.count)
            logger.info("Frequency range: \(String(format: "%.1f", minFreq))Hz - \(String(format: "%.1f", maxFreq))Hz, avg confidence: \(String(format: "%.2f", avgConfidence))")
        }

        return PitchAnalysisData(
            timeStamps: timeStamps,
            frequencies: frequencies,
            confidences: confidences,
            targetNotes: targetNotes,
            amplitudes: amplitudes
        )
    }

    // MARK: - Spectrogram Analysis

    private func analyzeSpectrogram(samples: [Float], duration: Double, progress: @escaping @MainActor (Double) async -> Void) async throws -> SpectrogramData {
        var timeStamps: [Double] = []
        var magnitudesArray: [[Float]] = []

        let hopSamples = Int(sampleRate * spectrogramSamplingInterval)
        var position = 0
        let totalSamples = samples.count
        var lastReportedProgress: Double = 0.0

        // Define frequency bins
        let binSize = spectrogramMaxFreq / Double(spectrogramFreqBins)
        let frequencyBins = (0..<spectrogramFreqBins).map { Float($0) * Float(binSize) + Float(binSize / 2) }

        while position + spectrogramBufferSize <= samples.count {
            let timestamp = Double(position) / sampleRate
            let chunk = Array(samples[position..<(position + spectrogramBufferSize)])

            if let (magnitudes, freqBinSize) = performFFT(samples: chunk) {
                // Group magnitudes into frequency bins
                var binMagnitudes = [Float](repeating: 0, count: spectrogramFreqBins)

                for i in 0..<spectrogramFreqBins {
                    let startFreq = Double(i) * binSize
                    let endFreq = Double(i + 1) * binSize
                    let startBin = Int(startFreq / freqBinSize)
                    let endBin = Int(endFreq / freqBinSize)

                    if endBin <= magnitudes.count {
                        let binSlice = magnitudes[startBin..<endBin]
                        binMagnitudes[i] = binSlice.max() ?? 0.0
                    }
                }

                timeStamps.append(timestamp)
                magnitudesArray.append(binMagnitudes)
            }

            position += hopSamples

            // Report progress every 10% to avoid UI update overhead
            let currentProgress = Double(position) / Double(totalSamples)
            if currentProgress - lastReportedProgress >= 0.1 {
                await progress(currentProgress)
                lastReportedProgress = currentProgress
            }
        }

        // Report final progress
        await progress(1.0)

        logger.info("Spectrogram analysis: \(timeStamps.count) time frames, \(self.spectrogramFreqBins) frequency bins")

        return SpectrogramData(
            timeStamps: timeStamps,
            frequencyBins: frequencyBins,
            magnitudes: magnitudesArray
        )
    }

    // MARK: - FFT Utilities

    private func performFFT(samples: [Float]) -> (magnitudes: [Float], freqBinSize: Double)? {
        let bufferSize = samples.count

        guard let fftSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(bufferSize), vDSP_DFT_Direction.FORWARD) else {
            return nil
        }

        defer {
            vDSP_DFT_DestroySetup(fftSetup)
        }

        var realPartIn = [Float](repeating: 0, count: bufferSize)
        var imagPartIn = [Float](repeating: 0, count: bufferSize)
        var realPartOut = [Float](repeating: 0, count: bufferSize)
        var imagPartOut = [Float](repeating: 0, count: bufferSize)

        // Apply Hanning window
        var window = [Float](repeating: 0, count: bufferSize)
        vDSP_hann_window(&window, vDSP_Length(bufferSize), Int32(vDSP_HANN_NORM))

        var windowedSamples = [Float](repeating: 0, count: bufferSize)
        vDSP_vmul(samples, 1, window, 1, &windowedSamples, 1, vDSP_Length(bufferSize))

        realPartIn = windowedSamples

        // Perform FFT
        vDSP_DFT_Execute(fftSetup, &realPartIn, &imagPartIn, &realPartOut, &imagPartOut)

        // Calculate magnitude spectrum
        var magnitudes = [Float](repeating: 0, count: bufferSize / 2)
        realPartOut.withUnsafeMutableBufferPointer { realPtr in
            imagPartOut.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                vDSP_zvabs(&splitComplex, 1, &magnitudes, 1, vDSP_Length(bufferSize / 2))
            }
        }

        let freqBinSize = sampleRate / Double(bufferSize)

        return (magnitudes, freqBinSize)
    }
}

// MARK: - Errors

enum AnalysisError: Error, LocalizedError {
    case bufferAllocationFailed
    case noChannelData
    case fftSetupFailed

    var errorDescription: String? {
        switch self {
        case .bufferAllocationFailed:
            return "Failed to allocate audio buffer"
        case .noChannelData:
            return "No channel data available in audio buffer"
        case .fftSetupFailed:
            return "Failed to create FFT setup"
        }
    }
}
