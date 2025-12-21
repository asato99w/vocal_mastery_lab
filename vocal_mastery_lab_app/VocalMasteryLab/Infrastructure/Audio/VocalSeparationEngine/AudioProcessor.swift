import Foundation
import AVFoundation
import Accelerate

/// Audio file processing utilities for vocal separation
final class AudioProcessor {

    // MARK: - Types

    struct AudioData {
        let samples: [[Float]]  // [channels][samples]
        let sampleRate: Double
        let frameCount: Int

        var channelCount: Int { samples.count }
    }

    enum ProcessingError: Error, LocalizedError {
        case fileNotFound(String)
        case unsupportedFormat(String)
        case readError(String)
        case writeError(String)
        case conversionError(String)

        var errorDescription: String? {
            switch self {
            case .fileNotFound(let message): return message
            case .unsupportedFormat(let message): return message
            case .readError(let message): return message
            case .writeError(let message): return message
            case .conversionError(let message): return message
            }
        }
    }

    // MARK: - Audio Loading

    /// Load audio file
    static func loadAudio(from url: URL, targetSampleRate: Double? = nil) throws -> AudioData {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ProcessingError.fileNotFound("File not found: \(url.path)")
        }

        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(audioFile.length)
        ) else {
            throw ProcessingError.readError("Failed to create buffer")
        }

        try audioFile.read(into: buffer)

        var samples = convertBufferToFloatArray(buffer)

        // Resample if needed
        if let targetRate = targetSampleRate, targetRate != format.sampleRate {
            samples = try resample(samples, from: format.sampleRate, to: targetRate)
        }

        let finalSampleRate = targetSampleRate ?? format.sampleRate

        return AudioData(
            samples: samples,
            sampleRate: finalSampleRate,
            frameCount: samples[0].count
        )
    }

    // MARK: - Audio Saving

    /// Save audio file
    static func saveAudio(
        _ audioData: AudioData,
        to url: URL,
        format: AVAudioCommonFormat = .pcmFormatFloat32
    ) throws {
        guard let outputFormat = AVAudioFormat(
            commonFormat: format,
            sampleRate: audioData.sampleRate,
            channels: AVAudioChannelCount(audioData.channelCount),
            interleaved: false
        ) else {
            throw ProcessingError.writeError("Failed to create format")
        }

        let audioFile = try AVAudioFile(
            forWriting: url,
            settings: outputFormat.settings
        )

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(audioData.frameCount)
        ) else {
            throw ProcessingError.writeError("Failed to create buffer")
        }

        buffer.frameLength = AVAudioFrameCount(audioData.frameCount)

        for channel in 0..<audioData.channelCount {
            guard let channelData = buffer.floatChannelData?[channel] else {
                throw ProcessingError.writeError("Failed to access channel data")
            }

            audioData.samples[channel].withUnsafeBufferPointer { srcBuffer in
                channelData.update(from: srcBuffer.baseAddress!, count: audioData.frameCount)
            }
        }

        try audioFile.write(from: buffer)
    }

    // MARK: - Format Conversion

    private static func convertBufferToFloatArray(_ buffer: AVAudioPCMBuffer) -> [[Float]] {
        let channelCount = Int(buffer.format.channelCount)
        let frameLength = Int(buffer.frameLength)

        var samples: [[Float]] = []

        for channel in 0..<channelCount {
            guard let channelData = buffer.floatChannelData?[channel] else {
                continue
            }

            let channelSamples = Array(UnsafeBufferPointer(
                start: channelData,
                count: frameLength
            ))

            samples.append(channelSamples)
        }

        // Convert mono to stereo
        if samples.count == 1 {
            samples.append(samples[0])
        }

        return samples
    }

    private static func resample(
        _ samples: [[Float]],
        from sourceSampleRate: Double,
        to targetSampleRate: Double
    ) throws -> [[Float]] {

        guard sourceSampleRate != targetSampleRate else {
            return samples
        }

        let ratio = targetSampleRate / sourceSampleRate
        let sourceLength = samples[0].count
        let targetLength = Int(Double(sourceLength) * ratio)

        var resampledSamples: [[Float]] = []

        for channel in samples {
            var resampled = [Float](repeating: 0, count: targetLength)

            // Linear interpolation resampling
            for i in 0..<targetLength {
                let srcIndex = Double(i) / ratio
                let index0 = Int(srcIndex)
                let index1 = min(index0 + 1, sourceLength - 1)
                let fraction = Float(srcIndex - Double(index0))

                resampled[i] = channel[index0] * (1.0 - fraction) + channel[index1] * fraction
            }

            resampledSamples.append(resampled)
        }

        return resampledSamples
    }

    // MARK: - Utility

    /// Convert stereo to mono
    static func convertToMono(_ audioData: AudioData) -> AudioData {
        guard audioData.channelCount > 1 else {
            return audioData
        }

        let monoSamples = zip(audioData.samples[0], audioData.samples[1]).map { (l, r) in
            (l + r) / 2.0
        }

        return AudioData(
            samples: [monoSamples],
            sampleRate: audioData.sampleRate,
            frameCount: monoSamples.count
        )
    }

    /// Convert mono to stereo
    static func convertToStereo(_ audioData: AudioData) -> AudioData {
        guard audioData.channelCount == 1 else {
            return audioData
        }

        return AudioData(
            samples: [audioData.samples[0], audioData.samples[0]],
            sampleRate: audioData.sampleRate,
            frameCount: audioData.frameCount
        )
    }

    /// Normalize audio to prevent clipping
    static func normalize(_ audioData: AudioData) -> AudioData {
        var normalizedSamples: [[Float]] = []

        // Find max value across all channels
        var maxValue: Float = 0
        for channel in audioData.samples {
            for sample in channel {
                maxValue = max(maxValue, abs(sample))
            }
        }

        guard maxValue > 0 else {
            return audioData
        }

        // Normalize
        let scale = 1.0 / maxValue
        for channel in audioData.samples {
            let normalized = channel.map { $0 * scale }
            normalizedSamples.append(normalized)
        }

        return AudioData(
            samples: normalizedSamples,
            sampleRate: audioData.sampleRate,
            frameCount: audioData.frameCount
        )
    }
}
