import Foundation
import AVFoundation
import VocalisDomain

/// Generates PCM audio buffers containing sine waves for specified MIDI notes
/// Used by AVAudioPlayerNodeScalePlayer for synthesized sound playback
public class SineWaveGenerator {

    /// Base amplitude for the sine wave (0.0 - 1.0)
    /// Default is 1.0 (maximum) because pure sine waves lack harmonics
    /// and sound quieter than complex waveforms at the same amplitude
    private let baseAmplitude: Double

    public init(amplitude: Double = 1.0) {
        self.baseAmplitude = max(0.0, min(1.0, amplitude))
    }

    /// Generate a PCM buffer containing a sine wave at the frequency of the given MIDI note
    /// - Parameters:
    ///   - note: MIDI note to generate
    ///   - duration: Duration of the buffer in seconds
    ///   - sampleRate: Sample rate for the buffer (typically 44100)
    /// - Returns: PCM buffer containing the generated sine wave with ADSR envelope
    public func generateBuffer(for note: MIDINote, duration: TimeInterval, sampleRate: Double) -> AVAudioPCMBuffer {
        let frameCount = AVAudioFrameCount(duration * sampleRate)

        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        buffer.frameLength = frameCount

        let frequency = note.frequency
        let data = buffer.floatChannelData![0]

        for i in 0..<Int(frameCount) {
            let t = Double(i) / sampleRate
            // Basic sine wave with base amplitude
            var sample = sin(2.0 * .pi * frequency * t) * baseAmplitude
            // Apply ADSR envelope
            sample *= envelope(t: t, duration: duration)
            data[i] = Float(sample)
        }

        return buffer
    }

    /// ADSR envelope for natural sound
    /// - Parameters:
    ///   - t: Current time in seconds
    ///   - duration: Total duration in seconds
    /// - Returns: Envelope amplitude (0.0 - 1.0)
    private func envelope(t: Double, duration: Double) -> Double {
        let attack = 0.01    // 10ms attack
        let decay = 0.02     // 20ms decay (shorter for more sustained loudness)
        let sustain = 0.95   // 95% sustain level (high to maintain volume)
        let release = 0.05   // 50ms release (shorter for cleaner note separation)

        let releaseStart = max(0, duration - release)

        if t < attack {
            // Attack phase: ramp up from 0 to 1
            return t / attack
        } else if t < attack + decay {
            // Decay phase: ramp down from 1 to sustain level
            let decayProgress = (t - attack) / decay
            return 1.0 - (1.0 - sustain) * decayProgress
        } else if t < releaseStart {
            // Sustain phase: maintain sustain level
            return sustain
        } else if t < duration {
            // Release phase: ramp down from sustain to 0
            let releaseProgress = (t - releaseStart) / release
            return sustain * (1.0 - releaseProgress)
        } else {
            return 0.0
        }
    }
}
