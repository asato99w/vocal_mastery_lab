import Foundation

/// Service for correcting octave errors in pitch detection data
/// YIN algorithm sometimes detects harmonics instead of fundamental frequency,
/// resulting in octave errors. This service corrects such errors.
public final class OctaveCorrectionService {

    public init() {}

    // MARK: - Single Frequency Correction

    /// Correct detected frequency to the closest octave of the target note
    /// - Parameters:
    ///   - detectedFrequency: The frequency detected by pitch analysis
    ///   - targetNote: The expected note at this timestamp
    /// - Returns: Corrected frequency in the closest octave to the target
    public func correctFrequency(_ detectedFrequency: Float, targetNote: MIDINote) -> Float {
        let targetFreq = Float(targetNote.frequency)

        // If already within half an octave, no correction needed
        // sqrt(0.5) ≈ 0.707, sqrt(2) ≈ 1.414
        let ratio = detectedFrequency / targetFreq
        if ratio > 0.707 && ratio < 1.414 {
            return detectedFrequency
        }

        // Find the octave shift that brings detected frequency closest to target
        var corrected = detectedFrequency

        // Shift up octaves if detected is too low
        while corrected < targetFreq * 0.707 {
            corrected *= 2
        }

        // Shift down octaves if detected is too high
        while corrected > targetFreq * 1.414 {
            corrected /= 2
        }

        return corrected
    }

    // MARK: - PitchAnalysisData Correction

    /// Apply octave correction to pitch analysis data based on note segments
    /// - Parameters:
    ///   - pitchData: Original pitch analysis data
    ///   - segments: Note segments defining target notes at each timestamp
    /// - Returns: New PitchAnalysisData with corrected frequencies
    public func applyCorrection(
        to pitchData: PitchAnalysisData,
        segments: [NoteSegment]
    ) -> PitchAnalysisData {
        guard !segments.isEmpty else {
            return pitchData
        }

        var correctedFrequencies = pitchData.frequencies

        for (index, timestamp) in pitchData.timeStamps.enumerated() {
            // Find segment containing this timestamp
            guard let segment = findSegment(at: timestamp, in: segments) else {
                continue  // No target note at this timestamp, keep original
            }

            let originalFreq = pitchData.frequencies[index]
            let correctedFreq = correctFrequency(originalFreq, targetNote: segment.note)
            correctedFrequencies[index] = correctedFreq
        }

        return PitchAnalysisData(
            timeStamps: pitchData.timeStamps,
            frequencies: correctedFrequencies,
            confidences: pitchData.confidences,
            targetNotes: pitchData.targetNotes,
            amplitudes: pitchData.amplitudes
        )
    }

    // MARK: - Private Methods

    /// Find the segment containing the given timestamp
    private func findSegment(at timestamp: Double, in segments: [NoteSegment]) -> NoteSegment? {
        for segment in segments {
            if timestamp >= segment.startTime && timestamp < segment.endTime {
                return segment
            }
        }
        return nil
    }
}
