import Foundation

/// Objective statistics calculated from pitch analysis data
/// All values are raw measurements without subjective scoring
public struct RecordingStatistics: Equatable {
    /// Overall statistics across all notes
    public let overall: OverallStatistics

    /// Per-position statistics (position within scale pattern)
    public let positionStatistics: [PositionStatistics]

    /// Per-pitch statistics (actual note frequencies)
    public let pitchStatistics: [PitchStatistics]

    /// Vibrato statistics (optional, nil if not enough data)
    public let vibratoStatistics: VibratoStatistics?

    /// Singer's Formant statistics (optional, nil if no spectrum data)
    public let singersFormantStatistics: SingersFormantStatistics?

    /// High frequency statistics (Brightness and Airiness)
    public let highFrequencyStatistics: HighFrequencyStatistics?

    /// Total recording duration
    public let totalDuration: TimeInterval

    public init(
        overall: OverallStatistics,
        positionStatistics: [PositionStatistics],
        pitchStatistics: [PitchStatistics],
        vibratoStatistics: VibratoStatistics? = nil,
        singersFormantStatistics: SingersFormantStatistics? = nil,
        highFrequencyStatistics: HighFrequencyStatistics? = nil,
        totalDuration: TimeInterval
    ) {
        self.overall = overall
        self.positionStatistics = positionStatistics
        self.pitchStatistics = pitchStatistics
        self.vibratoStatistics = vibratoStatistics
        self.singersFormantStatistics = singersFormantStatistics
        self.highFrequencyStatistics = highFrequencyStatistics
        self.totalDuration = totalDuration
    }

    // MARK: - Vibrato Statistics

    public struct VibratoStatistics: Equatable {
        /// Average vibrato rate in Hz (typical: 5-7 Hz)
        public let averageRate: Float

        /// Average vibrato extent in cents (typical: ±30-100 cents)
        public let averageExtent: Float

        /// Average regularity (0.0 - 1.0)
        public let averageRegularity: Float

        /// Percentage of time vibrato was detected
        public let presenceRate: Float

        /// Number of segments analyzed
        public let segmentsAnalyzed: Int

        public init(
            averageRate: Float,
            averageExtent: Float,
            averageRegularity: Float,
            presenceRate: Float,
            segmentsAnalyzed: Int
        ) {
            self.averageRate = averageRate
            self.averageExtent = averageExtent
            self.averageRegularity = averageRegularity
            self.presenceRate = presenceRate
            self.segmentsAnalyzed = segmentsAnalyzed
        }
    }

    // MARK: - Singer's Formant Statistics

    public struct SingersFormantStatistics: Equatable {
        /// Average energy ratio in SF band (2500-3500 Hz)
        /// Range: 0.0 - 1.0 (typically 0.05 - 0.15 for trained singers)
        public let averageRatio: Float

        /// Average intensity difference from surrounding bands (dB)
        public let averageIntensity: Float

        /// Whether singer's formant is detected
        public let isPresent: Bool

        /// Detection confidence (lower for high pitched voices)
        public let confidence: Float

        /// Ratio as percentage for display
        public var ratioPercentage: Float {
            averageRatio * 100
        }

        public init(
            averageRatio: Float,
            averageIntensity: Float,
            isPresent: Bool,
            confidence: Float
        ) {
            self.averageRatio = averageRatio
            self.averageIntensity = averageIntensity
            self.isPresent = isPresent
            self.confidence = confidence
        }
    }

    // MARK: - High Frequency Statistics (Brightness and Airiness)

    public struct HighFrequencyStatistics: Equatable {
        /// Average brightness ratio (4-6 kHz band)
        /// Range: 0.0 - 1.0 (typically 0.01 - 0.10)
        public let brightnessRatio: Float

        /// Average airiness ratio (6-9 kHz band)
        /// Range: 0.0 - 1.0 (typically 0.001 - 0.05)
        public let airinessRatio: Float

        /// Brightness ratio as percentage for display
        public var brightnessPercentage: Float {
            brightnessRatio * 100
        }

        /// Airiness ratio as percentage for display
        public var airinessPercentage: Float {
            airinessRatio * 100
        }

        public init(brightnessRatio: Float, airinessRatio: Float) {
            self.brightnessRatio = brightnessRatio
            self.airinessRatio = airinessRatio
        }
    }

    // MARK: - Overall Statistics

    public struct OverallStatistics: Equatable {
        /// Average pitch deviation in cents (absolute value)
        public let averageDeviationCents: Double

        /// Standard deviation of pitch deviation in cents
        public let deviationStdDev: Double

        /// Median pitch deviation in cents (absolute value)
        public let medianDeviationCents: Double

        /// Detection rate: percentage of samples with valid pitch during target notes
        public let detectionRate: Double

        /// Total number of valid pitch samples
        public let totalSamples: Int

        /// Detected vocal range - lowest frequency
        public let lowestFrequency: Double?

        /// Detected vocal range - highest frequency
        public let highestFrequency: Double?

        /// Lowest detected note name (e.g., "C3")
        public var lowestNoteName: String? {
            guard let freq = lowestFrequency else { return nil }
            return MIDINote.noteName(forFrequency: freq)
        }

        /// Highest detected note name (e.g., "G4")
        public var highestNoteName: String? {
            guard let freq = highestFrequency else { return nil }
            return MIDINote.noteName(forFrequency: freq)
        }

        public init(
            averageDeviationCents: Double,
            deviationStdDev: Double,
            medianDeviationCents: Double,
            detectionRate: Double,
            totalSamples: Int,
            lowestFrequency: Double?,
            highestFrequency: Double?
        ) {
            self.averageDeviationCents = averageDeviationCents
            self.deviationStdDev = deviationStdDev
            self.medianDeviationCents = medianDeviationCents
            self.detectionRate = detectionRate
            self.totalSamples = totalSamples
            self.lowestFrequency = lowestFrequency
            self.highestFrequency = highestFrequency
        }
    }

    // MARK: - Position Statistics (within scale pattern)

    public struct PositionStatistics: Equatable, Identifiable {
        public var id: Int { position }

        /// Position in scale (1 = first note, 2 = second, etc.)
        public let position: Int

        /// Average deviation in cents (signed: + = sharp, - = flat)
        public let averageDeviationCents: Double

        /// Standard deviation of pitch deviation
        public let deviationStdDev: Double

        /// Number of samples for this position
        public let sampleCount: Int

        public init(
            position: Int,
            averageDeviationCents: Double,
            deviationStdDev: Double,
            sampleCount: Int
        ) {
            self.position = position
            self.averageDeviationCents = averageDeviationCents
            self.deviationStdDev = deviationStdDev
            self.sampleCount = sampleCount
        }
    }

    // MARK: - Pitch Statistics (actual note frequencies)

    public struct PitchStatistics: Equatable, Identifiable {
        public var id: String { noteName }

        /// Note name (e.g., "C4", "D#5")
        public let noteName: String

        /// MIDI note number
        public let midiNoteNumber: Int

        /// Frequency in Hz
        public let frequency: Double

        /// Average deviation in cents (signed: + = sharp, - = flat)
        public let averageDeviationCents: Double

        /// Standard deviation of pitch deviation
        public let deviationStdDev: Double

        /// Number of occurrences (how many times this note appeared)
        public let occurrenceCount: Int

        /// Total sample count across all occurrences
        public let sampleCount: Int

        public init(
            noteName: String,
            midiNoteNumber: Int,
            frequency: Double,
            averageDeviationCents: Double,
            deviationStdDev: Double,
            occurrenceCount: Int,
            sampleCount: Int
        ) {
            self.noteName = noteName
            self.midiNoteNumber = midiNoteNumber
            self.frequency = frequency
            self.averageDeviationCents = averageDeviationCents
            self.deviationStdDev = deviationStdDev
            self.occurrenceCount = occurrenceCount
            self.sampleCount = sampleCount
        }
    }
}

// MARK: - MIDINote Extension for frequency to note name

extension MIDINote {
    /// Get note name for a given frequency
    public static func noteName(forFrequency frequency: Double) -> String? {
        guard frequency > 0 else { return nil }

        // A4 = 440Hz = MIDI 69
        let midiNumber = 69 + 12 * log2(frequency / 440.0)
        let roundedMidi = Int(round(midiNumber))

        guard roundedMidi >= 0 && roundedMidi <= 127 else { return nil }

        let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        let noteName = noteNames[roundedMidi % 12]
        let octave = (roundedMidi / 12) - 1

        return "\(noteName)\(octave)"
    }
}
