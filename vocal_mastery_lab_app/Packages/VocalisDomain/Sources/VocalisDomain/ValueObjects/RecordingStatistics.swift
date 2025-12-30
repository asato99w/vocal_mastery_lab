import Foundation

/// Objective statistics calculated from pitch analysis data
/// All values are raw measurements without subjective scoring
public struct RecordingStatistics: Equatable {
    /// Intonation statistics (deviation from nearest semitone)
    public let intonation: IntonationStatistics

    /// Pitch stability statistics (wavering within sustained notes)
    public let pitchStability: PitchStabilityStatistics

    /// Vocal range statistics (extended range info)
    public let vocalRange: VocalRangeStatistics

    /// Vibrato statistics (optional, nil if not enough data)
    public let vibratoStatistics: VibratoStatistics?

    /// Singer's Formant statistics (optional, nil if no spectrum data)
    public let singersFormantStatistics: SingersFormantStatistics?

    /// High frequency statistics (Brightness and Airiness)
    public let highFrequencyStatistics: HighFrequencyStatistics?

    /// Total recording duration
    public let totalDuration: TimeInterval

    /// Pitch detection rate (percentage of time with valid pitch)
    public let detectionRate: Double

    public init(
        intonation: IntonationStatistics,
        pitchStability: PitchStabilityStatistics,
        vocalRange: VocalRangeStatistics,
        vibratoStatistics: VibratoStatistics? = nil,
        singersFormantStatistics: SingersFormantStatistics? = nil,
        highFrequencyStatistics: HighFrequencyStatistics? = nil,
        totalDuration: TimeInterval,
        detectionRate: Double
    ) {
        self.intonation = intonation
        self.pitchStability = pitchStability
        self.vocalRange = vocalRange
        self.vibratoStatistics = vibratoStatistics
        self.singersFormantStatistics = singersFormantStatistics
        self.highFrequencyStatistics = highFrequencyStatistics
        self.totalDuration = totalDuration
        self.detectionRate = detectionRate
    }

    // MARK: - Intonation Statistics (deviation from nearest semitone)

    public struct IntonationStatistics: Equatable {
        /// Average deviation from nearest semitone in cents (absolute value)
        /// Lower is better. Typical range: 5-30 cents
        public let averageDeviationCents: Double

        /// Standard deviation of intonation deviation
        public let deviationStdDev: Double

        /// Percentage of samples within ±20 cents of nearest semitone
        /// Higher is better. Typical range: 60-95%
        public let accuracyRate: Double

        /// Percentage of samples within ±10 cents (excellent accuracy)
        public let excellentAccuracyRate: Double

        public init(
            averageDeviationCents: Double,
            deviationStdDev: Double,
            accuracyRate: Double,
            excellentAccuracyRate: Double
        ) {
            self.averageDeviationCents = averageDeviationCents
            self.deviationStdDev = deviationStdDev
            self.accuracyRate = accuracyRate
            self.excellentAccuracyRate = excellentAccuracyRate
        }
    }

    // MARK: - Pitch Stability Statistics

    public struct PitchStabilityStatistics: Equatable {
        /// Average pitch fluctuation within sustained notes (cents)
        /// Lower is better. Typical range: 5-25 cents
        public let averageFluctuation: Double

        /// Percentage of time with stable pitch (fluctuation < 15 cents)
        /// Higher is better
        public let stabilityRate: Double

        /// Number of sustained note segments analyzed
        public let segmentsAnalyzed: Int

        public init(
            averageFluctuation: Double,
            stabilityRate: Double,
            segmentsAnalyzed: Int
        ) {
            self.averageFluctuation = averageFluctuation
            self.stabilityRate = stabilityRate
            self.segmentsAnalyzed = segmentsAnalyzed
        }
    }

    // MARK: - Vocal Range Statistics

    public struct VocalRangeStatistics: Equatable {
        /// Lowest detected frequency (Hz)
        public let lowestFrequency: Double?

        /// Highest detected frequency (Hz)
        public let highestFrequency: Double?

        /// Total range in semitones
        public let rangeSemitones: Int

        /// Center frequency (geometric mean of used frequencies)
        public let centerFrequency: Double?

        /// Most frequently used note name (mode)
        public let mostUsedNote: String?

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

        /// Center note name
        public var centerNoteName: String? {
            guard let freq = centerFrequency else { return nil }
            return MIDINote.noteName(forFrequency: freq)
        }

        public init(
            lowestFrequency: Double?,
            highestFrequency: Double?,
            rangeSemitones: Int,
            centerFrequency: Double?,
            mostUsedNote: String?
        ) {
            self.lowestFrequency = lowestFrequency
            self.highestFrequency = highestFrequency
            self.rangeSemitones = rangeSemitones
            self.centerFrequency = centerFrequency
            self.mostUsedNote = mostUsedNote
        }
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
