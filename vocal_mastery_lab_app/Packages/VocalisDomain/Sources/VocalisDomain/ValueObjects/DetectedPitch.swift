import Foundation

/// Represents a detected pitch with note name, frequency, and confidence
public struct DetectedPitch: Equatable {
    public let noteName: String
    public let frequency: Double
    public let confidence: Double
    public let cents: Int?

    public init(noteName: String, frequency: Double, confidence: Double, cents: Int? = nil) {
        self.noteName = noteName
        self.frequency = frequency
        self.confidence = confidence
        self.cents = cents
    }

    /// Create DetectedPitch from frequency using MIDI note calculation
    /// A4 (440 Hz) = MIDI note 69
    public static func fromFrequency(_ frequency: Double, confidence: Double) -> DetectedPitch {
        // Calculate MIDI note number from frequency
        let midiNoteNumber = 12 * log2(frequency / 440.0) + 69
        let roundedNote = Int(round(midiNoteNumber))

        // Calculate cents deviation from the nearest note
        let centsDeviation = Int(round((midiNoteNumber - Double(roundedNote)) * 100))

        // Get note name
        let noteName = MIDINote.noteName(for: UInt8(clamping: roundedNote))

        return DetectedPitch(
            noteName: noteName,
            frequency: frequency,
            confidence: confidence,
            cents: centsDeviation
        )
    }
}

/// Classification of pitch accuracy based on cents deviation
public enum PitchAccuracy: Equatable {
    case accurate      // ±10 cents
    case slightlyOff   // ±25 cents
    case off           // > ±25 cents
    case none          // No pitch detected

    public static func from(cents: Int?) -> PitchAccuracy {
        guard let cents = cents else { return .none }
        let absCents = abs(cents)

        if absCents <= 10 { return .accurate }
        else if absCents <= 25 { return .slightlyOff }
        else { return .off }
    }

    public var displayColor: String {
        switch self {
        case .accurate: return "green"
        case .slightlyOff: return "orange"
        case .off: return "red"
        case .none: return "gray"
        }
    }
}
