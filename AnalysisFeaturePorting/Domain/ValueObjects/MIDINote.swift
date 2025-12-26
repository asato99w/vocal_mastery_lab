import Foundation

/// MIDI note number value object (0-127)
public struct MIDINote: Equatable, Comparable, Hashable {
    public let value: UInt8

    public init(_ value: UInt8) throws {
        guard value <= 127 else {
            throw MIDINoteError.outOfRange(value)
        }
        self.value = value
    }

    // Convenience initializers for common notes
    public static let middleC = try! MIDINote(60)  // C4
    public static let hiC = try! MIDINote(72)      // C5

    // Comparable
    public static func < (lhs: MIDINote, rhs: MIDINote) -> Bool {
        lhs.value < rhs.value
    }
}

public enum MIDINoteError: LocalizedError {
    case outOfRange(UInt8)

    public var errorDescription: String? {
        switch self {
        case .outOfRange(let value):
            return "MIDI note value \(value) is out of range (0-127)"
        }
    }
}

// MARK: - MIDINote Extensions

extension MIDINote {
    /// Get note name from MIDI note number
    public static func noteName(for midiNumber: UInt8) -> String {
        let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        let octave = Int(midiNumber / 12) - 1
        let noteIndex = Int(midiNumber % 12)
        return "\(noteNames[noteIndex])\(octave)"
    }

    /// Get note name for this MIDI note
    public var noteName: String {
        MIDINote.noteName(for: value)
    }

    /// Calculate frequency for this MIDI note
    public var frequency: Double {
        440.0 * pow(2.0, (Double(value) - 69.0) / 12.0)
    }
}
