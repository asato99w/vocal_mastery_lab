import Foundation

/// Helper for pitch name operations shared across the app
struct PitchNameHelper {
    /// All pitch names from C2 to C6 (49 pitches)
    static let pitchNames = [
        "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2",
        "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
        "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
        "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
        "C6"
    ]

    /// Base MIDI note number (C2 = 36)
    static let baseMIDI: UInt8 = 36

    /// Get pitch name for a given index (0 = C2, 48 = C6)
    /// - Parameter index: Index in the pitchNames array
    /// - Returns: Pitch name or "C3" if index is out of range
    static func name(forIndex index: Int) -> String {
        guard index >= 0, index < pitchNames.count else { return "C3" }
        return pitchNames[index]
    }

    /// Get pitch name for a given MIDI note number
    /// - Parameter midi: MIDI note number (36 = C2, 84 = C6)
    /// - Returns: Pitch name or "C3" if out of range
    static func name(forMIDI midi: UInt8) -> String {
        let index = Int(midi) - Int(baseMIDI)
        return name(forIndex: index)
    }

    /// Get index for a given MIDI note number
    /// - Parameter midi: MIDI note number
    /// - Returns: Index in pitchNames array (clamped to valid range)
    static func index(forMIDI midi: UInt8) -> Int {
        let index = Int(midi) - Int(baseMIDI)
        return max(0, min(index, pitchNames.count - 1))
    }

    /// Get MIDI note number for a given index
    /// - Parameter index: Index in pitchNames array
    /// - Returns: MIDI note number
    static func midi(forIndex index: Int) -> UInt8 {
        return UInt8(index) + baseMIDI
    }
}
