import Foundation
import VocalisDomain

/// ViewModel for recording settings configuration
/// Uses ScalePresetSettings as Single Source of Truth for type-safe preset persistence
public class RecordingSettingsViewModel: ObservableObject {
    /// The underlying settings - Single Source of Truth
    /// Preset save/load operates directly on this property
    @Published public var settings: ScalePresetSettings = ScalePresetSettings()

    // MARK: - Computed Properties for UI Binding
    // These provide convenient access to settings properties

    public var scaleType: ScaleType {
        get { settings.scaleType }
        set { settings.scaleType = newValue }
    }

    public var startPitchIndex: Int {
        get { settings.startPitchIndex }
        set { settings.startPitchIndex = newValue }
    }

    public var tempo: Int {
        get { settings.tempo }
        set { settings.tempo = newValue }
    }

    public var keyProgressionPattern: KeyProgressionPattern {
        get { settings.keyProgressionPattern }
        set { settings.keyProgressionPattern = newValue }
    }

    public var ascendingKeyCount: Int {
        get { settings.ascendingKeyCount }
        set { settings.ascendingKeyCount = newValue }
    }

    public var descendingKeyCount: Int {
        get { settings.descendingKeyCount }
        set { settings.descendingKeyCount = newValue }
    }

    public var ascendingKeyStepInterval: Int {
        get { settings.ascendingKeyStepInterval }
        set { settings.ascendingKeyStepInterval = newValue }
    }

    public var descendingKeyStepInterval: Int {
        get { settings.descendingKeyStepInterval }
        set { settings.descendingKeyStepInterval = newValue }
    }

    /// Backwards compatibility
    public var ascendingCount: Int {
        get { ascendingKeyCount }
        set { ascendingKeyCount = newValue }
    }

    /// Available pitches sorted from high to low (intuitive: top = high pitch)
    public let availablePitches = [
        "C6",
        "B5", "A#5", "A5", "G#5", "G5", "F#5", "F5", "E5", "D#5", "D5", "C#5", "C5",
        "B4", "A#4", "A4", "G#4", "G4", "F#4", "F4", "E4", "D#4", "D4", "C#4", "C4",
        "B3", "A#3", "A3", "G#3", "G3", "F#3", "F3", "E3", "D#3", "D3", "C#3", "C3",
        "B2", "A#2", "A2", "G#2", "G2", "F#2", "F2", "E2", "D#2", "D2", "C#2", "C2"
    ]

    public var isSettingsEnabled: Bool {
        scaleType != .off
    }

    /// Whether to show ascending key count control
    public var showsAscendingKeyCount: Bool {
        keyProgressionPattern.showsAscendingCount
    }

    /// Whether to show descending key count control
    public var showsDescendingKeyCount: Bool {
        keyProgressionPattern.showsDescendingCount
    }

    // MARK: - MIDI Range Validation

    /// Scale pattern offset (highest note relative to root)
    /// Uses ScaleType.patternOffset from domain layer
    private var scalePatternOffset: Int {
        scaleType.patternOffset
    }

    /// Starting MIDI note number based on pitch index
    /// Array is sorted high to low: index 0 = C6 (MIDI 84), index 48 = C2 (MIDI 36)
    private var startMIDINote: Int {
        84 - startPitchIndex  // C6 = MIDI 84, C2 = MIDI 36
    }

    /// Calculate the highest MIDI note that will be generated with current settings
    /// Formula: highestRoot + scalePatternOffset
    /// where highestRoot depends on key progression pattern
    /// NOTE: "N ascending" means N key transitions (not N-1)
    /// Example: 3 ascending from C3 = C3 → C#3 → D3 → D#3 (peak at D#3)
    public var highestMIDINote: Int {
        guard scaleType != .off else { return startMIDINote }

        let start = startMIDINote

        switch keyProgressionPattern {
        case .ascendingOnly:
            // Highest root: start + count * interval (N ascending = N transitions)
            let highestRoot = start + ascendingKeyCount * ascendingKeyStepInterval
            return highestRoot + scalePatternOffset

        case .descendingOnly:
            // Highest is the starting note
            return start + scalePatternOffset

        case .ascendingThenDescending:
            // Highest root is at the peak of ascending
            let highestRoot = start + ascendingKeyCount * ascendingKeyStepInterval
            return highestRoot + scalePatternOffset

        case .descendingThenAscending:
            // After descending, we ascend back up
            let valley = start - descendingKeyCount * descendingKeyStepInterval
            let highestRoot = valley + ascendingKeyCount * ascendingKeyStepInterval
            // Highest is either start or the return point, whichever is higher
            return max(start, highestRoot) + scalePatternOffset
        }
    }

    /// Calculate the lowest MIDI note that will be generated with current settings
    /// NOTE: "N descending" means N key transitions (not N-1)
    /// Example: 3 descending from D#3 = D#3 → D3 → C#3 → C3 (lowest at C3)
    public var lowestMIDINote: Int {
        guard scaleType != .off else { return startMIDINote }

        let start = startMIDINote

        switch keyProgressionPattern {
        case .ascendingOnly:
            // Lowest is the starting note
            return start

        case .descendingOnly:
            // Lowest root: start - count * interval (N descending = N transitions)
            return start - descendingKeyCount * descendingKeyStepInterval

        case .ascendingThenDescending:
            // After ascending, we descend from peak
            let peak = start + ascendingKeyCount * ascendingKeyStepInterval
            let lowestAfterDescend = peak - descendingKeyCount * descendingKeyStepInterval
            // Lowest is either start or the descent point, whichever is lower
            return min(start, lowestAfterDescend)

        case .descendingThenAscending:
            // Lowest root is at the valley of descending
            return start - descendingKeyCount * descendingKeyStepInterval
        }
    }

    /// Whether the current settings produce valid MIDI range (0-127)
    public var isValidMIDIRange: Bool {
        guard scaleType != .off else { return true }
        return lowestMIDINote >= 0 && highestMIDINote <= 127
    }

    /// Whether recording can be started with current settings
    public var canStartRecording: Bool {
        scaleType == .off || isValidMIDIRange
    }

    /// Warning message if MIDI range is invalid, nil if valid
    public var midiRangeWarning: String? {
        guard scaleType != .off else { return nil }

        if highestMIDINote > 127 {
            return "warning.midi_range.too_high".localized
        }
        if lowestMIDINote < 0 {
            return "warning.midi_range.too_low".localized
        }
        return nil
    }

    public init() {}

    /// Generate ScaleSettings from current UI settings
    public func generateScaleSettings() -> ScaleSettings? {
        guard scaleType != .off, let notePattern = scaleType.notePattern else {
            return nil // Scale off - no settings
        }

        // Calculate MIDI note number from reversed index (C6=0, C2=48)
        let midiNoteNumber = 84 - startPitchIndex

        // Calculate end note (one octave up) - kept for compatibility but not used
        let endNoteNumber = midiNoteNumber + 12

        // Calculate tempo (convert BPM to seconds per note)
        // At 120 BPM, each quarter note is 0.5 seconds
        let secondsPerNote = 60.0 / Double(tempo)

        do {
            let settings = ScaleSettings(
                startNote: try MIDINote(UInt8(midiNoteNumber)),
                endNote: try MIDINote(UInt8(endNoteNumber)),
                notePattern: notePattern,
                tempo: try Tempo(secondsPerNote: secondsPerNote),
                keyProgressionPattern: keyProgressionPattern,
                ascendingKeyCount: ascendingKeyCount,
                descendingKeyCount: descendingKeyCount,
                ascendingKeyStepInterval: ascendingKeyStepInterval,
                descendingKeyStepInterval: descendingKeyStepInterval
            )
            return settings
        } catch {
            return nil
        }
    }
}
