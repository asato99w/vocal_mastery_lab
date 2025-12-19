import Foundation

/// Note pattern for scale generation
public enum NotePattern: Equatable, Codable, Hashable {
    case fiveToneScale      // ドレミファソ (Root, +2, +4, +5, +7)
    case fiveToneDown       // ソファミレド (descending only: +7, +5, +4, +2, Root)
    case octaveRepeat       // オクターブリピート (Root, +4, +8, +12 with top repeat)
    case brokenScale        // ブロークン・スケール 1→5→3→8→5→3→1 (single)
    case brokenScaleDouble  // ブロークン・スケール 1→5→3→8→5→3→1 (x2)
    case rossiniScale       // ロッシーニ 1.5オクターブスケール
    case arpeggioDownTriple // アルペジオ下降3連 8→5→3→1→3→5→8→5→3→1→3→5→8→5→3→1

    /// Intervals from the root note (in semitones)
    public var intervals: [Int] {
        switch self {
        case .fiveToneScale, .fiveToneDown:
            return [0, 2, 4, 5, 7]  // C, D, E, F, G
        case .octaveRepeat:
            return [0, 4, 7, 12]  // C, E, G, C (major triad + octave)
        case .brokenScale, .brokenScaleDouble:
            return [0, 4, 7, 12]  // Chord tones: Root, 3rd, 5th, Octave
        case .rossiniScale:
            return [0, 4, 7, 12, 16, 19]  // 1.5 octave: Root, 3rd, 5th, 8va, 10th, 12th
        case .arpeggioDownTriple:
            return [0, 4, 7, 12]  // Chord tones: Root, 3rd, 5th, Octave
        }
    }

    /// Generate ascending then descending pattern
    /// Example: [0, 2, 4, 5, 7, 5, 4, 2, 0] for C-D-E-F-G-F-E-D-C
    public func ascendingDescending() -> [Int] {
        let ascending = intervals
        let descending = intervals.dropLast().reversed()
        return ascending + descending
    }

    /// Playback pattern for actual note sequence
    /// Allows complex patterns like top note repeats
    public var playbackPattern: [Int] {
        switch self {
        case .fiveToneScale:
            return [0, 2, 4, 5, 7, 5, 4, 2, 0]
        case .fiveToneDown:
            // Descending only: G-F-E-D-C = 5 notes
            return [7, 5, 4, 2, 0]
        case .octaveRepeat:
            return [0, 4, 7, 12, 12, 12, 12, 7, 4, 0]
        case .brokenScale:
            // 1→5→3→8→5→3→1 = 7 notes (single)
            return [0, 7, 4, 12, 7, 4, 0]
        case .brokenScaleDouble:
            // 1→5→3→8→5→3→1 (x2) = 13 notes
            return [0, 7, 4, 12, 7, 4, 0, 7, 4, 12, 7, 4, 0]
        case .rossiniScale:
            // 1→3→5→8→10→12→11→9→7→5→4→2→1 = 13 notes
            // Scale degrees to semitones: 1=0, 3=4, 5=7, 8=12, 10=16, 12=19, 11=17, 9=14, 7=11, 5=7, 4=5, 2=2, 1=0
            return [0, 4, 7, 12, 16, 19, 17, 14, 11, 7, 5, 2, 0]
        case .arpeggioDownTriple:
            // 8→5→3→1→3→5→8→5→3→1→3→5→8→5→3→1 = 16 notes
            // Scale degrees to semitones: 8=12, 5=7, 3=4, 1=0
            return [12, 7, 4, 0, 4, 7, 12, 7, 4, 0, 4, 7, 12, 7, 4, 0]
        }
    }

    /// Localization key for the pattern name
    /// Use this key to look up the localized display name in Localizable.strings
    /// Uses recording.scale_* keys for consistency across selection UI and recording display
    public var displayNameKey: String {
        switch self {
        case .fiveToneScale:
            return "recording.scale_five_tone"
        case .fiveToneDown:
            return "recording.scale_five_tone_down"
        case .octaveRepeat:
            return "recording.scale_octave_repeat"
        case .brokenScale:
            return "recording.scale_broken"
        case .brokenScaleDouble:
            return "recording.scale_broken_double"
        case .rossiniScale:
            return "recording.scale_rossini"
        case .arpeggioDownTriple:
            return "recording.scale_arpeggio_down_triple"
        }
    }

    /// Non-localized display name (for logging/debugging only)
    /// For user-facing display, use displayNameKey with localization
    public var displayName: String {
        switch self {
        case .fiveToneScale:
            return "Five-Tone Scale"
        case .fiveToneDown:
            return "Five-Tone Down"
        case .octaveRepeat:
            return "Octave Repeat"
        case .brokenScale:
            return "Broken Scale"
        case .brokenScaleDouble:
            return "Broken Scale (x2)"
        case .rossiniScale:
            return "Rossini Scale"
        case .arpeggioDownTriple:
            return "Arpeggio Down Triple"
        }
    }
}
