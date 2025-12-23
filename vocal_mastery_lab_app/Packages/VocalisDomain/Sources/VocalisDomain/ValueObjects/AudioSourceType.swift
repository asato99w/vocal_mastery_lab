import Foundation

/// Type of audio source for playback
public enum AudioSourceType: String, CaseIterable, Equatable, Hashable {
    case original
    case vocal
    case instrumental

    /// Display name for the audio source
    public var displayName: String {
        switch self {
        case .original:
            return "オリジナル"
        case .vocal:
            return "ボーカル"
        case .instrumental:
            return "伴奏"
        }
    }

    /// SF Symbol icon name
    public var iconName: String {
        switch self {
        case .original:
            return "waveform"
        case .vocal:
            return "person.wave.2"
        case .instrumental:
            return "music.note.list"
        }
    }
}
