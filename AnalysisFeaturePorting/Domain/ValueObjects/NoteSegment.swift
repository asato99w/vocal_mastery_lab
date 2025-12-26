import Foundation

/// Note playback segment representing a single target note bar for karaoke-style UI
/// Each segment represents the time period when a specific note was played
public struct NoteSegment: Identifiable, Equatable {
    /// Unique identifier for SwiftUI ForEach
    public let id: UUID

    /// The MIDI note that was played (determines Y-axis position)
    public let note: MIDINote

    /// Start time since recording began (determines X-axis left edge)
    public let startTime: TimeInterval

    /// End time since recording began (determines X-axis right edge)
    public let endTime: TimeInterval

    /// Duration of the note playback
    public var duration: TimeInterval {
        endTime - startTime
    }

    /// Frequency in Hz (for pitch comparison)
    public var frequency: Double {
        note.frequency
    }

    public init(id: UUID = UUID(), note: MIDINote, startTime: TimeInterval, endTime: TimeInterval) {
        self.id = id
        self.note = note
        self.startTime = startTime
        self.endTime = endTime
    }
}
