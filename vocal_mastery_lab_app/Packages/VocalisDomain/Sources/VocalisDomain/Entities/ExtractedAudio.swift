import Foundation

/// Extracted audio entity representing a separated vocal or instrumental track
public struct ExtractedAudio: Equatable, Identifiable, Codable, Hashable {
    public let id: ExtractedAudioId
    public let sourceRecordingId: RecordingId
    public let type: ExtractionType
    public let fileURL: URL
    public let createdAt: Date
    public let duration: Duration

    public init(
        id: ExtractedAudioId = ExtractedAudioId(),
        sourceRecordingId: RecordingId,
        type: ExtractionType,
        fileURL: URL,
        createdAt: Date = Date(),
        duration: Duration
    ) {
        self.id = id
        self.sourceRecordingId = sourceRecordingId
        self.type = type
        self.fileURL = fileURL
        self.createdAt = createdAt
        self.duration = duration
    }

    /// Formatted creation date for display (compact format)
    public var formattedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter.string(from: createdAt)
    }
}
