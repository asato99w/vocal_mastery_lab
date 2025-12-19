import Foundation

/// Recording entity
public struct Recording: Equatable, Identifiable, Codable, Hashable {
    public let id: RecordingId
    public let fileURL: URL
    public let createdAt: Date
    public let duration: Duration
    public var title: String?  // User-defined custom name
    public var analysisAlgorithm: PitchDetectionAlgorithm?  // Algorithm used for last analysis (nil = not analyzed yet)

    public init(
        id: RecordingId = RecordingId(),
        fileURL: URL,
        createdAt: Date = Date(),
        duration: Duration,
        title: String? = nil,
        analysisAlgorithm: PitchDetectionAlgorithm? = nil
    ) {
        self.id = id
        self.fileURL = fileURL
        self.createdAt = createdAt
        self.duration = duration
        self.title = title
        self.analysisAlgorithm = analysisAlgorithm
    }

    // MARK: - Codable (backward compatibility)

    private enum CodingKeys: String, CodingKey {
        case id, fileURL, createdAt, duration, title, analysisAlgorithm
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(RecordingId.self, forKey: .id)
        fileURL = try container.decode(URL.self, forKey: .fileURL)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        duration = try container.decode(Duration.self, forKey: .duration)
        title = try container.decodeIfPresent(String.self, forKey: .title)
        // Backward compatibility: old recordings without analysisAlgorithm default to nil
        analysisAlgorithm = try container.decodeIfPresent(PitchDetectionAlgorithm.self, forKey: .analysisAlgorithm)
    }

    /// Formatted creation date for display (compact format)
    public var formattedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter.string(from: createdAt)
    }
}

// MARK: - Codable conformance for Duration
extension Duration: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let seconds = try container.decode(TimeInterval.self)
        self.init(seconds: seconds)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(seconds)
    }
}
