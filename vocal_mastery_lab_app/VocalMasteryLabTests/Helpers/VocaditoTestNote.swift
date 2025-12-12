import Foundation

/// Represents a single note annotation from Vocadito dataset
struct VocaditoTestNote: Codable {
    let index: Int
    let startTime: Double
    let frequency: Double
    let duration: Double

    /// Get the midpoint time of this note (optimal detection point)
    var midTime: Double {
        return startTime + (duration / 2.0)
    }
}

/// Represents a track with its audio file and note annotations
struct VocaditoTrack: Codable {
    let audioFile: String
    let notes: [VocaditoTestNote]
}

/// Root structure of TestNotes.json
struct VocaditoTestData: Codable {
    let description: String
    let format: String
    let tracks: [String: VocaditoTrack]
}

/// Helper for loading and accessing Vocadito test data
enum VocaditoTestDataLoader {

    private static var cachedData: VocaditoTestData?

    /// Load TestNotes.json from test resources
    static func loadTestData() throws -> VocaditoTestData {
        if let cached = cachedData {
            return cached
        }

        let jsonPath = TestResourceLoader.getVocaditoTestNotesPath()
        let jsonURL = URL(fileURLWithPath: jsonPath)
        let jsonData = try Data(contentsOf: jsonURL)
        let decoder = JSONDecoder()
        let testData = try decoder.decode(VocaditoTestData.self, from: jsonData)

        cachedData = testData
        return testData
    }

    /// Get all notes for a specific track
    /// - Parameter trackName: Track identifier (e.g., "vocadito_1")
    /// - Returns: Array of test notes
    static func getNotes(for trackName: String) throws -> [VocaditoTestNote] {
        let testData = try loadTestData()
        guard let track = testData.tracks[trackName] else {
            throw VocaditoTestDataError.trackNotFound(trackName)
        }
        return track.notes
    }

    /// Get audio file name for a specific track
    /// - Parameter trackName: Track identifier (e.g., "vocadito_1")
    /// - Returns: Audio file name (e.g., "vocadito_1.wav")
    static func getAudioFileName(for trackName: String) throws -> String {
        let testData = try loadTestData()
        guard let track = testData.tracks[trackName] else {
            throw VocaditoTestDataError.trackNotFound(trackName)
        }
        return track.audioFile
    }

    /// Get all available track names
    static func getAllTrackNames() throws -> [String] {
        let testData = try loadTestData()
        return Array(testData.tracks.keys).sorted()
    }
}

/// Errors for Vocadito test data operations
enum VocaditoTestDataError: Error, LocalizedError {
    case trackNotFound(String)
    case invalidJSON

    var errorDescription: String? {
        switch self {
        case .trackNotFound(let trackName):
            return "Track '\(trackName)' not found in TestNotes.json"
        case .invalidJSON:
            return "Invalid JSON format in TestNotes.json"
        }
    }
}
