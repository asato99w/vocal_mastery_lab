import XCTest
import VocalisDomain
@testable import VocalMasteryLab

final class RecordingTests: XCTestCase {

    // MARK: - Codable Tests

    func testRecording_EncodesAndDecodesWithAlgorithm() throws {
        // Given: Recording with analysisAlgorithm
        let recording = Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0),
            scaleSettings: nil,
            title: "Test Recording",
            playbackTimeline: nil,
            analysisAlgorithm: .pyinDefault
        )

        // When: Encoding and decoding
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let data = try encoder.encode(recording)
        let decoded = try decoder.decode(Recording.self, from: data)

        // Then: analysisAlgorithm should be preserved
        XCTAssertEqual(decoded.analysisAlgorithm, .pyinDefault)
        XCTAssertEqual(decoded.id, recording.id)
        XCTAssertEqual(decoded.title, recording.title)
    }

    func testRecording_DecodesWithoutAlgorithm_DefaultsToNil() throws {
        // Given: JSON without analysisAlgorithm (old format)
        // Note: RecordingId uses singleValueContainer, so it's encoded as a plain UUID string
        let oldFormatJSON = """
        {
            "id": "12345678-1234-1234-1234-123456789ABC",
            "fileURL": "file:///tmp/test.m4a",
            "createdAt": 0,
            "duration": 10.0
        }
        """
        let data = oldFormatJSON.data(using: .utf8)!

        // When: Decoding
        let decoder = JSONDecoder()
        let recording = try decoder.decode(Recording.self, from: data)

        // Then: analysisAlgorithm should be nil (backward compatibility)
        XCTAssertNil(recording.analysisAlgorithm)
    }

    func testRecording_EncodesWithNilAlgorithm() throws {
        // Given: Recording without analysisAlgorithm
        let recording = Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0),
            scaleSettings: nil,
            analysisAlgorithm: nil
        )

        // When: Encoding
        let encoder = JSONEncoder()
        let data = try encoder.encode(recording)
        let jsonString = String(data: data, encoding: .utf8)!

        // Then: analysisAlgorithm should not be included or be null
        // (Either is acceptable for optional fields)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(Recording.self, from: data)
        XCTAssertNil(decoded.analysisAlgorithm)
    }

    func testRecording_DecodesYINAlgorithm() throws {
        // Given: Recording with YIN algorithm
        let recording = Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0),
            scaleSettings: nil,
            analysisAlgorithm: .yin
        )

        // When: Encoding and decoding
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let data = try encoder.encode(recording)
        let decoded = try decoder.decode(Recording.self, from: data)

        // Then: YIN should be preserved
        XCTAssertEqual(decoded.analysisAlgorithm, .yin)
    }

    // MARK: - Property Tests

    func testRecording_InitWithAllParameters() {
        // Given/When: Creating recording with all parameters
        let id = RecordingId()
        let fileURL = URL(fileURLWithPath: "/tmp/test.m4a")
        let createdAt = Date()
        let duration = Duration(seconds: 30.0)
        let scaleSettings = ScaleSettings(
            startNote: try! MIDINote(60),
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 0.5)
        )
        let title = "My Recording"
        let algorithm: PitchDetectionAlgorithm = .pyinDefault

        let recording = Recording(
            id: id,
            fileURL: fileURL,
            createdAt: createdAt,
            duration: duration,
            scaleSettings: scaleSettings,
            title: title,
            playbackTimeline: nil,
            analysisAlgorithm: algorithm
        )

        // Then: All properties should be set correctly
        XCTAssertEqual(recording.id, id)
        XCTAssertEqual(recording.fileURL, fileURL)
        XCTAssertEqual(recording.createdAt, createdAt)
        XCTAssertEqual(recording.duration, duration)
        XCTAssertEqual(recording.scaleSettings, scaleSettings)
        XCTAssertEqual(recording.title, title)
        XCTAssertEqual(recording.analysisAlgorithm, algorithm)
    }

    func testRecording_MutableAnalysisAlgorithm() {
        // Given: Recording without algorithm
        var recording = Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0),
            scaleSettings: nil,
            analysisAlgorithm: nil
        )

        // When: Updating algorithm
        recording.analysisAlgorithm = .pyinDefault

        // Then: Algorithm should be updated
        XCTAssertEqual(recording.analysisAlgorithm, .pyinDefault)

        // When: Changing to different algorithm
        recording.analysisAlgorithm = .yin

        // Then: Algorithm should be changed
        XCTAssertEqual(recording.analysisAlgorithm, .yin)
    }
}
