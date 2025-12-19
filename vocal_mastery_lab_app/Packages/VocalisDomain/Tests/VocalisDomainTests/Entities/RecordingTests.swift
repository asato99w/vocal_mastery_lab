import XCTest
@testable import VocalisDomain

final class RecordingTests: XCTestCase {
    func testInit_DefaultValues() {
        // Given
        let url = URL(fileURLWithPath: "/test/recording.m4a")
        let duration = Duration(seconds: 117)
        let settings = ScaleSettings.mvpDefault
        
        // When
        let recording = Recording(
            fileURL: url,
            duration: duration,
            scaleSettings: settings
        )
        
        // Then
        XCTAssertNotNil(recording.id)
        XCTAssertEqual(recording.fileURL, url)
        XCTAssertEqual(recording.duration, duration)
        XCTAssertEqual(recording.scaleSettings, settings)
    }
    
    func testIdentifiable() {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/test1.m4a"),
            duration: Duration(seconds: 100),
            scaleSettings: .mvpDefault
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/test2.m4a"),
            duration: Duration(seconds: 100),
            scaleSettings: .mvpDefault
        )
        
        // When & Then
        XCTAssertNotEqual(recording1.id, recording2.id)
    }
    
    func testFormattedDate() {
        // Given
        let date = Date()
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/test.m4a"),
            createdAt: date,
            duration: Duration(seconds: 100),
            scaleSettings: .mvpDefault
        )
        
        // When
        let formatted = recording.formattedDate
        
        // Then
        XCTAssertFalse(formatted.isEmpty)
    }
    
    func testCodable() throws {
        // Given
        let original = Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 117),
            scaleSettings: .mvpDefault
        )

        // When
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(Recording.self, from: encoded)

        // Then
        XCTAssertEqual(decoded.id, original.id)
        XCTAssertEqual(decoded.fileURL, original.fileURL)
        XCTAssertEqual(decoded.duration.seconds, original.duration.seconds, accuracy: 0.001)
        XCTAssertEqual(decoded.scaleSettings, original.scaleSettings)
    }

    func testScaleDisplayName_WithScaleSettings() throws {
        // Given
        let settings = ScaleSettings(
            startNote: try MIDINote(60),  // C4
            endNote: try MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: .standard,
            ascendingCount: 12
        )
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/test.m4a"),
            duration: Duration(seconds: 100),
            scaleSettings: settings
        )

        // When
        let displayName = recording.scaleDisplayName

        // Then: scaleDisplayName uses non-localized displayName (English)
        XCTAssertEqual(displayName, "C4 Five-Tone Scale")
    }

    func testScaleDisplayName_WithoutScaleSettings() {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/test.m4a"),
            duration: Duration(seconds: 100),
            scaleSettings: nil
        )

        // When
        let displayName = recording.scaleDisplayName

        // Then
        XCTAssertNil(displayName)
    }

    // MARK: - Title Tests

    func testTitle_DefaultIsNil() {
        // Given & When
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/test.m4a"),
            duration: Duration(seconds: 100),
            scaleSettings: .mvpDefault
        )

        // Then
        XCTAssertNil(recording.title)
    }

    func testTitle_CanBeSet() {
        // Given & When
        var recording = Recording(
            fileURL: URL(fileURLWithPath: "/test.m4a"),
            duration: Duration(seconds: 100),
            scaleSettings: .mvpDefault,
            title: "お気に入りのテイク"
        )

        // Then
        XCTAssertEqual(recording.title, "お気に入りのテイク")

        // When - title can be modified
        recording.title = "新しい名前"

        // Then
        XCTAssertEqual(recording.title, "新しい名前")
    }

    func testCodable_WithTitle() throws {
        // Given
        let original = Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 117),
            scaleSettings: .mvpDefault,
            title: "テスト録音"
        )

        // When
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(Recording.self, from: encoded)

        // Then
        XCTAssertEqual(decoded.title, "テスト録音")
    }

    func testCodable_WithNilTitle() throws {
        // Given
        let original = Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 117),
            scaleSettings: .mvpDefault,
            title: nil
        )

        // When
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(Recording.self, from: encoded)

        // Then
        XCTAssertNil(decoded.title)
    }

    // MARK: - PlaybackTimeline Tests

    func testPlaybackTimeline_DefaultIsNil() {
        // Given & When
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/test.m4a"),
            duration: Duration(seconds: 100),
            scaleSettings: .mvpDefault
        )

        // Then
        XCTAssertNil(recording.playbackTimeline)
    }

    func testPlaybackTimeline_CanBeSet() throws {
        // Given
        let note = try MIDINote(60)
        let events = [
            ScalePlaybackEvent(timestamp: 0.0, note: note, eventType: .noteStart),
            ScalePlaybackEvent(timestamp: 1.0, note: note, eventType: .noteEnd)
        ]
        let timeline = ScalePlaybackTimeline(events: events, recordingStartTime: Date())

        // When
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/test.m4a"),
            duration: Duration(seconds: 100),
            scaleSettings: .mvpDefault,
            playbackTimeline: timeline
        )

        // Then
        XCTAssertNotNil(recording.playbackTimeline)
        XCTAssertEqual(recording.playbackTimeline?.events.count, 2)
    }

    func testCodable_WithPlaybackTimeline() throws {
        // Given
        let note = try MIDINote(60)
        let events = [
            ScalePlaybackEvent(timestamp: 0.0, note: note, eventType: .noteStart),
            ScalePlaybackEvent(timestamp: 1.0, note: note, eventType: .noteEnd)
        ]
        let timeline = ScalePlaybackTimeline(events: events, recordingStartTime: Date())
        let original = Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 117),
            scaleSettings: .mvpDefault,
            playbackTimeline: timeline
        )

        // When
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(Recording.self, from: encoded)

        // Then
        XCTAssertNotNil(decoded.playbackTimeline)
        XCTAssertEqual(decoded.playbackTimeline?.events.count, 2)
    }

    func testCodable_WithNilPlaybackTimeline() throws {
        // Given
        let original = Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 117),
            scaleSettings: .mvpDefault,
            playbackTimeline: nil
        )

        // When
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(Recording.self, from: encoded)

        // Then
        XCTAssertNil(decoded.playbackTimeline)
    }

    func testBackwardCompatibility_OldRecordingWithoutTimeline() throws {
        // Given: JSON data from old format without playbackTimeline
        let oldFormatJSON = """
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "fileURL": "file:///test.m4a",
            "createdAt": 0,
            "duration": 100,
            "scaleSettings": null,
            "title": null
        }
        """
        let data = oldFormatJSON.data(using: .utf8)!

        // When
        let decoded = try JSONDecoder().decode(Recording.self, from: data)

        // Then: Should decode without error, playbackTimeline should be nil
        XCTAssertNil(decoded.playbackTimeline)
    }
}
