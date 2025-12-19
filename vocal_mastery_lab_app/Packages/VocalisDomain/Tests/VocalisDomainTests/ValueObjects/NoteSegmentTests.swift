import XCTest
@testable import VocalisDomain

final class NoteSegmentTests: XCTestCase {

    // MARK: - Initialization Tests

    func testInit_WithValidParameters_ShouldCreateSegment() throws {
        // Given
        let note = try MIDINote(60)
        let startTime: TimeInterval = 1.0
        let endTime: TimeInterval = 2.0

        // When
        let segment = NoteSegment(note: note, startTime: startTime, endTime: endTime)

        // Then
        XCTAssertEqual(segment.note, note)
        XCTAssertEqual(segment.startTime, 1.0)
        XCTAssertEqual(segment.endTime, 2.0)
    }

    func testInit_ShouldGenerateUniqueId() throws {
        // Given
        let note = try MIDINote(60)

        // When
        let segment1 = NoteSegment(note: note, startTime: 0.0, endTime: 1.0)
        let segment2 = NoteSegment(note: note, startTime: 0.0, endTime: 1.0)

        // Then
        XCTAssertNotEqual(segment1.id, segment2.id)
    }

    // MARK: - Duration Tests

    func testDuration_ShouldReturnDifference() throws {
        // Given
        let note = try MIDINote(60)
        let segment = NoteSegment(note: note, startTime: 1.5, endTime: 3.0)

        // When
        let duration = segment.duration

        // Then
        XCTAssertEqual(duration, 1.5, accuracy: 0.001)
    }

    func testDuration_WithZeroDuration_ShouldReturnZero() throws {
        // Given
        let note = try MIDINote(60)
        let segment = NoteSegment(note: note, startTime: 1.0, endTime: 1.0)

        // When
        let duration = segment.duration

        // Then
        XCTAssertEqual(duration, 0.0)
    }

    // MARK: - Frequency Tests

    func testFrequency_ShouldReturnNoteFrequency() throws {
        // Given: A4 (440 Hz)
        let note = try MIDINote(69)
        let segment = NoteSegment(note: note, startTime: 0.0, endTime: 1.0)

        // When
        let frequency = segment.frequency

        // Then
        XCTAssertEqual(frequency, 440.0, accuracy: 0.01)
    }

    func testFrequency_MiddleC_ShouldReturnCorrectFrequency() throws {
        // Given: C4 (261.63 Hz)
        let note = MIDINote.middleC
        let segment = NoteSegment(note: note, startTime: 0.0, endTime: 1.0)

        // When
        let frequency = segment.frequency

        // Then
        XCTAssertEqual(frequency, 261.63, accuracy: 0.01)
    }

    // MARK: - Identifiable Tests

    func testIdentifiable_IdShouldBeAccessible() throws {
        // Given
        let note = try MIDINote(60)
        let segment = NoteSegment(note: note, startTime: 0.0, endTime: 1.0)

        // When & Then
        XCTAssertNotNil(segment.id)
    }

    // MARK: - Equatable Tests

    func testEquatable_SameIdAndValues_ShouldBeEqual() throws {
        // Given
        let note = try MIDINote(60)
        let id = UUID()
        let segment1 = NoteSegment(id: id, note: note, startTime: 0.0, endTime: 1.0)
        let segment2 = NoteSegment(id: id, note: note, startTime: 0.0, endTime: 1.0)

        // When & Then
        XCTAssertEqual(segment1, segment2)
    }

    func testEquatable_DifferentId_ShouldNotBeEqual() throws {
        // Given
        let note = try MIDINote(60)
        let segment1 = NoteSegment(note: note, startTime: 0.0, endTime: 1.0)
        let segment2 = NoteSegment(note: note, startTime: 0.0, endTime: 1.0)

        // When & Then
        XCTAssertNotEqual(segment1, segment2)
    }

    // MARK: - Boundary Tests

    func testInit_WithLongDuration_ShouldWork() throws {
        // Given: 1 hour segment
        let note = try MIDINote(60)
        let segment = NoteSegment(note: note, startTime: 0.0, endTime: 3600.0)

        // When
        let duration = segment.duration

        // Then
        XCTAssertEqual(duration, 3600.0)
    }

    func testInit_WithBoundaryMIDINotes_ShouldWork() throws {
        // Given
        let minNote = try MIDINote(0)
        let maxNote = try MIDINote(127)

        // When
        let segmentMin = NoteSegment(note: minNote, startTime: 0.0, endTime: 1.0)
        let segmentMax = NoteSegment(note: maxNote, startTime: 0.0, endTime: 1.0)

        // Then
        XCTAssertEqual(segmentMin.note.value, 0)
        XCTAssertEqual(segmentMax.note.value, 127)
    }
}
