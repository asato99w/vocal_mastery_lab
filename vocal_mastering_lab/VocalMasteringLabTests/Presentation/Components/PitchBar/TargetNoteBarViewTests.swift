import XCTest
import SwiftUI
@testable import VocalMasteringLab
@testable import VocalisDomain

final class TargetNoteBarViewTests: XCTestCase {

    // MARK: - NoteSegment Tests

    func testNoteSegment_duration_calculatesCorrectly() throws {
        // Given
        let note = try MIDINote(60)  // C4
        let segment = NoteSegment(
            note: note,
            startTime: 1.0,
            endTime: 2.5
        )

        // Then
        XCTAssertEqual(segment.duration, 1.5)
    }

    func testNoteSegment_frequency_returnsNoteFrequency() throws {
        // Given: C4 = 261.63 Hz
        let note = try MIDINote(60)
        let segment = NoteSegment(
            note: note,
            startTime: 0.0,
            endTime: 1.0
        )

        // Then
        XCTAssertEqual(segment.frequency, note.frequency)
    }

    // MARK: - TargetNoteBarRenderer Tests

    func testTargetNoteBarRenderer_calculateBarWidth_returnsCorrectWidth() throws {
        // Given
        let segment = NoteSegment(
            note: try MIDINote(60),
            startTime: 0.0,
            endTime: 2.0  // 2 seconds
        )

        // When
        let width = TargetNoteBarRenderer.calculateBarWidth(for: segment)

        // Then: 2.0 seconds * pixelsPerSecond
        let expected = 2.0 * PitchBarConstants.pixelsPerSecond
        XCTAssertEqual(width, expected)
    }

    func testTargetNoteBarRenderer_calculateBarXPosition_returnsCorrectPosition() throws {
        // Given
        let segment = NoteSegment(
            note: try MIDINote(60),
            startTime: 1.5,  // starts at 1.5 seconds
            endTime: 2.5
        )
        let leftPadding: CGFloat = 50.0

        // When
        let x = TargetNoteBarRenderer.calculateBarXPosition(for: segment, leftPadding: leftPadding)

        // Then: 1.5 seconds * pixelsPerSecond + leftPadding
        let expected = 1.5 * PitchBarConstants.pixelsPerSecond + leftPadding
        XCTAssertEqual(x, expected)
    }

    func testTargetNoteBarRenderer_calculateBarYPosition_returnsCorrectPosition() throws {
        // Given: C4 = 261.63 Hz
        let note = try MIDINote(60)
        let segment = NoteSegment(
            note: note,
            startTime: 0.0,
            endTime: 1.0
        )
        let canvasHeight: CGFloat = 500.0

        // When
        let y = TargetNoteBarRenderer.calculateBarYPosition(for: segment, canvasHeight: canvasHeight)

        // Then: Should use logarithmic scale conversion
        let expectedY = PitchBarConstants.frequencyToY(frequency: note.frequency, canvasHeight: canvasHeight)
        XCTAssertEqual(y, expectedY, accuracy: 0.001)
    }

    // MARK: - Multiple Segments Tests

    func testTargetNoteBarRenderer_multipleSegments_correctPositioning() throws {
        // Given: Three consecutive notes
        let segments = [
            NoteSegment(note: try MIDINote(60), startTime: 0.0, endTime: 1.0),  // C4
            NoteSegment(note: try MIDINote(62), startTime: 1.0, endTime: 2.0),  // D4
            NoteSegment(note: try MIDINote(64), startTime: 2.0, endTime: 3.0)   // E4
        ]
        let leftPadding: CGFloat = 100.0

        // Then: Each bar should start where the previous one ends
        for (index, segment) in segments.enumerated() {
            let x = TargetNoteBarRenderer.calculateBarXPosition(for: segment, leftPadding: leftPadding)
            let expectedX = Double(index) * PitchBarConstants.pixelsPerSecond + leftPadding
            XCTAssertEqual(x, expectedX, accuracy: 0.001, "Segment \(index) X position incorrect")
        }
    }

    // MARK: - Edge Cases

    func testTargetNoteBarRenderer_zeroWidthSegment_handlesGracefully() throws {
        // Given: Zero duration segment (edge case)
        let segment = NoteSegment(
            note: try MIDINote(60),
            startTime: 1.0,
            endTime: 1.0  // Same start and end
        )

        // When
        let width = TargetNoteBarRenderer.calculateBarWidth(for: segment)

        // Then: Should return 0 width
        XCTAssertEqual(width, 0)
    }

    func testTargetNoteBarRenderer_highFrequencyNote_withinBounds() throws {
        // Given: High frequency note (C6 = 1046.5 Hz)
        let note = try MIDINote(84)
        let segment = NoteSegment(
            note: note,
            startTime: 0.0,
            endTime: 1.0
        )
        let canvasHeight: CGFloat = 500.0

        // When
        let y = TargetNoteBarRenderer.calculateBarYPosition(for: segment, canvasHeight: canvasHeight)

        // Then: Y should be near top (low value)
        XCTAssertGreaterThanOrEqual(y, 0)
        XCTAssertLessThanOrEqual(y, canvasHeight)
    }

    func testTargetNoteBarRenderer_lowFrequencyNote_withinBounds() throws {
        // Given: Low frequency note (C2 = 65.4 Hz)
        let note = try MIDINote(36)
        let segment = NoteSegment(
            note: note,
            startTime: 0.0,
            endTime: 1.0
        )
        let canvasHeight: CGFloat = 500.0

        // When
        let y = TargetNoteBarRenderer.calculateBarYPosition(for: segment, canvasHeight: canvasHeight)

        // Then: Y should be near bottom (high value)
        XCTAssertGreaterThanOrEqual(y, 0)
        XCTAssertLessThanOrEqual(y, canvasHeight)
    }
}
