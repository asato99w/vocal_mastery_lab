import XCTest
import SwiftUI
@testable import VocalMasteryLab
@testable import VocalisDomain

final class PitchBarViewTests: XCTestCase {

    // MARK: - PitchBarViewModel Tests

    func testPitchBarViewModel_initialization_withValidData() throws {
        // Given
        let pitchData = createSamplePitchData()
        let segments = try createSampleSegments()

        // When
        let viewModel = PitchBarViewModel(pitchData: pitchData, segments: segments)

        // Then
        XCTAssertEqual(viewModel.segments.count, 3)
        XCTAssertFalse(viewModel.deviationPoints.isEmpty)
    }

    func testPitchBarViewModel_initialization_emptyData() {
        // Given
        let pitchData = PitchAnalysisData(
            timeStamps: [],
            frequencies: [],
            confidences: [],
            targetNotes: []
        )
        let segments: [NoteSegment] = []

        // When
        let viewModel = PitchBarViewModel(pitchData: pitchData, segments: segments)

        // Then
        XCTAssertTrue(viewModel.segments.isEmpty)
        XCTAssertTrue(viewModel.deviationPoints.isEmpty)
    }

    func testPitchBarViewModel_overallAccuracy() throws {
        // Given: Pitch data perfectly matching target frequencies
        let c4 = try MIDINote(60)
        let pitchData = PitchAnalysisData(
            timeStamps: [0.5],
            frequencies: [Float(c4.frequency)],
            confidences: [0.9],
            targetNotes: [c4]
        )
        let segments = [
            NoteSegment(note: c4, startTime: 0.0, endTime: 1.0)
        ]

        // When
        let viewModel = PitchBarViewModel(pitchData: pitchData, segments: segments)

        // Then: Should have 100% accuracy
        XCTAssertEqual(viewModel.overallAccuracy, 100.0, accuracy: 0.1)
    }

    func testPitchBarViewModel_averageDeviation_perfect() throws {
        // Given: Pitch data perfectly matching target
        let c4 = try MIDINote(60)
        let pitchData = PitchAnalysisData(
            timeStamps: [0.5],
            frequencies: [Float(c4.frequency)],
            confidences: [0.9],
            targetNotes: [c4]
        )
        let segments = [
            NoteSegment(note: c4, startTime: 0.0, endTime: 1.0)
        ]

        // When
        let viewModel = PitchBarViewModel(pitchData: pitchData, segments: segments)

        // Then: Average deviation should be ~0
        XCTAssertNotNil(viewModel.averageDeviation)
        XCTAssertEqual(viewModel.averageDeviation!, 0.0, accuracy: 0.1)
    }

    func testPitchBarViewModel_noteScores() throws {
        // Given: Multiple segments with pitch data
        let c4 = try MIDINote(60)
        let d4 = try MIDINote(62)

        let pitchData = PitchAnalysisData(
            timeStamps: [0.5, 1.5],
            frequencies: [Float(c4.frequency), Float(d4.frequency)],
            confidences: [0.9, 0.9],
            targetNotes: [c4, d4]
        )

        let segments = [
            NoteSegment(note: c4, startTime: 0.0, endTime: 1.0),
            NoteSegment(note: d4, startTime: 1.0, endTime: 2.0)
        ]

        // When
        let viewModel = PitchBarViewModel(pitchData: pitchData, segments: segments)

        // Then
        XCTAssertEqual(viewModel.noteScores.count, 2)
    }

    func testPitchBarViewModel_canvasWidth() throws {
        // Given: Segments spanning 3 seconds
        let segments = [
            NoteSegment(note: try MIDINote(60), startTime: 0.0, endTime: 1.0),
            NoteSegment(note: try MIDINote(62), startTime: 1.0, endTime: 2.0),
            NoteSegment(note: try MIDINote(64), startTime: 2.0, endTime: 3.0)
        ]
        let pitchData = PitchAnalysisData(timeStamps: [], frequencies: [], confidences: [], targetNotes: [])

        // When
        let viewModel = PitchBarViewModel(pitchData: pitchData, segments: segments)
        let expectedWidth = 3.0 * PitchBarConstants.pixelsPerSecond + viewModel.leftPadding

        // Then
        XCTAssertEqual(viewModel.canvasWidth, expectedWidth, accuracy: 0.1)
    }

    // MARK: - Helper Methods

    func createSamplePitchData() -> PitchAnalysisData {
        return PitchAnalysisData(
            timeStamps: [0.5, 1.5, 2.5],
            frequencies: [261.6, 293.7, 329.6],  // C4, D4, E4 frequencies
            confidences: [0.9, 0.85, 0.88],
            targetNotes: [try? MIDINote(60), try? MIDINote(62), try? MIDINote(64)]
        )
    }

    func createSampleSegments() throws -> [NoteSegment] {
        return [
            NoteSegment(note: try MIDINote(60), startTime: 0.0, endTime: 1.0),  // C4
            NoteSegment(note: try MIDINote(62), startTime: 1.0, endTime: 2.0),  // D4
            NoteSegment(note: try MIDINote(64), startTime: 2.0, endTime: 3.0)   // E4
        ]
    }
}
