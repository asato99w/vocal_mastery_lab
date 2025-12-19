import XCTest
@testable import VocalisDomain

/// Tests for PitchAnalysisData value object
final class PitchAnalysisDataTests: XCTestCase {

    // MARK: - Initialization Tests

    func testInit_WithValidData_CreatesInstance() {
        // Given
        let timeStamps: [Double] = [0.0, 0.05, 0.10]
        let frequencies: [Float] = [261.6, 262.3, 261.9]
        let confidences: [Float] = [0.85, 0.92, 0.88]
        let targetNote = try! MIDINote(60) // C4
        let targetNotes: [MIDINote?] = [targetNote, targetNote, targetNote]

        // When
        let data = PitchAnalysisData(
            timeStamps: timeStamps,
            frequencies: frequencies,
            confidences: confidences,
            targetNotes: targetNotes
        )

        // Then
        XCTAssertEqual(data.timeStamps, timeStamps)
        XCTAssertEqual(data.frequencies, frequencies)
        XCTAssertEqual(data.confidences, confidences)
        XCTAssertEqual(data.targetNotes.count, targetNotes.count)
    }

    func testInit_WithEmptyArrays_CreatesInstance() {
        // Given
        let emptyTimeStamps: [Double] = []
        let emptyFrequencies: [Float] = []
        let emptyConfidences: [Float] = []
        let emptyTargetNotes: [MIDINote?] = []

        // When
        let data = PitchAnalysisData(
            timeStamps: emptyTimeStamps,
            frequencies: emptyFrequencies,
            confidences: emptyConfidences,
            targetNotes: emptyTargetNotes
        )

        // Then
        XCTAssertTrue(data.timeStamps.isEmpty)
        XCTAssertTrue(data.frequencies.isEmpty)
        XCTAssertTrue(data.confidences.isEmpty)
        XCTAssertTrue(data.targetNotes.isEmpty)
    }

    func testInit_WithNoTargetNotes_CreatesInstance() {
        // Given: Recording without scale (free recording)
        let timeStamps: [Double] = [0.0, 0.05]
        let frequencies: [Float] = [261.6, 262.3]
        let confidences: [Float] = [0.85, 0.92]
        let targetNotes: [MIDINote?] = [nil, nil]

        // When
        let data = PitchAnalysisData(
            timeStamps: timeStamps,
            frequencies: frequencies,
            confidences: confidences,
            targetNotes: targetNotes
        )

        // Then
        XCTAssertEqual(data.timeStamps.count, 2)
        XCTAssertEqual(data.targetNotes.count, 2)
        XCTAssertNil(data.targetNotes[0])
        XCTAssertNil(data.targetNotes[1])
    }

    // MARK: - Data Point Count Tests

    func testDataPointCount_ReturnsCorrectCount() {
        // Given
        let timeStamps: [Double] = [0.0, 0.05, 0.10, 0.15]
        let frequencies: [Float] = [261.6, 262.3, 261.9, 293.7]
        let confidences: [Float] = [0.85, 0.92, 0.88, 0.91]
        let targetNotes: [MIDINote?] = [nil, nil, nil, nil]

        let data = PitchAnalysisData(
            timeStamps: timeStamps,
            frequencies: frequencies,
            confidences: confidences,
            targetNotes: targetNotes
        )

        // When
        let count = data.dataPointCount

        // Then
        XCTAssertEqual(count, 4)
    }

    func testDataPointCount_WithEmptyData_ReturnsZero() {
        // Given
        let data = PitchAnalysisData(
            timeStamps: [],
            frequencies: [],
            confidences: [],
            targetNotes: []
        )

        // When
        let count = data.dataPointCount

        // Then
        XCTAssertEqual(count, 0)
    }

    // MARK: - Equatable Tests

    func testEquality_WithSameData_ReturnsTrue() {
        // Given
        let timeStamps: [Double] = [0.0, 0.05]
        let frequencies: [Float] = [261.6, 262.3]
        let confidences: [Float] = [0.85, 0.92]
        let targetNotes: [MIDINote?] = [nil, nil]

        let data1 = PitchAnalysisData(
            timeStamps: timeStamps,
            frequencies: frequencies,
            confidences: confidences,
            targetNotes: targetNotes
        )

        let data2 = PitchAnalysisData(
            timeStamps: timeStamps,
            frequencies: frequencies,
            confidences: confidences,
            targetNotes: targetNotes
        )

        // When & Then
        XCTAssertEqual(data1, data2)
    }

    func testEquality_WithDifferentData_ReturnsFalse() {
        // Given
        let data1 = PitchAnalysisData(
            timeStamps: [0.0],
            frequencies: [261.6],
            confidences: [0.85],
            targetNotes: [nil]
        )

        let data2 = PitchAnalysisData(
            timeStamps: [0.0],
            frequencies: [262.0], // Different frequency
            confidences: [0.85],
            targetNotes: [nil]
        )

        // When & Then
        XCTAssertNotEqual(data1, data2)
    }
}
