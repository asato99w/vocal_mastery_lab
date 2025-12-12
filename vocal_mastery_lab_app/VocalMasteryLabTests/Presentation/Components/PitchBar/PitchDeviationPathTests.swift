import XCTest
import SwiftUI
@testable import VocalMasteryLab
@testable import VocalisDomain

final class PitchDeviationPathTests: XCTestCase {

    // MARK: - PitchDeviationPoint Tests

    func testPitchDeviationPoint_creation() {
        // Given
        let point = PitchDeviationPoint(
            timestamp: 1.5,
            frequency: 440.0,
            confidence: 0.9,
            targetFrequency: 440.0
        )

        // Then
        XCTAssertEqual(point.timestamp, 1.5)
        XCTAssertEqual(point.frequency, 440.0)
        XCTAssertEqual(point.confidence, 0.9)
        XCTAssertEqual(point.targetFrequency, 440.0)
    }

    func testPitchDeviationPoint_deviationCalculation_perfect() {
        // Given: Perfectly matched pitch
        let point = PitchDeviationPoint(
            timestamp: 1.0,
            frequency: 440.0,
            confidence: 0.9,
            targetFrequency: 440.0
        )

        // Then
        XCTAssertNotNil(point.deviation)
        XCTAssertEqual(point.deviation!, 0.0, accuracy: 0.001)
    }

    func testPitchDeviationPoint_deviationCalculation_sharp() {
        // Given: Pitch is sharp by about 100 cents (one semitone)
        let targetFreq = 440.0
        let sharpFreq = targetFreq * pow(2, 1.0 / 12.0)  // One semitone higher
        let point = PitchDeviationPoint(
            timestamp: 1.0,
            frequency: sharpFreq,
            confidence: 0.9,
            targetFrequency: targetFreq
        )

        // Then: Should be approximately +100 cents
        XCTAssertNotNil(point.deviation)
        XCTAssertEqual(point.deviation!, 100.0, accuracy: 0.1)
    }

    func testPitchDeviationPoint_deviationCalculation_flat() {
        // Given: Pitch is flat by about 100 cents (one semitone)
        let targetFreq = 440.0
        let flatFreq = targetFreq / pow(2, 1.0 / 12.0)  // One semitone lower
        let point = PitchDeviationPoint(
            timestamp: 1.0,
            frequency: flatFreq,
            confidence: 0.9,
            targetFrequency: targetFreq
        )

        // Then: Should be approximately -100 cents
        XCTAssertNotNil(point.deviation)
        XCTAssertEqual(point.deviation!, -100.0, accuracy: 0.1)
    }

    func testPitchDeviationPoint_deviationCalculation_noTarget() {
        // Given: No target frequency (nil)
        let point = PitchDeviationPoint(
            timestamp: 1.0,
            frequency: 440.0,
            confidence: 0.9,
            targetFrequency: nil
        )

        // Then: Deviation should be nil
        XCTAssertNil(point.deviation)
    }

    func testPitchDeviationPoint_color_perfect() {
        // Given: Within ±10 cents (perfect)
        let point = PitchDeviationPoint(
            timestamp: 1.0,
            frequency: 440.0,
            confidence: 0.9,
            targetFrequency: 440.0
        )

        // Then: Should be green
        XCTAssertEqual(point.color, PitchBarConstants.perfectColor)
    }

    func testPitchDeviationPoint_color_noTarget() {
        // Given: No target frequency
        let point = PitchDeviationPoint(
            timestamp: 1.0,
            frequency: 440.0,
            confidence: 0.9,
            targetFrequency: nil
        )

        // Then: Should use default pitch line color (cyan from existing constants)
        XCTAssertEqual(point.color, PitchGraphConstants.pitchLineColor)
    }

    // MARK: - PitchDeviationPathRenderer Tests

    func testPitchDeviationPathRenderer_calculateXPosition() {
        // Given
        let timestamp = 2.5
        let leftPadding: CGFloat = 100.0

        // When
        let x = PitchDeviationPathRenderer.calculateXPosition(
            timestamp: timestamp,
            leftPadding: leftPadding
        )

        // Then
        let expected = CGFloat(timestamp) * PitchBarConstants.pixelsPerSecond + leftPadding
        XCTAssertEqual(x, expected)
    }

    func testPitchDeviationPathRenderer_calculateYPosition() {
        // Given
        let frequency = 440.0  // A4
        let canvasHeight: CGFloat = 500.0

        // When
        let y = PitchDeviationPathRenderer.calculateYPosition(
            frequency: frequency,
            canvasHeight: canvasHeight
        )

        // Then
        let expected = PitchBarConstants.frequencyToY(frequency: frequency, canvasHeight: canvasHeight)
        XCTAssertEqual(y, expected, accuracy: 0.001)
    }

    // MARK: - Point Conversion Tests

    func testPitchDeviationPathRenderer_convertPitchDataToPoints() throws {
        // Given: Sample pitch data
        let timestamps: [Double] = [0.0, 0.1, 0.2]
        let frequencies: [Float] = [261.6, 262.3, 261.9]
        let confidences: [Float] = [0.85, 0.92, 0.88]

        // Create simple segments for target lookup
        let segment = NoteSegment(
            note: try MIDINote(60),  // C4 = 261.63 Hz
            startTime: 0.0,
            endTime: 1.0
        )
        let segments = [segment]

        // When
        let points = PitchDeviationPathRenderer.convertPitchDataToPoints(
            timestamps: timestamps,
            frequencies: frequencies,
            confidences: confidences,
            segments: segments
        )

        // Then
        XCTAssertEqual(points.count, 3)
        XCTAssertEqual(points[0].timestamp, 0.0)
        XCTAssertEqual(points[0].frequency, 261.6, accuracy: 0.1)
        XCTAssertEqual(points[0].confidence, 0.85, accuracy: 0.01)
        XCTAssertNotNil(points[0].targetFrequency)
    }

    func testPitchDeviationPathRenderer_convertPitchDataToPoints_noTargetOutsideSegment() throws {
        // Given: Pitch data outside of any segment
        let timestamps: [Double] = [5.0]  // Outside segment time range
        let frequencies: [Float] = [440.0]
        let confidences: [Float] = [0.9]

        let segment = NoteSegment(
            note: try MIDINote(60),
            startTime: 0.0,
            endTime: 1.0  // Segment ends at 1.0
        )
        let segments = [segment]

        // When
        let points = PitchDeviationPathRenderer.convertPitchDataToPoints(
            timestamps: timestamps,
            frequencies: frequencies,
            confidences: confidences,
            segments: segments
        )

        // Then: Point should have no target frequency
        XCTAssertEqual(points.count, 1)
        XCTAssertNil(points[0].targetFrequency)
    }

    // MARK: - Target Frequency Lookup Tests

    func testPitchDeviationPathRenderer_findTargetFrequency_withinSegment() throws {
        // Given
        let segment = NoteSegment(
            note: try MIDINote(60),  // C4 = 261.63 Hz
            startTime: 1.0,
            endTime: 2.0
        )
        let segments = [segment]
        let timestamp = 1.5  // Within segment

        // When
        let target = PitchDeviationPathRenderer.findTargetFrequency(
            at: timestamp,
            segments: segments
        )

        // Then
        XCTAssertNotNil(target)
        XCTAssertEqual(target!, 261.63, accuracy: 0.1)
    }

    func testPitchDeviationPathRenderer_findTargetFrequency_atSegmentBoundary() throws {
        // Given
        let segment = NoteSegment(
            note: try MIDINote(60),
            startTime: 1.0,
            endTime: 2.0
        )
        let segments = [segment]

        // When: At exact start (inclusive)
        let targetStart = PitchDeviationPathRenderer.findTargetFrequency(at: 1.0, segments: segments)
        // When: At exact end (exclusive)
        let targetEnd = PitchDeviationPathRenderer.findTargetFrequency(at: 2.0, segments: segments)

        // Then
        XCTAssertNotNil(targetStart, "Should include start boundary")
        XCTAssertNil(targetEnd, "Should exclude end boundary")
    }

    func testPitchDeviationPathRenderer_findTargetFrequency_multipleSegments() throws {
        // Given: Multiple consecutive segments
        let segments = [
            NoteSegment(note: try MIDINote(60), startTime: 0.0, endTime: 1.0),  // C4
            NoteSegment(note: try MIDINote(62), startTime: 1.0, endTime: 2.0),  // D4
            NoteSegment(note: try MIDINote(64), startTime: 2.0, endTime: 3.0)   // E4
        ]

        // When
        let target0_5 = PitchDeviationPathRenderer.findTargetFrequency(at: 0.5, segments: segments)
        let target1_5 = PitchDeviationPathRenderer.findTargetFrequency(at: 1.5, segments: segments)
        let target2_5 = PitchDeviationPathRenderer.findTargetFrequency(at: 2.5, segments: segments)

        // Then: Should return correct frequency for each segment
        XCTAssertEqual(target0_5!, 261.63, accuracy: 0.1)  // C4
        XCTAssertEqual(target1_5!, 293.66, accuracy: 0.1)  // D4
        XCTAssertEqual(target2_5!, 329.63, accuracy: 0.1)  // E4
    }
}
