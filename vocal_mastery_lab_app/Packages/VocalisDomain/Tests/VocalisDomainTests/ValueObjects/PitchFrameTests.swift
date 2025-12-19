import XCTest
@testable import VocalisDomain

final class PitchFrameTests: XCTestCase {

    // MARK: - Initialization Tests

    func testInit_withAllValues_createsCorrectFrame() {
        // Given
        let timestamp = 0.5
        let frequency: Float = 440.0
        let confidence: Float = 0.85
        let amplitude: Float = 0.6

        // When
        let frame = PitchFrame(
            timestamp: timestamp,
            frequency: frequency,
            confidence: confidence,
            amplitude: amplitude
        )

        // Then
        XCTAssertEqual(frame.timestamp, timestamp)
        XCTAssertEqual(frame.frequency, frequency)
        XCTAssertEqual(frame.confidence, confidence)
        XCTAssertEqual(frame.amplitude, amplitude)
    }

    func testInit_withNilFrequency_createsUnvoicedFrame() {
        // Given
        let timestamp = 1.0
        let confidence: Float = 0.1
        let amplitude: Float = 0.05

        // When
        let frame = PitchFrame(
            timestamp: timestamp,
            frequency: nil,
            confidence: confidence,
            amplitude: amplitude
        )

        // Then
        XCTAssertEqual(frame.timestamp, timestamp)
        XCTAssertNil(frame.frequency)
        XCTAssertEqual(frame.confidence, confidence)
        XCTAssertEqual(frame.amplitude, amplitude)
    }

    // MARK: - isVoiced Property Tests

    func testIsVoiced_withFrequency_returnsTrue() {
        // Given
        let frame = PitchFrame(
            timestamp: 0.0,
            frequency: 440.0,
            confidence: 0.9,
            amplitude: 0.5
        )

        // Then
        XCTAssertTrue(frame.isVoiced)
    }

    func testIsVoiced_withNilFrequency_returnsFalse() {
        // Given
        let frame = PitchFrame(
            timestamp: 0.0,
            frequency: nil,
            confidence: 0.1,
            amplitude: 0.02
        )

        // Then
        XCTAssertFalse(frame.isVoiced)
    }

    // MARK: - Equatable Tests

    func testEquatable_withSameValues_returnsTrue() {
        // Given
        let frame1 = PitchFrame(timestamp: 0.5, frequency: 440.0, confidence: 0.9, amplitude: 0.6)
        let frame2 = PitchFrame(timestamp: 0.5, frequency: 440.0, confidence: 0.9, amplitude: 0.6)

        // Then
        XCTAssertEqual(frame1, frame2)
    }

    func testEquatable_withDifferentTimestamp_returnsFalse() {
        // Given
        let frame1 = PitchFrame(timestamp: 0.5, frequency: 440.0, confidence: 0.9, amplitude: 0.6)
        let frame2 = PitchFrame(timestamp: 1.0, frequency: 440.0, confidence: 0.9, amplitude: 0.6)

        // Then
        XCTAssertNotEqual(frame1, frame2)
    }

    func testEquatable_withBothNilFrequency_returnsTrue() {
        // Given
        let frame1 = PitchFrame(timestamp: 0.5, frequency: nil, confidence: 0.1, amplitude: 0.05)
        let frame2 = PitchFrame(timestamp: 0.5, frequency: nil, confidence: 0.1, amplitude: 0.05)

        // Then
        XCTAssertEqual(frame1, frame2)
    }
}
