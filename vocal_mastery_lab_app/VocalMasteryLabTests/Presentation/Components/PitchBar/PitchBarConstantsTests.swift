import XCTest
@testable import VocalMasteryLab

final class PitchBarConstantsTests: XCTestCase {

    // MARK: - Deviation Color Tests

    func testDeviationColor_within10Cents_returnsGreen() {
        // Given: Deviation within ±10 cents (perfect)
        let deviation1: Double = 0.0
        let deviation2: Double = 10.0
        let deviation3: Double = -10.0

        // When/Then
        XCTAssertEqual(PitchBarConstants.deviationColor(for: deviation1), PitchBarConstants.perfectColor)
        XCTAssertEqual(PitchBarConstants.deviationColor(for: deviation2), PitchBarConstants.perfectColor)
        XCTAssertEqual(PitchBarConstants.deviationColor(for: deviation3), PitchBarConstants.perfectColor)
    }

    func testDeviationColor_within25Cents_returnsBlue() {
        // Given: Deviation within ±25 cents (good)
        let deviation1: Double = 11.0
        let deviation2: Double = 25.0
        let deviation3: Double = -25.0

        // When/Then
        XCTAssertEqual(PitchBarConstants.deviationColor(for: deviation1), PitchBarConstants.goodColor)
        XCTAssertEqual(PitchBarConstants.deviationColor(for: deviation2), PitchBarConstants.goodColor)
        XCTAssertEqual(PitchBarConstants.deviationColor(for: deviation3), PitchBarConstants.goodColor)
    }

    func testDeviationColor_within50Cents_returnsYellow() {
        // Given: Deviation within ±50 cents (acceptable)
        let deviation1: Double = 26.0
        let deviation2: Double = 50.0
        let deviation3: Double = -50.0

        // When/Then
        XCTAssertEqual(PitchBarConstants.deviationColor(for: deviation1), PitchBarConstants.acceptableColor)
        XCTAssertEqual(PitchBarConstants.deviationColor(for: deviation2), PitchBarConstants.acceptableColor)
        XCTAssertEqual(PitchBarConstants.deviationColor(for: deviation3), PitchBarConstants.acceptableColor)
    }

    func testDeviationColor_beyond50Cents_returnsRed() {
        // Given: Deviation beyond ±50 cents (needs improvement)
        let deviation1: Double = 51.0
        let deviation2: Double = 100.0
        let deviation3: Double = -51.0

        // When/Then
        XCTAssertEqual(PitchBarConstants.deviationColor(for: deviation1), PitchBarConstants.needsImprovementColor)
        XCTAssertEqual(PitchBarConstants.deviationColor(for: deviation2), PitchBarConstants.needsImprovementColor)
        XCTAssertEqual(PitchBarConstants.deviationColor(for: deviation3), PitchBarConstants.needsImprovementColor)
    }

    // MARK: - Deviation Calculation Tests

    func testCalculateDeviation_perfectMatch_returnsZero() {
        // Given: Detected frequency equals target frequency
        let detected: Double = 440.0
        let target: Double = 440.0

        // When
        let deviation = PitchBarConstants.calculateDeviation(detected: detected, expected: target)

        // Then
        XCTAssertEqual(deviation, 0.0, accuracy: 0.001)
    }

    func testCalculateDeviation_oneSemitoneHigh_returns100Cents() {
        // Given: Detected frequency is one semitone higher
        let target: Double = 440.0  // A4
        let detected: Double = target * pow(2, 1.0 / 12.0)  // A#4

        // When
        let deviation = PitchBarConstants.calculateDeviation(detected: detected, expected: target)

        // Then
        XCTAssertEqual(deviation, 100.0, accuracy: 0.001)
    }

    func testCalculateDeviation_oneSemitoneLow_returnsMinus100Cents() {
        // Given: Detected frequency is one semitone lower
        let target: Double = 440.0  // A4
        let detected: Double = target / pow(2, 1.0 / 12.0)  // G#4

        // When
        let deviation = PitchBarConstants.calculateDeviation(detected: detected, expected: target)

        // Then
        XCTAssertEqual(deviation, -100.0, accuracy: 0.001)
    }

    // MARK: - Display Constants Tests

    func testDisplayConstants_haveValidValues() {
        // Verify display constants have reasonable values
        XCTAssertGreaterThan(PitchBarConstants.pixelsPerSecond, 0)
        XCTAssertGreaterThan(PitchBarConstants.noteBarHeight, 0)
        XCTAssertGreaterThan(PitchBarConstants.pixelsPerOctave, 0)
        XCTAssertGreaterThan(PitchBarConstants.minFrequency, 0)
        XCTAssertGreaterThan(PitchBarConstants.maxFrequency, PitchBarConstants.minFrequency)
    }

    // MARK: - Accuracy Evaluation Tests

    func testEvaluateAccuracy_perfectRange() {
        // Given
        let deviation: Double = 5.0

        // When
        let evaluation = PitchBarConstants.evaluateAccuracy(deviation: deviation)

        // Then
        XCTAssertEqual(evaluation, .perfect)
    }

    func testEvaluateAccuracy_goodRange() {
        // Given
        let deviation: Double = 20.0

        // When
        let evaluation = PitchBarConstants.evaluateAccuracy(deviation: deviation)

        // Then
        XCTAssertEqual(evaluation, .good)
    }

    func testEvaluateAccuracy_acceptableRange() {
        // Given
        let deviation: Double = 40.0

        // When
        let evaluation = PitchBarConstants.evaluateAccuracy(deviation: deviation)

        // Then
        XCTAssertEqual(evaluation, .acceptable)
    }

    func testEvaluateAccuracy_needsImprovementRange() {
        // Given
        let deviation: Double = 60.0

        // When
        let evaluation = PitchBarConstants.evaluateAccuracy(deviation: deviation)

        // Then
        XCTAssertEqual(evaluation, .needsImprovement)
    }
}
