import XCTest
@testable import VocalisDomain

/// Tests for SpectrogramData value object
final class SpectrogramDataTests: XCTestCase {

    // MARK: - Initialization Tests

    func testInit_WithValidData_CreatesInstance() {
        // Given
        let timeStamps: [Double] = [0.0, 0.1, 0.2]
        let frequencyBins: [Float] = [80, 180, 280, 380]
        let magnitudes: [[Float]] = [
            [0.1, 0.3, 0.8, 0.5],  // t=0.0s
            [0.2, 0.4, 0.7, 0.6],  // t=0.1s
            [0.3, 0.5, 0.6, 0.7]   // t=0.2s
        ]

        // When
        let data = SpectrogramData(
            timeStamps: timeStamps,
            frequencyBins: frequencyBins,
            magnitudes: magnitudes
        )

        // Then
        XCTAssertEqual(data.timeStamps, timeStamps)
        XCTAssertEqual(data.frequencyBins, frequencyBins)
        XCTAssertEqual(data.magnitudes.count, magnitudes.count)
        XCTAssertEqual(data.magnitudes[0], magnitudes[0])
    }

    func testInit_WithEmptyArrays_CreatesInstance() {
        // Given
        let emptyTimeStamps: [Double] = []
        let emptyFrequencyBins: [Float] = []
        let emptyMagnitudes: [[Float]] = []

        // When
        let data = SpectrogramData(
            timeStamps: emptyTimeStamps,
            frequencyBins: emptyFrequencyBins,
            magnitudes: emptyMagnitudes
        )

        // Then
        XCTAssertTrue(data.timeStamps.isEmpty)
        XCTAssertTrue(data.frequencyBins.isEmpty)
        XCTAssertTrue(data.magnitudes.isEmpty)
    }

    // MARK: - Data Point Count Tests

    func testTimeFrameCount_ReturnsCorrectCount() {
        // Given
        let timeStamps: [Double] = [0.0, 0.1, 0.2, 0.3]
        let frequencyBins: [Float] = [80, 180]
        let magnitudes: [[Float]] = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8]
        ]

        let data = SpectrogramData(
            timeStamps: timeStamps,
            frequencyBins: frequencyBins,
            magnitudes: magnitudes
        )

        // When
        let count = data.timeFrameCount

        // Then
        XCTAssertEqual(count, 4)
    }

    func testFrequencyBinCount_ReturnsCorrectCount() {
        // Given
        let timeStamps: [Double] = [0.0]
        let frequencyBins: [Float] = [80, 180, 280]
        let magnitudes: [[Float]] = [[0.1, 0.2, 0.3]]

        let data = SpectrogramData(
            timeStamps: timeStamps,
            frequencyBins: frequencyBins,
            magnitudes: magnitudes
        )

        // When
        let count = data.frequencyBinCount

        // Then
        XCTAssertEqual(count, 3)
    }

    // MARK: - Equatable Tests

    func testEquality_WithSameData_ReturnsTrue() {
        // Given
        let timeStamps: [Double] = [0.0, 0.1]
        let frequencyBins: [Float] = [80, 180]
        let magnitudes: [[Float]] = [[0.1, 0.2], [0.3, 0.4]]

        let data1 = SpectrogramData(
            timeStamps: timeStamps,
            frequencyBins: frequencyBins,
            magnitudes: magnitudes
        )

        let data2 = SpectrogramData(
            timeStamps: timeStamps,
            frequencyBins: frequencyBins,
            magnitudes: magnitudes
        )

        // When & Then
        XCTAssertEqual(data1, data2)
    }

    func testEquality_WithDifferentData_ReturnsFalse() {
        // Given
        let data1 = SpectrogramData(
            timeStamps: [0.0],
            frequencyBins: [80, 180],
            magnitudes: [[0.1, 0.2]]
        )

        let data2 = SpectrogramData(
            timeStamps: [0.0],
            frequencyBins: [80, 180],
            magnitudes: [[0.1, 0.3]]  // Different magnitude
        )

        // When & Then
        XCTAssertNotEqual(data1, data2)
    }
}
