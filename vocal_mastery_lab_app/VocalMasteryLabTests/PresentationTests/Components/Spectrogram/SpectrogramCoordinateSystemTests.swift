import XCTest
@testable import VocalMasteryLab

/// Unit tests for SpectrogramCoordinateSystem
/// Validates coordinate calculations and conversions
final class SpectrogramCoordinateSystemTests: XCTestCase {
    var sut: SpectrogramCoordinateSystem!

    override func setUp() {
        super.setUp()
        sut = SpectrogramCoordinateSystem()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    // MARK: - Canvas Height Tests

    func testCalculateCanvasHeight_withStandardFrequency() {
        // Given: Standard 6kHz frequency range
        let maxFreq = 6000.0
        let viewportHeight: CGFloat = 500  // Unused but required

        // When: Calculate canvas height
        let result = sut.calculateCanvasHeight(maxFreq: maxFreq, viewportHeight: viewportHeight)

        // Then: Height should be 6kHz × 576pt/kHz = 3456pt
        XCTAssertEqual(result, 3456.0, accuracy: 0.1)
    }

    func testCalculateCanvasHeight_withDifferentFrequency() {
        // Given: 8kHz frequency range
        let maxFreq = 8000.0
        let viewportHeight: CGFloat = 500

        // When: Calculate canvas height
        let result = sut.calculateCanvasHeight(maxFreq: maxFreq, viewportHeight: viewportHeight)

        // Then: Height should be 8kHz × 576pt/kHz = 4608pt
        XCTAssertEqual(result, 4608.0, accuracy: 0.1)
    }

    func testCalculateCanvasHeight_respectsMaximumLimit() {
        // Given: Extremely high frequency (would exceed limit)
        let maxFreq = 20000.0  // 20kHz
        let viewportHeight: CGFloat = 500

        // When: Calculate canvas height
        let result = sut.calculateCanvasHeight(maxFreq: maxFreq, viewportHeight: viewportHeight)

        // Then: Height should be clamped to maximum (10000pt)
        XCTAssertEqual(result, 10000.0, accuracy: 0.1)
    }

    // MARK: - Frequency to Y Coordinate Tests

    func testFrequencyToCanvasY_atMaxFrequency() {
        // Given: Frequency at maximum (6kHz)
        let frequency = 6000.0
        let canvasHeight: CGFloat = 3456.0
        let maxFreq = 6000.0

        // When: Convert to canvas Y
        let result = sut.frequencyToCanvasY(frequency: frequency, canvasHeight: canvasHeight, maxFreq: maxFreq)

        // Then: Should be at top (Y=0)
        XCTAssertEqual(result, 0.0, accuracy: 0.1)
    }

    func testFrequencyToCanvasY_atZeroHz() {
        // Given: Frequency at 0Hz
        let frequency = 0.0
        let canvasHeight: CGFloat = 3456.0
        let maxFreq = 6000.0

        // When: Convert to canvas Y
        let result = sut.frequencyToCanvasY(frequency: frequency, canvasHeight: canvasHeight, maxFreq: maxFreq)

        // Then: Should be at bottom (Y=canvasHeight)
        XCTAssertEqual(result, canvasHeight, accuracy: 0.1)
    }

    func testFrequencyToCanvasY_atMiddleFrequency() {
        // Given: Frequency at middle (3kHz)
        let frequency = 3000.0
        let canvasHeight: CGFloat = 3456.0
        let maxFreq = 6000.0

        // When: Convert to canvas Y
        let result = sut.frequencyToCanvasY(frequency: frequency, canvasHeight: canvasHeight, maxFreq: maxFreq)

        // Then: Should be at middle (Y=canvasHeight/2)
        XCTAssertEqual(result, canvasHeight / 2, accuracy: 0.1)
    }

    func testFrequencyToCanvasY_linearMapping() {
        // Given: Various frequencies
        let canvasHeight: CGFloat = 3456.0
        let maxFreq = 6000.0

        // When/Then: Verify linear mapping
        let freq1 = 1000.0  // 1/6 from bottom
        let y1 = sut.frequencyToCanvasY(frequency: freq1, canvasHeight: canvasHeight, maxFreq: maxFreq)
        XCTAssertEqual(y1, canvasHeight * 5/6, accuracy: 0.1)

        let freq2 = 4500.0  // 3/4 from bottom
        let y2 = sut.frequencyToCanvasY(frequency: freq2, canvasHeight: canvasHeight, maxFreq: maxFreq)
        XCTAssertEqual(y2, canvasHeight * 1/4, accuracy: 0.1)
    }

    // MARK: - Max Frequency Tests

    func testGetMaxFrequency_returnsStandard8kHz() {
        // When: Get maximum frequency
        let result = sut.getMaxFrequency()

        // Then: Should return 8000.0 (8kHz) for high frequency analysis
        XCTAssertEqual(result, 8000.0, accuracy: 0.1)
    }

    // MARK: - Canvas Width Tests

    func testCalculateCanvasWidth_withStandardDuration() {
        // Given: 10 second duration
        let dataDuration = 10.0
        let leftPadding: CGFloat = 200.0

        // When: Calculate canvas width
        let result = sut.calculateCanvasWidth(dataDuration: dataDuration, leftPadding: leftPadding)

        // Then: Width should be (10s × 300pt/s) + 200pt = 3200pt
        XCTAssertEqual(result, 3200.0, accuracy: 0.1)
    }

    func testCalculateCanvasWidth_withShortDuration() {
        // Given: Very short duration
        let dataDuration = 0.1  // 100ms
        let leftPadding: CGFloat = 50.0

        // When: Calculate canvas width
        let result = sut.calculateCanvasWidth(dataDuration: dataDuration, leftPadding: leftPadding)

        // Then: Should respect minimum width (100pt)
        XCTAssertEqual(result, 100.0, accuracy: 0.1)
    }

    func testCalculateCanvasWidth_withZeroDuration() {
        // Given: Zero duration
        let dataDuration = 0.0
        let leftPadding: CGFloat = 50.0

        // When: Calculate canvas width
        let result = sut.calculateCanvasWidth(dataDuration: dataDuration, leftPadding: leftPadding)

        // Then: Should return minimum width (100pt)
        XCTAssertEqual(result, 100.0, accuracy: 0.1)
    }

    // MARK: - Time to X Coordinate Tests

    func testTimeToCanvasX_atZeroTime() {
        // Given: Time at 0s
        let time = 0.0
        let leftPadding: CGFloat = 200.0

        // When: Convert to canvas X
        let result = sut.timeToCanvasX(time: time, leftPadding: leftPadding)

        // Then: Should be at leftPadding position
        XCTAssertEqual(result, leftPadding, accuracy: 0.1)
    }

    func testTimeToCanvasX_atOneSecond() {
        // Given: Time at 1s
        let time = 1.0
        let leftPadding: CGFloat = 200.0

        // When: Convert to canvas X
        let result = sut.timeToCanvasX(time: time, leftPadding: leftPadding)

        // Then: Should be leftPadding + (1s × 300pt/s) = 500pt
        XCTAssertEqual(result, 500.0, accuracy: 0.1)
    }

    func testTimeToCanvasX_linearMapping() {
        // Given: Various times
        let leftPadding: CGFloat = 100.0

        // When/Then: Verify linear mapping (300pt/s)
        let time1 = 2.5
        let x1 = sut.timeToCanvasX(time: time1, leftPadding: leftPadding)
        XCTAssertEqual(x1, 100 + 2.5 * 300, accuracy: 0.1)

        let time2 = 10.0
        let x2 = sut.timeToCanvasX(time: time2, leftPadding: leftPadding)
        XCTAssertEqual(x2, 100 + 10.0 * 300, accuracy: 0.1)
    }

    // MARK: - Pixels Per Second Tests

    func testGetPixelsPerSecond_returnsStandard300() {
        // When: Get pixels per second
        let result = sut.getPixelsPerSecond()

        // Then: Should return 300.0
        XCTAssertEqual(result, 300.0, accuracy: 0.1)
    }
}
