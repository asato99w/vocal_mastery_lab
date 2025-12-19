import XCTest
@testable import VocalisDomain

final class VibratoAnalyzerTests: XCTestCase {

    // MARK: - VibratoAnalysis Structure Tests

    func testVibratoAnalysisInit() {
        // Given/When
        let analysis = VibratoAnalysis(
            rate: 6.0,
            extent: 50.0,
            regularity: 0.85,
            isPresent: true
        )

        // Then
        XCTAssertEqual(analysis.rate, 6.0)
        XCTAssertEqual(analysis.extent, 50.0)
        XCTAssertEqual(analysis.regularity, 0.85)
        XCTAssertTrue(analysis.isPresent)
    }

    func testVibratoAnalysisNoVibrato() {
        // Given/When
        let analysis = VibratoAnalysis(
            rate: 0,
            extent: 0,
            regularity: 0,
            isPresent: false
        )

        // Then
        XCTAssertFalse(analysis.isPresent)
    }

    // MARK: - VibratoAnalyzer Tests

    func testAnalyzeWithSteadyPitch_shouldReturnNoVibrato() {
        // Given: Steady pitch (no vibrato)
        let sut = VibratoAnalyzer()
        let steadyFrequencies = [Float](repeating: 440.0, count: 20)
        let timeStamps = (0..<20).map { Double($0) * 0.05 }  // 50ms intervals

        // When
        let result = sut.analyze(frequencies: steadyFrequencies, timeStamps: timeStamps)

        // Then
        XCTAssertFalse(result.isPresent)
        XCTAssertLessThan(result.extent, 10.0)  // Less than 10 cents
    }

    func testAnalyzeWithVibrato_shouldDetectRate() {
        // Given: Simulated vibrato at 6Hz with ±50 cents
        let sut = VibratoAnalyzer()
        let baseFreq: Float = 440.0
        let vibratoRate: Float = 6.0  // Hz
        let extentCents: Float = 50.0  // cents

        // Calculate frequency deviation from cents
        // cents = 1200 * log2(f2/f1)
        // f2 = f1 * 2^(cents/1200)
        let extentRatio = pow(2.0, extentCents / 1200.0)

        // Generate 2 seconds of data at 40Hz sample rate (25ms intervals)
        let sampleRate: Float = 40.0
        let duration: Float = 2.0
        let sampleCount = Int(sampleRate * duration)

        var frequencies: [Float] = []
        var timeStamps: [Double] = []

        for i in 0..<sampleCount {
            let time = Float(i) / sampleRate
            // Sine wave modulation
            let phase = 2.0 * Float.pi * vibratoRate * time
            let modulation = sin(phase) * (extentRatio - 1.0)
            let freq = baseFreq * (1.0 + modulation)
            frequencies.append(freq)
            timeStamps.append(Double(time))
        }

        // When
        let result = sut.analyze(frequencies: frequencies, timeStamps: timeStamps)

        // Then
        XCTAssertTrue(result.isPresent)
        XCTAssertEqual(result.rate, vibratoRate, accuracy: 1.5)  // Within 1.5 Hz
        XCTAssertEqual(result.extent, extentCents, accuracy: 15.0)  // Within 15 cents
    }

    func testAnalyzeWithFastVibrato_shouldDetectHighRate() {
        // Given: Fast vibrato at 7.5Hz (rock style)
        let sut = VibratoAnalyzer()
        let baseFreq: Float = 440.0
        let vibratoRate: Float = 7.5  // Slightly below 8Hz to stay in detection range
        let extentCents: Float = 40.0

        let extentRatio = pow(2.0, extentCents / 1200.0)
        let sampleRate: Float = 40.0  // Higher sample rate for fast vibrato
        let duration: Float = 1.0
        let sampleCount = Int(sampleRate * duration)

        var frequencies: [Float] = []
        var timeStamps: [Double] = []

        for i in 0..<sampleCount {
            let time = Float(i) / sampleRate
            let phase = 2.0 * Float.pi * vibratoRate * time
            let modulation = sin(phase) * (extentRatio - 1.0)
            frequencies.append(baseFreq * (1.0 + modulation))
            timeStamps.append(Double(time))
        }

        // When
        let result = sut.analyze(frequencies: frequencies, timeStamps: timeStamps)

        // Then
        XCTAssertTrue(result.isPresent)
        XCTAssertGreaterThan(result.rate, 6.0)
    }

    func testAnalyzeWithSlowVibrato_shouldDetectLowRate() {
        // Given: Slow vibrato at 4.5Hz (enka style)
        let sut = VibratoAnalyzer()
        let baseFreq: Float = 440.0
        let vibratoRate: Float = 4.5
        let extentCents: Float = 80.0

        let extentRatio = pow(2.0, extentCents / 1200.0)
        let sampleRate: Float = 40.0  // Higher sample rate for better resolution
        let duration: Float = 2.0     // 2 seconds for slow vibrato
        let sampleCount = Int(sampleRate * duration)

        var frequencies: [Float] = []
        var timeStamps: [Double] = []

        for i in 0..<sampleCount {
            let time = Float(i) / sampleRate
            let phase = 2.0 * Float.pi * vibratoRate * time
            let modulation = sin(phase) * (extentRatio - 1.0)
            frequencies.append(baseFreq * (1.0 + modulation))
            timeStamps.append(Double(time))
        }

        // When
        let result = sut.analyze(frequencies: frequencies, timeStamps: timeStamps)

        // Then
        XCTAssertTrue(result.isPresent)
        XCTAssertLessThan(result.rate, 6.0)
    }

    func testAnalyzeWithEmptyData_shouldReturnNoVibrato() {
        // Given
        let sut = VibratoAnalyzer()

        // When
        let result = sut.analyze(frequencies: [], timeStamps: [])

        // Then
        XCTAssertFalse(result.isPresent)
    }

    func testAnalyzeWithInsufficientData_shouldReturnNoVibrato() {
        // Given: Only 3 samples (not enough for vibrato detection)
        let sut = VibratoAnalyzer()
        let frequencies: [Float] = [440.0, 445.0, 440.0]
        let timeStamps: [Double] = [0.0, 0.05, 0.1]

        // When
        let result = sut.analyze(frequencies: frequencies, timeStamps: timeStamps)

        // Then
        XCTAssertFalse(result.isPresent)
    }

    // MARK: - Boundary Value Tests (Regression tests for lag calculation fix)

    func testAnalyzeWithVibratoNearMaxRate_shouldDetect() {
        // Given: Vibrato at 9Hz - near the maxVibratoRate=10Hz boundary
        // This tests the ceil() fix in minLag calculation
        let sut = VibratoAnalyzer()
        let baseFreq: Float = 440.0
        let vibratoRate: Float = 9.0  // Near boundary
        let extentCents: Float = 50.0

        let extentRatio = pow(2.0, extentCents / 1200.0)
        let sampleRate: Float = 100.0  // FCPE-like sample rate
        let duration: Float = 2.0
        let sampleCount = Int(sampleRate * duration)

        var frequencies: [Float] = []
        var timeStamps: [Double] = []

        for i in 0..<sampleCount {
            let time = Float(i) / sampleRate
            let phase = 2.0 * Float.pi * vibratoRate * time
            let modulation = sin(phase) * (extentRatio - 1.0)
            frequencies.append(baseFreq * (1.0 + modulation))
            timeStamps.append(Double(time))
        }

        // When
        let result = sut.analyze(frequencies: frequencies, timeStamps: timeStamps)

        // Then: Should detect vibrato (rate within 4-10Hz range)
        XCTAssertTrue(result.isPresent, "Vibrato at 9Hz should be detected within 4-10Hz range")
        XCTAssertGreaterThanOrEqual(result.rate, 4.0)
        XCTAssertLessThanOrEqual(result.rate, 10.0)
    }

    func testAnalyzeWithLowSampleRate_shouldStillDetect() {
        // Given: YIN-like low sample rate (~20Hz) with relaxed regularity
        let sut = VibratoAnalyzer(minimumRegularity: 0.15)  // YIN parameter
        let baseFreq: Float = 440.0
        let vibratoRate: Float = 6.0
        let extentCents: Float = 60.0

        let extentRatio = pow(2.0, extentCents / 1200.0)
        let sampleRate: Float = 20.0  // YIN-like low sample rate
        let duration: Float = 3.0     // Longer duration for low sample rate
        let sampleCount = Int(sampleRate * duration)

        var frequencies: [Float] = []
        var timeStamps: [Double] = []

        for i in 0..<sampleCount {
            let time = Float(i) / sampleRate
            let phase = 2.0 * Float.pi * vibratoRate * time
            let modulation = sin(phase) * (extentRatio - 1.0)
            frequencies.append(baseFreq * (1.0 + modulation))
            timeStamps.append(Double(time))
        }

        // When
        let result = sut.analyze(frequencies: frequencies, timeStamps: timeStamps)

        // Then: Should detect vibrato even at low sample rate
        XCTAssertTrue(result.isPresent, "Vibrato should be detected even at 20Hz sample rate with relaxed regularity")
    }

    // MARK: - Large Extent False Positive Prevention Tests

    func testAnalyzeWithLargeExtent_shouldNotDetectVibrato() {
        // Given: Piano-like large pitch changes (C4 to C5 scale = 1200 cents)
        // This simulates piano accompaniment that was causing false positives
        let sut = VibratoAnalyzer()

        // Piano scale frequencies (C major scale ascending and descending)
        let pianoFrequencies: [Float] = [
            261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25,  // C4 to C5
            493.88, 440.00, 392.00, 349.23, 329.63, 293.66, 261.63,          // C5 back to C4
            293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25,          // C4 to C5 again
            493.88, 440.00, 392.00, 349.23, 329.63, 293.66, 261.63           // C5 back to C4
        ]

        // 50ms intervals (20Hz sample rate)
        let timeStamps = pianoFrequencies.indices.map { Double($0) * 0.05 }

        // When
        let result = sut.analyze(frequencies: pianoFrequencies, timeStamps: timeStamps)

        // Then: Should NOT detect vibrato - extent is too large (600+ cents)
        XCTAssertFalse(result.isPresent, "Piano scale (600+ cents extent) should NOT be detected as vibrato")
        XCTAssertGreaterThan(result.extent, 200.0, "Extent should be greater than 200 cents threshold")
    }

    func testAnalyzeWithExtentAtBoundary_shouldNotDetectVibrato() {
        // Given: Modulation with extent at exactly 201 cents (just above 200 threshold)
        let sut = VibratoAnalyzer()
        let baseFreq: Float = 440.0
        let vibratoRate: Float = 6.0
        let extentCents: Float = 201.0  // Just above 200 threshold

        let extentRatio = pow(2.0, extentCents / 1200.0)
        let sampleRate: Float = 40.0
        let duration: Float = 2.0
        let sampleCount = Int(sampleRate * duration)

        var frequencies: [Float] = []
        var timeStamps: [Double] = []

        for i in 0..<sampleCount {
            let time = Float(i) / sampleRate
            let phase = 2.0 * Float.pi * vibratoRate * time
            let modulation = sin(phase) * (extentRatio - 1.0)
            frequencies.append(baseFreq * (1.0 + modulation))
            timeStamps.append(Double(time))
        }

        // When
        let result = sut.analyze(frequencies: frequencies, timeStamps: timeStamps)

        // Then: Should NOT detect vibrato - extent exceeds 200 cents threshold
        XCTAssertFalse(result.isPresent, "Extent of 201 cents should NOT be detected as vibrato")
    }

    func testAnalyzeWithExtentJustBelowThreshold_shouldDetectVibrato() {
        // Given: Modulation with extent at 95 cents (results in ~100 cents peak-to-peak/2)
        // Note: The analyzer calculates extent as (max-min)/2 in cents deviation from mean
        // For a sine wave with amplitude A in frequency ratio: extent ≈ 1200 * log2(1+A) * 2
        let sut = VibratoAnalyzer()
        let baseFreq: Float = 440.0
        let vibratoRate: Float = 6.0
        let extentCents: Float = 95.0  // Results in ~100 cents extent

        let extentRatio = pow(2.0, extentCents / 1200.0)
        let sampleRate: Float = 40.0
        let duration: Float = 2.0
        let sampleCount = Int(sampleRate * duration)

        var frequencies: [Float] = []
        var timeStamps: [Double] = []

        for i in 0..<sampleCount {
            let time = Float(i) / sampleRate
            let phase = 2.0 * Float.pi * vibratoRate * time
            let modulation = sin(phase) * (extentRatio - 1.0)
            frequencies.append(baseFreq * (1.0 + modulation))
            timeStamps.append(Double(time))
        }

        // When
        let result = sut.analyze(frequencies: frequencies, timeStamps: timeStamps)

        // Then: Should detect vibrato - extent is within valid range (15-200 cents)
        XCTAssertTrue(result.isPresent, "Extent of ~100 cents should be detected as vibrato")
        XCTAssertGreaterThanOrEqual(result.extent, 15.0, "Extent should be above minimum threshold")
        XCTAssertLessThanOrEqual(result.extent, 200.0, "Extent should be below maximum threshold")
    }
}
