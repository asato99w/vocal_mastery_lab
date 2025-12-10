import XCTest
@testable import VocalMasteringLab
@testable import VocalisDomain

/// Tests for FCPEStrategy - Neural network based pitch detection
/// Note: Full integration tests require CoreML model in bundle, which is not available in unit tests
/// These tests verify the strategy structure and behavior without model loading
final class FCPEStrategyTests: XCTestCase {

    // MARK: - Protocol Conformance Tests

    func testConformsTo_PitchDetectionStrategy() {
        // When: FCPEStrategy is created
        // Then: It should conform to PitchDetectionStrategy
        let strategy: PitchDetectionStrategy = FCPEStrategy()
        XCTAssertNotNil(strategy)
    }

    func testName_returnsFCPE() {
        // Given: A FCPEStrategy
        let sut = FCPEStrategy()

        // When: Accessing the name
        let name = sut.name

        // Then: Should return "FCPE"
        XCTAssertEqual(name, "FCPE")
    }

    func testRequiresOctaveCorrection_returnsFalse() {
        // Given: A FCPEStrategy
        let sut = FCPEStrategy()

        // When: Checking if octave correction is required
        let requires = sut.requiresOctaveCorrection

        // Then: FCPE uses neural network with good octave stability
        XCTAssertFalse(requires)
    }

    // MARK: - Pitch Detection Tests (Without Model)

    func testDetectPitch_withEmptySamples_returnsEmptyArray() {
        // Given: A FCPEStrategy and empty samples
        let sut = FCPEStrategy()
        let samples: [Float] = []

        // When: Detecting pitch
        let result = sut.detectPitch(samples: samples, sampleRate: 44100.0)

        // Then: Should return empty array
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - Helpers

    private func generateSinusoid(frequency: Double, sampleRate: Double, duration: Double) -> [Float] {
        let sampleCount = Int(sampleRate * duration)
        return (0..<sampleCount).map { i in
            let t = Double(i) / sampleRate
            return Float(sin(2.0 * .pi * frequency * t) * 0.8)  // Amplitude 0.8
        }
    }
}
