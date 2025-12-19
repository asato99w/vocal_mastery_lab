import XCTest
@testable import VocalisDomain

// MARK: - Mock Implementation for Protocol Testing

final class MockPitchDetectionStrategy: PitchDetectionStrategy {
    var name: String
    var requiresOctaveCorrection: Bool
    var framesToReturn: [PitchFrame] = []

    init(name: String = "MockStrategy", requiresOctaveCorrection: Bool = false) {
        self.name = name
        self.requiresOctaveCorrection = requiresOctaveCorrection
    }

    func detectPitch(samples: [Float], sampleRate: Double) -> [PitchFrame] {
        return framesToReturn
    }
}

// MARK: - Protocol Tests

final class PitchDetectionStrategyTests: XCTestCase {

    // MARK: - Protocol Conformance Tests

    func testProtocol_hasNameProperty() {
        // Given
        let strategy = MockPitchDetectionStrategy(name: "TestStrategy")

        // Then
        XCTAssertEqual(strategy.name, "TestStrategy")
    }

    func testProtocol_hasRequiresOctaveCorrectionProperty() {
        // Given
        let strategyNeedsCorrection = MockPitchDetectionStrategy(requiresOctaveCorrection: true)
        let strategyNoCorrection = MockPitchDetectionStrategy(requiresOctaveCorrection: false)

        // Then
        XCTAssertTrue(strategyNeedsCorrection.requiresOctaveCorrection)
        XCTAssertFalse(strategyNoCorrection.requiresOctaveCorrection)
    }

    func testProtocol_detectPitch_returnsFrames() {
        // Given
        let strategy = MockPitchDetectionStrategy()
        let expectedFrames = [
            PitchFrame(timestamp: 0.0, frequency: 440.0, confidence: 0.9, amplitude: 0.5),
            PitchFrame(timestamp: 0.05, frequency: 450.0, confidence: 0.85, amplitude: 0.6),
            PitchFrame(timestamp: 0.1, frequency: nil, confidence: 0.1, amplitude: 0.02)
        ]
        strategy.framesToReturn = expectedFrames

        // When
        let samples: [Float] = Array(repeating: 0.0, count: 4410) // 0.1s at 44100Hz
        let frames = strategy.detectPitch(samples: samples, sampleRate: 44100.0)

        // Then
        XCTAssertEqual(frames.count, 3)
        XCTAssertEqual(frames[0].frequency, 440.0)
        XCTAssertNil(frames[2].frequency)
    }

    // MARK: - Usage Pattern Tests

    func testStrategy_canBeUsedPolymorphically() {
        // Given
        let strategies: [PitchDetectionStrategy] = [
            MockPitchDetectionStrategy(name: "YIN", requiresOctaveCorrection: true),
            MockPitchDetectionStrategy(name: "pYIN", requiresOctaveCorrection: false)
        ]

        // Then
        XCTAssertEqual(strategies[0].name, "YIN")
        XCTAssertTrue(strategies[0].requiresOctaveCorrection)
        XCTAssertEqual(strategies[1].name, "pYIN")
        XCTAssertFalse(strategies[1].requiresOctaveCorrection)
    }
}
