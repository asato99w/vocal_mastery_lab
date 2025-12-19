import XCTest
@testable import VocalisDomain

/// Tests for AnalysisResult entity
final class AnalysisResultTests: XCTestCase {

    // MARK: - Initialization Tests

    func testInit_WithAllData_CreatesInstance() {
        // Given
        let pitchData = PitchAnalysisData(
            timeStamps: [0.0, 0.05],
            frequencies: [261.6, 262.3],
            confidences: [0.85, 0.92],
            targetNotes: [try! MIDINote(60), try! MIDINote(60)]
        )

        let spectrogramData = SpectrogramData(
            timeStamps: [0.0, 0.1],
            frequencyBins: [80, 180],
            magnitudes: [[0.1, 0.2], [0.3, 0.4]]
        )

        let scaleSettings = ScaleSettings(
            startNote: try! MIDINote(60),
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 0.5)
        )

        // When
        let result = AnalysisResult(
            pitchData: pitchData,
            spectrogramData: spectrogramData,
            scaleSettings: scaleSettings
        )

        // Then
        XCTAssertEqual(result.pitchData, pitchData)
        XCTAssertEqual(result.spectrogramData, spectrogramData)
        XCTAssertEqual(result.scaleSettings, scaleSettings)
    }

    func testInit_WithoutScaleSettings_CreatesInstance() {
        // Given: Free recording without scale
        let pitchData = PitchAnalysisData(
            timeStamps: [0.0],
            frequencies: [261.6],
            confidences: [0.85],
            targetNotes: [nil]
        )

        let spectrogramData = SpectrogramData(
            timeStamps: [0.0],
            frequencyBins: [80],
            magnitudes: [[0.1]]
        )

        // When
        let result = AnalysisResult(
            pitchData: pitchData,
            spectrogramData: spectrogramData,
            scaleSettings: nil
        )

        // Then
        XCTAssertEqual(result.pitchData, pitchData)
        XCTAssertEqual(result.spectrogramData, spectrogramData)
        XCTAssertNil(result.scaleSettings)
    }

    // MARK: - Equatable Tests

    func testEquality_WithSameData_ReturnsTrue() {
        // Given
        let pitchData = PitchAnalysisData(
            timeStamps: [0.0],
            frequencies: [261.6],
            confidences: [0.85],
            targetNotes: [nil]
        )

        let spectrogramData = SpectrogramData(
            timeStamps: [0.0],
            frequencyBins: [80],
            magnitudes: [[0.1]]
        )

        let result1 = AnalysisResult(
            pitchData: pitchData,
            spectrogramData: spectrogramData,
            scaleSettings: nil
        )

        let result2 = AnalysisResult(
            pitchData: pitchData,
            spectrogramData: spectrogramData,
            scaleSettings: nil
        )

        // When & Then
        XCTAssertEqual(result1, result2)
    }

    func testEquality_WithDifferentPitchData_ReturnsFalse() {
        // Given
        let pitchData1 = PitchAnalysisData(
            timeStamps: [0.0],
            frequencies: [261.6],
            confidences: [0.85],
            targetNotes: [nil]
        )

        let pitchData2 = PitchAnalysisData(
            timeStamps: [0.0],
            frequencies: [262.0],  // Different frequency
            confidences: [0.85],
            targetNotes: [nil]
        )

        let spectrogramData = SpectrogramData(
            timeStamps: [0.0],
            frequencyBins: [80],
            magnitudes: [[0.1]]
        )

        let result1 = AnalysisResult(
            pitchData: pitchData1,
            spectrogramData: spectrogramData,
            scaleSettings: nil
        )

        let result2 = AnalysisResult(
            pitchData: pitchData2,
            spectrogramData: spectrogramData,
            scaleSettings: nil
        )

        // When & Then
        XCTAssertNotEqual(result1, result2)
    }
}
