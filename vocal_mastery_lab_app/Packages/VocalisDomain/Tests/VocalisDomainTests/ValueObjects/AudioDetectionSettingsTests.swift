import XCTest
@testable import VocalisDomain

/// Tests for AudioDetectionSettings Value Object
final class AudioDetectionSettingsTests: XCTestCase {

    // MARK: - Initialization Tests

    func testInit_shouldCreateWithValidValues() {
        // When
        let settings = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4
        )

        // Then
        XCTAssertEqual(settings.scalePlaybackVolume, 0.8)
        XCTAssertEqual(settings.recordingPlaybackVolume, 0.7)
        XCTAssertEqual(settings.rmsSilenceThreshold, 0.02)
        XCTAssertEqual(settings.confidenceThreshold, 0.4)
    }

    // MARK: - Volume Validation Tests

    func testInit_shouldClampScalePlaybackVolumeToMinimum() {
        // When
        let settings = AudioDetectionSettings(
            scalePlaybackVolume: -0.5,  // Invalid: too low
            recordingPlaybackVolume: 0.8,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4
        )

        // Then
        XCTAssertEqual(settings.scalePlaybackVolume, 0.0, "Scale playback volume should be clamped to minimum 0.0")
    }

    func testInit_shouldClampScalePlaybackVolumeToMaximum() {
        // When
        let settings = AudioDetectionSettings(
            scalePlaybackVolume: 1.5,  // Invalid: too high
            recordingPlaybackVolume: 0.8,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4
        )

        // Then
        XCTAssertEqual(settings.scalePlaybackVolume, 1.0, "Scale playback volume should be clamped to maximum 1.0")
    }

    func testInit_shouldClampRecordingPlaybackVolumeToMinimum() {
        // When
        let settings = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: -0.5,  // Invalid: too low
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4
        )

        // Then
        XCTAssertEqual(settings.recordingPlaybackVolume, 0.0, "Recording playback volume should be clamped to minimum 0.0")
    }

    func testInit_shouldClampRecordingPlaybackVolumeToMaximum() {
        // When
        let settings = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 1.5,  // Invalid: too high
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4
        )

        // Then
        XCTAssertEqual(settings.recordingPlaybackVolume, 1.0, "Recording playback volume should be clamped to maximum 1.0")
    }

    // MARK: - RMS Threshold Validation Tests

    func testInit_shouldClampRMSThresholdToMinimum() {
        // When
        let settings = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: -0.1,  // Invalid: too low
            confidenceThreshold: 0.4
        )

        // Then
        XCTAssertEqual(settings.rmsSilenceThreshold, 0.001, "RMS threshold should be clamped to minimum 0.001")
    }

    func testInit_shouldClampRMSThresholdToMaximum() {
        // When
        let settings = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.5,  // Invalid: too high
            confidenceThreshold: 0.4
        )

        // Then
        XCTAssertEqual(settings.rmsSilenceThreshold, 0.1, "RMS threshold should be clamped to maximum 0.1")
    }

    // MARK: - Confidence Threshold Validation Tests

    func testInit_shouldClampConfidenceThresholdToMinimum() {
        // When
        let settings = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.05  // Invalid: too low
        )

        // Then
        XCTAssertEqual(settings.confidenceThreshold, 0.1, "Confidence threshold should be clamped to minimum 0.1")
    }

    func testInit_shouldClampConfidenceThresholdToMaximum() {
        // When
        let settings = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 1.5  // Invalid: too high
        )

        // Then
        XCTAssertEqual(settings.confidenceThreshold, 1.0, "Confidence threshold should be clamped to maximum 1.0")
    }

    // MARK: - Default Settings Tests

    func testDefaultSettings_shouldHaveReasonableValues() {
        // When
        let settings = AudioDetectionSettings.default

        // Then
        XCTAssertEqual(settings.scalePlaybackVolume, 0.8)
        XCTAssertEqual(settings.recordingPlaybackVolume, 0.8)
        XCTAssertEqual(settings.rmsSilenceThreshold, 0.02)
        XCTAssertEqual(settings.confidenceThreshold, 0.4)
    }

    func testSimulatorSettings_shouldHaveLowerThresholds() {
        // When
        let settings = AudioDetectionSettings.simulator

        // Then
        XCTAssertEqual(settings.scalePlaybackVolume, 0.8)
        XCTAssertEqual(settings.recordingPlaybackVolume, 0.8)
        XCTAssertEqual(settings.rmsSilenceThreshold, 0.005, "Simulator should have lower RMS threshold")
        XCTAssertEqual(settings.confidenceThreshold, 0.3, "Simulator should have lower confidence threshold")
    }

    // MARK: - Detection Sensitivity Tests

    func testSensitivity_whenRMSThresholdIs005_shouldReturnHigh() {
        // Given
        let settings = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.005,
            confidenceThreshold: 0.4
        )

        // When
        let sensitivity = settings.sensitivity

        // Then
        XCTAssertEqual(sensitivity, .high)
    }

    func testSensitivity_whenRMSThresholdIs002_shouldReturnNormal() {
        // Given
        let settings = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4
        )

        // When
        let sensitivity = settings.sensitivity

        // Then
        XCTAssertEqual(sensitivity, .normal)
    }

    func testSensitivity_whenRMSThresholdIs005High_shouldReturnLow() {
        // Given
        let settings = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.05,
            confidenceThreshold: 0.4
        )

        // When
        let sensitivity = settings.sensitivity

        // Then
        XCTAssertEqual(sensitivity, .low)
    }

    // MARK: - DetectionSensitivity Enum Tests

    func testDetectionSensitivityLow_shouldReturnCorrectRMSThreshold() {
        // When
        let threshold = AudioDetectionSettings.DetectionSensitivity.low.rmsThreshold

        // Then
        XCTAssertEqual(threshold, 0.05)
    }

    func testDetectionSensitivityNormal_shouldReturnCorrectRMSThreshold() {
        // When
        let threshold = AudioDetectionSettings.DetectionSensitivity.normal.rmsThreshold

        // Then
        XCTAssertEqual(threshold, 0.02)
    }

    func testDetectionSensitivityHigh_shouldReturnCorrectRMSThreshold() {
        // When
        let threshold = AudioDetectionSettings.DetectionSensitivity.high.rmsThreshold

        // Then
        XCTAssertEqual(threshold, 0.005)
    }

    func testDetectionSensitivityInit_withLowThreshold_shouldReturnHigh() {
        // When
        let sensitivity = AudioDetectionSettings.DetectionSensitivity(fromRMSThreshold: 0.008)

        // Then
        XCTAssertEqual(sensitivity, .high)
    }

    func testDetectionSensitivityInit_withMidThreshold_shouldReturnNormal() {
        // When
        let sensitivity = AudioDetectionSettings.DetectionSensitivity(fromRMSThreshold: 0.025)

        // Then
        XCTAssertEqual(sensitivity, .normal)
    }

    func testDetectionSensitivityInit_withHighThreshold_shouldReturnLow() {
        // When
        let sensitivity = AudioDetectionSettings.DetectionSensitivity(fromRMSThreshold: 0.045)

        // Then
        XCTAssertEqual(sensitivity, .low)
    }

    // MARK: - Codable Tests

    func testCodable_shouldEncodeAndDecode() throws {
        // Given
        let original = AudioDetectionSettings(
            scalePlaybackVolume: 0.75,
            recordingPlaybackVolume: 0.65,
            rmsSilenceThreshold: 0.015,
            confidenceThreshold: 0.35
        )

        // When
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(AudioDetectionSettings.self, from: data)

        // Then
        XCTAssertEqual(original, decoded)
    }

    // MARK: - Equatable Tests

    func testEquatable_whenAllPropertiesSame_shouldReturnTrue() {
        // Given
        let settings1 = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4
        )
        let settings2 = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4
        )

        // Then
        XCTAssertEqual(settings1, settings2)
    }

    func testEquatable_whenScalePlaybackVolumeDifferent_shouldReturnFalse() {
        // Given
        let settings1 = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4
        )
        let settings2 = AudioDetectionSettings(
            scalePlaybackVolume: 0.6,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4
        )

        // Then
        XCTAssertNotEqual(settings1, settings2)
    }

    func testEquatable_whenRecordingPlaybackVolumeDifferent_shouldReturnFalse() {
        // Given
        let settings1 = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4
        )
        let settings2 = AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.5,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4
        )

        // Then
        XCTAssertNotEqual(settings1, settings2)
    }

    // MARK: - ScaleSoundType Tests

    func testDefaultSettingsIncludesScaleSoundType() {
        // Given: デフォルト設定
        let settings = AudioDetectionSettings.default

        // Then: scaleSoundTypeが含まれる
        XCTAssertEqual(settings.scaleSoundType, .acousticGrandPiano)
    }

    func testCustomScaleSoundTypeInitialization() {
        // すべての音源タイプで初期化できることを確認

        for soundType in ScaleSoundType.allCases {
            // When: カスタム音源で初期化
            let settings = AudioDetectionSettings(
                scalePlaybackVolume: 0.5,
                recordingPlaybackVolume: 0.5,
                rmsSilenceThreshold: 0.02,
                confidenceThreshold: 0.4,
                scaleSoundType: soundType
            )

            // Then: 指定した音源が設定される
            XCTAssertEqual(settings.scaleSoundType, soundType)
        }
    }

    func testScaleSoundTypeEncodingAndDecoding() throws {
        // Given: カスタム音源を含む設定
        let originalSettings = AudioDetectionSettings(
            scalePlaybackVolume: 0.7,
            recordingPlaybackVolume: 0.6,
            rmsSilenceThreshold: 0.015,
            confidenceThreshold: 0.35,
            scaleSoundType: .flute
        )

        // When: エンコード
        let encoder = JSONEncoder()
        let data = try encoder.encode(originalSettings)

        // When: デコード
        let decoder = JSONDecoder()
        let decodedSettings = try decoder.decode(
            AudioDetectionSettings.self,
            from: data
        )

        // Then: すべてのプロパティが正しく復元される
        XCTAssertEqual(decodedSettings, originalSettings)
        XCTAssertEqual(decodedSettings.scaleSoundType, .flute)
    }

    func testBackwardCompatibility() throws {
        // Given: scaleSoundTypeを含まない古いJSON（既存ユーザーのデータ）
        let oldJSON = """
        {
            "scalePlaybackVolume": 0.5,
            "recordingPlaybackVolume": 0.5,
            "rmsSilenceThreshold": 0.02,
            "confidenceThreshold": 0.4
        }
        """

        let data = oldJSON.data(using: .utf8)!

        // When: デコード
        let decoder = JSONDecoder()
        let settings = try decoder.decode(
            AudioDetectionSettings.self,
            from: data
        )

        // Then: デフォルト値が使用される
        XCTAssertEqual(settings.scaleSoundType, .acousticGrandPiano)

        // 他のプロパティは正しくデコードされる
        XCTAssertEqual(settings.scalePlaybackVolume, 0.5)
        XCTAssertEqual(settings.recordingPlaybackVolume, 0.5)
    }

    func testEqualityWithDifferentScaleSoundType() {
        // Given: 音源のみ異なる設定
        let settings1 = AudioDetectionSettings(
            scalePlaybackVolume: 0.5,
            recordingPlaybackVolume: 0.5,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4,
            scaleSoundType: .acousticGrandPiano
        )

        let settings2 = AudioDetectionSettings(
            scalePlaybackVolume: 0.5,
            recordingPlaybackVolume: 0.5,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4,
            scaleSoundType: .electricPiano
        )

        // Then: 等しくない
        XCTAssertNotEqual(settings1, settings2)
    }
}
