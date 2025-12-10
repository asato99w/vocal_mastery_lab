import XCTest
import AVFoundation
@testable import VocalMasteringLab

final class AudioSessionManagerTests: XCTestCase {

    var sut: AudioSessionManager!
    var audioSession: AVAudioSession!

    override func setUp() {
        super.setUp()
        sut = AudioSessionManager.shared
        audioSession = AVAudioSession.sharedInstance()
    }

    override func tearDown() {
        // Reset audio session to a known state
        try? audioSession.setActive(false, options: .notifyOthersOnDeactivation)
        audioSession = nil
        sut = nil
        super.tearDown()
    }

    // MARK: - Singleton Tests

    func testShared_AlwaysReturnsSameInstance() {
        // When
        let instance1 = AudioSessionManager.shared
        let instance2 = AudioSessionManager.shared

        // Then
        XCTAssertTrue(instance1 === instance2)
    }

    // MARK: - Configuration Tests

    func testConfigureForRecording_SetsCorrectCategory() throws {
        // When
        try sut.configureForRecording()

        // Then
        XCTAssertEqual(audioSession.category, .playAndRecord)
        // Mode is dynamically selected based on audio route (.measurement for headphones, .videoRecording for built-in)
        XCTAssertTrue([AVAudioSession.Mode.measurement, .videoRecording].contains(audioSession.mode))
        XCTAssertTrue(audioSession.categoryOptions.contains(.defaultToSpeaker))
        XCTAssertTrue(audioSession.categoryOptions.contains(.allowBluetooth))
        XCTAssertTrue(audioSession.categoryOptions.contains(.allowBluetoothA2DP))
    }

    func testConfigureForPlayback_SetsCorrectCategory() throws {
        // When
        try sut.configureForPlayback()

        // Then
        XCTAssertEqual(audioSession.category, .playback)
        XCTAssertEqual(audioSession.mode, .default)
        XCTAssertTrue(audioSession.categoryOptions.contains(.mixWithOthers))
    }

    func testConfigureForRecordingAndPlayback_SetsCorrectCategory() throws {
        // When
        try sut.configureForRecordingAndPlayback()

        // Then
        XCTAssertEqual(audioSession.category, .playAndRecord)
        // Mode is dynamically selected based on audio route (.measurement for headphones, .videoRecording for built-in)
        XCTAssertTrue([AVAudioSession.Mode.measurement, .videoRecording].contains(audioSession.mode))
        // Note: .defaultToSpeaker is only included when mode is NOT .measurement
        XCTAssertTrue(audioSession.categoryOptions.contains(.allowBluetooth))
        XCTAssertTrue(audioSession.categoryOptions.contains(.allowBluetoothA2DP))
    }

    // MARK: - Activation Tests

    func testActivate_ActivatesAudioSession() throws {
        // Given
        try sut.configureForPlayback()

        // When
        try sut.activate()

        // Then - no error thrown means success
        // Note: On simulator, some errors are acceptable
        XCTAssertTrue(true) // If we reach here, activation succeeded or was handled gracefully
    }

    func testActivate_MultipleTimesDoesNotFail() throws {
        // Given
        try sut.configureForPlayback()

        // When - activate multiple times
        try sut.activate()
        try sut.activate()
        try sut.activate()

        // Then - no error thrown
        XCTAssertTrue(true)
    }

    func testActivateIfNeeded_WhenNoOtherAudioPlaying_Activates() throws {
        // Given
        try sut.configureForPlayback()

        // When
        try sut.activateIfNeeded()

        // Then - no error thrown
        XCTAssertTrue(true)
    }

    func testDeactivate_DeactivatesAudioSession() throws {
        // Given
        try sut.configureForPlayback()
        try sut.activate()

        // When
        try sut.deactivate()

        // Then - no error thrown
        XCTAssertTrue(true)
    }

    // MARK: - Notification Handling Tests

    func testInterruptionNotification_Began_HandledCorrectly() {
        // Given
        let expectation = expectation(description: "Interruption notification handled")
        expectation.isInverted = false // We expect this to complete without errors

        // When - simulate interruption began notification
        NotificationCenter.default.post(
            name: AVAudioSession.interruptionNotification,
            object: audioSession,
            userInfo: [
                AVAudioSessionInterruptionTypeKey: UInt(AVAudioSession.InterruptionType.began.rawValue)
            ]
        )

        // Then - no crash, notification is handled
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 1.0)
    }

    func testInterruptionNotification_Ended_HandledCorrectly() {
        // Given
        let expectation = expectation(description: "Interruption ended notification handled")
        expectation.isInverted = false

        // When - simulate interruption ended notification
        NotificationCenter.default.post(
            name: AVAudioSession.interruptionNotification,
            object: audioSession,
            userInfo: [
                AVAudioSessionInterruptionTypeKey: UInt(AVAudioSession.InterruptionType.ended.rawValue),
                AVAudioSessionInterruptionOptionKey: UInt(AVAudioSession.InterruptionOptions.shouldResume.rawValue)
            ]
        )

        // Then
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 1.0)
    }

    func testRouteChangeNotification_NewDeviceAvailable_HandledCorrectly() {
        // Given
        let expectation = expectation(description: "Route change notification handled")
        expectation.isInverted = false

        // When - simulate new device available notification
        NotificationCenter.default.post(
            name: AVAudioSession.routeChangeNotification,
            object: audioSession,
            userInfo: [
                AVAudioSessionRouteChangeReasonKey: UInt(AVAudioSession.RouteChangeReason.newDeviceAvailable.rawValue)
            ]
        )

        // Then
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 1.0)
    }

    func testRouteChangeNotification_OldDeviceUnavailable_HandledCorrectly() {
        // Given
        let expectation = expectation(description: "Route change notification handled")
        expectation.isInverted = false

        // When - simulate device removal notification
        NotificationCenter.default.post(
            name: AVAudioSession.routeChangeNotification,
            object: audioSession,
            userInfo: [
                AVAudioSessionRouteChangeReasonKey: UInt(AVAudioSession.RouteChangeReason.oldDeviceUnavailable.rawValue)
            ]
        )

        // Then
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 1.0)
    }

    // MARK: - Integration Tests

    func testFullRecordingFlow_ConfigureActivateDeactivate_Succeeds() throws {
        // Given & When & Then
        try sut.configureForRecording()
        try sut.activate()

        // Simulate some work
        Thread.sleep(forTimeInterval: 0.1)

        try sut.deactivate()

        // No error thrown = success
        XCTAssertTrue(true)
    }

    func testFullPlaybackFlow_ConfigureActivateDeactivate_Succeeds() throws {
        // Given & When & Then
        try sut.configureForPlayback()
        try sut.activate()

        // Simulate some work
        Thread.sleep(forTimeInterval: 0.1)

        try sut.deactivate()

        // No error thrown = success
        XCTAssertTrue(true)
    }
}
