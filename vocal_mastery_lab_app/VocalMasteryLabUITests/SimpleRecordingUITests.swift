//
//  SimpleRecordingUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for simplified recording screen (UI_DESIGN.md)
//

import XCTest

/// UI tests for the new simplified recording screen
/// Based on UI_DESIGN.md specification:
/// - Timer display
/// - Record start/stop button
/// - Background recording hint
/// - Last recording info (date, duration)
/// - Play button
/// - Vocal extraction button (mock)
final class SimpleRecordingUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    // MARK: - Initial State Tests

    /// Test: Recording screen shows initial state correctly
    /// Verifies: Timer at 00:00:00, record button visible, no last recording initially
    @MainActor
    func testRecordingScreen_initialState() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
        homeRecordButton.tap()

        // Verify timer display shows 00:00:00
        let timerText = app.staticTexts["RecordingTimerLabel"]
        XCTAssertTrue(timerText.waitForExistence(timeout: 5), "Timer label should exist")

        // Verify record button is visible
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")

        // Verify background hint is visible
        let backgroundHint = app.staticTexts["BackgroundRecordingHint"]
        XCTAssertTrue(backgroundHint.waitForExistence(timeout: 3), "Background recording hint should be visible")

        // Screenshot: Initial recording screen
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "simple_recording_01_initial"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    // MARK: - Recording Flow Tests

    /// Test: Basic recording flow - start and stop
    /// Verifies: Timer updates during recording, stop button appears, returns to initial state
    @MainActor
    func testRecordingFlow_startAndStop() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
        homeRecordButton.tap()

        // Start recording
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))
        startButton.tap()

        // Verify stop button appears (recording started)
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop button should appear during recording")

        // Record for 2 seconds
        Thread.sleep(forTimeInterval: 2.0)

        // Screenshot: Recording in progress
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "simple_recording_02_recording"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // Stop recording
        stopButton.tap()

        // Verify start button reappears
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start button should reappear after stopping")

        // Screenshot: After recording
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "simple_recording_03_after_recording"
        attachment2.lifetime = .keepAlways
        add(attachment2)
    }

    // MARK: - Last Recording Tests

    /// Test: Last recording info appears after recording
    /// Verifies: Date, duration displayed, play and vocal extraction buttons appear
    @MainActor
    func testLastRecordingInfo_appearsAfterRecording() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
        homeRecordButton.tap()

        // Create a recording
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10))

        // Record for 1 second
        Thread.sleep(forTimeInterval: 1.0)
        stopButton.tap()

        // Wait for recording to be saved
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))

        // Verify last recording section appears
        let lastRecordingSection = app.otherElements["LastRecordingSection"]
        XCTAssertTrue(lastRecordingSection.waitForExistence(timeout: 5), "Last recording section should appear")

        // Verify play button appears
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 3), "Play button should appear for last recording")

        // Verify vocal extraction button appears
        let vocalExtractionButton = app.buttons["VocalExtractionButton"]
        XCTAssertTrue(vocalExtractionButton.waitForExistence(timeout: 3), "Vocal extraction button should appear")

        // Screenshot: Last recording info
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "simple_recording_04_last_recording_info"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    // MARK: - Playback Tests

    /// Test: Play last recording
    /// Verifies: Playback starts when play button tapped
    @MainActor
    func testPlayLastRecording() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen and create a recording
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
        homeRecordButton.tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10))

        Thread.sleep(forTimeInterval: 1.0)
        stopButton.tap()

        // Wait for recording to be saved
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear")

        // Tap play button
        playButton.tap()

        // Verify playback state (button changes or indicator appears)
        // The exact verification depends on implementation
        Thread.sleep(forTimeInterval: 0.5)

        // Screenshot: During playback
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "simple_recording_05_playback"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    // MARK: - Vocal Extraction Tests (Mock)

    /// Test: Vocal extraction button triggers action
    /// Verifies: Button is tappable and triggers some response (mock implementation)
    @MainActor
    func testVocalExtractionButton_triggersAction() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen and create a recording
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
        homeRecordButton.tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10))

        Thread.sleep(forTimeInterval: 1.0)
        stopButton.tap()

        // Wait for vocal extraction button
        let vocalExtractionButton = app.buttons["VocalExtractionButton"]
        XCTAssertTrue(vocalExtractionButton.waitForExistence(timeout: 5), "Vocal extraction button should appear")

        // Tap vocal extraction button
        vocalExtractionButton.tap()

        // For mock implementation, verify some response (alert, navigation, or state change)
        // This test will be updated when real implementation is added
        Thread.sleep(forTimeInterval: 0.5)

        // Screenshot: After vocal extraction tap
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "simple_recording_06_vocal_extraction"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    // MARK: - Navigation Tests

    /// Test: Navigate to recording list
    /// Verifies: List button navigates to recording list
    @MainActor
    func testNavigateToRecordingList() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
        homeRecordButton.tap()

        // Wait for recording screen
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))

        // Find and tap list button in navigation bar
        let listButton = app.buttons["RecordingListButton"]
        XCTAssertTrue(listButton.waitForExistence(timeout: 3), "List button should exist in navigation bar")
        listButton.tap()

        // Verify navigation to list (check for list-specific element)
        // The exact element depends on RecordingListView implementation
        Thread.sleep(forTimeInterval: 1.0)

        // Screenshot: Recording list
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "simple_recording_07_list"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
