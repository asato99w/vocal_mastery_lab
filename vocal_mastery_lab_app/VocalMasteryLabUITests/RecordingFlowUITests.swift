//
//  RecordingFlowUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for recording flow (start/stop recording)
//

import XCTest

final class RecordingFlowUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    /// Test 1: Basic recording flow - record start and stop
    /// Expected: ~10 seconds execution time
    @MainActor
    func testBasicRecordingFlow() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Navigate to Recording screen from Home
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist on initial screen")
        homeRecordButton.tap()

        // 2. Wait for recording screen to load
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist after navigating to Recording screen")

        // Screenshot: Initial recording screen
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "basic_flow_01_initial_screen"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // 3. Start recording
        startButton.tap()

        // 4. Verify immediate visual feedback - the UI should change within 1 second
        // This ensures user knows the app is responding to their tap
        // Check for either: loading indicator (preparing state) OR countdown message (countdown state)
        let loadingIndicator = app.activityIndicators["RecordingLoadingIndicator"]
        let countdownText = app.staticTexts[L10n.countdownMessage]

        let hasImmediateFeedback = loadingIndicator.waitForExistence(timeout: 0.5) || countdownText.waitForExistence(timeout: 0.5)
        XCTAssertTrue(hasImmediateFeedback, "Either loading indicator or countdown message should appear immediately after tapping start button to show the app is responding")

        // 5. Wait for recording to start by checking StopButton appearance
        // No Thread.sleep needed - waitForExistence will wait for countdown + initialization
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear during recording")

        // Continue recording for a moment to ensure valid audio data
        Thread.sleep(forTimeInterval: 1.0)

        // Screenshot: Recording in progress
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "basic_flow_02_recording_in_progress"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // 6. Stop recording
        stopButton.tap()

        // Wait for recording to finish and be saved by checking PlayButton appearance
        // No Thread.sleep needed - waitForExistence will wait for save completion
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play last recording button should appear after recording, confirming save was successful")

        // 7. Verify we're back to initial state (start button should appear again)
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should reappear after stopping recording")

        // Screenshot: After recording completion
        let screenshot3 = app.screenshot()
        let attachment3 = XCTAttachment(screenshot: screenshot3)
        attachment3.name = "basic_flow_03_after_recording"
        attachment3.lifetime = .keepAlways
        add(attachment3)
    }
}
