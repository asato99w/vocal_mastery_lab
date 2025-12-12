//
//  LandscapeLayoutUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for landscape orientation layout verification
//

import XCTest

final class LandscapeLayoutUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    /// Test landscape layout during recording and playback
    /// Captures screenshots in landscape orientation to verify UI layout
    @MainActor
    func testLandscapeRecordingAndPlayback() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Navigate to Recording screen from Home
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // 2. Wait for recording screen to load
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")

        // Screenshot 1: Portrait - Initial recording screen
        takeScreenshot(app, name: "landscape_01_portrait_initial")

        // 3. Rotate to landscape
        XCUIDevice.shared.orientation = .landscapeLeft
        Thread.sleep(forTimeInterval: 0.5)  // Wait for rotation animation

        // Screenshot 2: Landscape - Initial recording screen
        takeScreenshot(app, name: "landscape_02_landscape_initial")

        // 4. Start recording
        startButton.tap()

        // Wait for recording to start
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Record for a moment
        Thread.sleep(forTimeInterval: 2.0)

        // Screenshot 3: Landscape - Recording in progress
        takeScreenshot(app, name: "landscape_03_recording_in_progress")

        // 5. Stop recording
        stopButton.tap()

        // Wait for recording to finish
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after recording")

        // Screenshot 4: Landscape - After recording (ready to play)
        takeScreenshot(app, name: "landscape_04_after_recording")

        // 6. Play the recording
        playButton.tap()

        // Wait for playback to start
        Thread.sleep(forTimeInterval: 0.5)

        // Screenshot 5: Landscape - During playback
        takeScreenshot(app, name: "landscape_05_during_playback")

        // Wait a bit more during playback
        Thread.sleep(forTimeInterval: 1.0)

        // Screenshot 6: Landscape - Playback continued
        takeScreenshot(app, name: "landscape_06_playback_continued")

        // 7. Rotate back to portrait
        XCUIDevice.shared.orientation = .portrait
        Thread.sleep(forTimeInterval: 0.5)

        // Screenshot 7: Portrait - During playback
        takeScreenshot(app, name: "landscape_07_portrait_playback")
    }

    /// Test landscape layout with scale settings visible
    @MainActor
    func testLandscapeWithSettingsPanel() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
        homeRecordButton.tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))

        // Show settings panel
        let showSettingsButton = app.buttons["ShowSettingsButton"]
        if showSettingsButton.waitForExistence(timeout: 2) {
            showSettingsButton.tap()
            Thread.sleep(forTimeInterval: 0.3)
        }

        // Screenshot: Portrait with settings
        takeScreenshot(app, name: "landscape_settings_01_portrait")

        // Rotate to landscape
        XCUIDevice.shared.orientation = .landscapeLeft
        Thread.sleep(forTimeInterval: 0.5)

        // Screenshot: Landscape with settings
        takeScreenshot(app, name: "landscape_settings_02_landscape")

        // Rotate back to portrait
        XCUIDevice.shared.orientation = .portrait
    }

    // MARK: - Helper Methods

    /// Take screenshot using XCUIScreen.main for proper landscape orientation support
    /// Using app.screenshot() in landscape mode captures only partial screen
    /// See: https://developer.apple.com/forums/thread/665859
    private func takeScreenshot(_ app: XCUIApplication, name: String) {
        // Use XCUIScreen.main.screenshot() instead of app.screenshot()
        // This properly captures the full screen in landscape orientation
        let screenshot = XCUIScreen.main.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
