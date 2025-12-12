//
//  NavigationUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for navigation and multi-recording scenarios
//

import XCTest

final class NavigationUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    // MARK: - Helper Methods

    /// Create a recording and return to home screen
    @MainActor
    private func createRecordingAndReturnToHome(_ app: XCUIApplication) {
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        // Wait for recording to start by checking StopButton appearance
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Continue recording for a moment to ensure valid audio data
        Thread.sleep(forTimeInterval: 1.0)

        stopButton.tap()

        // Wait for recording to be saved by checking PlayButton appearance
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // Navigate back to Home
        app.navigationBars.buttons.element(boundBy: 0).tap()
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should appear after navigation back")
    }

    // MARK: - Tests

    /// Test: Multiple recordings creation and management
    /// Expected: ~20 seconds execution time
    @MainActor
    func testMultipleRecordings() throws {
        let app = launchAppWithResetRecordingCount()

        // Create first recording
        createRecordingAndReturnToHome(app)

        // Create second recording
        createRecordingAndReturnToHome(app)

        // Navigate to Recording List
        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // Wait for list to load by checking for analysis buttons (visible in rows)
        let analysisLinks = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisLinks.firstMatch.waitForExistence(timeout: 5), "Analysis button should appear in list")

        // Screenshot: Recording list with multiple recordings
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "multiple_01_recording_list"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // 4. Verify both recordings are displayed
        let recordingCount = analysisLinks.count

        XCTAssertGreaterThanOrEqual(recordingCount, 2, "At least 2 recordings should be displayed in the list (found \(recordingCount))")

        // 5. Verify recordings are ordered correctly (most recent first)
        // Note: We can't easily verify the exact order in UI tests without specific timestamps,
        // but we can verify that multiple recordings are present and accessible
    }

    /// Test: Full navigation flow through the app
    /// Expected: ~15 seconds execution time
    @MainActor
    func testFullNavigationFlow() throws {
        let app = launchAppWithResetRecordingCount()

        // Screenshot: Home screen
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "nav_01_home_screen"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // 1. Home -> Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist on Home screen")
        homeRecordButton.tap()

        // Wait for recording screen to load by checking start button
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist on Recording screen")

        // Screenshot: Recording screen
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "nav_02_recording_screen"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // 2. Recording -> back to Home
        app.navigationBars.buttons.element(boundBy: 0).tap()
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Should be back at Home screen")

        // 3. Home -> Recording List screen
        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // Wait for recording list to load by checking navigation title
        let listTitle = app.navigationBars.staticTexts.element(matching: NSPredicate(format: "label CONTAINS[c] %@", L10n.listTitle))
        XCTAssertTrue(listTitle.waitForExistence(timeout: 5), "Recording List navigation title should be visible")

        // Screenshot: Recording List screen
        let screenshot3 = app.screenshot()
        let attachment3 = XCTAttachment(screenshot: screenshot3)
        attachment3.name = "nav_03_recording_list_screen"
        attachment3.lifetime = .keepAlways
        add(attachment3)

        // 4. Recording List -> back to Home
        app.navigationBars.buttons.element(boundBy: 0).tap()
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Should be back at Home screen from Recording List")

        // 5. Home -> Settings screen
        let homeSettingsButton = app.buttons["HomeSettingsButton"]
        XCTAssertTrue(homeSettingsButton.waitForExistence(timeout: 5), "Home settings button should exist")
        homeSettingsButton.tap()

        // Wait for settings screen to load by checking navigation title
        let settingsTitle = app.navigationBars.staticTexts.element(matching: NSPredicate(format: "label CONTAINS[c] %@", L10n.settingsTitle))
        let settingsTitleEn = app.navigationBars.staticTexts.element(matching: NSPredicate(format: "label CONTAINS[c] %@", "Settings"))
        let settingsLoaded = settingsTitle.waitForExistence(timeout: 5) || settingsTitleEn.waitForExistence(timeout: 1)
        XCTAssertTrue(settingsLoaded, "Settings navigation title should be visible")

        // Screenshot: Settings screen
        let screenshot4 = app.screenshot()
        let attachment4 = XCTAttachment(screenshot: screenshot4)
        attachment4.name = "nav_04_settings_screen"
        attachment4.lifetime = .keepAlways
        add(attachment4)

        // 6. Settings -> back to Home
        app.navigationBars.buttons.element(boundBy: 0).tap()
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Should be back at Home screen from Settings")

        // Screenshot: Back to Home
        let screenshot5 = app.screenshot()
        let attachment5 = XCTAttachment(screenshot: screenshot5)
        attachment5.name = "nav_05_back_to_home"
        attachment5.lifetime = .keepAlways
        add(attachment5)
    }
}
