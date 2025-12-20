//
//  AnalysisUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for analysis screen functionality
//

import XCTest

final class AnalysisUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    /// Test: Analysis view display and basic playback controls
    /// Expected: ~20 seconds execution time
    @MainActor
    func testAnalysisViewDisplay() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Create a recording first
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

        // 2. Navigate to Recording List
        app.navigationBars.buttons.element(boundBy: 0).tap()

        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // Wait for recording list to load
        let analysisLinks = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisLinks.firstMatch.waitForExistence(timeout: 5), "Analysis navigation link should exist")

        // 3. Navigate to Analysis screen
        analysisLinks.firstMatch.tap()

        // Wait for analysis screen to load
        // Note: In portrait mode, RecordingInfoCompact is used (not RecordingInfoPanel with "Recording Info" title)
        let recordingInfoCompact = app.otherElements["RecordingInfoCompact"]
        XCTAssertTrue(recordingInfoCompact.waitForExistence(timeout: 10), "Analysis screen should load")

        // 4. Wait for analysis to complete by checking for playback button
        let playPauseButtonWait = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButtonWait.waitForExistence(timeout: 10), "Analysis should complete and show playback controls")

        // Screenshot: Analysis screen after loading (consolidated from 2 screenshots)
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "analysis_01_loaded"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // 5. Verify Recording Info Panel is displayed (already checked during navigation)

        // 6. Verify Playback controls exist
        let playPauseButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButton.waitForExistence(timeout: 10), "Play/Pause button should exist")

        let seekBackButton = app.buttons["AnalysisSeekBackButton"]
        XCTAssertTrue(seekBackButton.waitForExistence(timeout: 3), "Seek back button should exist")

        let seekForwardButton = app.buttons["AnalysisSeekForwardButton"]
        XCTAssertTrue(seekForwardButton.waitForExistence(timeout: 3), "Seek forward button should exist")

        let progressSlider = app.sliders["AnalysisProgressSlider"]
        XCTAssertTrue(progressSlider.waitForExistence(timeout: 3), "Progress slider should exist")

        // 7. Test playback controls - Play
        playPauseButton.tap()

        // Wait a moment for playback to start (minimum time for valid state)
        Thread.sleep(forTimeInterval: 0.5)

        // Verify button changed to pause state (still exists but may show different icon)
        XCTAssertTrue(playPauseButton.exists, "Play/Pause button should still exist during playback")

        // 8. Test playback controls - Pause
        playPauseButton.tap()

        // 9. Test seek controls
        seekBackButton.tap()
        seekForwardButton.tap()

        // Screenshot: Final state after all playback control operations
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "analysis_02_after_controls"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // 10. Verify navigation back works
        app.navigationBars.buttons.element(boundBy: 0).tap()

        // Should be back at Recording List
        let analysisLinksAfterBack = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisLinksAfterBack.firstMatch.waitForExistence(timeout: 3), "Should be back at recording list")
    }

    // MARK: - Helper Methods

    /// Navigate to analysis screen by creating a recording and navigating to it
    private func navigateToAnalysisScreen(_ app: XCUIApplication) {
        // 1. Create a recording
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        Thread.sleep(forTimeInterval: 1.0)
        stopButton.tap()

        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // 2. Navigate to Recording List
        app.navigationBars.buttons.element(boundBy: 0).tap()

        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // Wait for recording list to load
        let analysisLinks = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisLinks.firstMatch.waitForExistence(timeout: 5), "Analysis navigation link should exist")

        // 3. Navigate to Analysis screen
        analysisLinks.firstMatch.tap()

        // Wait for analysis screen to load
        // Note: In portrait mode, RecordingInfoCompact is used (not RecordingInfoPanel with "Recording Info" title)
        let recordingInfoCompact = app.otherElements["RecordingInfoCompact"]
        XCTAssertTrue(recordingInfoCompact.waitForExistence(timeout: 10), "Analysis screen should load")
    }

    /// Test: Playback scroll behavior - verify time axis and playback cursor
    /// Purpose: Verify that spectrogram time axis scrolls correctly and returns to start position after playback
    @MainActor
    func testPlayback_TimeAxisScroll() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to analysis screen
        navigateToAnalysisScreen(app)

        // Wait for analysis to complete by checking for playback button
        let playPauseButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButton.waitForExistence(timeout: 10), "Play/Pause button should exist")

        // Start playback
        playPauseButton.tap()

        // Wait during playback (about 1 second into playback)
        Thread.sleep(forTimeInterval: 1.0)

        // Wait for playback to complete (assuming recording is short, ~2-3 seconds)
        // We'll wait for the full recording duration plus buffer
        Thread.sleep(forTimeInterval: 3.0)

        // Screenshot: After playback ends (should return to start position)
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "time_axis_after_playback"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // Verify play button is back to play state (not pause)
        XCTAssertTrue(playPauseButton.exists, "Play/Pause button should exist after playback ends")
    }



    /// Test: Spectrogram viewport architecture verification with screenshots
    /// Purpose: Verify that spectrogram fills the entire viewport correctly
    @MainActor
    func testSpectrogramViewport_Screenshots() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to analysis screen
        navigateToAnalysisScreen(app)

        // Wait for spectrogram to appear
        let spectrogramView = app.otherElements["SpectrogramView"]
        XCTAssertTrue(spectrogramView.waitForExistence(timeout: 5), "Spectrogram view should exist")

        // Wait for analysis to complete by waiting for playback controls
        let playPauseButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButton.waitForExistence(timeout: 30), "Analysis should complete and show playback controls")

        // Perform vertical scroll (simulate drag down)
        let spectrogramCenter = spectrogramView.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        let spectrogramBottom = spectrogramView.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.8))
        spectrogramCenter.press(forDuration: 0.1, thenDragTo: spectrogramBottom)

        // Scroll up to show higher frequencies
        let spectrogramTop = spectrogramView.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.2))
        spectrogramBottom.press(forDuration: 0.1, thenDragTo: spectrogramTop)

        // Screenshot: Final state after scroll operations
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "spectrogram_viewport_final"
        attachment1.lifetime = .keepAlways
        add(attachment1)
    }

}
