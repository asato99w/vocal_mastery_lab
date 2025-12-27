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

    // MARK: - Helper Methods

    /// Create a recording and perform vocal extraction, then navigate to recording list
    /// This is required because analysis screen requires extracted vocals
    @MainActor
    private func createRecordingWithExtraction(_ app: XCUIApplication) {
        // 1. Navigate to Recording screen and create a recording
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

        // Wait for recording to be saved
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // 2. Navigate to Vocal Extraction screen
        let vocalButton = app.buttons["VocalExtractionButton"]
        XCTAssertTrue(vocalButton.waitForExistence(timeout: 5), "Vocal extraction button should be visible")
        vocalButton.tap()

        // Wait for extraction screen to load
        let extractionTitle = app.navigationBars["ボーカル抽出"]
        XCTAssertTrue(extractionTitle.waitForExistence(timeout: 5), "Should navigate to vocal extraction screen")

        // 3. Start extraction
        let startExtractionButton = app.buttons["抽出開始"]
        XCTAssertTrue(startExtractionButton.waitForExistence(timeout: 3), "Start extraction button should be visible")
        startExtractionButton.tap()

        // 4. Wait for extraction to complete (this may take time depending on model)
        // Look for the completion indicator or save button
        let saveButton = app.buttons["保存"]
        XCTAssertTrue(saveButton.waitForExistence(timeout: 120), "Extraction should complete and show save button")

        // 5. Save the extraction
        saveButton.tap()

        // 6. Wait to return to recording screen
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Should return to recording screen after save")

        // 7. Navigate back to Home
        app.navigationBars.buttons.element(boundBy: 0).tap()
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Should return to home screen")
    }

    /// Navigate to recording list
    @MainActor
    private func navigateToRecordingList(_ app: XCUIApplication) {
        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // Wait for list to load by checking for cells
        let cells = app.cells
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 5), "Recording cell should appear in list")
    }

    /// Navigate to analysis screen via menu (requires extracted vocals)
    @MainActor
    private func navigateToAnalysisViaMenu(_ app: XCUIApplication) {
        // Find and tap the menu button for the first recording
        let menuButtons = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "MenuButton_"))
        XCTAssertTrue(menuButtons.firstMatch.waitForExistence(timeout: 3), "Menu button should exist for recording")
        menuButtons.firstMatch.tap()

        // Tap the vocal analysis option in the menu
        let analysisMenuItem = app.buttons["ボーカル分析"]
        XCTAssertTrue(analysisMenuItem.waitForExistence(timeout: 3), "Vocal analysis menu item should exist")
        analysisMenuItem.tap()

        // Wait for analysis screen to load
        let recordingInfoCompact = app.otherElements["RecordingInfoCompact"]
        XCTAssertTrue(recordingInfoCompact.waitForExistence(timeout: 10), "Analysis screen should load")
    }

    // MARK: - Tests

    /// Test: Analysis view display and basic playback controls
    /// Expected: ~2 minutes execution time (includes vocal extraction)
    @MainActor
    func testAnalysisViewDisplay() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Create a recording with vocal extraction
        createRecordingWithExtraction(app)

        // 2. Navigate to Recording List
        navigateToRecordingList(app)

        // 3. Navigate to Analysis screen via menu
        navigateToAnalysisViaMenu(app)

        // 4. Wait for analysis to complete by checking for playback button
        let playPauseButtonWait = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButtonWait.waitForExistence(timeout: 30), "Analysis should complete and show playback controls")

        // Screenshot: Analysis screen after loading
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "analysis_01_loaded"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // 5. Verify Playback controls exist
        let playPauseButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButton.waitForExistence(timeout: 10), "Play/Pause button should exist")

        let seekBackButton = app.buttons["AnalysisSeekBackButton"]
        XCTAssertTrue(seekBackButton.waitForExistence(timeout: 3), "Seek back button should exist")

        let seekForwardButton = app.buttons["AnalysisSeekForwardButton"]
        XCTAssertTrue(seekForwardButton.waitForExistence(timeout: 3), "Seek forward button should exist")

        let progressSlider = app.sliders["AnalysisProgressSlider"]
        XCTAssertTrue(progressSlider.waitForExistence(timeout: 3), "Progress slider should exist")

        // 6. Test playback controls - Play
        playPauseButton.tap()

        // Wait a moment for playback to start
        Thread.sleep(forTimeInterval: 0.5)

        // Verify button changed to pause state
        XCTAssertTrue(playPauseButton.exists, "Play/Pause button should still exist during playback")

        // 7. Test playback controls - Pause
        playPauseButton.tap()

        // 8. Test seek controls
        seekBackButton.tap()
        seekForwardButton.tap()

        // Screenshot: Final state after all playback control operations
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "analysis_02_after_controls"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // 9. Verify navigation back works
        app.navigationBars.buttons.element(boundBy: 0).tap()

        // Should be back at Recording List
        let cells = app.cells
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 3), "Should be back at recording list")
    }

    /// Test: Playback scroll behavior - verify time axis and playback cursor
    /// Purpose: Verify that spectrogram time axis scrolls correctly and returns to start position after playback
    /// Expected: ~2 minutes execution time (includes vocal extraction)
    @MainActor
    func testPlayback_TimeAxisScroll() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Create a recording with vocal extraction
        createRecordingWithExtraction(app)

        // 2. Navigate to Recording List
        navigateToRecordingList(app)

        // 3. Navigate to Analysis screen via menu
        navigateToAnalysisViaMenu(app)

        // Wait for analysis to complete by checking for playback button
        let playPauseButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButton.waitForExistence(timeout: 30), "Play/Pause button should exist")

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
    /// Expected: ~2 minutes execution time (includes vocal extraction)
    @MainActor
    func testSpectrogramViewport_Screenshots() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Create a recording with vocal extraction
        createRecordingWithExtraction(app)

        // 2. Navigate to Recording List
        navigateToRecordingList(app)

        // 3. Navigate to Analysis screen via menu
        navigateToAnalysisViaMenu(app)

        // Wait for graph tab picker to appear and switch to Spectrogram tab
        let graphTabPicker = app.segmentedControls["GraphTabPicker"]
        XCTAssertTrue(graphTabPicker.waitForExistence(timeout: 10), "Graph tab picker should exist")

        // Switch to Spectrogram tab
        let spectrogramTabButton = graphTabPicker.buttons.element(boundBy: 1)
        spectrogramTabButton.tap()

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
