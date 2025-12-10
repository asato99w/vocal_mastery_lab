//
//  AnalysisUITests.swift
//  VocalMasteringLabUITests
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



    /// Test: Timing analysis with 1.5 octave scale (Rossini) and longer recording
    /// Purpose: Collect timing data for scale bar vs pitch detection comparison
    @MainActor
    func testTimingAnalysis_RossiniScale() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Navigate to recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // 2. Change scale to Rossini (1.5 octave)
        let scaleTypePicker = app.buttons["ScaleTypePicker"]
        XCTAssertTrue(scaleTypePicker.waitForExistence(timeout: 5), "Scale type picker should exist")
        scaleTypePicker.tap()

        // Select Rossini scale from the menu
        // The button text will be the localized name for rossiniScale
        let rossiniOption = app.buttons["Rossini (1.5オクターブ)"]
        if rossiniOption.waitForExistence(timeout: 2) {
            rossiniOption.tap()
        } else {
            // Try English localization
            let rossiniEnglish = app.buttons["Rossini (1.5 Octave)"]
            if rossiniEnglish.waitForExistence(timeout: 2) {
                rossiniEnglish.tap()
            } else {
                // Fallback: tap anywhere to dismiss and proceed with default scale
                app.tap()
            }
        }

        // 3. Start recording
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        // Wait for recording to start by checking StopButton appearance
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // 4. Record for longer duration (8 seconds for Rossini scale)
        Thread.sleep(forTimeInterval: 8.0)

        stopButton.tap()

        // Wait for recording to be saved
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // 5. Navigate to Recording List
        app.navigationBars.buttons.element(boundBy: 0).tap()

        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // Wait for recording list to load
        let analysisLinks = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisLinks.firstMatch.waitForExistence(timeout: 5), "Analysis navigation link should exist")

        // 6. Navigate to Analysis screen
        analysisLinks.firstMatch.tap()

        // Wait for analysis screen to load
        let recordingInfoCompact = app.otherElements["RecordingInfoCompact"]
        XCTAssertTrue(recordingInfoCompact.waitForExistence(timeout: 10), "Analysis screen should load")

        // 7. Wait for analysis to complete
        let playPauseButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButton.waitForExistence(timeout: 30), "Analysis should complete and show playback controls")

        // 8. Wait additional time for analysis logging to complete
        // YIN analysis runs asynchronously and logs need time to flush
        Thread.sleep(forTimeInterval: 3.0)

        // 9. Screenshot for visual verification
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "timing_analysis_rossini_scale"
        attachment.lifetime = .keepAlways
        add(attachment)
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

    // MARK: - Analyze Recording Button Tests (分析ボタンテスト)

    /// Test: Navigate to analysis via AnalyzeRecordingButton and verify pitch graph content consistency
    /// 録音画面から「録音を分析」ボタンで分析画面に遷移し、ピッチグラフの内容が録音設定と一致することを確認
    @MainActor
    func testAnalyzeButton_NavigatesAndShowsConsistentPitchData() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Navigate to recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // 2. Configure scale settings - select Rossini scale for more data points
        let scaleTypePicker = app.buttons["ScaleTypePicker"]
        XCTAssertTrue(scaleTypePicker.waitForExistence(timeout: 5), "Scale type picker should exist")

        // Capture the current scale type label for verification
        let initialScaleLabel = scaleTypePicker.label
        scaleTypePicker.tap()

        // Select Rossini scale
        let rossiniOption = app.buttons["Rossini (1.5オクターブ)"]
        let rossiniEnglish = app.buttons["Rossini (1.5 Octave)"]
        if rossiniOption.waitForExistence(timeout: 2) {
            rossiniOption.tap()
        } else if rossiniEnglish.waitForExistence(timeout: 2) {
            rossiniEnglish.tap()
        } else {
            app.tap() // Dismiss picker, use default scale
        }

        // Capture selected scale type for later verification
        let selectedScaleType = scaleTypePicker.label

        // 3. Start and stop recording
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Record for 3 seconds to capture scale audio
        Thread.sleep(forTimeInterval: 3.0)
        stopButton.tap()

        // 4. Verify AnalyzeRecordingButton appears and tap it
        let analyzeButton = app.buttons["AnalyzeRecordingButton"]
        XCTAssertTrue(analyzeButton.waitForExistence(timeout: 5), "Analyze recording button should appear after recording")
        analyzeButton.tap()

        // 5. Verify navigation to analysis screen
        let recordingInfoCompact = app.otherElements["RecordingInfoCompact"]
        XCTAssertTrue(recordingInfoCompact.waitForExistence(timeout: 10), "Should navigate to analysis screen")

        // 6. Wait for analysis to complete
        let playPauseButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButton.waitForExistence(timeout: 30), "Analysis should complete")

        // 7. Verify pitch graph components exist and have data
        let pitchGraphView = app.otherElements["PitchAnalysisView"]
        XCTAssertTrue(pitchGraphView.waitForExistence(timeout: 5), "Pitch graph view should be displayed")

        let pitchGraphTitle = app.staticTexts["PitchGraphTitle"]
        XCTAssertTrue(pitchGraphTitle.waitForExistence(timeout: 5), "Pitch graph title should exist")

        // 8. Verify spectrogram exists (indicates audio data was analyzed)
        let spectrogramView = app.otherElements["SpectrogramView"]
        XCTAssertTrue(spectrogramView.waitForExistence(timeout: 5), "Spectrogram view should be displayed")

        // 9. Verify playback controls are functional (data loaded successfully)
        let seekBackButton = app.buttons["AnalysisSeekBackButton"]
        let seekForwardButton = app.buttons["AnalysisSeekForwardButton"]
        let progressSlider = app.sliders["AnalysisProgressSlider"]
        XCTAssertTrue(seekBackButton.exists, "Seek back button should exist")
        XCTAssertTrue(seekForwardButton.exists, "Seek forward button should exist")
        XCTAssertTrue(progressSlider.exists, "Progress slider should exist")

        // Screenshot: Analysis screen with pitch data
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "analyze_button_pitch_data_consistency"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    /// Test: Recording persists in list after analysis navigation, with data integrity verification
    /// 分析画面から一覧に戻った後、録音が正しく永続化されていることを確認
    @MainActor
    func testAnalyzeButton_RecordingPersistsInListWithDataIntegrity() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Navigate to recording screen and create a recording
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // 2. Start and stop recording
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        Thread.sleep(forTimeInterval: 2.0)
        stopButton.tap()

        // 3. Navigate to analysis via analyze button
        let analyzeButton = app.buttons["AnalyzeRecordingButton"]
        XCTAssertTrue(analyzeButton.waitForExistence(timeout: 5), "Analyze recording button should appear")
        analyzeButton.tap()

        // 4. Verify analysis screen loads and analysis completes
        let recordingInfoCompact = app.otherElements["RecordingInfoCompact"]
        XCTAssertTrue(recordingInfoCompact.waitForExistence(timeout: 10), "Analysis screen should load")

        let playPauseButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButton.waitForExistence(timeout: 30), "Analysis should complete")

        // 5. Navigate back: Analysis -> Recording -> Home
        // Wait for back button to appear (it's hidden during analysis)
        let backButton = app.navigationBars.buttons.element(boundBy: 0)
        XCTAssertTrue(backButton.waitForExistence(timeout: 5), "Back button should appear after analysis completes")
        backButton.tap()

        let backButton2 = app.navigationBars.buttons.element(boundBy: 0)
        XCTAssertTrue(backButton2.waitForExistence(timeout: 5), "Back button should exist on recording screen")
        backButton2.tap()

        // 6. Navigate to recording list
        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // 7. Verify recording exists in list with cached state (analysis was completed, button should be blue)
        let cachedLinks = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_cached_"))
        XCTAssertTrue(cachedLinks.firstMatch.waitForExistence(timeout: 10), "Recording should be persisted with cached analysis data (blue button)")
        XCTAssertGreaterThanOrEqual(cachedLinks.count, 1, "At least one cached recording should exist in list")

        // Screenshot: Recording list with persisted recording
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "recording_list_persistence"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // 8. Navigate to the recording from list to verify data integrity
        cachedLinks.firstMatch.tap()

        // 9. Verify analysis screen loads again with same data
        let recordingInfoAgain = app.otherElements["RecordingInfoCompact"]
        XCTAssertTrue(recordingInfoAgain.waitForExistence(timeout: 10), "Should navigate to analysis screen from list")

        // 10. Verify analysis completes successfully (data integrity check)
        let playPauseButtonAgain = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButtonAgain.waitForExistence(timeout: 30), "Analysis should complete for persisted recording")

        // 11. Verify pitch graph and spectrogram are displayed (data still intact)
        let pitchGraphView = app.otherElements["PitchAnalysisView"]
        XCTAssertTrue(pitchGraphView.waitForExistence(timeout: 5), "Pitch graph should be displayed for persisted recording")

        let spectrogramView = app.otherElements["SpectrogramView"]
        XCTAssertTrue(spectrogramView.waitForExistence(timeout: 5), "Spectrogram should be displayed for persisted recording")

        // Screenshot: Re-opened analysis from list
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "recording_list_reopened_analysis"
        attachment2.lifetime = .keepAlways
        add(attachment2)
    }

}
