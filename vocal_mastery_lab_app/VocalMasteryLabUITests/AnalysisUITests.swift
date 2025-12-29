//
//  AnalysisUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for analysis screen functionality
//  Based on AnalysisUITests_Scenarios.md specifications
//

import XCTest

final class AnalysisUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    // MARK: - Helper Methods

    /// Create a recording (2 seconds) and perform vocal extraction, then navigate to recording list
    /// This is required because analysis screen requires extracted vocals
    @MainActor
    private func createRecordingWithExtraction(_ app: XCUIApplication) {
        // 1. Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // 2. Start recording
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        // Wait for recording to start
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Record for 2 seconds (as specified in scenario)
        Thread.sleep(forTimeInterval: 2.0)

        // 3. Stop recording
        stopButton.tap()

        // Wait for recording to be saved
        let vocalButton = app.buttons["VocalExtractionButton"]
        XCTAssertTrue(vocalButton.waitForExistence(timeout: 10), "Vocal extraction button should appear after save")

        // 4. Navigate to Vocal Extraction screen
        vocalButton.tap()

        // Wait for extraction screen to load
        let extractionTitle = app.navigationBars["ボーカル抽出"]
        XCTAssertTrue(extractionTitle.waitForExistence(timeout: 5), "Should navigate to vocal extraction screen")

        // 5. Start extraction
        let startExtractionButton = app.buttons["抽出開始"]
        XCTAssertTrue(startExtractionButton.waitForExistence(timeout: 3), "Start extraction button should be visible")
        startExtractionButton.tap()

        // 6. Wait for extraction to complete
        let saveButton = app.buttons["保存"]
        XCTAssertTrue(saveButton.waitForExistence(timeout: 120), "Extraction should complete and show save button")

        // 7. Save the extraction
        saveButton.tap()

        // 8. Wait to return to recording screen
        XCTAssertTrue(vocalButton.waitForExistence(timeout: 5), "Should return to recording screen after save")

        // 9. Navigate to Recording List via toolbar button
        let listButton = app.buttons["RecordingListButton"]
        XCTAssertTrue(listButton.waitForExistence(timeout: 5), "Recording list button should exist")
        listButton.tap()

        // Wait for list to load
        let cells = app.cells
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 5), "Recording cell should appear in list")
    }

    /// Navigate to analysis screen via menu from recording list
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
    }

    /// Wait for analysis to complete by checking for playback controls
    @MainActor
    private func waitForAnalysisCompletion(_ app: XCUIApplication, timeout: TimeInterval = 60) {
        let playPauseButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButton.waitForExistence(timeout: timeout), "Analysis should complete and show playback controls")
    }

    // MARK: - Test 1: Comprehensive Flow

    /// Test: Analysis start, progress display, playback, and seek operations
    /// Covers: Phases 1-8 from AnalysisUITests_Scenarios.md
    /// Expected: ~50 seconds execution time
    @MainActor
    func testAnalysisComprehensiveFlow() throws {
        let app = launchAppWithResetRecordingCount()

        // ========================================
        // Phase 1: Preparation (Create recording with extraction)
        // ========================================
        createRecordingWithExtraction(app)

        // ========================================
        // Phase 2: Navigate to Analysis screen
        // ========================================
        navigateToAnalysisViaMenu(app)

        // Verify navigation succeeded - check for analysis UI elements
        // Note: RecordingInfoPanel (Landscape) or RecordingInfoCompact (Portrait)
        let infoPanel = app.otherElements["RecordingInfoPanel"]
        let infoCompact = app.otherElements["RecordingInfoCompact"]
        let hasInfoPanel = infoPanel.waitForExistence(timeout: 10) || infoCompact.exists
        XCTAssertTrue(hasInfoPanel, "Analysis screen should show recording info (Panel or Compact)")

        // ========================================
        // Phase 3: Verify Analysis Progress
        // ========================================
        // Check for progress indicator (may not be visible if analysis is fast)
        let progressText = app.staticTexts["分析中..."]
        let progressExists = progressText.waitForExistence(timeout: 2)
        if progressExists {
            print("✅ Analysis progress text is visible")
        } else {
            print("ℹ️ Analysis progress text not found - analysis may have completed quickly")
        }

        // ========================================
        // Phase 4: Analysis Complete - Verify Initial State
        // ========================================
        waitForAnalysisCompletion(app, timeout: 60)

        // Verify all required UI elements exist
        let graphTabPicker = app.segmentedControls["GraphTabPicker"]
        XCTAssertTrue(graphTabPicker.waitForExistence(timeout: 5), "GraphTabPicker should exist")

        let playPauseButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButton.exists, "AnalysisPlayPauseButton should exist")

        // Note: AnalysisProgressSlider may be a native Slider or custom control
        let progressSlider = app.sliders["AnalysisProgressSlider"]
        let progressSliderOther = app.otherElements["AnalysisProgressSlider"]
        let hasProgressSlider = progressSlider.exists || progressSliderOther.exists
        XCTAssertTrue(hasProgressSlider, "AnalysisProgressSlider should exist")

        let seekBackButton = app.buttons["AnalysisSeekBackButton"]
        XCTAssertTrue(seekBackButton.exists, "AnalysisSeekBackButton should exist")

        let seekForwardButton = app.buttons["AnalysisSeekForwardButton"]
        XCTAssertTrue(seekForwardButton.exists, "AnalysisSeekForwardButton should exist")

        // Verify default tab is Pitch Analysis
        let pitchAnalysisView = app.otherElements["PitchAnalysisView"]
        XCTAssertTrue(pitchAnalysisView.waitForExistence(timeout: 3), "PitchAnalysisView should be visible by default")

        // Screenshot: Analysis complete initial state
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "analysis_01_complete_initial"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // ========================================
        // Phase 5: Playback Operations
        // ========================================

        // Play
        playPauseButton.tap()
        Thread.sleep(forTimeInterval: 0.5)

        // Verify playback started (button should still exist, icon changed internally)
        XCTAssertTrue(playPauseButton.exists, "Play/Pause button should exist during playback")

        // Pause
        playPauseButton.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Resume
        playPauseButton.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Pause again for seek tests
        playPauseButton.tap()

        // ========================================
        // Phase 6: Seek Button Operations
        // ========================================

        // Seek back (5 seconds)
        seekBackButton.tap()
        Thread.sleep(forTimeInterval: 0.2)

        // Seek forward (5 seconds)
        seekForwardButton.tap()
        Thread.sleep(forTimeInterval: 0.2)

        // ========================================
        // Phase 7: Slider Seek Operation
        // ========================================
        // Use the available slider element
        if progressSlider.exists {
            progressSlider.adjust(toNormalizedSliderPosition: 0.5)
        } else if progressSliderOther.exists {
            // For custom slider, use coordinate-based interaction
            let sliderCenter = progressSliderOther.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
            sliderCenter.tap()
        }

        Thread.sleep(forTimeInterval: 0.3)

        // Screenshot: After playback and seek operations
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "analysis_02_after_playback"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // ========================================
        // Phase 8: Navigate Back
        // ========================================
        app.navigationBars.buttons.element(boundBy: 0).tap()

        // Verify returned to recording list
        let cells = app.cells
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 5), "Should return to recording list")
    }

    // MARK: - Test 2: Graph Visualization and Statistics

    /// Test: Graph tab switching, expand/collapse, and statistics sheet
    /// Covers: Phases 1-9 from AnalysisUITests_Scenarios.md
    /// Expected: ~60 seconds execution time
    @MainActor
    func testGraphVisualizationAndStatistics() throws {
        let app = launchAppWithResetRecordingCount()

        // ========================================
        // Phase 1: Preparation (Create recording, extraction, analysis)
        // ========================================
        createRecordingWithExtraction(app)
        navigateToAnalysisViaMenu(app)
        waitForAnalysisCompletion(app, timeout: 60)

        // ========================================
        // Phase 2: Verify Initial Tab (Pitch Analysis)
        // ========================================
        let graphTabPicker = app.segmentedControls["GraphTabPicker"]
        XCTAssertTrue(graphTabPicker.waitForExistence(timeout: 5), "GraphTabPicker should exist")

        let pitchAnalysisView = app.otherElements["PitchAnalysisView"]
        XCTAssertTrue(pitchAnalysisView.waitForExistence(timeout: 3), "PitchAnalysisView should be visible by default")

        let autoFollowToggle = app.switches["AutoFollowToggle"]
        let autoFollowToggleOther = app.otherElements["AutoFollowToggle"]
        let hasAutoFollow = autoFollowToggle.exists || autoFollowToggleOther.exists
        if !hasAutoFollow {
            print("⚠️ AutoFollowToggle not found - may be hidden in current layout")
        }

        let pitchExpandButton = app.buttons["PitchGraphExpandButton"]
        XCTAssertTrue(pitchExpandButton.waitForExistence(timeout: 3), "PitchGraphExpandButton should exist")

        // ========================================
        // Phase 3: Pitch Graph Expand/Collapse
        // ========================================

        // Expand
        pitchExpandButton.tap()

        let expandedPitchView = app.otherElements["ExpandedPitchGraphView"]
        XCTAssertTrue(expandedPitchView.waitForExistence(timeout: 5), "ExpandedPitchGraphView should appear")

        let pitchCollapseButton = app.buttons["PitchGraphCollapseButton"]
        XCTAssertTrue(pitchCollapseButton.waitForExistence(timeout: 3), "PitchGraphCollapseButton should exist in expanded view")

        let expandedPlayButton = app.buttons["ExpandedAnalysisPlayPauseButton"]
        XCTAssertTrue(expandedPlayButton.waitForExistence(timeout: 3), "ExpandedAnalysisPlayPauseButton should exist")

        // Screenshot: Expanded pitch graph
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "analysis_03_pitch_expanded"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // Collapse
        pitchCollapseButton.tap()

        // Verify returned to normal view
        XCTAssertTrue(pitchAnalysisView.waitForExistence(timeout: 3), "PitchAnalysisView should reappear after collapse")

        // ========================================
        // Phase 4: Switch to Spectrogram Tab
        // ========================================
        let spectrogramTabButton = graphTabPicker.buttons.element(boundBy: 1)
        spectrogramTabButton.tap()

        let spectrogramView = app.otherElements["SpectrogramView"]
        XCTAssertTrue(spectrogramView.waitForExistence(timeout: 5), "SpectrogramView should appear")

        let spectrogramCanvas = app.otherElements["SpectrogramCanvas"]
        let hasCanvas = spectrogramCanvas.waitForExistence(timeout: 3)
        if !hasCanvas {
            print("ℹ️ SpectrogramCanvas not found separately - may be part of SpectrogramView")
        }

        let spectrogramExpandButton = app.buttons["SpectrogramExpandButton"]
        XCTAssertTrue(spectrogramExpandButton.waitForExistence(timeout: 3), "SpectrogramExpandButton should exist")

        // Verify Pitch view is no longer visible
        XCTAssertFalse(pitchAnalysisView.exists, "PitchAnalysisView should not be visible on Spectrogram tab")

        // Screenshot: Spectrogram view
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "analysis_04_spectrogram"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // ========================================
        // Phase 5: Spectrogram Expand/Collapse
        // ========================================

        // Expand
        spectrogramExpandButton.tap()

        let expandedSpectrogramView = app.otherElements["ExpandedSpectrogramView"]
        XCTAssertTrue(expandedSpectrogramView.waitForExistence(timeout: 5), "ExpandedSpectrogramView should appear")

        let spectrogramCollapseButton = app.buttons["SpectrogramCollapseButton"]
        XCTAssertTrue(spectrogramCollapseButton.waitForExistence(timeout: 3), "SpectrogramCollapseButton should exist")

        let expandedPlayButton2 = app.buttons["ExpandedAnalysisPlayPauseButton"]
        XCTAssertTrue(expandedPlayButton2.waitForExistence(timeout: 3), "ExpandedAnalysisPlayPauseButton should exist in expanded spectrogram")

        // Test playback in expanded view
        expandedPlayButton2.tap()
        Thread.sleep(forTimeInterval: 0.5)
        expandedPlayButton2.tap() // Pause

        // Screenshot: Expanded spectrogram
        let screenshot3 = app.screenshot()
        let attachment3 = XCTAttachment(screenshot: screenshot3)
        attachment3.name = "analysis_05_spectrogram_expanded"
        attachment3.lifetime = .keepAlways
        add(attachment3)

        // Collapse
        spectrogramCollapseButton.tap()

        // Verify returned to normal view
        XCTAssertTrue(spectrogramView.waitForExistence(timeout: 3), "SpectrogramView should reappear after collapse")

        // ========================================
        // Phase 6: Switch Back to Pitch Tab
        // ========================================
        let pitchTabButton = graphTabPicker.buttons.element(boundBy: 0)
        pitchTabButton.tap()

        XCTAssertTrue(pitchAnalysisView.waitForExistence(timeout: 3), "PitchAnalysisView should reappear")
        XCTAssertFalse(spectrogramView.exists, "SpectrogramView should not be visible on Pitch tab")

        // ========================================
        // Phase 7: Statistics Sheet
        // ========================================
        // Try both Portrait and Landscape identifiers
        let statisticsButton = app.buttons["StatisticsButton"]
        let statisticsButtonCompact = app.buttons["StatisticsButtonCompact"]

        if statisticsButton.waitForExistence(timeout: 2) {
            statisticsButton.tap()
        } else if statisticsButtonCompact.waitForExistence(timeout: 2) {
            statisticsButtonCompact.tap()
        } else {
            XCTFail("Statistics button not found (neither StatisticsButton nor StatisticsButtonCompact)")
        }

        // Verify statistics sheet appeared
        let statisticsSheet = app.otherElements["StatisticsSheetView"]
        XCTAssertTrue(statisticsSheet.waitForExistence(timeout: 5), "StatisticsSheetView should appear")

        // Verify required sections exist (they should be expanded by default)
        let pitchAnalysisSection = app.otherElements["PitchAnalysisSection"]
        XCTAssertTrue(pitchAnalysisSection.waitForExistence(timeout: 5), "PitchAnalysisSection should exist in statistics sheet")

        let spectrumAnalysisSection = app.otherElements["SpectrumAnalysisSection"]
        XCTAssertTrue(spectrumAnalysisSection.waitForExistence(timeout: 3), "SpectrumAnalysisSection should exist in statistics sheet")

        let closeButton = app.buttons["StatisticsSheetCloseButton"]
        XCTAssertTrue(closeButton.waitForExistence(timeout: 3), "StatisticsSheetCloseButton should exist")

        // Screenshot: Statistics sheet
        let screenshot4 = app.screenshot()
        let attachment4 = XCTAttachment(screenshot: screenshot4)
        attachment4.name = "analysis_06_statistics_sheet"
        attachment4.lifetime = .keepAlways
        add(attachment4)

        // ========================================
        // Phase 8: Section Toggle Verification
        // ========================================

        // Verify PitchAnalysisSectionToggleButton exists and can be toggled
        let pitchSectionToggle = app.buttons["PitchAnalysisSectionToggleButton"]
        XCTAssertTrue(pitchSectionToggle.waitForExistence(timeout: 3), "PitchAnalysisSectionToggleButton should exist")
        pitchSectionToggle.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Verify PositionSectionToggleButton exists (inside PitchAnalysisSection)
        let positionToggle = app.buttons["PositionSectionToggleButton"]
        XCTAssertTrue(positionToggle.waitForExistence(timeout: 3), "PositionSectionToggleButton should exist")
        positionToggle.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Verify PositionSectionContent appears after toggle
        let positionContent = app.otherElements["PositionSectionContent"]
        XCTAssertTrue(positionContent.waitForExistence(timeout: 3), "PositionSectionContent should appear after toggle")

        // Verify PitchSectionToggleButton exists
        let pitchToggle = app.buttons["PitchSectionToggleButton"]
        XCTAssertTrue(pitchToggle.waitForExistence(timeout: 3), "PitchSectionToggleButton should exist")
        pitchToggle.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Verify PitchSectionContent appears after toggle
        let pitchContent = app.otherElements["PitchSectionContent"]
        XCTAssertTrue(pitchContent.waitForExistence(timeout: 3), "PitchSectionContent should appear after toggle")

        // Verify VibratoSectionToggleButton exists
        let vibratoToggle = app.buttons["VibratoSectionToggleButton"]
        XCTAssertTrue(vibratoToggle.waitForExistence(timeout: 3), "VibratoSectionToggleButton should exist")
        vibratoToggle.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Verify VibratoSectionContent or VibratoSectionNoData appears
        // (depends on whether vibrato was detected in the recording)
        let vibratoContent = app.otherElements["VibratoSectionContent"]
        let vibratoNoData = app.otherElements["VibratoSectionNoData"]
        let hasVibratoSection = vibratoContent.waitForExistence(timeout: 2) || vibratoNoData.waitForExistence(timeout: 2)
        XCTAssertTrue(hasVibratoSection, "Either VibratoSectionContent or VibratoSectionNoData should appear after toggle")

        // Screenshot: Statistics sheet with sections expanded
        let screenshot5 = app.screenshot()
        let attachment5 = XCTAttachment(screenshot: screenshot5)
        attachment5.name = "analysis_07_statistics_expanded"
        attachment5.lifetime = .keepAlways
        add(attachment5)

        // ========================================
        // Phase 9: Close Statistics Sheet
        // ========================================
        closeButton.tap()

        // Verify sheet is closed
        XCTAssertFalse(statisticsSheet.waitForExistence(timeout: 2), "StatisticsSheetView should be dismissed")

        // Verify analysis screen is still visible
        XCTAssertTrue(pitchAnalysisView.waitForExistence(timeout: 3), "Should return to analysis screen after closing statistics")

        // Final screenshot
        let screenshot6 = app.screenshot()
        let attachment6 = XCTAttachment(screenshot: screenshot6)
        attachment6.name = "analysis_08_final"
        attachment6.lifetime = .keepAlways
        add(attachment6)
    }

}
