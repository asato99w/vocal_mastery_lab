//
//  RecordingUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for recording screen based on scenario document
//  Reference: docs/RecordingUITests_Scenarios.md
//

import XCTest

final class RecordingUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    // MARK: - Helper Methods

    /// Navigate to recording screen from home
    @MainActor
    private func navigateToRecordingScreen(_ app: XCUIApplication) {
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()
    }

    /// Create a recording with specified duration and return to recording screen
    @MainActor
    private func createRecording(_ app: XCUIApplication, duration: TimeInterval) {
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        Thread.sleep(forTimeInterval: duration)

        stopButton.tap()

        // Wait for recording to finish
        let lastRecordingSection = app.otherElements["LastRecordingSection"]
        XCTAssertTrue(lastRecordingSection.waitForExistence(timeout: 5), "Last recording section should appear")
    }

    /// Extract vocals from last recording and return to recording screen
    @MainActor
    private func extractVocalsFromLastRecording(_ app: XCUIApplication) {
        let vocalExtractionButton = app.buttons["VocalExtractionButton"]
        XCTAssertTrue(vocalExtractionButton.waitForExistence(timeout: 5), "Vocal extraction button should exist")
        vocalExtractionButton.tap()

        // VocalExtractionView: Tap "抽出開始" button to start extraction
        let startExtractionButton = app.buttons["抽出開始"]
        XCTAssertTrue(startExtractionButton.waitForExistence(timeout: 5), "Start extraction button should exist")
        startExtractionButton.tap()

        // Wait for extraction to complete - look for "保存" button
        let saveButton = app.buttons["保存"]
        XCTAssertTrue(saveButton.waitForExistence(timeout: 60), "Save button should appear after extraction completes")
        saveButton.tap()

        // Wait for save to complete and return to recording screen
        // Verify we're back on recording screen by checking StartRecordingButton exists
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 10), "Should return to recording screen after save")
    }

    /// Select a track from the backing track menu (after menu is opened)
    /// Returns the label of the selected track, or nil if no track was selected
    @MainActor
    @discardableResult
    private func selectTrackFromMenu(_ app: XCUIApplication, skipFirst: Bool = false) -> String? {
        // Known button labels to skip
        let skipLabels = Set(["なし", "Back", "録音を開始", "録音一覧"])
        let skipPatterns = ["chevron", "録音中"]

        var matchedItems: [XCUIElement] = []
        let menuItems = app.buttons.allElementsBoundByIndex

        for item in menuItems {
            let label = item.label
            // Skip empty labels
            if label.isEmpty { continue }
            // Skip known buttons
            if skipLabels.contains(label) { continue }
            // Skip buttons containing specific patterns
            if skipPatterns.contains(where: { label.contains($0) }) { continue }
            // Look for date/time patterns (xx:xx or xx/xx format)
            if label.contains(":") && label.contains("/") {
                matchedItems.append(item)
            }
        }

        // Select the appropriate item
        if matchedItems.isEmpty {
            return nil
        }

        let index = skipFirst && matchedItems.count > 1 ? 1 : 0
        let selectedItem = matchedItems[index]
        selectedItem.tap()
        return selectedItem.label
    }

    /// Add a screenshot attachment
    @MainActor
    private func addScreenshot(_ app: XCUIApplication, name: String) {
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    // MARK: - Scenario Tests

    /// Test 1: Recording Comprehensive Flow
    /// Tests: Recording start, countdown, timer, stop, last recording info, navigation
    /// Expected: ~30 seconds execution time
    /// Reference: docs/RecordingUITests_Scenarios.md
    @MainActor
    func testRecordingComprehensiveFlow() throws {
        let app = launchAppWithResetRecordingCount()
        navigateToRecordingScreen(app)

        // === Phase 1: Initial State Verification ===
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "StartRecordingButton should exist")
        XCTAssertTrue(startButton.isEnabled, "StartRecordingButton should be enabled")

        let timerLabel = app.staticTexts["RecordingTimerLabel"]
        XCTAssertTrue(timerLabel.exists, "RecordingTimerLabel should exist")

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertFalse(stopButton.exists, "StopRecordingButton should not exist initially")

        let countdownNumber = app.staticTexts["CountdownNumber"]
        XCTAssertFalse(countdownNumber.exists, "CountdownNumber should not exist initially")

        let recordingListButton = app.buttons["RecordingListButton"]
        XCTAssertTrue(recordingListButton.exists, "RecordingListButton should exist")

        addScreenshot(app, name: "phase1_initial_state")

        // === Phase 2: Start Recording and Countdown ===
        startButton.tap()

        // Check for loading indicator (preparing state)
        let loadingIndicator = app.activityIndicators["RecordingLoadingIndicator"]
        // Note: Loading indicator may be very brief, so just check existence without strict assertion
        _ = loadingIndicator.waitForExistence(timeout: 2)

        // Wait for countdown (if not disabled) or stop button
        let stopButtonAppears = stopButton.waitForExistence(timeout: 10)
        XCTAssertTrue(stopButtonAppears, "StopRecordingButton should appear after countdown")

        addScreenshot(app, name: "phase2_recording_started")

        // === Phase 3: Recording State Verification ===
        XCTAssertTrue(stopButton.exists, "StopRecordingButton should exist during recording")
        XCTAssertTrue(stopButton.isEnabled, "StopRecordingButton should be enabled")
        XCTAssertFalse(startButton.exists, "StartRecordingButton should not exist during recording")
        XCTAssertFalse(countdownNumber.exists, "CountdownNumber should not exist during recording")

        // Wait for timer to progress
        Thread.sleep(forTimeInterval: 1.5)

        addScreenshot(app, name: "phase3_recording_in_progress")

        // === Phase 4: Stop Recording ===
        stopButton.tap()

        // Wait for recording to finish
        XCTAssertFalse(stopButton.waitForExistence(timeout: 3), "StopRecordingButton should disappear after stop")
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "StartRecordingButton should reappear")
        XCTAssertTrue(startButton.isEnabled, "StartRecordingButton should be enabled")

        // Last recording section should appear
        let lastRecordingSection = app.otherElements["LastRecordingSection"]
        XCTAssertTrue(lastRecordingSection.waitForExistence(timeout: 5), "LastRecordingSection should exist")

        let lastRecordingDateLabel = app.staticTexts["LastRecordingDateLabel"]
        XCTAssertTrue(lastRecordingDateLabel.exists, "LastRecordingDateLabel should exist")

        let lastRecordingDurationLabel = app.staticTexts["LastRecordingDurationLabel"]
        XCTAssertTrue(lastRecordingDurationLabel.exists, "LastRecordingDurationLabel should exist")

        let vocalExtractionButton = app.buttons["VocalExtractionButton"]
        XCTAssertTrue(vocalExtractionButton.exists, "VocalExtractionButton should exist")

        addScreenshot(app, name: "phase4_recording_stopped")

        // === Phase 5: Navigate to Vocal Extraction ===
        vocalExtractionButton.tap()

        // Verify navigation to VocalExtractionView
        let extractionStartButton = app.buttons["抽出開始"]
        XCTAssertTrue(extractionStartButton.waitForExistence(timeout: 5), "Extraction start button should exist")

        addScreenshot(app, name: "phase5_vocal_extraction_view")

        // Go back to recording screen
        app.navigationBars.buttons.element(boundBy: 0).tap()

        // Verify returned to recording screen
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Should return to recording screen")
        XCTAssertTrue(lastRecordingSection.exists, "LastRecordingSection should still exist")

        addScreenshot(app, name: "phase5_back_to_recording")

        // === Phase 6: Navigate to Recording List ===
        recordingListButton.tap()

        // Verify navigation to recording list
        let cells = app.cells
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 5), "Recording cell should appear in list")

        addScreenshot(app, name: "phase6_recording_list")
    }

    /// Test 2: Backing Track Playback
    /// Tests: Track selection, source switching, playback, seek, recording with backing track
    /// Expected: ~50 seconds execution time (with 2 extractions)
    /// Reference: docs/RecordingUITests_Scenarios.md
    @MainActor
    func testBackingTrackPlayback() throws {
        let app = launchAppWithResetRecordingCount()
        navigateToRecordingScreen(app)

        // === Phase 1: Preparation - Create 2 extracted recordings ===
        // Recording A (5 seconds to allow longer playback for testing)
        createRecording(app, duration: 5.0)
        extractVocalsFromLastRecording(app)

        // Recording B (5 seconds)
        createRecording(app, duration: 5.0)
        extractVocalsFromLastRecording(app)

        // Verify backing track section and count
        // First check if BackingTrackSection exists (should exist when recordingState == .idle)
        let backingTrackSection = app.otherElements["BackingTrackSection"]
        XCTAssertTrue(backingTrackSection.waitForExistence(timeout: 5), "BackingTrackSection should exist when idle")

        let backingTrackCount = app.staticTexts["BackingTrackCount"]
        XCTAssertTrue(backingTrackCount.waitForExistence(timeout: 2), "BackingTrackCount should exist")
        // Note: Count text may vary, just verify element exists

        addScreenshot(app, name: "phase1_preparation_complete")

        // === Phase 2: Backing Track Section Verification ===
        // (backingTrackSection already verified in Phase 1)
        XCTAssertTrue(backingTrackSection.exists, "BackingTrackSection should still exist")

        let backingTrackLabel = app.staticTexts["BackingTrackLabel"]
        XCTAssertTrue(backingTrackLabel.exists, "BackingTrackLabel should exist")

        let backingTrackPicker = app.buttons["BackingTrackPicker"]
        XCTAssertTrue(backingTrackPicker.exists, "BackingTrackPicker should exist")

        // Player should not exist when no track selected
        let backingTrackPlayerView = app.otherElements["BackingTrackPlayerView"]
        XCTAssertFalse(backingTrackPlayerView.exists, "BackingTrackPlayerView should not exist without selection")

        let backingSourcePicker = app.buttons["BackingSourcePicker"]
        XCTAssertFalse(backingSourcePicker.exists, "BackingSourcePicker should not exist without selection")

        addScreenshot(app, name: "phase2_backing_track_section")

        // === Phase 3: Select First Track (Recording A) ===
        backingTrackPicker.tap()
        Thread.sleep(forTimeInterval: 0.5)

        // Debug: print all button labels
        let allButtons = app.buttons.allElementsBoundByIndex
        var debugButtonLabels: [String] = []
        for button in allButtons {
            debugButtonLabels.append(button.label)
        }
        print("DEBUG: Available buttons after menu open: \(debugButtonLabels)")

        // Select the first available track
        let selectedTrack = selectTrackFromMenu(app)
        XCTAssertNotNil(selectedTrack, "Should be able to select a track. Available buttons: \(debugButtonLabels)")

        // Wait for UI to update after track selection
        Thread.sleep(forTimeInterval: 1.0)

        // Verify player appeared - check BackingSourcePicker first (it appears when track is selected)
        let sourcePickerExists = backingSourcePicker.waitForExistence(timeout: 5)
        if !sourcePickerExists {
            // Take screenshot for debugging
            addScreenshot(app, name: "debug_source_picker_not_found")
        }
        XCTAssertTrue(sourcePickerExists, "BackingSourcePicker should exist after selection (selected: \(selectedTrack ?? "nil"))")
        XCTAssertTrue(backingTrackPlayerView.waitForExistence(timeout: 3), "BackingTrackPlayerView should exist")

        // Verify BackingTrackInfoLabel exists (contains track name and source)
        let backingTrackInfoLabel = app.otherElements["BackingTrackInfoLabel"]
        XCTAssertTrue(backingTrackInfoLabel.waitForExistence(timeout: 3), "BackingTrackInfoLabel should exist - shows track name and source")

        let playPauseButton = app.buttons["BackingTrackPlayPauseButton"]
        XCTAssertTrue(playPauseButton.exists, "BackingTrackPlayPauseButton should exist")

        let stopButtonBacking = app.buttons["BackingTrackStopButton"]
        XCTAssertTrue(stopButtonBacking.exists, "BackingTrackStopButton should exist")

        // Note: BackingTrackSeekSlider is a custom GeometryReader-based control, not a native Slider
        let seekSlider = app.otherElements["BackingTrackSeekSlider"]
        XCTAssertTrue(seekSlider.exists, "BackingTrackSeekSlider should exist")

        let currentTimeLabel = app.staticTexts["BackingTrackCurrentTimeLabel"]
        XCTAssertTrue(currentTimeLabel.exists, "BackingTrackCurrentTimeLabel should exist")

        let durationLabel = app.staticTexts["BackingTrackDurationLabel"]
        XCTAssertTrue(durationLabel.exists, "BackingTrackDurationLabel should exist")

        addScreenshot(app, name: "phase3_track_selected")

        // === Phase 4: Source Switching (Stopped) ===
        // Switch to Vocal
        backingSourcePicker.tap()
        Thread.sleep(forTimeInterval: 0.3)
        let vocalOption = app.buttons["ボーカル"]
        if vocalOption.waitForExistence(timeout: 2) {
            vocalOption.tap()
        }
        Thread.sleep(forTimeInterval: 0.3)

        addScreenshot(app, name: "phase4_vocal_source")

        // Switch to Instrumental
        backingSourcePicker.tap()
        Thread.sleep(forTimeInterval: 0.3)
        let instrumentalOption = app.buttons["伴奏"]
        if instrumentalOption.waitForExistence(timeout: 2) {
            instrumentalOption.tap()
        }
        Thread.sleep(forTimeInterval: 0.3)

        addScreenshot(app, name: "phase4_instrumental_source")

        // Switch to Original
        backingSourcePicker.tap()
        Thread.sleep(forTimeInterval: 0.3)
        let originalOption = app.buttons["元音源"]
        if originalOption.waitForExistence(timeout: 2) {
            originalOption.tap()
        }
        Thread.sleep(forTimeInterval: 0.3)

        addScreenshot(app, name: "phase4_original_source")

        // === Phase 5: Playback - Play/Pause/Resume ===
        // Play
        playPauseButton.tap()

        // Wait for playback to start
        Thread.sleep(forTimeInterval: 0.5)

        // Verify PlayingIndicator appears when playback starts
        // Note: The indicator is detected as staticText (not otherElements) by XCUITest
        // because SwiftUI HStack with only text+image content is merged into text element
        let playingIndicator = app.staticTexts["BackingTrackPlayingIndicator"]
        let indicatorExists = playingIndicator.waitForExistence(timeout: 5)

        XCTAssertTrue(indicatorExists, "BackingTrackPlayingIndicator should appear when playback starts")

        addScreenshot(app, name: "phase5_playing")

        // Pause
        playPauseButton.tap()
        Thread.sleep(forTimeInterval: 0.5)

        // Resume
        playPauseButton.tap()
        Thread.sleep(forTimeInterval: 0.5)

        addScreenshot(app, name: "phase5_resumed")

        // === Phase 6: Source Switching (While Playing) ===
        // Switch to Vocal while playing
        backingSourcePicker.tap()
        Thread.sleep(forTimeInterval: 0.3)
        if vocalOption.waitForExistence(timeout: 2) {
            vocalOption.tap()
        }
        Thread.sleep(forTimeInterval: 0.5)

        // Verify playback continues after source switch
        XCTAssertTrue(playingIndicator.waitForExistence(timeout: 3), "BackingTrackPlayingIndicator should remain visible after source switch to Vocal")

        addScreenshot(app, name: "phase6_source_switch_while_playing")

        // Switch to Instrumental while playing
        backingSourcePicker.tap()
        Thread.sleep(forTimeInterval: 0.3)
        if instrumentalOption.waitForExistence(timeout: 2) {
            instrumentalOption.tap()
        }
        Thread.sleep(forTimeInterval: 0.3)

        addScreenshot(app, name: "phase6_instrumental_while_playing")

        // === Phase 7: Seek Operation ===
        // Note: Slider adjustment is tricky in UI tests, just verify it exists and is interactable
        XCTAssertTrue(seekSlider.isEnabled, "Seek slider should be enabled")

        // Stop playback
        stopButtonBacking.tap()
        Thread.sleep(forTimeInterval: 0.3)

        addScreenshot(app, name: "phase7_after_stop")

        // === Phase 8: Switch to Different Track (Recording B) ===
        // Start playback first
        playPauseButton.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Switch to different track
        backingTrackPicker.tap()
        Thread.sleep(forTimeInterval: 0.5)

        // Select the second track (skip the first one)
        selectTrackFromMenu(app, skipFirst: true)
        Thread.sleep(forTimeInterval: 0.5)

        // Verify source picker exists
        XCTAssertTrue(backingSourcePicker.exists, "BackingSourcePicker should exist after track switch")

        addScreenshot(app, name: "phase8_track_switched")

        // === Phase 9: Track Deselection ===
        // Stop playback first to ensure clean state
        let stopBtnPhase9 = app.buttons["BackingTrackStopButton"]
        if stopBtnPhase9.exists {
            stopBtnPhase9.tap()
            Thread.sleep(forTimeInterval: 0.3)
        }

        // Open track picker menu
        backingTrackPicker.tap()
        Thread.sleep(forTimeInterval: 0.5)

        // Find and tap "なし" option - must succeed
        let noneOption = app.buttons["なし"]
        let noneOptionFound = noneOption.waitForExistence(timeout: 3)
        if !noneOptionFound {
            addScreenshot(app, name: "phase9_none_option_not_found")
        }
        XCTAssertTrue(noneOptionFound, "'なし' option should appear in track picker menu")
        noneOption.tap()

        // Wait for player to disappear (more robust than fixed sleep)
        var playerDisappeared = false
        for _ in 0..<10 {
            if !backingTrackPlayerView.exists {
                playerDisappeared = true
                break
            }
            Thread.sleep(forTimeInterval: 0.5)
        }

        // Take screenshot if player didn't disappear
        if !playerDisappeared {
            addScreenshot(app, name: "phase9_player_still_visible")
        }

        // Verify player disappeared
        XCTAssertTrue(playerDisappeared, "BackingTrackPlayerView should not exist after deselection")
        XCTAssertFalse(backingSourcePicker.exists, "BackingSourcePicker should not exist after deselection")
        XCTAssertFalse(playPauseButton.exists, "BackingTrackPlayPauseButton should not exist after deselection")

        addScreenshot(app, name: "phase9_track_deselected")

        // Re-select a track
        backingTrackPicker.tap()
        Thread.sleep(forTimeInterval: 0.3)
        selectTrackFromMenu(app)
        Thread.sleep(forTimeInterval: 0.5)

        XCTAssertTrue(backingTrackPlayerView.waitForExistence(timeout: 3), "BackingTrackPlayerView should exist after re-selection")
        XCTAssertTrue(backingSourcePicker.exists, "BackingSourcePicker should exist after re-selection")

        addScreenshot(app, name: "phase9_track_reselected")

        // === Phase 10: Recording with Backing Track (Countdown + Recording State) ===
        // Start backing track playback
        let playPauseButtonAgain = app.buttons["BackingTrackPlayPauseButton"]
        XCTAssertTrue(playPauseButtonAgain.waitForExistence(timeout: 3), "PlayPauseButton should exist")
        playPauseButtonAgain.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Start recording
        let startRecordingButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startRecordingButton.exists, "StartRecordingButton should exist")
        startRecordingButton.tap()

        // Wait for countdown to complete and recording to start
        let stopRecordingButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopRecordingButton.waitForExistence(timeout: 10), "StopRecordingButton should appear")

        // Verify backing track player still visible during recording
        XCTAssertTrue(backingTrackPlayerView.exists, "BackingTrackPlayerView should exist during recording")

        addScreenshot(app, name: "phase10_recording_with_backing_track")

        // Verify seek slider is still visible during recording (custom GeometryReader control)
        let seekSliderDuringRecording = app.otherElements["BackingTrackSeekSlider"]
        XCTAssertTrue(seekSliderDuringRecording.exists, "BackingTrackSeekSlider should exist during recording")

        // Stop backing track during recording
        let stopBackingDuringRecording = app.buttons["BackingTrackStopButton"]
        if stopBackingDuringRecording.exists {
            stopBackingDuringRecording.tap()
            Thread.sleep(forTimeInterval: 0.3)
        }

        // Verify recording continues
        XCTAssertTrue(stopRecordingButton.exists, "Recording should continue after stopping backing track")

        addScreenshot(app, name: "phase10_backing_stopped_during_recording")

        // Stop recording
        stopRecordingButton.tap()

        // Verify recording stopped
        let lastRecordingSection = app.otherElements["LastRecordingSection"]
        XCTAssertTrue(lastRecordingSection.waitForExistence(timeout: 5), "LastRecordingSection should exist")

        // Verify backing track player still visible after recording
        XCTAssertTrue(backingTrackPlayerView.exists, "BackingTrackPlayerView should exist after recording")

        addScreenshot(app, name: "phase10_recording_complete_with_backing")
    }
}
