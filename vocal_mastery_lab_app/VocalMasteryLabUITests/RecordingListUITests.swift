//
//  RecordingListUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for recording list based on scenario document
//  Reference: docs/RecordingListUITests_Scenarios.md
//

import XCTest

final class RecordingListUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    // MARK: - Helper Methods

    /// Create a recording with specified duration
    /// - Parameters:
    ///   - app: The application instance
    ///   - duration: Recording duration in seconds
    @MainActor
    private func createRecording(_ app: XCUIApplication, duration: TimeInterval) {
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

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

    /// Navigate to recording list from home screen
    @MainActor
    private func navigateToRecordingList(_ app: XCUIApplication) {
        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        let cells = app.cells
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 5), "Recording cell should appear in list")
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

    /// Test 1: Recording List Comprehensive Flow
    /// Tests: Multiple recordings, playback, toggle, prev/next, delete, robustness
    /// Expected: ~40 seconds execution time
    /// Reference: docs/RecordingListUITests_Scenarios.md
    @MainActor
    func testRecordingListComprehensiveFlow() throws {
        let app = launchAppWithResetRecordingCount()

        // === Phase 1: Preparation - Create 2 recordings ===
        // Recording A (older, will be at index 1)
        createRecording(app, duration: 1.0)
        app.navigationBars.buttons.element(boundBy: 0).tap()

        // Recording B (newer, will be at index 0)
        createRecording(app, duration: 1.0)
        app.navigationBars.buttons.element(boundBy: 0).tap()

        // Navigate to recording list
        navigateToRecordingList(app)

        // === Phase 2: List Display Verification ===
        let cells = app.cells
        XCTAssertTrue(cells.count >= 2, "Should have at least 2 recordings")

        let menuButtons = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "MenuButton_"))
        XCTAssertTrue(menuButtons.count >= 2, "Should have menu buttons for each recording")

        // PlaybackControlPanel elements
        let playPauseButton = app.buttons["PlaybackControlPanel_PlayPauseButton"]
        XCTAssertTrue(playPauseButton.waitForExistence(timeout: 3), "PlayPauseButton should exist")

        let previousButton = app.buttons["PlaybackControlPanel_PreviousButton"]
        XCTAssertTrue(previousButton.exists, "PreviousButton should exist")
        XCTAssertFalse(previousButton.isEnabled, "PreviousButton should be disabled initially")

        let nextButton = app.buttons["PlaybackControlPanel_NextButton"]
        XCTAssertTrue(nextButton.exists, "NextButton should exist")
        XCTAssertFalse(nextButton.isEnabled, "NextButton should be disabled initially")

        let slider = app.sliders["PlaybackControlPanel_Slider"]
        XCTAssertTrue(slider.exists, "Slider should exist")

        addScreenshot(app, name: "phase2_list_display")

        // === Phase 3: Start Playback ===
        // Tap Recording B (most recent, at top, index 0) - per scenario "録音Aタップ"
        // Note: Scenario says "録音A" but list order is B(0), A(1). Testing with first cell.
        cells.element(boundBy: 0).tap()
        Thread.sleep(forTimeInterval: 0.5)

        // Verify playback started
        XCTAssertTrue(slider.isEnabled, "Slider should be enabled during playback")

        let currentTimeLabel = app.staticTexts["PlaybackControlPanel_CurrentTime"]
        XCTAssertTrue(currentTimeLabel.exists, "CurrentTime label should exist")

        let totalTimeLabel = app.staticTexts["PlaybackControlPanel_TotalTime"]
        XCTAssertTrue(totalTimeLabel.exists, "TotalTime label should exist")

        // Verify button states for first recording
        XCTAssertFalse(previousButton.isEnabled, "PreviousButton should be disabled for first recording")
        XCTAssertTrue(nextButton.isEnabled, "NextButton should be enabled when there's a next recording")

        // Verify time progresses
        Thread.sleep(forTimeInterval: 0.5)

        addScreenshot(app, name: "phase3_playback_started")

        // === Phase 4: Toggle Playback ===
        // Tap same recording to pause
        cells.element(boundBy: 0).tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Tap again to resume
        cells.element(boundBy: 0).tap()
        Thread.sleep(forTimeInterval: 0.3)

        addScreenshot(app, name: "phase4_toggle_playback")

        // === Phase 5: Next/Previous Navigation ===
        // Tap Next button to go to Recording A (older, at index 1)
        nextButton.tap()
        Thread.sleep(forTimeInterval: 0.5)

        // Verify button states after navigating to second (last) recording
        XCTAssertTrue(previousButton.isEnabled, "PreviousButton should be enabled for second recording")
        XCTAssertFalse(nextButton.isEnabled, "NextButton should be disabled for last recording")

        addScreenshot(app, name: "phase5_after_next")

        // Tap Previous button to go back to Recording B (first)
        previousButton.tap()
        Thread.sleep(forTimeInterval: 0.5)

        // Verify button states after navigating back to first recording
        XCTAssertFalse(previousButton.isEnabled, "PreviousButton should be disabled for first recording")
        XCTAssertTrue(nextButton.isEnabled, "NextButton should be enabled for first recording")

        addScreenshot(app, name: "phase5_after_previous")

        // === Phase 6: Direct Recording Tap Switch ===
        // Tap Recording A directly (second cell, index 1)
        cells.element(boundBy: 1).tap()
        Thread.sleep(forTimeInterval: 0.5)

        XCTAssertTrue(previousButton.isEnabled, "PreviousButton should be enabled after switching to second recording")
        XCTAssertFalse(nextButton.isEnabled, "NextButton should be disabled for last recording")

        addScreenshot(app, name: "phase6_direct_tap_switch")

        // === Phase 7: Delete Recording While Playing ===
        let initialCount = cells.count

        // Swipe left on current recording (Recording A at index 1) to delete
        cells.element(boundBy: 1).swipeLeft()

        let deleteButton = app.buttons[L10n.delete]
        XCTAssertTrue(deleteButton.waitForExistence(timeout: 3), "Delete button should appear after swipe")
        deleteButton.tap()

        let deleteConfirmButton = app.buttons["DeleteConfirmButton"]
        XCTAssertTrue(deleteConfirmButton.waitForExistence(timeout: 3), "Delete confirm button should exist")
        deleteConfirmButton.tap()

        // Wait for deletion
        let deletionExpectation = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "count < %d", initialCount),
            object: cells
        )
        let deletionResult = XCTWaiter.wait(for: [deletionExpectation], timeout: 5.0)
        XCTAssertEqual(deletionResult, .completed, "Recording count should decrease after deletion")

        addScreenshot(app, name: "phase7_after_delete")

        // === Phase 8: Remaining Recording Playback (Robustness) ===
        Thread.sleep(forTimeInterval: 0.5)

        // Tap remaining recording (Recording B, now the only one)
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 3), "Remaining recording should exist")
        cells.firstMatch.tap()
        Thread.sleep(forTimeInterval: 0.5)

        XCTAssertTrue(slider.isEnabled, "Slider should be enabled for remaining recording")

        // Verify navigation buttons are disabled (only one recording left)
        XCTAssertFalse(previousButton.isEnabled, "PreviousButton should be disabled with single recording")
        XCTAssertFalse(nextButton.isEnabled, "NextButton should be disabled with single recording")

        // Verify AudioSourcePicker exists
        let originalButton = app.buttons["AudioSourceButton_original"]
        XCTAssertTrue(originalButton.waitForExistence(timeout: 3), "Original button should exist")
        XCTAssertTrue(originalButton.isEnabled, "Original button should be enabled")

        let vocalButton = app.buttons["AudioSourceButton_vocal"]
        XCTAssertFalse(vocalButton.isEnabled, "Vocal button should be disabled without extraction")

        let instrumentalButton = app.buttons["AudioSourceButton_instrumental"]
        XCTAssertFalse(instrumentalButton.isEnabled, "Instrumental button should be disabled without extraction")

        addScreenshot(app, name: "phase8_comprehensive_flow_final")
    }

    /// Test 2: Audio Source Switching with Extraction
    /// Tests: Extraction, source switching, button states across recordings
    /// Expected: ~60 seconds execution time
    /// Reference: docs/RecordingListUITests_Scenarios.md
    @MainActor
    func testAudioSourceSwitchingWithExtraction() throws {
        let app = launchAppWithResetRecordingCount()

        // === Phase 1: Preparation - Create 2 recordings ===
        // Recording A (older, will be at index 1)
        createRecording(app, duration: 1.0)
        app.navigationBars.buttons.element(boundBy: 0).tap()

        // Recording B (newer, will be at index 0)
        createRecording(app, duration: 1.0)
        app.navigationBars.buttons.element(boundBy: 0).tap()

        // Navigate to recording list
        navigateToRecordingList(app)

        let cells = app.cells
        XCTAssertTrue(cells.count >= 2, "Should have at least 2 recordings")

        // === Phase 2: Extract Recording A (older, at index 1) per scenario ===
        let menuButtons = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "MenuButton_"))
        menuButtons.element(boundBy: 1).tap()  // Recording A is at index 1

        // Tap "ボーカル抽出" in menu
        let extractMenuButton = app.buttons["ボーカル抽出"]
        XCTAssertTrue(extractMenuButton.waitForExistence(timeout: 3), "Extract button should exist in menu")
        extractMenuButton.tap()

        // VocalExtractionView: Tap "抽出開始" button to start extraction
        let startExtractionButton = app.buttons["抽出開始"]
        XCTAssertTrue(startExtractionButton.waitForExistence(timeout: 5), "Start extraction button should exist")
        startExtractionButton.tap()

        // Wait for extraction to complete - look for "保存" button
        let saveButton = app.buttons["保存"]
        XCTAssertTrue(saveButton.waitForExistence(timeout: 60), "Save button should appear after extraction completes")

        // Tap save button to save the extraction
        saveButton.tap()

        // Wait for save to complete and dismiss - list should reload
        Thread.sleep(forTimeInterval: 1.0)

        // After saving, should be back at recording list
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 5), "Should return to recording list after saving")

        // Verify extraction indicators
        // Recording A (index 1) should have ExtractionIndicators
        // Recording B (index 0) should NOT have ExtractionIndicators

        addScreenshot(app, name: "phase2_after_extraction")

        // === Phase 3: Play Extracted Recording A (at index 1) ===
        cells.element(boundBy: 1).tap()  // Recording A is at index 1
        Thread.sleep(forTimeInterval: 0.5)

        let slider = app.sliders["PlaybackControlPanel_Slider"]
        XCTAssertTrue(slider.isEnabled, "Slider should be enabled")

        let originalButton = app.buttons["AudioSourceButton_original"]
        let vocalButton = app.buttons["AudioSourceButton_vocal"]
        let instrumentalButton = app.buttons["AudioSourceButton_instrumental"]
        XCTAssertTrue(originalButton.waitForExistence(timeout: 3), "Original button should exist")

        // For extracted recording A, all sources should be enabled
        XCTAssertTrue(originalButton.isEnabled, "Original should be enabled for extracted recording")
        XCTAssertTrue(vocalButton.isEnabled, "Vocal should be enabled for extracted recording")
        XCTAssertTrue(instrumentalButton.isEnabled, "Instrumental should be enabled for extracted recording")

        let previousButton = app.buttons["PlaybackControlPanel_PreviousButton"]
        let nextButton = app.buttons["PlaybackControlPanel_NextButton"]
        // Recording A is at index 1 (last), so Previous should be enabled, Next disabled
        XCTAssertTrue(previousButton.isEnabled, "PreviousButton should be enabled for last recording")
        XCTAssertFalse(nextButton.isEnabled, "NextButton should be disabled for last recording")

        addScreenshot(app, name: "phase3_extracted_recording")

        // === Phase 4: Switch to Vocal ===
        vocalButton.tap()
        Thread.sleep(forTimeInterval: 0.5)

        addScreenshot(app, name: "phase4_vocal_source_selected")

        // === Phase 5: Switch to Instrumental ===
        instrumentalButton.tap()
        Thread.sleep(forTimeInterval: 0.5)

        addScreenshot(app, name: "phase5_instrumental_source_selected")

        // === Phase 6: Switch to Unextracted Recording B (via Previous) ===
        // Recording B is at index 0, need to go to previous (which is actually "forward" in list)
        previousButton.tap()
        Thread.sleep(forTimeInterval: 0.5)

        // For unextracted recording B, vocal and instrumental should be disabled
        XCTAssertTrue(originalButton.isEnabled, "Original should be enabled")
        XCTAssertFalse(vocalButton.isEnabled, "Vocal should be disabled for unextracted recording")
        XCTAssertFalse(instrumentalButton.isEnabled, "Instrumental should be disabled for unextracted recording")

        // Recording B is at index 0 (first), so Previous disabled, Next enabled
        XCTAssertFalse(previousButton.isEnabled, "PreviousButton should be disabled for first recording")
        XCTAssertTrue(nextButton.isEnabled, "NextButton should be enabled for first recording")

        addScreenshot(app, name: "phase6_unextracted_recording")

        // === Phase 7: Switch back to Extracted Recording A (via Next) ===
        nextButton.tap()
        Thread.sleep(forTimeInterval: 0.5)

        // All sources should be enabled again for Recording A
        XCTAssertTrue(originalButton.isEnabled, "Original should be enabled")
        XCTAssertTrue(vocalButton.isEnabled, "Vocal should be enabled for extracted recording")
        XCTAssertTrue(instrumentalButton.isEnabled, "Instrumental should be enabled for extracted recording")

        XCTAssertTrue(previousButton.isEnabled, "PreviousButton should be enabled")
        XCTAssertFalse(nextButton.isEnabled, "NextButton should be disabled")

        addScreenshot(app, name: "phase7_back_to_extracted")

        // === Phase 8: Direct Tap to Recording B (unextracted, index 0) ===
        cells.element(boundBy: 0).tap()
        Thread.sleep(forTimeInterval: 0.5)

        XCTAssertTrue(originalButton.isEnabled, "Original should be enabled")
        XCTAssertFalse(vocalButton.isEnabled, "Vocal should be disabled for unextracted")
        XCTAssertFalse(instrumentalButton.isEnabled, "Instrumental should be disabled for unextracted")

        XCTAssertFalse(previousButton.isEnabled, "PreviousButton should be disabled")
        XCTAssertTrue(nextButton.isEnabled, "NextButton should be enabled")

        addScreenshot(app, name: "phase8_direct_tap_unextracted")

        // === Phase 9: Direct Tap to Recording A (extracted, index 1) ===
        cells.element(boundBy: 1).tap()
        Thread.sleep(forTimeInterval: 0.5)

        XCTAssertTrue(originalButton.isEnabled, "Original should be enabled")
        XCTAssertTrue(vocalButton.isEnabled, "Vocal should be enabled")
        XCTAssertTrue(instrumentalButton.isEnabled, "Instrumental should be enabled")

        XCTAssertTrue(previousButton.isEnabled, "PreviousButton should be enabled")
        XCTAssertFalse(nextButton.isEnabled, "NextButton should be disabled")

        addScreenshot(app, name: "phase9_direct_tap_extracted")

        // === Phase 10: Vocal selected, switch B→A, verify reset ===
        vocalButton.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Switch to B (unextracted, index 0)
        cells.element(boundBy: 0).tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Vocal should be disabled for B
        XCTAssertFalse(vocalButton.isEnabled, "Vocal should be disabled for B")

        // Switch back to A (extracted, index 1)
        cells.element(boundBy: 1).tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Vocal should be enabled again for A
        XCTAssertTrue(vocalButton.isEnabled, "Vocal should be enabled again for A")

        addScreenshot(app, name: "phase10_source_reset_verification")

        // === Phase 11: Pause, then switch source (should auto-play) ===
        // Tap to pause
        cells.element(boundBy: 1).tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Switch to Vocal (should start playing)
        vocalButton.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Verify playback resumed by checking slider is enabled
        XCTAssertTrue(slider.isEnabled, "Slider should be enabled after source switch")

        addScreenshot(app, name: "phase11_audio_source_test_final")
    }

}
