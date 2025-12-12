//
//  RecordingListUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for recording list (navigation, deletion)
//

import XCTest

final class RecordingListUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    // MARK: - Helper Methods

    /// Create a recording and navigate to the recording list
    /// Used by tests that need a pre-existing recording
    @MainActor
    private func createRecordingAndNavigateToList(_ app: XCUIApplication) {
        // 1. Create a recording
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

        // Wait for recording to finish and be saved
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play last recording button should appear after recording")

        // 2. Navigate back to Home
        app.navigationBars.buttons.element(boundBy: 0).tap()

        // 3. Navigate to Recording List screen
        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // 4. Wait for list to load by checking for analysis buttons (visible in rows)
        let analysisButtons = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisButtons.firstMatch.waitForExistence(timeout: 5), "Analysis button should appear in list")
    }

    /// Navigate directly to the recording list (assumes recording already exists)
    @MainActor
    private func navigateToRecordingList(_ app: XCUIApplication) {
        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // Wait for list to load by checking for analysis buttons (visible in rows)
        let analysisButtons = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisButtons.firstMatch.waitForExistence(timeout: 5), "Analysis button should appear in list")
    }

    // MARK: - Tests

    /// Test 2: Recording list navigation - create recording and navigate to list
    /// Expected: ~15 seconds execution time
    @MainActor
    func testRecordingListNavigation() throws {
        let app = launchAppWithResetRecordingCount()

        // Create recording and navigate to list
        createRecordingAndNavigateToList(app)

        // 4. Verify recording appears in the list
        // Wait for list to load by checking for delete buttons

        // Screenshot: Recording list
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "list_nav_01_recording_list"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // Verify at least one recording exists in the list
        // Use prefix match for dynamic identifier that includes recording UUID
        let analysisLinks = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisLinks.firstMatch.waitForExistence(timeout: 5), "At least one recording should exist in the list")

        // 5. Navigate to Analysis screen by tapping the analysis button
        analysisLinks.firstMatch.tap()

        // Wait for analysis screen to load by checking for analysis UI elements
        let analysisPlayButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(analysisPlayButton.waitForExistence(timeout: 5), "Analysis play button should appear")

        // Screenshot: Analysis screen
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "list_nav_02_analysis_screen"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // 6. Navigate back to list using back button
        app.navigationBars.buttons.element(boundBy: 0).tap()

        // Wait for list to reload by checking analysis buttons visibility
        let analysisLinksAfterBack = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisLinksAfterBack.firstMatch.waitForExistence(timeout: 3), "Analysis button should reappear after navigation back")

        // Screenshot: Back to list
        let screenshot3 = app.screenshot()
        let attachment3 = XCTAttachment(screenshot: screenshot3)
        attachment3.name = "list_nav_03_back_to_list"
        attachment3.lifetime = .keepAlways
        add(attachment3)

        // Verify we're back at the list
        XCTAssertTrue(analysisLinksAfterBack.firstMatch.waitForExistence(timeout: 3), "Should be back at recording list with analysis button visible")
    }

    /// Test: Recording list shows scale name for scale recordings
    /// Expected: ~12 seconds execution time
    @MainActor
    func testRecordingListShowsScaleName() throws {
        let app = launchAppWithResetRecordingCount()

        // Create recording and navigate to list
        createRecordingAndNavigateToList(app)

        // Screenshot: Recording list with scale name
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "scale_name_display"
        attachment.lifetime = .keepAlways
        add(attachment)

        // Verify scale name is displayed (e.g., "C4 5トーン")
        // The scale name should contain the note name pattern and scale pattern name
        // Note: Scale names are now unified across selection UI and recording display
        let scaleNameTexts = app.staticTexts.matching(NSPredicate(format: "label CONTAINS[c] %@", L10n.scaleFiveTone))
        XCTAssertTrue(scaleNameTexts.firstMatch.waitForExistence(timeout: 3), "Scale name containing '\(L10n.scaleFiveTone)' should be displayed in the recording list")
    }

    /// Test: Playback position slider appears during playback
    /// Expected: ~15 seconds execution time
    @MainActor
    func testPlaybackPositionSliderAppearsWhenPlaying() throws {
        let app = launchAppWithResetRecordingCount()

        // Create recording and navigate to list
        createRecordingAndNavigateToList(app)

        // Screenshot: List before playback
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "slider_01_before_playback"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // Tap on recording cell to start playback
        let cells = app.cells
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 3), "Recording cell should exist in the list")
        cells.firstMatch.tap()

        // Wait for playback to start - slider should be visible in PlaybackControlPanel
        let sliders = app.sliders
        XCTAssertTrue(sliders.firstMatch.waitForExistence(timeout: 5), "Position slider should exist in PlaybackControlPanel")

        // Screenshot: During playback (slider should be visible)
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "slider_02_during_playback"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // Verify slider is visible during playback
        XCTAssertTrue(sliders.firstMatch.exists, "Position slider should be visible during playback")

        // Verify time display exists (format: "M:SS")
        // Time labels should show current position and total duration
        let timeLabels = app.staticTexts.matching(NSPredicate(format: "label MATCHES %@", "[0-9]:[0-9]{2}"))
        XCTAssertGreaterThan(timeLabels.count, 0, "Time labels should be displayed during playback")

        // Wait for playback to finish naturally (1 second recording)
        Thread.sleep(forTimeInterval: 1.0)

        // Screenshot: After playback (slider should disappear)
        let screenshot3 = app.screenshot()
        let attachment3 = XCTAttachment(screenshot: screenshot3)
        attachment3.name = "slider_03_after_playback"
        attachment3.lifetime = .keepAlways
        add(attachment3)
    }

    /// Test 3: Delete recording functionality
    /// Expected: ~15 seconds execution time
    @MainActor
    func testDeleteRecording() throws {
        let app = launchAppWithResetRecordingCount()

        // Create recording and navigate to list
        createRecordingAndNavigateToList(app)

        // Screenshot: Recording list before deletion
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "delete_01_before_delete"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // Verify recording exists and count recordings before deletion
        // Use cells for swipe actions on List items
        let cells = app.cells
        let initialCount = cells.count

        // 5. Swipe left to reveal delete button
        cells.firstMatch.swipeLeft()

        // Wait for delete button to appear after swipe
        let deleteButton = app.buttons[L10n.delete]
        XCTAssertTrue(deleteButton.waitForExistence(timeout: 3), "Delete button should appear after swipe")
        deleteButton.tap()

        // Wait for confirmation dialog to appear by checking for confirm button
        let deleteConfirmButton = app.buttons["DeleteConfirmButton"]
        XCTAssertTrue(deleteConfirmButton.waitForExistence(timeout: 3), "Delete confirm button should exist in confirmation dialog")

        // Screenshot: Confirmation dialog
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "delete_02_confirmation_dialog"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // 6. Confirm deletion by tapping the delete confirm button
        deleteConfirmButton.tap()

        // Wait for deletion to complete by checking count change
        // Re-query the cells after deletion to ensure fresh count
        let cellsAfterDeletion = app.cells

        // Wait for the count to decrease (delete animation to complete)
        let expectation = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "count < %d", initialCount),
            object: cellsAfterDeletion
        )
        let result = XCTWaiter.wait(for: [expectation], timeout: 5.0)
        XCTAssertEqual(result, .completed, "Recording count should decrease after deletion")

        // Screenshot: After deletion
        let screenshot3 = app.screenshot()
        let attachment3 = XCTAttachment(screenshot: screenshot3)
        attachment3.name = "delete_03_after_delete"
        attachment3.lifetime = .keepAlways
        add(attachment3)

        // 7. Verify recording is deleted by checking final count
        let finalCount = cellsAfterDeletion.count
        XCTAssertEqual(finalCount, initialCount - 1, "Recording count should decrease by 1 after deletion (was \(initialCount), now \(finalCount))")
    }

    /// Test: Rename recording functionality
    /// Expected: ~15 seconds execution time
    @MainActor
    func testRenameRecording() throws {
        let app = launchAppWithResetRecordingCount()

        // Create recording and navigate to list
        createRecordingAndNavigateToList(app)

        // Screenshot: Recording list before rename
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "rename_01_before_rename"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // Get cells for swipe action
        let cells = app.cells
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 5), "Cell should exist")

        // Swipe right to reveal rename button
        cells.firstMatch.swipeRight()

        // Wait for rename button to appear after swipe
        let renameButton = app.buttons[L10n.rename]
        XCTAssertTrue(renameButton.waitForExistence(timeout: 3), "Rename button should appear after swipe")
        renameButton.tap()

        // Wait for rename alert to appear
        let textField = app.textFields.firstMatch
        XCTAssertTrue(textField.waitForExistence(timeout: 3), "Text field should appear in rename alert")

        // Verify text field shows current name (scale display name)
        // Note: Scale names are now unified across selection UI and recording display
        let currentValue = textField.value as? String ?? ""
        XCTAssertFalse(currentValue.isEmpty, "Text field should show current recording name, not be empty")
        XCTAssertTrue(currentValue.contains(L10n.scaleFiveTone), "Text field should contain scale name '\(L10n.scaleFiveTone)', but got '\(currentValue)'")

        // Screenshot: Rename dialog
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "rename_02_rename_dialog"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // Clear existing text and enter new name
        let newName = "Test Recording"
        textField.tap()
        // Delete existing text character by character
        let existingText = currentValue
        for _ in existingText {
            textField.typeText(XCUIKeyboardKey.delete.rawValue)
        }
        textField.typeText(newName)

        // Tap save button
        let saveButton = app.buttons[L10n.save]
        XCTAssertTrue(saveButton.waitForExistence(timeout: 3), "Save button should exist")
        saveButton.tap()

        // Wait for alert to dismiss
        Thread.sleep(forTimeInterval: 0.5)

        // Screenshot: After rename
        let screenshot3 = app.screenshot()
        let attachment3 = XCTAttachment(screenshot: screenshot3)
        attachment3.name = "rename_03_after_rename"
        attachment3.lifetime = .keepAlways
        add(attachment3)

        // Verify the new name is displayed
        let renamedText = app.staticTexts[newName]
        XCTAssertTrue(renamedText.waitForExistence(timeout: 3), "Renamed recording should display the new name '\(newName)'")
    }
}
