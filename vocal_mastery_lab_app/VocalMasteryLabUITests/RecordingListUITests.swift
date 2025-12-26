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

        // 4. Wait for list to load by checking for cells (recording rows)
        let cells = app.cells
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 5), "Recording cell should appear in list")
    }

    /// Navigate directly to the recording list (assumes recording already exists)
    @MainActor
    private func navigateToRecordingList(_ app: XCUIApplication) {
        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // Wait for list to load by checking for cells (recording rows)
        let cells = app.cells
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 5), "Recording cell should appear in list")
    }

    // MARK: - Tests

    /// Test 2: Recording list navigation - create recording and navigate to list
    /// Expected: ~15 seconds execution time
    @MainActor
    func testRecordingListNavigation() throws {
        let app = launchAppWithResetRecordingCount()

        // Create recording and navigate to list
        createRecordingAndNavigateToList(app)

        // Screenshot: Recording list
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "list_nav_01_recording_list"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // Verify at least one recording exists in the list
        let cells = app.cells
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 5), "At least one recording should exist in the list")

        // Verify menu button exists for the recording
        let menuButtons = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "MenuButton_"))
        XCTAssertTrue(menuButtons.firstMatch.waitForExistence(timeout: 3), "Menu button should exist for recording")

        // Screenshot: Recording list with menu button visible
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "list_nav_02_with_menu"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // Navigate back to home
        app.navigationBars.buttons.element(boundBy: 0).tap()

        // Verify we're back at home
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 3), "Should be back at home screen")

        // Screenshot: Back to home
        let screenshot3 = app.screenshot()
        let attachment3 = XCTAttachment(screenshot: screenshot3)
        attachment3.name = "list_nav_03_back_to_home"
        attachment3.lifetime = .keepAlways
        add(attachment3)
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

}
