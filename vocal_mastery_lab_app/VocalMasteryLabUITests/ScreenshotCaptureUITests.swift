import XCTest

/// UI test for capturing App Store screenshots
///
/// **IMPORTANT**: These tests are SKIPPED by default to avoid slowing down regular test runs.
/// To run these tests, set the environment variable `RUN_SCREENSHOT_TESTS=1`:
///
/// ```bash
/// # Run screenshot tests only
/// RUN_SCREENSHOT_TESTS=1 xcodebuild test \
///   -project VocalMasteryLab.xcodeproj \
///   -scheme VocalMasteryLab-UIOnly \
///   -destination 'id=<SIMULATOR_UUID>' \
///   -only-testing:VocalMasteryLabUITests/ScreenshotCaptureUITests \
///   -parallel-testing-enabled NO
///
/// # Or via test-runner.sh
/// RUN_SCREENSHOT_TESTS=1 ./scripts/test-runner.sh ui ScreenshotCaptureUITests
/// ```
final class ScreenshotCaptureUITests: XCTestCase {

    override func setUpWithError() throws {
        // Skip screenshot tests by default unless RUN_SCREENSHOT_TESTS=1 is set
        try skipUnlessScreenshotTestsEnabled()
        continueAfterFailure = false
    }

    /// Skip test unless RUN_SCREENSHOT_TESTS environment variable is set
    private func skipUnlessScreenshotTestsEnabled() throws {
        guard ProcessInfo.processInfo.environment["RUN_SCREENSHOT_TESTS"] == "1" else {
            throw XCTSkip("Screenshot tests are skipped by default. Set RUN_SCREENSHOT_TESTS=1 to run.")
        }
    }

    /// Capture all required screenshots for App Store submission (uses device's current language)
    /// Launches with premium tier to show full functionality
    @MainActor
    func testCaptureAllScreenshots() throws {
        let app = launchAppWithResetRecordingCount(premium: true)

        // Screenshot 1: Home screen
        captureHomeScreen(app)

        // Screenshot 2: Recording screen
        captureRecordingScreen(app)

        // Screenshot 3: Recording list (create multiple recordings with varied settings)
        createMultipleRecordingsWithVariedSettings(app)
        captureRecordingList(app)

        // Screenshot 4: Analysis screen
        captureAnalysisScreen(app)

        // Screenshot 5: Statistics sheet
        captureStatisticsSheet(app)

        // Screenshot 6: Settings screen
        captureSettingsScreen(app)
    }

    /// Capture screenshots for Japanese App Store submission
    /// Launches with premium tier to show full functionality
    @MainActor
    func testCaptureAllScreenshots_Japanese() throws {
        let app = launchAppWithResetRecordingCount(language: "ja", locale: "ja_JP", premium: true)

        // Screenshot 1: Home screen
        captureHomeScreen(app)

        // Screenshot 2: Recording screen
        captureRecordingScreen(app)

        // Screenshot 3: Recording list (create recordings without varying settings to avoid scroll issues)
        createSimpleRecordings(app, count: 3)
        captureRecordingList(app)

        // Screenshot 4: Analysis screen
        captureAnalysisScreen(app)

        // Screenshot 5: Statistics sheet
        captureStatisticsSheet(app)

        // Screenshot 6: Settings screen
        captureSettingsScreen(app)
    }

    /// Create simple recordings without changing settings (for Japanese screenshots)
    @MainActor
    private func createSimpleRecordings(_ app: XCUIApplication, count: Int) {
        for _ in 0..<count {
            let homeRecordButton = app.buttons["HomeRecordButton"]
            XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
            homeRecordButton.tap()

            let startButton = app.buttons["StartRecordingButton"]
            XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")

            // Start recording immediately without changing settings
            startButton.tap()

            // Wait for recording to start
            let stopButton = app.buttons["StopRecordingButton"]
            XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

            // Record for 3 seconds (longer recording for better pitch analysis)
            Thread.sleep(forTimeInterval: 3.0)

            stopButton.tap()

            // Wait for recording to be saved
            let lastRecordingSection = app.otherElements["LastRecordingSection"]
            XCTAssertTrue(lastRecordingSection.waitForExistence(timeout: 5), "Last recording section should appear after save")

            // Navigate back to Home
            app.navigationBars.buttons.element(boundBy: 0).tap()
            XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Should return to home")
        }
    }

    // MARK: - Screenshot Helper

    /// Take screenshot using XCUIScreen.main for proper landscape orientation support
    private func takeScreenshot(name: String) {
        let screenshot = XCUIScreen.main.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    // MARK: - Screenshot Capture Methods

    @MainActor
    private func captureHomeScreen(_ app: XCUIApplication) {
        // Verify home screen is displayed
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home screen should be visible")

        // Wait for UI to stabilize
        Thread.sleep(forTimeInterval: 1.0)

        takeScreenshot(name: "01_home")
    }

    @MainActor
    private func captureRecordingScreen(_ app: XCUIApplication) {
        // Navigate to recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.exists, "Home record button should exist")
        homeRecordButton.tap()

        // Wait for recording screen to appear
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Recording screen should be visible")

        // Wait for UI to stabilize
        Thread.sleep(forTimeInterval: 1.0)

        // Screenshot: Recording screen (idle state)
        takeScreenshot(name: "02_recording_idle")

        // Start recording and capture during recording
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Wait for spectrum to show audio data (longer recording for better pitch analysis)
        Thread.sleep(forTimeInterval: 3.0)

        // Screenshot: Recording in progress
        takeScreenshot(name: "02_recording_in_progress")

        // Stop recording
        stopButton.tap()

        // Wait for recording to be saved
        let lastRecordingSection = app.otherElements["LastRecordingSection"]
        XCTAssertTrue(lastRecordingSection.waitForExistence(timeout: 5), "Last recording section should appear after save")

        // Screenshot: Recording complete
        Thread.sleep(forTimeInterval: 0.5)
        takeScreenshot(name: "02_recording_complete")

        // Note: 再生ボタンはRecordingViewから削除済み。再生機能はRecordingListViewで提供。

        // Navigate back to home
        app.navigationBars.buttons.element(boundBy: 0).tap()
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Should return to home")
    }

    /// Settings configuration for creating varied recordings
    private struct RecordingConfig {
        let scaleType: String  // L10n value for picker menu item
        let startPitch: String? // e.g., "C3", "E3", "G3" - nil to skip pitch change
    }

    @MainActor
    private func createMultipleRecordingsWithVariedSettings(_ app: XCUIApplication) {
        // Create 3 recordings with different settings for a varied screenshot
        // Note: nil for startPitch skips pitch change (uses default C3)
        let configs = [
            RecordingConfig(scaleType: L10n.scaleFiveTone, startPitch: nil),  // Uses default C3
            RecordingConfig(scaleType: L10n.scaleOctaveRepeat, startPitch: "E3"),
            RecordingConfig(scaleType: L10n.scaleFiveTone, startPitch: "G3")
        ]

        for config in configs {
            createRecordingWithSettings(app, config: config)
        }
    }

    @MainActor
    private func createRecordingWithSettings(_ app: XCUIApplication, config: RecordingConfig) {
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")

        // Show settings panel if hidden (portrait layout)
        let showSettingsButton = app.buttons[L10n.showSettings]
        if showSettingsButton.waitForExistence(timeout: 2) {
            showSettingsButton.tap()
        }

        // Change scale type if needed
        let scaleTypePicker = app.buttons["ScaleTypePicker"]
        if scaleTypePicker.waitForExistence(timeout: 5) {
            scaleTypePicker.tap()
            let menuItem = app.menuItems[config.scaleType]
            if menuItem.waitForExistence(timeout: 2) {
                menuItem.tap()
            } else {
                // Fallback to buttons
                let button = app.buttons.matching(identifier: config.scaleType)
                if button.firstMatch.waitForExistence(timeout: 2) {
                    button.firstMatch.tap()
                }
            }
        }

        // Change start pitch if specified
        if let startPitch = config.startPitch {
            let startPitchPicker = app.buttons["StartPitchPicker"]
            if startPitchPicker.waitForExistence(timeout: 5) {
                startPitchPicker.tap()

                // Wait for menu/popover to appear
                Thread.sleep(forTimeInterval: 1.0)

                // Try to find and tap the pitch
                var found = false

                // Method 1: Try accessibility identifier (e.g., "Pitch_E3")
                let pitchId = "Pitch_\(startPitch)"
                let pitchButton = app.buttons[pitchId]
                if pitchButton.waitForExistence(timeout: 2) {
                    pitchButton.tap()
                    found = true
                }

                // Method 2: Try menuItems by label
                if !found {
                    let pitchMenuItem = app.menuItems[startPitch]
                    if pitchMenuItem.waitForExistence(timeout: 1) {
                        pitchMenuItem.tap()
                        found = true
                    }
                }

                // Method 3: Try staticTexts within menus
                if !found {
                    let menuTexts = app.staticTexts[startPitch]
                    if menuTexts.waitForExistence(timeout: 1) {
                        menuTexts.tap()
                        found = true
                    }
                }

                // If still not found, dismiss the menu by tapping outside
                if !found {
                    let coordinate = app.coordinate(withNormalizedOffset: CGVector(dx: 0.1, dy: 0.1))
                    coordinate.tap()
                    Thread.sleep(forTimeInterval: 0.5)
                }
            }
        }

        // Hide settings panel if visible (to make start button accessible)
        // Try both English and Japanese labels for the hide settings button
        let hideSettingsLabels = ["Hide Settings", "設定を隠す"]
        for label in hideSettingsLabels {
            let hideButton = app.buttons[label]
            if hideButton.waitForExistence(timeout: 1) && hideButton.isHittable {
                hideButton.tap()
                Thread.sleep(forTimeInterval: 0.5)
                break
            }
        }

        // Scroll to make start button visible if needed
        // Use swipeUp to bring the button into view
        var attempts = 0
        while !startButton.isHittable && attempts < 3 {
            app.swipeUp()
            Thread.sleep(forTimeInterval: 0.3)
            attempts += 1
        }

        // If still not hittable, try coordinate-based tap
        if startButton.isHittable {
            startButton.tap()
        } else {
            // Tap using coordinate as fallback
            let coordinate = startButton.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
            coordinate.tap()
        }

        // Wait for recording to start
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Record for 3 seconds (longer recording for better pitch analysis)
        Thread.sleep(forTimeInterval: 3.0)

        stopButton.tap()

        // Wait for recording to be saved
        let lastRecordingSection = app.otherElements["LastRecordingSection"]
        XCTAssertTrue(lastRecordingSection.waitForExistence(timeout: 5), "Last recording section should appear after save")

        // Navigate back to Home
        app.navigationBars.buttons.element(boundBy: 0).tap()
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Should return to home")
    }

    @MainActor
    private func captureRecordingList(_ app: XCUIApplication) {
        // Navigate to Recording List
        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // Wait for list to load
        let analysisLinks = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisLinks.firstMatch.waitForExistence(timeout: 5), "Recording list should be visible")

        // Wait for UI to stabilize
        Thread.sleep(forTimeInterval: 1.0)

        // Screenshot: Recording list (idle state)
        takeScreenshot(name: "03_recording_list_idle")

        // Tap on first recording row to start playback (row tap triggers selectAndPlay)
        // The row itself is tappable, not just the analysis button
        let firstRow = app.cells.firstMatch
        if firstRow.waitForExistence(timeout: 2) {
            firstRow.tap()
        } else {
            // Fallback: tap analysis link which is in the row
            analysisLinks.firstMatch.tap()
        }

        // Wait for playback to start and progress
        Thread.sleep(forTimeInterval: 1.5)

        // Screenshot: Recording list with playback in progress
        takeScreenshot(name: "03_recording_list_playback")

        // Stop playback using the PlaybackControlPanel
        let playbackPlayButton = app.buttons["PlaybackControlPanel_PlayPauseButton"]
        if playbackPlayButton.waitForExistence(timeout: 2) {
            playbackPlayButton.tap()
        }
    }

    @MainActor
    private func captureAnalysisScreen(_ app: XCUIApplication) {
        // Tap on first recording in the list
        let analysisLinks = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisLinks.firstMatch.exists, "Recording should exist in list")
        analysisLinks.firstMatch.tap()

        // Wait for analysis screen to load
        let playButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Analysis screen should be visible")

        // Wait for UI to stabilize and pitch analysis to complete
        // Analysis takes time - wait for spectrogram and pitch data to fully render
        Thread.sleep(forTimeInterval: 8.0)

        // Scroll down to show pitch analysis graph with data
        app.swipeUp()
        Thread.sleep(forTimeInterval: 0.5)

        // Screenshot: Analysis screen (idle state with pitch data visible)
        takeScreenshot(name: "04_analysis_idle")

        // Scroll down the screen to show the Pitch Analysis graph
        app.swipeUp()
        Thread.sleep(forTimeInterval: 0.3)

        // Start playback using the play button
        playButton.tap()

        // Wait for playback to progress - longer wait for pitch detection to process
        // The recording is short (3-4 seconds), so wait until playback reaches middle
        Thread.sleep(forTimeInterval: 1.5)

        // Screenshot: Analysis playback in progress
        // Note: Pitch Analysis graph may not show data if pitch detection didn't find clear pitch
        takeScreenshot(name: "04_analysis_playback")

        // Stop playback
        if playButton.waitForExistence(timeout: 1) {
            playButton.tap()
        }

        // --- Fullscreen Screenshots ---

        // Scroll back up to find expand buttons
        app.swipeDown()
        Thread.sleep(forTimeInterval: 0.5)

        // Capture fullscreen spectrogram
        let spectrogramExpandButton = app.buttons["SpectrogramExpandButton"]
        if spectrogramExpandButton.waitForExistence(timeout: 3) {
            spectrogramExpandButton.tap()
            Thread.sleep(forTimeInterval: 1.0)

            // Screenshot: Fullscreen spectrogram
            takeScreenshot(name: "04_spectrogram_fullscreen")

            // Collapse spectrogram
            let spectrogramCollapseButton = app.buttons["SpectrogramCollapseButton"]
            if spectrogramCollapseButton.waitForExistence(timeout: 2) {
                spectrogramCollapseButton.tap()
                Thread.sleep(forTimeInterval: 0.5)
            }
        }

        // Scroll down to find pitch graph expand button
        app.swipeUp()
        Thread.sleep(forTimeInterval: 0.5)

        // Capture fullscreen pitch graph
        let pitchGraphExpandButton = app.buttons["PitchGraphExpandButton"]
        if pitchGraphExpandButton.waitForExistence(timeout: 3) {
            pitchGraphExpandButton.tap()
            Thread.sleep(forTimeInterval: 1.0)

            // Screenshot: Fullscreen pitch graph
            takeScreenshot(name: "04_pitch_graph_fullscreen")

            // Collapse pitch graph
            let pitchGraphCollapseButton = app.buttons["PitchGraphCollapseButton"]
            if pitchGraphCollapseButton.waitForExistence(timeout: 2) {
                pitchGraphCollapseButton.tap()
                Thread.sleep(forTimeInterval: 0.5)
            }
        }

        // Navigate back to list
        app.navigationBars.buttons.element(boundBy: 0).tap()
        Thread.sleep(forTimeInterval: 0.5)
    }

    @MainActor
    private func captureStatisticsSheet(_ app: XCUIApplication) {
        // Navigate to analysis screen from recording list
        // After captureAnalysisScreen, we're back at the recording list

        // Tap the first recording to open analysis screen
        let firstRecordingLink = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'AnalysisNavigationLink_'")).firstMatch
        if firstRecordingLink.waitForExistence(timeout: 3) {
            firstRecordingLink.tap()
            Thread.sleep(forTimeInterval: 1.0)
        }

        // Scroll up to ensure statistics button is visible (it's at the top of analysis screen)
        app.swipeDown()
        Thread.sleep(forTimeInterval: 0.5)

        // Try the regular statistics button first
        let statisticsButton = app.buttons["StatisticsButton"]
        let statisticsButtonCompact = app.buttons["StatisticsButtonCompact"]

        if statisticsButton.waitForExistence(timeout: 3) {
            statisticsButton.tap()
        } else if statisticsButtonCompact.waitForExistence(timeout: 3) {
            statisticsButtonCompact.tap()
        } else {
            // Fallback: scroll up again to find the button
            app.swipeDown()
            Thread.sleep(forTimeInterval: 0.5)
            if statisticsButton.waitForExistence(timeout: 2) {
                statisticsButton.tap()
            } else if statisticsButtonCompact.waitForExistence(timeout: 2) {
                statisticsButtonCompact.tap()
            }
        }

        // Wait for statistics sheet to appear
        let statisticsSheetView = app.otherElements["StatisticsSheetView"]
        XCTAssertTrue(statisticsSheetView.waitForExistence(timeout: 5), "Statistics sheet should appear")

        // Wait for UI to stabilize
        Thread.sleep(forTimeInterval: 1.0)

        // Screenshot: Statistics sheet (Pitch Analysis section)
        takeScreenshot(name: "05_statistics_pitch")

        // Scroll down to show Spectrum Analysis section
        // The sheet is scrollable, so we need to scroll within the sheet
        let scrollView = app.scrollViews.firstMatch
        if scrollView.exists {
            scrollView.swipeUp()
        } else {
            app.swipeUp()
        }
        Thread.sleep(forTimeInterval: 0.5)

        // Screenshot: Statistics sheet (Spectrum Analysis section)
        takeScreenshot(name: "05_statistics_spectrum")

        // Close the sheet
        let closeButton = app.buttons["StatisticsSheetCloseButton"]
        if closeButton.waitForExistence(timeout: 2) {
            closeButton.tap()
        } else {
            // Fallback: swipe down to dismiss
            app.swipeDown()
        }

        Thread.sleep(forTimeInterval: 0.5)
    }

    @MainActor
    private func captureSettingsScreen(_ app: XCUIApplication) {
        // Navigate back to home first
        // Tap back button repeatedly until we reach home
        while app.navigationBars.buttons.element(boundBy: 0).exists {
            app.navigationBars.buttons.element(boundBy: 0).tap()
            Thread.sleep(forTimeInterval: 0.3)
        }

        // Wait for home screen
        let homeSettingsButton = app.buttons["HomeSettingsButton"]
        XCTAssertTrue(homeSettingsButton.waitForExistence(timeout: 5), "Home settings button should exist")
        homeSettingsButton.tap()

        // Wait for settings screen to load
        Thread.sleep(forTimeInterval: 1.0)

        takeScreenshot(name: "06_settings")
    }
}
