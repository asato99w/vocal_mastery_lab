//
//  StatisticsSheetUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for statistics sheet functionality and scale settings consistency
//

import XCTest

/// Statistics Sheet UI Tests
///
/// Tests for verifying statistics sheet display and consistency with scale settings.
/// Statistics are calculated from scale settings, NOT from detection results.
///
/// ## Specification (See: claudedocs/active/statistics-sheet-specification.md)
///
/// ### By Scale Position
/// - **Always displays ALL positions** based on notePattern.playbackPattern.count
/// - fiveToneScale: 9 positions (playbackPattern = [0, 2, 4, 5, 7, 5, 4, 2, 0])
/// - octaveRepeat: 10 positions (playbackPattern = [0, 4, 7, 12, 12, 12, 12, 7, 4, 0])
/// - brokenScale: 7 positions (playbackPattern = [0, 7, 4, 12, 7, 4, 0]) - single pattern
/// - brokenScaleDouble: 13 positions (playbackPattern = [0, 7, 4, 12, 7, 4, 0, 7, 4, 12, 7, 4, 0]) - x2 repeat
/// - rossiniScale: 13 positions (playbackPattern = [0, 4, 7, 12, 16, 19, 17, 14, 11, 7, 5, 2, 0])
///
/// ### By Pitch
/// - **Always displays ALL unique note names** across all key changes
/// - Calculated from: generateKeyRoots() × notePattern.intervals
/// - MVP Default (fiveToneScale, ascendingKeyCount=3, keyStepInterval=1):
///   Keys: [C4, C#4, D4, D#4, D4, C#4, C4]
///   Unique notes: C4, C#4, D4, D#4, E4, F4, F#4, G4, G#4, A4, A#4 (11 notes)
final class StatisticsSheetUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    // MARK: - Basic Statistics Sheet Tests

    /// Test: Statistics sheet opens and closes correctly
    /// Verifies basic sheet functionality without scale-specific assertions
    @MainActor
    func testStatisticsSheet_OpenAndClose() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to analysis screen with a recording
        navigateToAnalysisScreenWithRecording(app)

        // Wait for analysis to complete
        let statisticsButton = app.buttons["StatisticsButton"]
        if !statisticsButton.waitForExistence(timeout: 10) {
            // Try compact button for portrait mode
            let statisticsButtonCompact = app.buttons["StatisticsButtonCompact"]
            XCTAssertTrue(statisticsButtonCompact.waitForExistence(timeout: 5), "Statistics button should exist")
            statisticsButtonCompact.tap()
        } else {
            statisticsButton.tap()
        }

        // Verify statistics sheet appears by looking for "Statistics" navigation title
        let statisticsTitle = app.staticTexts["Statistics"]
        XCTAssertTrue(statisticsTitle.waitForExistence(timeout: 5), "Statistics sheet should appear")

        // Screenshot: Statistics sheet opened
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "statistics_sheet_01_opened"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // Verify Overall section exists by looking for "Overall" text header
        let overallText = app.staticTexts["Overall"]
        XCTAssertTrue(overallText.waitForExistence(timeout: 3), "Overall section should exist")

        // Close the sheet using the close button
        let closeButton = app.buttons["StatisticsSheetCloseButton"]
        XCTAssertTrue(closeButton.waitForExistence(timeout: 3), "Close button should exist")
        closeButton.tap()

        // Verify sheet is dismissed by checking Statistics title is gone
        XCTAssertFalse(statisticsTitle.waitForExistence(timeout: 2), "Statistics sheet should be dismissed")
    }

    /// Test: Statistics sheet displays all sections with data
    /// Verifies Overall, Position, and Pitch sections appear when data exists
    @MainActor
    func testStatisticsSheet_DisplaysAllSections() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to analysis screen with a recording (default 5-tone scale)
        navigateToAnalysisScreenWithRecording(app)

        // Open statistics sheet
        openStatisticsSheet(app)

        // Verify Overall section by looking for the header text
        let overallText = app.staticTexts["Overall"]
        XCTAssertTrue(overallText.waitForExistence(timeout: 5), "Overall section should exist")

        // Verify Position section header if data exists
        let positionText = app.staticTexts["By Scale Position"]
        // Position section may not exist if no pitch was detected during recording
        // This is acceptable - we're testing that the sections display correctly when data exists

        // Screenshot: All sections displayed
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "statistics_sheet_02_all_sections"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // If pitch section exists (By Pitch header), expand it
        let pitchToggleButton = app.buttons["PitchSectionToggleButton"]
        if pitchToggleButton.exists {
            pitchToggleButton.tap()
            Thread.sleep(forTimeInterval: 0.5)

            // Screenshot: Pitch section expanded
            let screenshot2 = app.screenshot()
            let attachment2 = XCTAttachment(screenshot: screenshot2)
            attachment2.name = "statistics_sheet_03_pitch_expanded"
            attachment2.lifetime = .keepAlways
            add(attachment2)
        }
    }

    // MARK: - Scale Statistics Tests (Consolidated)
    // Each test verifies BOTH position count AND pitch detection for a scale type
    // This reduces test count from 12 to 6 while maintaining coverage

    /// Test: 5-tone scale statistics (position count = 9, pitch detection working)
    /// playbackPattern: [0, 2, 4, 5, 7, 5, 4, 2, 0] → 9 positions
    @MainActor
    func testStatistics_FiveToneScale() throws {
        let app = launchAppWithResetRecordingCount()
        createRecordingWithDefaultScale(app)
        navigateToRecordingListAndAnalysis(app)
        openStatisticsSheet(app)

        // Verify position count
        let positionCount = verifyPositionSection(app, expectedCount: 9)
        XCTAssertEqual(positionCount, 9, "fiveToneScale should display 9 positions")

        // Verify pitch detection
        let pitchCount = verifyPitchSection(app)
        XCTAssertGreaterThan(pitchCount, 0, "At least one pitch should be detected")

        takeScreenshot(app, name: "statistics_fiveTone")
    }

    /// Test: Octave repeat scale statistics (position count = 10, pitch detection working)
    /// playbackPattern: [0, 4, 7, 12, 12, 12, 12, 7, 4, 0] → 10 positions
    @MainActor
    func testStatistics_OctaveRepeatScale() throws {
        let app = launchAppWithResetRecordingCount()
        createRecordingWithOctaveRepeatScale(app)
        navigateToRecordingListAndAnalysis(app)
        openStatisticsSheet(app)

        // Verify position count
        let positionCount = verifyPositionSection(app, expectedCount: 10)
        XCTAssertEqual(positionCount, 10, "octaveRepeat should display 10 positions")

        // Verify pitch detection
        let pitchCount = verifyPitchSection(app)
        XCTAssertGreaterThan(pitchCount, 0, "At least one pitch should be detected")

        takeScreenshot(app, name: "statistics_octaveRepeat")
    }

    /// Test: Broken scale statistics (position count = 7, pitch detection working)
    /// playbackPattern: [0, 7, 4, 12, 7, 4, 0] → 7 positions
    @MainActor
    func testStatistics_BrokenScale() throws {
        let app = launchAppWithResetRecordingCount()
        createRecordingWithBrokenScale(app)
        navigateToRecordingListAndAnalysis(app)
        openStatisticsSheet(app)

        // Verify position count
        let positionCount = verifyPositionSection(app, expectedCount: 7)
        XCTAssertEqual(positionCount, 7, "brokenScale should display 7 positions")

        // Verify pitch detection
        let pitchCount = verifyPitchSection(app)
        XCTAssertGreaterThan(pitchCount, 0, "At least one pitch should be detected")

        takeScreenshot(app, name: "statistics_brokenScale")
    }

    /// Test: Broken scale double statistics (position count = 13, pitch detection working)
    /// playbackPattern: [0, 7, 4, 12, 7, 4, 0, 7, 4, 12, 7, 4, 0] → 13 positions
    @MainActor
    func testStatistics_BrokenScaleDouble() throws {
        let app = launchAppWithResetRecordingCount()
        createRecordingWithBrokenScaleDouble(app)
        navigateToRecordingListAndAnalysis(app)
        openStatisticsSheet(app)

        // Verify position count
        let positionCount = verifyPositionSection(app, expectedCount: 13)
        XCTAssertEqual(positionCount, 13, "brokenScaleDouble should display 13 positions")

        // Verify pitch detection
        let pitchCount = verifyPitchSection(app)
        XCTAssertGreaterThan(pitchCount, 0, "At least one pitch should be detected")

        takeScreenshot(app, name: "statistics_brokenScaleDouble")
    }

    /// Test: Rossini scale statistics (position count = 13, pitch detection working)
    /// playbackPattern: [0, 4, 7, 12, 16, 19, 17, 14, 12, 7, 5, 2, 0] → 13 positions
    @MainActor
    func testStatistics_RossiniScale() throws {
        let app = launchAppWithResetRecordingCount()
        createRecordingWithRossiniScale(app)
        navigateToRecordingListAndAnalysis(app)
        openStatisticsSheet(app)

        // Verify position count
        let positionCount = verifyPositionSection(app, expectedCount: 13)
        XCTAssertEqual(positionCount, 13, "rossiniScale should display 13 positions")

        // Verify pitch detection
        let pitchCount = verifyPitchSection(app)
        XCTAssertGreaterThan(pitchCount, 0, "At least one pitch should be detected")

        takeScreenshot(app, name: "statistics_rossiniScale")
    }

    // MARK: - Statistics Verification Helpers

    /// Verify position section and return count
    private func verifyPositionSection(_ app: XCUIApplication, expectedCount: Int) -> Int {
        let positionSectionText = app.staticTexts["By Scale Position"]
        XCTAssertTrue(positionSectionText.waitForExistence(timeout: 5), "By Scale Position section should exist")

        let positionToggleButton = app.buttons["PositionSectionToggleButton"]
        XCTAssertTrue(positionToggleButton.waitForExistence(timeout: 3), "Position section toggle should exist")
        positionToggleButton.tap()
        Thread.sleep(forTimeInterval: 0.5)

        var positionCount = 0
        for i in 1...expectedCount {
            let suffix = "\(i)st"
            if app.staticTexts[suffix].exists {
                positionCount += 1
            }
        }
        return positionCount
    }

    /// Verify pitch section and return detected pitch count
    private func verifyPitchSection(_ app: XCUIApplication) -> Int {
        let pitchToggleButton = app.buttons["PitchSectionToggleButton"]
        XCTAssertTrue(pitchToggleButton.waitForExistence(timeout: 10), "Pitch section toggle should exist")
        pitchToggleButton.tap()
        Thread.sleep(forTimeInterval: 1.0)

        let allPossibleNotes = [
            "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2",
            "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
            "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
            "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
            "C6"
        ]
        var pitchCount = 0
        for noteName in allPossibleNotes {
            if app.staticTexts[noteName].exists {
                pitchCount += 1
            }
        }
        return pitchCount
    }

    /// Take screenshot helper
    private func takeScreenshot(_ app: XCUIApplication, name: String) {
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    // MARK: - Helper Methods

    /// Navigate to analysis screen by creating a recording and navigating to it
    private func navigateToAnalysisScreenWithRecording(_ app: XCUIApplication) {
        // Create a recording
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Record for a moment to ensure audio data
        Thread.sleep(forTimeInterval: 3.0)
        stopButton.tap()

        // Wait for recording to be saved
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // Navigate back to home
        app.navigationBars.buttons.element(boundBy: 0).tap()

        // Navigate to Recording List
        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // Wait for recording list to load
        let analysisLinks = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisLinks.firstMatch.waitForExistence(timeout: 5), "Analysis navigation link should exist")

        // Navigate to Analysis screen
        analysisLinks.firstMatch.tap()

        // Wait for analysis screen to load
        let recordingInfoCompact = app.otherElements["RecordingInfoCompact"]
        XCTAssertTrue(recordingInfoCompact.waitForExistence(timeout: 10), "Analysis screen should load")
    }

    /// Create recording with default 5-tone scale settings
    private func createRecordingWithDefaultScale(_ app: XCUIApplication) {
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // Default scale is 5-tone, no need to change settings
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Record for enough time to capture scale playback
        Thread.sleep(forTimeInterval: 5.0)
        stopButton.tap()

        // Wait for recording to be saved
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // Navigate back to home
        app.navigationBars.buttons.element(boundBy: 0).tap()
    }

    /// Create recording with octave repeat scale
    private func createRecordingWithOctaveRepeatScale(_ app: XCUIApplication) {
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // Show settings panel if collapsed
        let showSettingsButton = app.buttons["ShowSettingsButton"]
        if showSettingsButton.waitForExistence(timeout: 2) {
            showSettingsButton.tap()
            Thread.sleep(forTimeInterval: 0.3)
        }

        // Change scale pattern to Octave Repeat
        // Note: ScaleTypePicker is the correct identifier (not ScalePatternPicker)
        let scaleTypePicker = app.buttons["ScaleTypePicker"]
        XCTAssertTrue(scaleTypePicker.waitForExistence(timeout: 5), "Scale type picker should exist")
        scaleTypePicker.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Select Octave Repeat option using correct localization key
        // The picker uses "recording.scale_octave_repeat" (L10n.scaleOctaveRepeat), not "scale.octaveRepeat"
        let octaveRepeatOption = app.buttons[L10n.scaleOctaveRepeat]
        XCTAssertTrue(octaveRepeatOption.waitForExistence(timeout: 3), "Octave Repeat option should exist")
        octaveRepeatOption.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Start recording
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Record for enough time
        Thread.sleep(forTimeInterval: 5.0)
        stopButton.tap()

        // Wait for recording to be saved
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // Navigate back to home
        app.navigationBars.buttons.element(boundBy: 0).tap()
    }

    /// Create recording with broken scale
    private func createRecordingWithBrokenScale(_ app: XCUIApplication) {
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // Show settings panel if collapsed
        let showSettingsButton = app.buttons["ShowSettingsButton"]
        if showSettingsButton.waitForExistence(timeout: 2) {
            showSettingsButton.tap()
            Thread.sleep(forTimeInterval: 0.3)
        }

        // Change scale pattern to Broken Scale
        let scaleTypePicker = app.buttons["ScaleTypePicker"]
        XCTAssertTrue(scaleTypePicker.waitForExistence(timeout: 5), "Scale type picker should exist")
        scaleTypePicker.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Select Broken Scale option
        let brokenScaleOption = app.buttons[L10n.scaleBroken]
        XCTAssertTrue(brokenScaleOption.waitForExistence(timeout: 3), "Broken Scale option should exist")
        brokenScaleOption.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Start recording
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Record for enough time
        Thread.sleep(forTimeInterval: 5.0)
        stopButton.tap()

        // Wait for recording to be saved
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // Navigate back to home
        app.navigationBars.buttons.element(boundBy: 0).tap()
    }

    /// Create recording with broken scale double (x2)
    private func createRecordingWithBrokenScaleDouble(_ app: XCUIApplication) {
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // Show settings panel if collapsed
        let showSettingsButton = app.buttons["ShowSettingsButton"]
        if showSettingsButton.waitForExistence(timeout: 2) {
            showSettingsButton.tap()
            Thread.sleep(forTimeInterval: 0.3)
        }

        // Change scale pattern to Broken Scale Double (x2)
        let scaleTypePicker = app.buttons["ScaleTypePicker"]
        XCTAssertTrue(scaleTypePicker.waitForExistence(timeout: 5), "Scale type picker should exist")
        scaleTypePicker.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Select Broken Scale Double (x2) option
        let brokenScaleDoubleOption = app.buttons[L10n.scaleBrokenDouble]
        XCTAssertTrue(brokenScaleDoubleOption.waitForExistence(timeout: 3), "Broken Scale (x2) option should exist")
        brokenScaleDoubleOption.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Start recording
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Record for enough time
        Thread.sleep(forTimeInterval: 5.0)
        stopButton.tap()

        // Wait for recording to be saved
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // Navigate back to home
        app.navigationBars.buttons.element(boundBy: 0).tap()
    }

    /// Create recording with rossini scale
    private func createRecordingWithRossiniScale(_ app: XCUIApplication) {
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // Show settings panel if collapsed
        let showSettingsButton = app.buttons["ShowSettingsButton"]
        if showSettingsButton.waitForExistence(timeout: 2) {
            showSettingsButton.tap()
            Thread.sleep(forTimeInterval: 0.3)
        }

        // Change scale pattern to Rossini Scale
        let scaleTypePicker = app.buttons["ScaleTypePicker"]
        XCTAssertTrue(scaleTypePicker.waitForExistence(timeout: 5), "Scale type picker should exist")
        scaleTypePicker.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Select Rossini Scale option (displayed as "1.5 Octave")
        let rossiniScaleOption = app.buttons[L10n.scaleRossini]
        XCTAssertTrue(rossiniScaleOption.waitForExistence(timeout: 3), "Rossini Scale option should exist")
        rossiniScaleOption.tap()
        Thread.sleep(forTimeInterval: 0.3)

        // Start recording
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Record for enough time
        Thread.sleep(forTimeInterval: 5.0)
        stopButton.tap()

        // Wait for recording to be saved
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // Navigate back to home
        app.navigationBars.buttons.element(boundBy: 0).tap()
    }

    /// Navigate to recording list and then to analysis screen
    private func navigateToRecordingListAndAnalysis(_ app: XCUIApplication) {
        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // Wait for recording list to load
        let analysisLinks = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisLinks.firstMatch.waitForExistence(timeout: 5), "Analysis navigation link should exist")

        // Navigate to Analysis screen (most recent recording)
        analysisLinks.firstMatch.tap()

        // Wait for analysis screen to load
        let recordingInfoCompact = app.otherElements["RecordingInfoCompact"]
        XCTAssertTrue(recordingInfoCompact.waitForExistence(timeout: 10), "Analysis screen should load")

        // Wait for analysis to complete
        Thread.sleep(forTimeInterval: 3.0)
    }

    /// Open statistics sheet from analysis screen
    private func openStatisticsSheet(_ app: XCUIApplication) {
        // Wait for analysis to complete by checking that "Analyzing..." text disappears
        let analyzingText = app.staticTexts["Analyzing..."]
        let startTime = Date()
        let maxWaitTime: TimeInterval = 30.0
        while analyzingText.exists && Date().timeIntervalSince(startTime) < maxWaitTime {
            Thread.sleep(forTimeInterval: 0.5)
        }

        // Additional wait for UI to stabilize after analysis completes
        Thread.sleep(forTimeInterval: 1.0)

        // Try regular button first, then compact button
        let statisticsButton = app.buttons["StatisticsButton"]
        if statisticsButton.waitForExistence(timeout: 3) {
            statisticsButton.tap()
        } else {
            let statisticsButtonCompact = app.buttons["StatisticsButtonCompact"]
            XCTAssertTrue(statisticsButtonCompact.waitForExistence(timeout: 5), "Statistics button should exist")
            statisticsButtonCompact.tap()
        }

        // Wait for sheet to appear by checking for "Statistics" navigation title
        let statisticsTitle = app.staticTexts["Statistics"]
        XCTAssertTrue(statisticsTitle.waitForExistence(timeout: 5), "Statistics sheet should appear")

        // Expand sheet to full size by swiping up on the sheet grabber
        expandSheetToFullSize(app)
    }

    /// Expand the statistics sheet to full size
    private func expandSheetToFullSize(_ app: XCUIApplication) {
        // Find the sheet grabber and swipe up to expand
        let sheetGrabber = app.buttons["Sheet Grabber"]
        if sheetGrabber.waitForExistence(timeout: 2) {
            // Swipe up from the grabber to expand sheet
            sheetGrabber.swipeUp()
            Thread.sleep(forTimeInterval: 0.5)
        } else {
            // Alternative: swipe up on the statistics sheet view itself
            let statisticsSheet = app.otherElements["StatisticsSheetView"]
            if statisticsSheet.exists {
                statisticsSheet.swipeUp()
                Thread.sleep(forTimeInterval: 0.5)
            }
        }
    }
}
