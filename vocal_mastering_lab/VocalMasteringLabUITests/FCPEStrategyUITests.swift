//
//  FCPEStrategyUITests.swift
//  VocalMasteringLabUITests
//
//  UI tests for FCPE pitch detection strategy investigation
//

import XCTest

final class FCPEStrategyUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    /// Test: FCPE algorithm selection and pitch detection verification
    /// Purpose: Investigate why FCPE returns no pitch data
    @MainActor
    func testFCPE_PitchDetectionInvestigation() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Navigate to Settings -> Input Settings to select FCPE algorithm
        let homeSettingsButton = app.buttons["HomeSettingsButton"]
        XCTAssertTrue(homeSettingsButton.waitForExistence(timeout: 5), "Settings button should exist")
        homeSettingsButton.tap()

        // Wait for settings screen - use label since no identifier is set
        let inputSettingsButton = app.buttons["Input Settings"]
        let inputSettingsButtonJa = app.buttons["入力設定"]

        if inputSettingsButton.waitForExistence(timeout: 5) {
            inputSettingsButton.tap()
        } else if inputSettingsButtonJa.waitForExistence(timeout: 3) {
            inputSettingsButtonJa.tap()
        } else {
            XCTFail("Input settings button should exist")
        }

        // Wait for input settings screen and take screenshot
        Thread.sleep(forTimeInterval: 1.0)

        // Screenshot: Input settings screen
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "fcpe_01_input_settings"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // 2. Select FCPE algorithm
        // Look for FCPE option in the picker or navigation view
        let fcpeOption = app.staticTexts["FCPE (Neural Net)"]
        let fcpeOptionJa = app.staticTexts["FCPE (ニューラルネット)"]

        if fcpeOption.waitForExistence(timeout: 3) {
            fcpeOption.tap()
        } else if fcpeOptionJa.waitForExistence(timeout: 3) {
            fcpeOptionJa.tap()
        } else {
            // Try tapping on the algorithm picker row to open selection
            let algorithmRow = app.cells.containing(NSPredicate(format: "label CONTAINS %@", "YIN")).firstMatch
            if algorithmRow.waitForExistence(timeout: 3) {
                algorithmRow.tap()

                // Now try to find FCPE option again
                Thread.sleep(forTimeInterval: 0.5)
                if fcpeOption.waitForExistence(timeout: 3) {
                    fcpeOption.tap()
                } else if fcpeOptionJa.waitForExistence(timeout: 3) {
                    fcpeOptionJa.tap()
                }
            }
        }

        // Screenshot: After selecting FCPE
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "fcpe_02_algorithm_selected"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // 3. Save settings - settings auto-save, just navigate back
        // Navigate back to home

        // Navigate back to home if needed
        let backButton = app.navigationBars.buttons.element(boundBy: 0)
        while backButton.exists && !app.buttons["HomeRecordButton"].exists {
            backButton.tap()
            Thread.sleep(forTimeInterval: 0.3)
        }

        // 4. Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // 5. Start and complete a recording
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Record for 3 seconds to capture audio with scale playback
        Thread.sleep(forTimeInterval: 3.0)
        stopButton.tap()

        // Wait for recording to complete
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // Screenshot: After recording
        let screenshot3 = app.screenshot()
        let attachment3 = XCTAttachment(screenshot: screenshot3)
        attachment3.name = "fcpe_03_after_recording"
        attachment3.lifetime = .keepAlways
        add(attachment3)

        // 6. Navigate to analysis via analyze button
        let analyzeButton = app.buttons["AnalyzeRecordingButton"]
        XCTAssertTrue(analyzeButton.waitForExistence(timeout: 5), "Analyze button should exist")
        analyzeButton.tap()

        // 7. Wait for analysis screen to load
        let recordingInfoCompact = app.otherElements["RecordingInfoCompact"]
        XCTAssertTrue(recordingInfoCompact.waitForExistence(timeout: 10), "Analysis screen should load")

        // 8. Wait for analysis to complete (this is where FCPE should run)
        let playPauseButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButton.waitForExistence(timeout: 60), "Analysis should complete")

        // Wait additional time for FCPE processing (neural network is slower)
        Thread.sleep(forTimeInterval: 5.0)

        // 9. Verify pitch graph components
        let pitchGraphView = app.otherElements["PitchAnalysisView"]
        XCTAssertTrue(pitchGraphView.waitForExistence(timeout: 5), "Pitch graph view should be displayed")

        // Screenshot: Analysis screen with FCPE results
        let screenshot4 = app.screenshot()
        let attachment4 = XCTAttachment(screenshot: screenshot4)
        attachment4.name = "fcpe_04_analysis_result"
        attachment4.lifetime = .keepAlways
        add(attachment4)

        // 10. Verify spectrogram exists (audio was processed)
        let spectrogramView = app.otherElements["SpectrogramView"]
        XCTAssertTrue(spectrogramView.waitForExistence(timeout: 5), "Spectrogram view should be displayed")

        // 11. Check for pitch line visibility
        // The pitch line is drawn on the spectrogram - if no pitch data, it won't be visible
        // We can't directly verify pixel content, but we can take another screenshot after scrolling

        // Scroll spectrogram to see different parts
        let spectrogramCenter = spectrogramView.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        let spectrogramLeft = spectrogramView.coordinate(withNormalizedOffset: CGVector(dx: 0.2, dy: 0.5))
        spectrogramCenter.press(forDuration: 0.1, thenDragTo: spectrogramLeft)

        // Screenshot: After scrolling spectrogram
        let screenshot5 = app.screenshot()
        let attachment5 = XCTAttachment(screenshot: screenshot5)
        attachment5.name = "fcpe_05_scrolled_spectrogram"
        attachment5.lifetime = .keepAlways
        add(attachment5)

        // Test passed - check logs for FCPE pitch detection details
        // The FileLogger will capture:
        // - Whether FCPE model was loaded
        // - Number of pitch frames detected
        // - Any errors during processing
    }

    /// Test: Compare YIN vs FCPE pitch detection on same recording
    /// Purpose: Verify if the issue is FCPE-specific or general
    @MainActor
    func testYIN_vs_FCPE_Comparison() throws {
        let app = launchAppWithResetRecordingCount()

        // Part 1: Record and analyze with YIN (default)
        // 1. Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // 2. Start and complete a recording (YIN is default)
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        Thread.sleep(forTimeInterval: 3.0)
        stopButton.tap()

        // Wait for recording to complete
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // Navigate to analysis
        let analyzeButton = app.buttons["AnalyzeRecordingButton"]
        XCTAssertTrue(analyzeButton.waitForExistence(timeout: 5), "Analyze button should exist")
        analyzeButton.tap()

        // Wait for analysis to complete
        let playPauseButton = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButton.waitForExistence(timeout: 60), "YIN Analysis should complete")

        // Screenshot: YIN analysis result
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "comparison_01_yin_result"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // Part 2: Change to FCPE and re-analyze
        // Navigate back to home
        let backButton = app.navigationBars.buttons.element(boundBy: 0)
        backButton.tap()
        backButton.tap()

        // Go to Settings -> Input Settings
        let homeSettingsButton = app.buttons["HomeSettingsButton"]
        XCTAssertTrue(homeSettingsButton.waitForExistence(timeout: 5), "Settings button should exist")
        homeSettingsButton.tap()

        let inputSettingsButton2 = app.buttons["Input Settings"]
        let inputSettingsButtonJa2 = app.buttons["入力設定"]
        if inputSettingsButton2.waitForExistence(timeout: 5) {
            inputSettingsButton2.tap()
        } else if inputSettingsButtonJa2.waitForExistence(timeout: 3) {
            inputSettingsButtonJa2.tap()
        }

        // Select FCPE algorithm
        let fcpeOption = app.staticTexts["FCPE (Neural Net)"]
        let fcpeOptionJa = app.staticTexts["FCPE (ニューラルネット)"]

        // Find and tap the algorithm row to open selection
        let algorithmRow = app.cells.containing(NSPredicate(format: "label CONTAINS %@", "YIN")).firstMatch
        if algorithmRow.waitForExistence(timeout: 3) {
            algorithmRow.tap()
            Thread.sleep(forTimeInterval: 0.5)
        }

        if fcpeOption.waitForExistence(timeout: 3) {
            fcpeOption.tap()
        } else if fcpeOptionJa.waitForExistence(timeout: 3) {
            fcpeOptionJa.tap()
        }

        // Save settings
        let saveButton = app.buttons["Save"]
        let saveButtonJa = app.buttons["保存"]
        if saveButton.waitForExistence(timeout: 2) {
            saveButton.tap()
        } else if saveButtonJa.waitForExistence(timeout: 2) {
            saveButtonJa.tap()
        }

        // Navigate to Recording List
        while !app.buttons["HomeListButton"].exists && backButton.exists {
            backButton.tap()
            Thread.sleep(forTimeInterval: 0.3)
        }

        let homeListButton = app.buttons["HomeListButton"]
        XCTAssertTrue(homeListButton.waitForExistence(timeout: 5), "Home list button should exist")
        homeListButton.tap()

        // Open the same recording again (now with FCPE algorithm)
        let analysisLinks = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "AnalysisNavigationLink_"))
        XCTAssertTrue(analysisLinks.firstMatch.waitForExistence(timeout: 5), "Analysis link should exist")
        analysisLinks.firstMatch.tap()

        // Wait for FCPE analysis to complete (algorithm change invalidates cache)
        let playPauseButton2 = app.buttons["AnalysisPlayPauseButton"]
        XCTAssertTrue(playPauseButton2.waitForExistence(timeout: 90), "FCPE Analysis should complete")

        // Wait for FCPE processing
        Thread.sleep(forTimeInterval: 5.0)

        // Screenshot: FCPE analysis result on same recording
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "comparison_02_fcpe_result"
        attachment2.lifetime = .keepAlways
        add(attachment2)
    }
}
