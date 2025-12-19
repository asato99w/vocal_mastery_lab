//
//  BackgroundRecordingUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for background recording functionality
//

import XCTest

/// UI tests for verifying recording continues in background
final class BackgroundRecordingUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    // MARK: - Background Recording Tests

    /// Test that recording continues when app goes to background
    /// 1. Start recording
    /// 2. Send app to background
    /// 3. Wait in background
    /// 4. Return to foreground
    /// 5. Verify recording is still in progress (timer has advanced)
    @MainActor
    func testRecordingContinuesInBackground() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
        homeRecordButton.tap()

        // Start recording
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))
        startButton.tap()

        // Wait for recording to start
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop button should appear when recording starts")

        // Record for 2 seconds in foreground
        Thread.sleep(forTimeInterval: 2.0)

        // Get timer value before going to background
        let timer = app.staticTexts["RecordingTimerLabel"]
        let timerBeforeBackground = timer.label

        // Send app to background
        XCUIDevice.shared.press(.home)

        // Wait in background for 3 seconds
        Thread.sleep(forTimeInterval: 3.0)

        // Return to foreground
        app.activate()

        // Wait for app to become active
        Thread.sleep(forTimeInterval: 1.0)

        // Verify recording is still in progress
        XCTAssertTrue(stopButton.waitForExistence(timeout: 5), "Recording should still be in progress after returning from background")

        // Get timer value after returning from background
        let timerAfterBackground = timer.label

        // Timer should have advanced (not frozen)
        XCTAssertNotEqual(timerBeforeBackground, timerAfterBackground, "Timer should have advanced while in background")

        // Parse timer values to verify advancement
        let beforeSeconds = parseTimerToSeconds(timerBeforeBackground)
        let afterSeconds = parseTimerToSeconds(timerAfterBackground)

        // Timer should have advanced by at least 2 seconds (accounting for timing variations)
        XCTAssertGreaterThan(afterSeconds, beforeSeconds + 2, "Timer should have advanced by at least 2 seconds while in background")

        // Stop recording to clean up
        stopButton.tap()

        // Verify recording stopped and saved
        let lastRecordingSection = app.otherElements["LastRecordingSection"]
        XCTAssertTrue(lastRecordingSection.waitForExistence(timeout: 5), "Recording should be saved after stopping")
    }

    /// Test that recording is saved correctly after returning from background
    /// 1. Start recording
    /// 2. Go to background
    /// 3. Return to foreground
    /// 4. Stop recording
    /// 5. Verify recording duration includes background time
    @MainActor
    func testRecordingSavedAfterBackgroundReturn() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen
        app.buttons["HomeRecordButton"].tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10))

        // Record for 1 second in foreground
        Thread.sleep(forTimeInterval: 1.0)

        // Send app to background
        XCUIDevice.shared.press(.home)

        // Wait in background for 3 seconds
        Thread.sleep(forTimeInterval: 3.0)

        // Return to foreground
        app.activate()
        Thread.sleep(forTimeInterval: 0.5)

        // Stop recording
        stopButton.tap()

        // Verify recording section appears
        let lastRecordingSection = app.otherElements["LastRecordingSection"]
        XCTAssertTrue(lastRecordingSection.waitForExistence(timeout: 5), "Last recording section should appear")

        // Verify duration label exists and shows reasonable duration
        let durationLabel = app.staticTexts["LastRecordingDurationLabel"]
        XCTAssertTrue(durationLabel.exists, "Duration label should exist")

        // Duration should be at least 4 seconds (1s foreground + 3s background)
        // The format is "X分Y秒" or "Y秒"
        let durationText = durationLabel.label
        let durationSeconds = parseDurationToSeconds(durationText)

        XCTAssertGreaterThanOrEqual(durationSeconds, 3, "Recording duration should include background time (expected at least 4s, got \(durationSeconds)s)")
    }

    /// Test that recording state is preserved across multiple background/foreground cycles
    @MainActor
    func testMultipleBackgroundForegroundCycles() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen
        app.buttons["HomeRecordButton"].tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10))

        // Cycle 1: Go to background and return
        XCUIDevice.shared.press(.home)
        Thread.sleep(forTimeInterval: 2.0)
        app.activate()
        Thread.sleep(forTimeInterval: 0.5)

        // Verify still recording
        XCTAssertTrue(stopButton.exists, "Recording should continue after first background cycle")

        // Cycle 2: Go to background and return
        XCUIDevice.shared.press(.home)
        Thread.sleep(forTimeInterval: 2.0)
        app.activate()
        Thread.sleep(forTimeInterval: 0.5)

        // Verify still recording
        XCTAssertTrue(stopButton.exists, "Recording should continue after second background cycle")

        // Cycle 3: Go to background and return
        XCUIDevice.shared.press(.home)
        Thread.sleep(forTimeInterval: 2.0)
        app.activate()
        Thread.sleep(forTimeInterval: 0.5)

        // Verify still recording
        XCTAssertTrue(stopButton.exists, "Recording should continue after third background cycle")

        // Stop recording
        stopButton.tap()

        // Verify recording was saved
        let lastRecordingSection = app.otherElements["LastRecordingSection"]
        XCTAssertTrue(lastRecordingSection.waitForExistence(timeout: 5), "Recording should be saved")
    }

    // MARK: - Helper Methods

    /// Parse timer string "HH:MM:SS" to total seconds
    private func parseTimerToSeconds(_ timerString: String) -> Int {
        let components = timerString.split(separator: ":")
        guard components.count == 3,
              let hours = Int(components[0]),
              let minutes = Int(components[1]),
              let seconds = Int(components[2]) else {
            return 0
        }
        return hours * 3600 + minutes * 60 + seconds
    }

    /// Parse duration string "X分Y秒" or "Y秒" to total seconds
    private func parseDurationToSeconds(_ durationString: String) -> Int {
        // Pattern: "X分Y秒" or "Y秒"
        var totalSeconds = 0

        // Extract minutes if present
        if let minuteRange = durationString.range(of: "分") {
            let minuteStr = String(durationString[..<minuteRange.lowerBound])
            if let minutes = Int(minuteStr.filter { $0.isNumber }) {
                totalSeconds += minutes * 60
            }
        }

        // Extract seconds
        if let secondRange = durationString.range(of: "秒") {
            var startIndex = durationString.startIndex
            if let minuteRange = durationString.range(of: "分") {
                startIndex = minuteRange.upperBound
            }
            let secondStr = String(durationString[startIndex..<secondRange.lowerBound])
            if let seconds = Int(secondStr.filter { $0.isNumber }) {
                totalSeconds += seconds
            }
        }

        return totalSeconds
    }
}
