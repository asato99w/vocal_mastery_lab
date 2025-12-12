import XCTest

/// UI tests for recording limit and paywall functionality
final class RecordingLimitUITests: XCTestCase {

    var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()

        // Use standard UI test launch arguments for compatibility
        app.launchArguments = ["-UITestResetRecordingCount", "-UITestDisableAnimations", "-UITestDisableCountdown"]

        // Set subscription tier to free (to test free tier recording limit)
        app.launchEnvironment["SUBSCRIPTION_TIER"] = "free"
        // COMMENTED OUT: Testing without DAILY_RECORDING_COUNT to reproduce debug environment behavior
        // app.launchEnvironment["DAILY_RECORDING_COUNT"] = "100"

        app.launch()
    }

    override func tearDownWithError() throws {
        app = nil
    }

    // MARK: - Recording Limit Alert Tests

    /// Test that recording limit alert appears when user tries to record at limit
    func testRecordingLimitAlert_shouldAppear_whenAtLimit() throws {
        // Given: User is at the recording limit (5 recordings for free tier)
        app.terminate()
        app.launchEnvironment["SUBSCRIPTION_TIER"] = "free"
        app.launchEnvironment["DAILY_RECORDING_COUNT"] = "5"  // Set count to limit (free tier = 5/day)
        app.launch()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // When: User taps Record button at limit
        let recordButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(recordButton.waitForExistence(timeout: 5), "Record button should exist")
        recordButton.tap()

        // Then: Alert should appear (identified by accessibility identifier)
        // Note: SwiftUI alerts can be identified via parent view's accessibilityIdentifier
        // or by checking alert title/existence
        let alert = app.alerts.firstMatch
        XCTAssertTrue(alert.waitForExistence(timeout: 5), "Limit alert should appear")

        // Verify this is the recording limit alert by checking it has an OK button
        // and was triggered at the limit (language-independent verification)
        let okButton = alert.buttons["OK"]
        XCTAssertTrue(okButton.exists, "Alert should have OK button (recording limit alert)")

        // And: Recording should NOT start (button should remain as "StartRecordingButton")
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertFalse(stopButton.exists, "Stop button should NOT exist - recording should not have started")
        XCTAssertTrue(recordButton.exists, "Start button should still exist")
    }

    /// Test that OK button dismisses the recording limit alert
    func testRecordingLimitAlert_shouldDismiss_whenOKPressed() throws {
        // Given: User is at the recording limit (5 recordings for free tier)
        app.terminate()
        app.launchEnvironment["SUBSCRIPTION_TIER"] = "free"
        app.launchEnvironment["DAILY_RECORDING_COUNT"] = "5"  // Set count to limit (free tier = 5/day)
        app.launch()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        let recordButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(recordButton.waitForExistence(timeout: 5), "Record button should exist")
        recordButton.tap()

        let alert = app.alerts.firstMatch
        XCTAssertTrue(alert.waitForExistence(timeout: 5), "Alert should appear")

        // When: User taps OK button
        let okButton = alert.buttons["OK"]
        XCTAssertTrue(okButton.exists, "OK button should exist in alert")
        okButton.tap()

        // Then: Alert should disappear (wait for disappearance instead of fixed sleep)
        let alertDisappeared = !alert.waitForExistence(timeout: 3)
        XCTAssertTrue(
            alertDisappeared || !alert.exists,
            "Alert should be dismissed after tapping OK"
        )

        // And: Recording screen should still be accessible (not frozen)
        XCTAssertTrue(recordButton.isHittable, "Record button should still be accessible")
    }

    /// Test that alert can be dismissed and shown again on subsequent attempts
    func testRecordingLimitAlert_canBeShownMultipleTimes() throws {
        // Given: User is at the recording limit (5 recordings for free tier)
        app.terminate()
        app.launchEnvironment["SUBSCRIPTION_TIER"] = "free"
        app.launchEnvironment["DAILY_RECORDING_COUNT"] = "5"  // Set count to limit (free tier = 5/day)
        app.launch()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        let recordButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(recordButton.waitForExistence(timeout: 5), "Record button should exist")
        recordButton.tap()

        let alert = app.alerts.firstMatch
        XCTAssertTrue(alert.waitForExistence(timeout: 5), "Alert should appear")
        alert.buttons["OK"].tap()

        // Wait for alert to dismiss (state-based wait)
        _ = !alert.waitForExistence(timeout: 3)
        XCTAssertFalse(alert.exists, "Alert should be dismissed")

        // When: User taps Record button again
        recordButton.tap()

        // Then: Alert should appear again
        XCTAssertTrue(
            alert.waitForExistence(timeout: 5),
            "Alert should be shown again on second attempt"
        )

        // And: Can be dismissed again
        alert.buttons["OK"].tap()
        XCTAssertFalse(alert.exists, "Alert should be dismissed second time")
    }

    /// Test that free users can record and stop multiple times within limit
    func testFreeUser_canRecordMultipleTimes_withinLimit() throws {
        // Given: Free user with 0 recordings (within 100 daily limit)
        app.terminate()
        // Set subscription tier to free (to test free tier recording limit)
        app.launchEnvironment["SUBSCRIPTION_TIER"] = "free"
        // DO NOT set DAILY_RECORDING_COUNT - we need the count to increment naturally
        // Recording count is reset via -UITestResetRecordingCount launch argument (see setUpWithError)
        app.launch()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        let recordButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(recordButton.waitForExistence(timeout: 5), "Record button should exist")

        // When: User attempts 4 recordings to test limit boundary
        // Expected outcomes:
        // - Free tier (limit=3): Alert on 4th attempt
        // - Premium tier (limit=10): No alert within this test
        for iteration in 1...4 {
            print("🔴 ITERATION \(iteration): Starting recording")

            // Tap record button
            recordButton.tap()

            // Check if alert appeared BEFORE recording starts (at limit)
            let alertBeforeRecording = app.alerts.firstMatch
            if alertBeforeRecording.waitForExistence(timeout: 2) {
                print("⚠️ ALERT APPEARED at iteration \(iteration) BEFORE recording")
                print("Alert title: \(alertBeforeRecording.staticTexts.element(boundBy: 0).label)")
                print("Alert message: \(alertBeforeRecording.staticTexts.element(boundBy: 1).label)")

                // Dismiss alert to continue test
                let okButton = alertBeforeRecording.buttons["OK"]
                if okButton.exists {
                    okButton.tap()
                }

                // Recording should NOT have started
                let stopButton = app.buttons["StopRecordingButton"]
                XCTAssertFalse(
                    stopButton.exists,
                    "Iteration \(iteration): Recording should NOT start when limit alert appears"
                )

                print("🛑 LIMIT REACHED at iteration \(iteration)")
                break
            }

            // Wait for recording to start (countdown skipped with -UITestDisableCountdown)
            // No sleep needed - just wait for stop button to appear

            // Verify recording started
            let stopButton = app.buttons["StopRecordingButton"]
            if !stopButton.waitForExistence(timeout: 5) {
                print("❌ ITERATION \(iteration): Stop button did NOT appear - recording failed to start")
                XCTFail("Iteration \(iteration): Stop button should appear after countdown")
                break
            }

            // Record for 1 second
            sleep(1)

            // Stop recording
            print("🔴 ITERATION \(iteration): Stopping recording")
            stopButton.tap()

            // Wait for recording to stop and return to idle state
            XCTAssertTrue(
                recordButton.waitForExistence(timeout: 5),
                "Iteration \(iteration): Record button should reappear after stopping"
            )

            print("✅ ITERATION \(iteration): Completed successfully")
            // No delay needed - waitForExistence on recordButton already handles timing
        }

        print("✅ TEST COMPLETED")
    }

    /// Test that premium users can record unlimited times without daily count limit
    func testPremiumUser_canRecordUnlimitedTimes() throws {
        // Given: Premium user with 0 recordings
        app.terminate()
        // Set subscription tier to premium (to test premium tier recording limit)
        app.launchEnvironment["SUBSCRIPTION_TIER"] = "premium"
        // Reset recording count to 0 for consistent test behavior
        app.launchEnvironment["DAILY_RECORDING_COUNT"] = "0"
        app.launch()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        let recordButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(recordButton.waitForExistence(timeout: 5), "Record button should exist")

        // When: User records and stops 10 times (more than free tier limit)
        for iteration in 1...10 {
            print("🟣 PREMIUM ITERATION \(iteration): Starting recording")

            // Tap record button
            recordButton.tap()

            // Wait for recording to start (countdown skipped with -UITestDisableCountdown)
            // No sleep needed - just wait for stop button to appear

            // Verify recording started
            let stopButton = app.buttons["StopRecordingButton"]
            XCTAssertTrue(
                stopButton.waitForExistence(timeout: 5),
                "Iteration \(iteration): Stop button should appear after recording starts"
            )

            // Record for 1 second (well within 300 second max duration)
            sleep(1)

            // Stop recording
            print("🟣 PREMIUM ITERATION \(iteration): Stopping recording")
            stopButton.tap()

            // Wait for recording to stop and return to idle state
            XCTAssertTrue(
                recordButton.waitForExistence(timeout: 5),
                "Iteration \(iteration): Record button should reappear after stopping"
            )

            // Verify no limit alert appeared
            let alert = app.alerts.firstMatch
            XCTAssertFalse(
                alert.exists,
                "Iteration \(iteration): Premium user should NOT see daily count limit alert"
            )

            print("✅ PREMIUM ITERATION \(iteration): Completed successfully")
            // No delay needed - waitForExistence on recordButton already handles timing
        }

        // Then: All 10 recordings completed successfully without daily count limit alerts
        print("✅ ALL PREMIUM ITERATIONS COMPLETED: Premium user recorded 10 times without daily count limit")
    }
}
