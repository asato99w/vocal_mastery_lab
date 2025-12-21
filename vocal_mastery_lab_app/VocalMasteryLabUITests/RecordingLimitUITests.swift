import XCTest

/// UI tests for recording limit and paywall functionality
/// Note: Recording limit tests are skipped while all features are free (unlimited recording).
/// Structure preserved for future paid plan restoration.
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

    // MARK: - Recording Limit Alert Tests (SKIPPED - Unlimited recording for "all free" policy)

    /// Test that recording limit alert appears when user tries to record at limit
    func testRecordingLimitAlert_shouldAppear_whenAtLimit() throws {
        throw XCTSkip("Skipped: Recording limits removed while all features are free. Preserved for future paid plan.")
    }

    /// Test that OK button dismisses the recording limit alert
    func testRecordingLimitAlert_shouldDismiss_whenOKPressed() throws {
        throw XCTSkip("Skipped: Recording limits removed while all features are free. Preserved for future paid plan.")
    }

    /// Test that alert can be dismissed and shown again on subsequent attempts
    func testRecordingLimitAlert_canBeShownMultipleTimes() throws {
        throw XCTSkip("Skipped: Recording limits removed while all features are free. Preserved for future paid plan.")
    }

    // MARK: - Unlimited Recording Tests (Current "all free" policy)

    /// Test that free users can record and stop multiple times without limit
    /// Note: Updated to reflect "all free" policy - no recording limits for any tier
    func testFreeUser_canRecordMultipleTimes_unlimitedRecording() throws {
        // Given: Free user (no recording limits in current policy)
        app.terminate()
        app.launchEnvironment["SUBSCRIPTION_TIER"] = "free"
        app.launch()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        let recordButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(recordButton.waitForExistence(timeout: 5), "Record button should exist")

        // When: User records multiple times (current policy: unlimited)
        for iteration in 1...3 {
            // Tap record button
            recordButton.tap()

            // Verify recording started (no limit alerts should appear)
            let stopButton = app.buttons["StopRecordingButton"]
            XCTAssertTrue(
                stopButton.waitForExistence(timeout: 5),
                "Iteration \(iteration): Stop button should appear (no limit alerts)"
            )

            // Record briefly
            sleep(1)

            // Stop recording
            stopButton.tap()

            // Wait for recording to stop
            XCTAssertTrue(
                recordButton.waitForExistence(timeout: 5),
                "Iteration \(iteration): Record button should reappear after stopping"
            )

            // Verify no limit alert appeared
            let alert = app.alerts.firstMatch
            XCTAssertFalse(
                alert.exists,
                "Iteration \(iteration): Free user should NOT see limit alert (current policy: unlimited)"
            )
        }
    }

    /// Test that premium users can record unlimited times without daily count limit
    /// Note: All tiers now have unlimited recording (current policy: all free)
    func testPremiumUser_canRecordUnlimitedTimes() throws {
        // Given: Premium user (no recording limits in current policy)
        app.terminate()
        app.launchEnvironment["SUBSCRIPTION_TIER"] = "premium"
        app.launch()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        let recordButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(recordButton.waitForExistence(timeout: 5), "Record button should exist")

        // When: User records multiple times (current policy: unlimited)
        for iteration in 1...3 {
            // Tap record button
            recordButton.tap()

            // Verify recording started
            let stopButton = app.buttons["StopRecordingButton"]
            XCTAssertTrue(
                stopButton.waitForExistence(timeout: 5),
                "Iteration \(iteration): Stop button should appear after recording starts"
            )

            // Record briefly
            sleep(1)

            // Stop recording
            stopButton.tap()

            // Wait for recording to stop
            XCTAssertTrue(
                recordButton.waitForExistence(timeout: 5),
                "Iteration \(iteration): Record button should reappear after stopping"
            )

            // Verify no limit alert appeared
            let alert = app.alerts.firstMatch
            XCTAssertFalse(
                alert.exists,
                "Iteration \(iteration): Premium user should NOT see limit alert"
            )
        }
    }
}
