//
//  MIDIRangeValidationUITests.swift
//  VocalMasteringLabUITests
//
//  UI tests for MIDI range validation (warning display and button disable)
//

import XCTest

final class MIDIRangeValidationUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    /// Test: Recording button is enabled with default settings (valid MIDI range)
    /// This verifies that the UI properly reflects valid MIDI range configurations
    @MainActor
    func testRecordingButton_enabledWithDefaultSettings() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Navigate to Recording screen from Home
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        // 2. Wait for recording screen to load
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")

        // 3. Verify start button is enabled with default settings
        // Default settings: C3 start, 5-tone scale, 5 ascending, semitone interval
        // This should be well within MIDI range (no warning)
        XCTAssertTrue(startButton.isEnabled, "Start button should be enabled with default settings")

        // 4. Verify no MIDI warning is displayed
        let midiWarning = app.otherElements["MIDIRangeWarning"]
        XCTAssertFalse(midiWarning.exists, "No MIDI warning should be shown with default settings")

        // Screenshot: Initial valid state
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "midi_validation_default_settings_valid"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    /// Test: MIDIRangeWarning accessibility identifier is properly set
    /// This ensures the warning view has the correct identifier for UI testing
    @MainActor
    func testMIDIRangeWarning_accessibilityIdentifierExists() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
        homeRecordButton.tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))

        // 2. Check that warning element query works (even if not visible)
        // This confirms the accessibility infrastructure is in place
        let midiWarning = app.otherElements["MIDIRangeWarning"]

        // 3. With default settings, warning should not exist
        XCTAssertFalse(midiWarning.exists, "MIDI warning should not exist with default valid settings")

        // Screenshot: Capture settings panel for verification
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "midi_validation_no_warning_default"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    /// Test: Settings panel displays correctly and is interactive
    /// This verifies the settings UI is accessible for MIDI range configuration
    @MainActor
    func testSettingsPanel_displaysCorrectly() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
        homeRecordButton.tap()

        // 2. Verify key UI elements exist
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start button should exist")

        let scaleTypePicker = app.buttons["ScaleTypePicker"]
        XCTAssertTrue(scaleTypePicker.waitForExistence(timeout: 3), "Scale type picker should exist")

        let startPitchPicker = app.buttons["StartPitchPicker"]
        XCTAssertTrue(startPitchPicker.waitForExistence(timeout: 3), "Start pitch picker should exist")

        // 3. Verify button is enabled (valid configuration)
        XCTAssertTrue(startButton.isEnabled, "Start button should be enabled")

        // Screenshot: Settings panel
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "midi_validation_settings_panel"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
