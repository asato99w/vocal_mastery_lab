//
//  PlaybackUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for playback functionality
//

import XCTest

/// Playback UI tests
///
/// ⚠️ IMPORTANT: This test class contains pitch detection tests that require
/// speaker → microphone audio feedback loop in iOS Simulator. Due to AVAudioSession
/// limitations, these tests CANNOT run in parallel with other tests.
///
/// **Run these tests individually, not in parallel execution mode.**
///
/// Parallel execution issue:
/// - When 5 simulator clones run simultaneously, AVAudioSession conflicts occur
/// - Speaker output → Microphone input feedback loop fails
/// - Pitch detection tests fail even though the implementation is correct
///
/// Verified working:
/// - Single test execution: ✅ PASS
/// - Parallel execution (5 clones): ❌ FAIL (expected)
///
/// Usage:
/// ```bash
/// # Run this test class individually
/// xcodebuild test -only-testing:VocalMasteryLabUITests/PlaybackUITests
/// ```
final class PlaybackUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    /// Test 4: Full playback completion (natural playback end)
    /// Expected: ~8 seconds execution time
    func testPlaybackFullCompletion() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. Create a short recording first
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        // Wait for recording to start by checking StopButton appearance
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // Continue recording for playback verification
        // 3 seconds needed to allow playback initialization (2s) + buffer for verification
        Thread.sleep(forTimeInterval: 3.0)

        stopButton.tap()

        // Wait for recording to finish and be saved by checking PlayButton appearance
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // Screenshot: Before playback
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "playback_01_before_play"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // 3. Start playback
        playButton.tap()

        // Wait for playback to start and scale playback to initialize
        // Note: Playback with scale requires initialization time
        Thread.sleep(forTimeInterval: 2.0)

        // 4. Verify Stop Playback button appears (playback is in progress)
        let stopPlaybackButton = app.buttons["StopPlaybackButton"]
        XCTAssertTrue(stopPlaybackButton.waitForExistence(timeout: 2), "Stop playback button should appear during playback")

        // Screenshot: During playback
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "playback_02_during_playback"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // 5. Wait for playback to complete naturally (recording was ~1 second)
        // Give enough time for the short recording to finish playing
        Thread.sleep(forTimeInterval: 3.0)

        // 6. Verify Play button reappears after playback completion
        XCTAssertTrue(playButton.waitForExistence(timeout: 3), "Play last recording button should reappear after playback completes")

        // Screenshot: After playback completion
        let screenshot3 = app.screenshot()
        let attachment3 = XCTAttachment(screenshot: screenshot3)
        attachment3.name = "playback_03_after_completion"
        attachment3.lifetime = .keepAlways
        add(attachment3)
    }

    // MARK: - Bug Reproduction Tests

    /// BUG REPRODUCTION: 2回目の再生が開始されない
    ///
    /// **症状**: 録音後の再生を2回行うと、2回目の再生が開始されない
    ///
    /// **エラーメッセージ** (コンソール):
    /// ```
    /// required condition is false: IsFormatSampleRateAndChannelCountValid(format)
    /// ```
    ///
    /// **根本原因の仮説**:
    /// - `AVAudioPlayerWrapper.stop()` が `AudioSessionManager.shared.deactivate()` を呼び出す
    /// - これにより `AVAudioEngineScalePlayer` のノード接続フォーマットが無効になる
    /// - 2回目の `engine.start()` でフォーマット検証エラー
    ///
    /// **関連**: 再生時のピッチ検出が動作しない問題と関連がある可能性
    ///
    /// **TDD Red Phase**: このテストは現在失敗する（バグの存在を証明）
    func testPlayTwice_shouldNotCrash() throws {
        let app = launchAppWithResetRecordingCount()

        // 1. 録音を作成
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "Home record button should exist")
        homeRecordButton.tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "Start recording button should exist")
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop recording button should appear")

        // 3秒間録音
        Thread.sleep(forTimeInterval: 3.0)
        stopButton.tap()

        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should appear after save")

        // 2. 1回目の再生
        playButton.tap()
        let stopPlaybackButton = app.buttons["StopPlaybackButton"]
        XCTAssertTrue(stopPlaybackButton.waitForExistence(timeout: 5), "Stop playback button should appear during FIRST playback")

        // Screenshot: 1回目の再生中
        let screenshot1 = app.screenshot()
        let attachment1 = XCTAttachment(screenshot: screenshot1)
        attachment1.name = "play_twice_01_first_playback"
        attachment1.lifetime = .keepAlways
        add(attachment1)

        // 再生完了を待つ
        Thread.sleep(forTimeInterval: 5.0)
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should reappear after first playback completes")

        // 3. 2回目の再生 - ここでバグ発生
        playButton.tap()

        // Screenshot: 2回目の再生試行後
        let screenshot2 = app.screenshot()
        let attachment2 = XCTAttachment(screenshot: screenshot2)
        attachment2.name = "play_twice_02_second_playback_attempt"
        attachment2.lifetime = .keepAlways
        add(attachment2)

        // BUG: このアサーションが失敗する（StopPlaybackButtonが表示されない）
        XCTAssertTrue(stopPlaybackButton.waitForExistence(timeout: 5), "Stop playback button should appear during SECOND playback (BUG: fails here)")

        // 2回目の再生完了を待つ
        Thread.sleep(forTimeInterval: 5.0)
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "Play button should reappear after SECOND playback completes")

        // Screenshot: 2回目の再生後
        let screenshot3 = app.screenshot()
        let attachment3 = XCTAttachment(screenshot: screenshot3)
        attachment3.name = "play_twice_03_after_second_playback"
        attachment3.lifetime = .keepAlways
        add(attachment3)
    }

}
