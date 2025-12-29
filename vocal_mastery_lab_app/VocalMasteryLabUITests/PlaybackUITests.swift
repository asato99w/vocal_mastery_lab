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
    /// Note: 再生ボタンはRecordingViewから削除されたため、このテストはスキップ
    func testPlaybackFullCompletion() throws {
        throw XCTSkip("Skipped: 再生ボタンはRecordingViewから削除済み。録音一覧画面での再生機能はRecordingListViewで提供。")
    }

    // MARK: - Bug Reproduction Tests

    /// BUG REPRODUCTION: 2回目の再生が開始されない
    /// Note: 再生ボタンはRecordingViewから削除されたため、このテストはスキップ
    func testPlayTwice_shouldNotCrash() throws {
        throw XCTSkip("Skipped: 再生ボタンはRecordingViewから削除済み。録音一覧画面での再生機能はRecordingListViewで提供。")
    }

}
