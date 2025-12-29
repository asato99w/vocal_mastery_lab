//
//  SimpleRecordingUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for simplified recording screen (UI_DESIGN.md)
//

import XCTest

/// UI tests for recording screen based on UI_DESIGN.md specification
final class SimpleRecordingUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    // MARK: - 1. Initial State

    /// 初期状態: タイマー、録音開始ボタン、バックグラウンドヒントが表示される
    @MainActor
    func testInitialState() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
        homeRecordButton.tap()

        // タイマー「00:00:00」が表示される
        let timer = app.staticTexts["RecordingTimerLabel"]
        XCTAssertTrue(timer.waitForExistence(timeout: 5), "Timer should be visible")
        XCTAssertEqual(timer.label, "00:00:00", "Timer should show 00:00:00 initially")

        // 録音開始ボタンが表示される
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.exists, "Start recording button should be visible")

        // バックグラウンドヒントが表示される
        let hint = app.staticTexts["BackgroundRecordingHint"]
        XCTAssertTrue(hint.exists, "Background recording hint should be visible")
    }

    // MARK: - 2. Recording Flow

    /// 録音フロー: 開始→タイマー進行→停止→最後の録音セクション表示
    @MainActor
    func testRecordingFlow() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen
        app.buttons["HomeRecordButton"].tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))

        // 録音開始
        startButton.tap()

        // 停止ボタンに切り替わる
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "Stop button should appear during recording")

        // タイマーが進む（00:00:00ではなくなる）
        let timer = app.staticTexts["RecordingTimerLabel"]
        Thread.sleep(forTimeInterval: 2.0)
        XCTAssertNotEqual(timer.label, "00:00:00", "Timer should progress during recording")

        // 録音停止
        stopButton.tap()

        // 最後の録音セクションが表示される
        let lastRecordingSection = app.otherElements["LastRecordingSection"]
        XCTAssertTrue(lastRecordingSection.waitForExistence(timeout: 5), "Last recording section should appear after recording")
    }

    // MARK: - 3. Last Recording

    /// 最後の録音: 日付・再生時間表示、再生ボタン、ボーカル抽出ボタン
    @MainActor
    func testLastRecordingSection() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate and create a recording
        app.buttons["HomeRecordButton"].tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10))
        Thread.sleep(forTimeInterval: 1.0)
        stopButton.tap()

        // 最後の録音セクションを確認
        let lastRecordingSection = app.otherElements["LastRecordingSection"]
        XCTAssertTrue(lastRecordingSection.waitForExistence(timeout: 5))

        // 日付が表示される
        let dateLabel = app.staticTexts["LastRecordingDateLabel"]
        XCTAssertTrue(dateLabel.exists, "Recording date should be displayed")

        // 再生時間が表示される
        let durationLabel = app.staticTexts["LastRecordingDurationLabel"]
        XCTAssertTrue(durationLabel.exists, "Recording duration should be displayed")

        // ボーカル抽出ボタンが表示される（再生ボタンはUIから削除済み）
        let vocalButton = app.buttons["VocalExtractionButton"]
        XCTAssertTrue(vocalButton.exists, "Vocal extraction button should be visible")
    }

    /// 再生ボタンをタップすると再生状態になる
    /// Note: 再生ボタンはUIから削除されたため、このテストはスキップ
    @MainActor
    func testPlayback() throws {
        throw XCTSkip("Skipped: 再生ボタンはRecordingViewから削除済み。録音一覧画面での再生機能はRecordingListViewで提供。")
    }

    // MARK: - 4. Navigation

    /// ナビゲーション: 一覧ボタンで録音一覧に遷移
    @MainActor
    func testNavigateToRecordingList() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen
        app.buttons["HomeRecordButton"].tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))

        // 一覧ボタンをタップ
        let listButton = app.buttons["RecordingListButton"]
        XCTAssertTrue(listButton.exists, "Recording list button should be visible")
        listButton.tap()

        // 録音一覧画面に遷移
        let listTitle = app.staticTexts[L10n.listTitle]
        XCTAssertTrue(listTitle.waitForExistence(timeout: 5), "Should navigate to recording list")
    }

    // MARK: - 5. Vocal Extraction Navigation

    /// ボーカル抽出ボタンをタップするとボーカル抽出画面に遷移
    @MainActor
    func testNavigateToVocalExtraction() throws {
        let app = launchAppWithResetRecordingCount()

        // Navigate to Recording screen and create a recording
        app.buttons["HomeRecordButton"].tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))
        startButton.tap()

        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10))
        Thread.sleep(forTimeInterval: 1.0)
        stopButton.tap()

        // ボーカル抽出ボタンをタップ
        let vocalButton = app.buttons["VocalExtractionButton"]
        XCTAssertTrue(vocalButton.waitForExistence(timeout: 5), "Vocal extraction button should be visible")
        vocalButton.tap()

        // ボーカル抽出画面に遷移
        let extractionTitle = app.navigationBars["ボーカル抽出"]
        XCTAssertTrue(extractionTitle.waitForExistence(timeout: 5), "Should navigate to vocal extraction screen")

        // 抽出開始ボタンが表示される
        let startExtractionButton = app.buttons["抽出開始"]
        XCTAssertTrue(startExtractionButton.waitForExistence(timeout: 3), "Start extraction button should be visible")
    }
}
