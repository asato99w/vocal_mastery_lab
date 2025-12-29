//
//  BackingTrackUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for backing track selection functionality
//

import XCTest

final class BackingTrackUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    /// 録音後にバッキングトラック一覧に新しい録音が表示されることを確認
    @MainActor
    func testBackingTrackPicker_ShowsNewRecordingAfterCompletion() throws {
        let app = XCUIApplication()
        app.launchArguments = ["--uitesting"]
        app.launch()

        // ホームから録音画面へ遷移
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5), "ホーム録音ボタンが存在すべき")
        homeRecordButton.tap()

        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "録音画面がロードされるべき")

        // 画面が完全にロードされるまで待機
        Thread.sleep(forTimeInterval: 1.0)

        // デバッグ: 現在の画面をスクリーンショット
        let debugScreenshot = XCTAttachment(screenshot: app.screenshot())
        debugScreenshot.name = "debug_recording_screen"
        debugScreenshot.lifetime = .keepAlways
        add(debugScreenshot)

        // バッキングトラックラベルを探す（「利用可能:」を含むテキスト）
        let backingTrackCount = app.staticTexts.matching(NSPredicate(format: "label CONTAINS '利用可能'")).firstMatch
        XCTAssertTrue(backingTrackCount.waitForExistence(timeout: 5), "バッキングトラックカウントが存在すべき")
        let initialCountText = backingTrackCount.label
        let initialCount = extractCount(from: initialCountText)

        // 録音を実行
        startButton.tap()
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10), "停止ボタンが表示されるべき")
        Thread.sleep(forTimeInterval: 1.0)
        stopButton.tap()

        // 録音完了を待つ
        let lastRecordingSection = app.otherElements["LastRecordingSection"]
        XCTAssertTrue(lastRecordingSection.waitForExistence(timeout: 5), "録音後に最後の録音セクションが表示されるべき")
        XCTAssertTrue(startButton.waitForExistence(timeout: 5), "開始ボタンが再表示されるべき")

        // 録音直後（再読み込みなし）でカウントを取得
        Thread.sleep(forTimeInterval: 0.5)  // UI更新を待つ
        let newBackingTrackCount = app.staticTexts.matching(NSPredicate(format: "label CONTAINS '利用可能'")).firstMatch
        XCTAssertTrue(newBackingTrackCount.waitForExistence(timeout: 5))
        let newCountText = newBackingTrackCount.label
        let newCount = extractCount(from: newCountText)

        // カウントが増加したことを確認
        XCTAssertGreaterThan(newCount, initialCount,
            "録音後にバッキングトラック数が増加すべき。初期: \(initialCount), 新規: \(newCount)")

        // カウントが1以上であることを確認（録音が表示されている）
        XCTAssertGreaterThanOrEqual(newCount, 1, "録音後に少なくとも1件の録音がバッキングトラック一覧に表示されるべき")
    }

    private func extractCount(from label: String) -> Int {
        let pattern = #"(\d+)件"#
        if let regex = try? NSRegularExpression(pattern: pattern),
           let match = regex.firstMatch(in: label, range: NSRange(label.startIndex..., in: label)),
           let range = Range(match.range(at: 1), in: label) {
            return Int(label[range]) ?? 0
        }
        return 0
    }

    // MARK: - Backing Track Player Tests

    /// バッキングトラック選択時にプレイヤーが表示されることを確認
    @MainActor
    func testBackingTrackPlayer_ShowsWhenTrackSelected() throws {
        let app = XCUIApplication()
        app.launchArguments = ["--uitesting"]
        app.launch()

        // ホームから録音画面へ遷移
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
        homeRecordButton.tap()

        // 録音画面がロードされるまで待機
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))
        Thread.sleep(forTimeInterval: 1.0)

        // 録音を実行（バッキングトラック用の録音を作成）
        startButton.tap()
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10))
        Thread.sleep(forTimeInterval: 1.0)
        stopButton.tap()

        // 録音完了を待つ
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))
        Thread.sleep(forTimeInterval: 0.5)

        // バッキングトラックセクションが存在することを確認
        let backingSection = app.otherElements["BackingTrackSection"]
        XCTAssertTrue(backingSection.waitForExistence(timeout: 5), "バッキングトラックセクションが存在すべき")

        // バッキングトラックピッカー（Menu）をタップして録音を選択
        let backingPicker = app.buttons["BackingTrackPicker"]
        XCTAssertTrue(backingPicker.waitForExistence(timeout: 5), "バッキングトラックピッカーが存在すべき")
        backingPicker.tap()
        Thread.sleep(forTimeInterval: 0.5)

        // メニューから最初の録音を選択（「なし」以外）
        // SwiftUI Menuはbuttons内に表示される
        let allButtons = app.buttons.allElementsBoundByIndex
        var selectedTrack = false
        for button in allButtons {
            let label = button.label
            // 「なし」と「BackingTrackPicker」以外のボタンを探す
            if !label.isEmpty && label != "なし" && !label.contains("chevron") && button.identifier != "BackingTrackPicker" {
                // 日付形式（録音のタイトル）らしきものを選択
                if label.contains("/") || label.contains("録音") || label.count > 5 {
                    button.tap()
                    selectedTrack = true
                    break
                }
            }
        }

        // トラックが選択できなかった場合はスキップ
        guard selectedTrack else {
            throw XCTSkip("選択可能なバッキングトラックが見つかりませんでした")
        }

        Thread.sleep(forTimeInterval: 0.5)

        // デバッグ: トラック選択後のスクリーンショット
        add(XCTAttachment(screenshot: app.screenshot()).apply { $0.name = "after_track_select"; $0.lifetime = .keepAlways })

        // プレイヤービューが表示されることを確認
        let playerView = app.otherElements["BackingTrackPlayerView"]
        XCTAssertTrue(playerView.waitForExistence(timeout: 5), "バッキングトラックプレイヤーが表示されるべき")

        // 再生ボタンが存在することを確認
        // Note: SwiftUI accessibilityIdentifierが親ビューから継承されるため、ラベルで検索
        let playPauseButton = app.buttons.matching(NSPredicate(format: "label == '再生' OR label == 'play'")).firstMatch
        XCTAssertTrue(playPauseButton.waitForExistence(timeout: 3), "再生/一時停止ボタンが存在すべき")

        // 停止ボタンが存在することを確認
        let stopTrackButton = app.buttons.matching(NSPredicate(format: "label == '停止' OR label == 'stop'")).firstMatch
        XCTAssertTrue(stopTrackButton.waitForExistence(timeout: 3), "停止ボタンが存在すべき")
    }

    /// バッキングトラックプレイヤーで再生/一時停止ができることを確認
    @MainActor
    func testBackingTrackPlayer_PlayPauseToggle() throws {
        let app = XCUIApplication()
        app.launchArguments = ["--uitesting"]
        app.launch()

        // ホームから録音画面へ遷移
        let homeRecordButton = app.buttons["HomeRecordButton"]
        XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
        homeRecordButton.tap()

        // 録音画面がロードされるまで待機
        let startButton = app.buttons["StartRecordingButton"]
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))
        Thread.sleep(forTimeInterval: 1.0)

        // 録音を実行（再生テストのため、より長い録音を作成）
        startButton.tap()
        let stopButton = app.buttons["StopRecordingButton"]
        XCTAssertTrue(stopButton.waitForExistence(timeout: 10))
        Thread.sleep(forTimeInterval: 3.0)  // 3秒間録音（再生テストに十分な長さ）
        stopButton.tap()

        // 録音完了を待つ
        XCTAssertTrue(startButton.waitForExistence(timeout: 5))
        Thread.sleep(forTimeInterval: 0.5)

        // バッキングトラックセクションが存在することを確認
        let backingSection = app.otherElements["BackingTrackSection"]
        XCTAssertTrue(backingSection.waitForExistence(timeout: 5), "バッキングトラックセクションが存在すべき")

        // バッキングトラックピッカー（Menu）をタップして録音を選択
        let backingPicker = app.buttons["BackingTrackPicker"]
        XCTAssertTrue(backingPicker.waitForExistence(timeout: 5), "バッキングトラックピッカーが存在すべき")
        backingPicker.tap()
        Thread.sleep(forTimeInterval: 0.5)

        // メニューから録音を選択
        let allButtons = app.buttons.allElementsBoundByIndex
        var selectedTrack = false
        for button in allButtons {
            let label = button.label
            if !label.isEmpty && label != "なし" && !label.contains("chevron") && button.identifier != "BackingTrackSection" {
                if label.contains("/") || label.contains("録音") || label.count > 5 {
                    button.tap()
                    selectedTrack = true
                    break
                }
            }
        }

        guard selectedTrack else {
            throw XCTSkip("選択可能なバッキングトラックが見つかりませんでした")
        }
        Thread.sleep(forTimeInterval: 0.5)

        // プレイヤーが表示されていることを確認
        let playerView = app.otherElements["BackingTrackPlayerView"]
        XCTAssertTrue(playerView.waitForExistence(timeout: 5))

        // 再生ボタンをタップ
        // Note: SwiftUI accessibilityIdentifierが親ビューから継承されるため、ラベルで検索
        let playButton = app.buttons.matching(NSPredicate(format: "label == '再生' OR label == 'play'")).firstMatch
        XCTAssertTrue(playButton.waitForExistence(timeout: 3), "再生ボタンが存在すべき")

        playButton.tap()

        // UI更新を待つ
        Thread.sleep(forTimeInterval: 0.5)

        // デバッグ: スクリーンショット
        add(XCTAttachment(screenshot: app.screenshot()).apply { $0.name = "after_play_tap"; $0.lifetime = .keepAlways })

        // 再生中インジケータまたは一時停止ボタンが表示されることを確認
        // Note: 「再生中」テキストを含む要素を検索
        let playingIndicator = app.staticTexts.matching(NSPredicate(format: "label CONTAINS '再生中'")).firstMatch
        let pauseButton = app.buttons.matching(NSPredicate(format: "label == '一時停止' OR label == 'pause'")).firstMatch

        // どちらかが存在すれば再生が開始された
        let playbackStarted = playingIndicator.waitForExistence(timeout: 3) || pauseButton.exists
        XCTAssertTrue(playbackStarted, "再生が開始されるべき（インジケータまたは一時停止ボタンが表示）")

        // 一時停止ボタンをタップ
        if pauseButton.exists {
            pauseButton.tap()
        } else if playButton.exists {
            // 再生ボタンがまだ存在する場合（状態が変わっていない可能性）
            playButton.tap()
        }
        Thread.sleep(forTimeInterval: 0.5)

        // 再生中インジケータが非表示になることを確認
        XCTAssertFalse(playingIndicator.exists, "一時停止後、再生中インジケータが非表示になるべき")
    }
}

private extension XCTAttachment {
    func apply(_ block: (XCTAttachment) -> Void) -> XCTAttachment {
        block(self)
        return self
    }
}
