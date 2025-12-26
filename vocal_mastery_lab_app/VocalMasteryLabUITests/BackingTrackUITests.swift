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
        let playButton = app.buttons["PlayLastRecordingButton"]
        XCTAssertTrue(playButton.waitForExistence(timeout: 5), "録音後に再生ボタンが表示されるべき")
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
}

private extension XCTAttachment {
    func apply(_ block: (XCTAttachment) -> Void) -> XCTAttachment {
        block(self)
        return self
    }
}
