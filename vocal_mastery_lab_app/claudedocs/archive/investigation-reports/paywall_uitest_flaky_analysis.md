# PaywallUITests Flaky Test 分析レポート

## 概要

2件のUIテストが一時的に失敗したが、再実行では成功した事象についての詳細分析。

## 対象テスト

1. **testDebugMenu_tierSwitch_shouldPersistAcrossScreens()** (Line 276-330)
2. **testPurchaseButton_isAccessible()** (Line 75-82)

## 実行結果

### 個別実行
- testDebugMenu_tierSwitch_shouldPersistAcrossScreens(): ✅ PASSED
- testPurchaseButton_isAccessible(): ✅ PASSED

### バッチ実行
- 12個のPaywallUITestsすべて: ✅ PASSED

### 結論
**Flaky Test（不安定なテスト）** - 環境依存で時々失敗する可能性がある

---

## Flaky Testの原因候補

### 🔴 高確率の原因

#### 1. 固定sleepに依存したタイミング制御 (Critical)

**問題箇所**:
```swift
// PaywallUITests.swift:284
Thread.sleep(forTimeInterval: 0.5)  // Debugメニュー表示待ち

// PaywallUITests.swift:292
Thread.sleep(forTimeInterval: 1)    // ステータス更新待ち

// PaywallUITests.swift:202, 220, etc.
Thread.sleep(forTimeInterval: 0.5)  // 各種画面遷移待ち
```

**なぜFlaky?**:
- シミュレーターのCPU/メモリ状態によって、0.5秒や1秒では不十分な場合がある
- バックグラウンドプロセスの影響でレンダリングが遅延
- 他のテストと並行実行時にリソース競合

**修正案**:
```swift
// ❌ BAD: 固定sleep
Thread.sleep(forTimeInterval: 0.5)
let purchaseButton = app.buttons["購入する"]

// ✅ GOOD: waitForExistence使用
let purchaseButton = app.buttons["購入する"]
XCTAssertTrue(purchaseButton.waitForExistence(timeout: 5))
```

---

#### 2. 古いUI要素への参照の再利用 (testDebugMenu_tierSwitch_shouldPersistAcrossScreens)

**問題箇所**:
```swift
// Line 295: 最初の確認
let currentTierLabel = app.staticTexts.containing(NSPredicate(format: "label CONTAINS[cd] %@", "現在: Premium"))
XCTAssertTrue(currentTierLabel.firstMatch.exists, "Should show current tier as Premium")

// [画面遷移を複数回実行]

// Line 328: 同じ変数を再利用
XCTAssertTrue(currentTierLabel.firstMatch.exists, "Tier should still be Premium in debug menu")
```

**なぜFlaky?**:
- `currentTierLabel`はLine 295で作成された時点のUI要素への参照
- その後、複数回の画面遷移（Home → Settings → Subscription → Settings → Home → Debug）が発生
- Line 328で同じ変数を使っているが、UI階層が再構築されている可能性
- 古い参照が無効になり、存在確認が失敗する可能性

**修正案**:
```swift
// Line 295: 最初の確認
let currentTierLabel = app.staticTexts.containing(NSPredicate(format: "label CONTAINS[cd] %@", "現在: Premium"))
XCTAssertTrue(currentTierLabel.firstMatch.exists, "Should show current tier as Premium")

// [画面遷移]

// Line 328: 新しいクエリで再取得
let currentTierLabelAgain = app.staticTexts.containing(NSPredicate(format: "label CONTAINS[cd] %@", "現在: Premium"))
XCTAssertTrue(currentTierLabelAgain.firstMatch.waitForExistence(timeout: 5), "Tier should still be Premium in debug menu")
```

---

### 🟡 中確率の原因

#### 3. navigateToPaywall()の2つのパスの不安定性

**問題箇所**:
```swift
// Line 195-224
private func navigateToPaywall() {
    // Option 1: Upgrade Banner (timeout: 2秒)
    let upgradeBanner = app.buttons.containing(NSPredicate(format: "label CONTAINS[cd] %@", "無制限録音を解放"))
    if upgradeBanner.firstMatch.waitForExistence(timeout: 2) {
        upgradeBanner.firstMatch.tap()
        Thread.sleep(forTimeInterval: 0.5)
        return
    }

    // Option 2: Debug Menu (timeout: 2秒)
    #if DEBUG
    let debugButton = app.staticTexts["Debug"]
    if debugButton.waitForExistence(timeout: 2) {
        // ...複数のsleepとwaitForExistence
    }
    #endif
}
```

**なぜFlaky?**:
- Option 1が失敗してOption 2にフォールバックする際、合計で約6秒のタイムアウト
- Upgrade Bannerの表示タイミングが環境依存
- Debug Menuパスも複数のステップがあり、各ステップで失敗の可能性

**修正案**:
```swift
private func navigateToPaywall() {
    // DEBUGビルドでは常にDebug Menuを使う（確実性）
    #if DEBUG
    let debugButton = app.staticTexts["Debug"]
    XCTAssertTrue(debugButton.waitForExistence(timeout: 5), "Debug button should exist")
    debugButton.tap()

    let paywallLink = app.buttons.containing(NSPredicate(format: "label CONTAINS[cd] %@", "プレミアムプラン"))
    XCTAssertTrue(paywallLink.firstMatch.waitForExistence(timeout: 5), "Paywall link should exist")
    paywallLink.firstMatch.tap()

    // Paywallが表示されるまで待つ
    let purchaseButton = app.buttons["購入する"]
    XCTAssertTrue(purchaseButton.waitForExistence(timeout: 5), "Paywall should be displayed")
    #else
    // Releaseビルドの場合のみUpgrade Bannerを使う
    let upgradeBanner = app.buttons.containing(NSPredicate(format: "label CONTAINS[cd] %@", "無制限録音を解放"))
    XCTAssertTrue(upgradeBanner.firstMatch.waitForExistence(timeout: 5))
    upgradeBanner.firstMatch.tap()
    #endif
}
```

---

#### 4. testPurchaseButton_isAccessible()の即座の要素検証

**問題箇所**:
```swift
// Line 75-82
func testPurchaseButton_isAccessible() throws {
    navigateToPaywall()  // 内部で0.5秒sleep

    // すぐに要素を検証
    let purchaseButton = app.buttons["購入する"]
    XCTAssertTrue(purchaseButton.exists, "Purchase button should exist")
    XCTAssertTrue(purchaseButton.isEnabled, "Purchase button should be enabled")
}
```

**なぜFlaky?**:
- `navigateToPaywall()`が完了してもPaywallのレンダリングが完了していない可能性
- SwiftUIのシート表示アニメーションが完了する前にボタンを探している

**修正案**:
```swift
func testPurchaseButton_isAccessible() throws {
    navigateToPaywall()

    // Paywallの表示を確実に待つ
    let purchaseButton = app.buttons["購入する"]
    XCTAssertTrue(purchaseButton.waitForExistence(timeout: 5), "Purchase button should exist")
    XCTAssertTrue(purchaseButton.isEnabled, "Purchase button should be enabled")
}
```

---

### 🟢 低確率の原因

#### 5. テスト間の状態汚染

**可能性**:
- 前のテストが完全にクリーンアップされていない
- アラートやシートが残っている
- ナビゲーションスタックが正しくリセットされていない

**検証方法**:
```swift
override func setUpWithError() throws {
    continueAfterFailure = false
    app = XCUIApplication()
    app.launchArguments = ["UI-Testing", "--reset-state"]  // 状態リセット引数を追加
    app.launch()

    // ホーム画面に戻っていることを確認
    let homeSettingsButton = app.buttons["HomeSettingsButton"]
    XCTAssertTrue(homeSettingsButton.waitForExistence(timeout: 5), "Should start at home screen")
}
```

---

#### 6. NSPredicateクエリのパフォーマンス

**問題箇所**:
```swift
// 複数箇所で使用
let currentTierLabel = app.staticTexts.containing(NSPredicate(format: "label CONTAINS[cd] %@", "現在: Premium"))
```

**なぜFlaky?**:
- `CONTAINS[cd]`は大文字小文字を区別しない部分一致検索
- UI階層が深い場合、検索に時間がかかる
- `firstMatch`を使っているが、見つかるまでの時間が不定

**修正案**:
```swift
// Accessibility Identifierを使う（より高速）
// PaywallView.swift側:
Text("現在: \(tier)")
    .accessibilityIdentifier("CurrentTierLabel")

// テストコード:
let currentTierLabel = app.staticTexts["CurrentTierLabel"]
XCTAssertTrue(currentTierLabel.waitForExistence(timeout: 5))
```

---

## 修正優先順位

### 🔴 最優先（今すぐ修正すべき）

1. **固定sleepをwaitForExistence()に置き換え**
   - 影響範囲：全テスト
   - 修正難易度：低
   - 効果：大

2. **testDebugMenu_tierSwitch_shouldPersistAcrossScreensの古い参照を修正**
   - 影響範囲：1テスト
   - 修正難易度：低
   - 効果：中

### 🟡 中優先（時間があれば修正）

3. **navigateToPaywall()のロジックを単純化**
   - 影響範囲：複数テスト
   - 修正難易度：中
   - 効果：中

4. **Accessibility Identifierの追加**
   - 影響範囲：全テスト
   - 修正難易度：中（本体コードも変更必要）
   - 効果：大（長期的）

### 🟢 低優先（様子見）

5. **テスト間の状態リセット強化**
   - 影響範囲：全テスト
   - 修正難易度：低
   - 効果：小（問題が頻発した場合のみ）

---

## 推奨アクション

### すぐに実施すべき修正

```swift
// 1. testPurchaseButton_isAccessible()
func testPurchaseButton_isAccessible() throws {
    navigateToPaywall()

    let purchaseButton = app.buttons["購入する"]
    XCTAssertTrue(purchaseButton.waitForExistence(timeout: 5), "Purchase button should exist")
    XCTAssertTrue(purchaseButton.isEnabled, "Purchase button should be enabled")
}

// 2. testDebugMenu_tierSwitch_shouldPersistAcrossScreens()
// Line 328付近を修正
// 古い参照を使わず、新しいクエリを実行
let currentTierLabelAgain = app.staticTexts.containing(NSPredicate(format: "label CONTAINS[cd] %@", "現在: Premium"))
XCTAssertTrue(currentTierLabelAgain.firstMatch.waitForExistence(timeout: 5), "Tier should still be Premium")

// 3. navigateToPaywall()のsleepをwaitに置き換え
private func navigateToPaywall() {
    #if DEBUG
    let debugButton = app.staticTexts["Debug"]
    XCTAssertTrue(debugButton.waitForExistence(timeout: 5))
    debugButton.tap()

    let paywallLink = app.buttons.containing(NSPredicate(format: "label CONTAINS[cd] %@", "プレミアムプラン"))
    XCTAssertTrue(paywallLink.firstMatch.waitForExistence(timeout: 5))
    paywallLink.firstMatch.tap()

    // Paywallが表示されるまで待つ
    let purchaseButton = app.buttons["購入する"]
    XCTAssertTrue(purchaseButton.waitForExistence(timeout: 5))
    #endif
}
```

---

## モニタリング計画

### 短期（今後1週間）
- 修正なしで様子見
- 失敗発生時のログを収集
- 失敗パターンの特定

### 中期（修正実施後）
- 修正を適用し、10回以上の連続テスト実行で検証
- 失敗率を測定（目標：0%）

### 長期
- CIパイプラインでの失敗率モニタリング
- 新しいFlaky Testの早期検出

---

## 結論

**現時点の判断**：
- 2件のテストは**Flaky Test**である可能性が高い
- 主な原因は**固定sleepへの依存**と**古いUI要素参照の再利用**
- 修正は容易だが、現時点では**様子見推奨**（再発頻度が不明）

**次のアクション**：
1. 今後同じテストが失敗した場合は、すぐに修正を実施
2. 頻発する場合（週1回以上）は、全テストのsleep→waitForExistence変換を実施
3. CIパイプラインでのテスト実行回数を増やし、Flaky率を測定
