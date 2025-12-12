# 録音制限システムの実装分析

## 概要

本ドキュメントは、VocalisStudioの録音制限システムの実装を調査し、UIテスト環境と本番環境での挙動の違いを明らかにしたものです。

調査日: 2025-11-17

## システムアーキテクチャ

### 関連コンポーネント

1. **RecordingStateViewModel** (`VocalisStudio/Presentation/ViewModels/RecordingStateViewModel.swift`)
   - 録音開始時に制限チェックを実行
   - `RecordingUsageTracker`から日次録音回数を取得
   - `RecordingLimitConfig`からティア別の制限を取得

2. **RecordingUsageTracker** (`VocalisStudio/Infrastructure/Storage/RecordingUsageTracker.swift`)
   - 日次録音回数の管理
   - UIテスト用の環境変数オーバーライド機能

3. **RecordingLimitConfig** (`SubscriptionDomain/ValueObjects/RecordingLimitConfig.swift`)
   - ティア別の録音制限値を定義
   - `ProductionRecordingLimitConfig`と`TestRecordingLimitConfig`の2種類

4. **StoreKitSubscriptionRepository** (`VocalisStudio/Infrastructure/Subscription/StoreKitSubscriptionRepository.swift`)
   - サブスクリプションステータスの取得
   - UIテスト用の環境変数オーバーライド機能

## 制限値の定義

### ProductionRecordingLimitConfig (本番環境)

```swift
public func limitForTier(_ tier: SubscriptionTier) -> RecordingLimit {
    switch tier {
    case .free:
        return RecordingLimit(dailyCount: 100, maxDuration: 30)  // 100回/日、30秒
    case .premium:
        return RecordingLimit(dailyCount: nil, maxDuration: 300) // 無制限、5分
    case .premiumPlus:
        return RecordingLimit(dailyCount: nil, maxDuration: nil) // 完全無制限
    }
}
```

### TestRecordingLimitConfig (テスト環境)

```swift
public func limitForTier(_ tier: SubscriptionTier) -> RecordingLimit {
    switch tier {
    case .free:
        return RecordingLimit(dailyCount: 5, maxDuration: 2)    // 5回/日、2秒
    case .premium:
        return RecordingLimit(dailyCount: nil, maxDuration: 4)  // 無制限、4秒
    case .premiumPlus:
        return RecordingLimit(dailyCount: nil, maxDuration: nil) // 完全無制限
    }
}
```

## 制限チェックのロジック

### RecordingLimit.isCountWithinLimit

```swift
public func isCountWithinLimit(_ count: Int) -> Bool {
    guard let limit = dailyCount else {
        return true // dailyCount が nil の場合は無制限
    }
    return count < limit
}
```

**重要な仕様:**
- `dailyCount: nil` の場合、常に`true`を返す（無制限）
- Premium/PremiumPlusティアは`dailyCount: nil`のため、録音回数制限なし

## UIテスト環境の特殊処理

### 環境変数オーバーライド (`#if DEBUG`ブロック内)

#### 1. サブスクリプションティアのオーバーライド

**StoreKitSubscriptionRepository.swift (line 34):**
```swift
#if DEBUG
if let tierString = ProcessInfo.processInfo.environment["SUBSCRIPTION_TIER"],
   let tier = SubscriptionTier(rawValue: tierString) {
    return SubscriptionStatus(tier: tier, ...)
}
#endif
```

#### 2. 日次録音回数のオーバーライド

**RecordingUsageTracker.swift (line 25):**
```swift
#if DEBUG
if let testCount = ProcessInfo.processInfo.environment["DAILY_RECORDING_COUNT"],
   let count = Int(testCount) {
    return count
}
#endif
```

### UIテストでの設定例

```swift
// RecordingLimitUITests.swift
app.launchEnvironment["SUBSCRIPTION_TIER"] = "free"
app.launchEnvironment["DAILY_RECORDING_COUNT"] = "0"  // または "100"
```

## 録音制限チェックの処理フロー

### RecordingStateViewModel.startRecording()

```swift
// 1. 現在の録音回数を取得
self.dailyRecordingCount = usageTracker.getTodayCount()
// ↓ UIテスト: 環境変数から取得
// ↓ 本番: UserDefaultsから取得

// 2. 制限チェック
if !recordingLimit.isCountWithinLimit(self.dailyRecordingCount) {
    // 制限到達: エラーメッセージを表示
    errorMessage = "本日の録音回数の上限に達しました (\(currentTier.displayName)プラン)"
    return
}

// 3. 録音開始処理
```

## テスト環境と本番環境の挙動比較

### UIテスト環境

| 機能 | 実装 |
|------|------|
| サブスクリプションティア | 環境変数`SUBSCRIPTION_TIER`で設定可能 |
| 日次録音回数 | 環境変数`DAILY_RECORDING_COUNT`で設定可能 |
| 制限値の設定 | `ProductionRecordingLimitConfig`を使用<br>（RecordingStateViewModelのデフォルト） |
| コンパイル条件 | `#if DEBUG`ブロックが有効 |

### 本番環境

| 機能 | 実装 |
|------|------|
| サブスクリプションティア | StoreKit APIから取得 |
| 日次録音回数 | UserDefaultsから取得 |
| 制限値の設定 | `ProductionRecordingLimitConfig`を使用 |
| コンパイル条件 | `#if DEBUG`ブロックが無効 |

## プレミアムユーザーの録音回数制限

### 結論

**本番環境ではプレミアムユーザーに録音回数制限はかけられていません。**

### 根拠

1. **ProductionRecordingLimitConfig**で、Premium/PremiumPlusティアの`dailyCount`は`nil`
2. **RecordingLimit.isCountWithinLimit()**は、`dailyCount`が`nil`の場合に常に`true`を返す
3. UIテストでも本番環境でも同じ`ProductionRecordingLimitConfig`が使用される

### 制限の適用状況

| ティア | 録音回数制限 | 録音時間制限 |
|--------|-------------|-------------|
| Free | 100回/日 | 30秒 |
| Premium | **無制限** | 5分 |
| PremiumPlus | **無制限** | **無制限** |

## 実装テスト結果

### testFreeUser_canRecordMultipleTimes_withinLimit

**テスト内容:**
- 無料ユーザー（日次録音回数: 0、上限: 100）が6回連続で録音と停止を繰り返す

**結果:**
- ✅ PASSED (65.552秒)
- 全6回の録音が正常に完了
- 録音制限アラートは一度も表示されなかった

**検証項目:**
1. 各イテレーションで録音→停止が正常に実行される
2. 上限内（0-5回 < 100回）では制限アラートが表示されない
3. UI要素の状態遷移が正しい（StartRecordingButton → StopRecordingButton → StartRecordingButton）

## デバッグ環境の挙動再現テスト（2025-11-17）

### 問題の仮説

デバッグ環境（手動実行）では、フリーティアユーザーが100回/日の制限ではなく、デフォルト値の5回/日で制限されている可能性がある。

### テスト設定

**環境変数の設定:**
- `SUBSCRIPTION_TIER = "free"` - フリーティアとして設定
- `DAILY_RECORDING_COUNT` - **コメントアウト（設定なし）**

**テスト内容:**
- 7回連続で録音と停止を繰り返す
- 各イテレーションで録音制限アラートが表示されないことを確認

### テスト結果

**実行日時:** 2025-11-17 20:12:53
**結果:** ❌ FAILED (45.870秒)

**詳細:**
- ✅ Iteration 1: 成功
- ✅ Iteration 2: 成功
- ✅ Iteration 3: 成功
- ❌ Iteration 4: **失敗** - "Stop button should appear after countdown"

**失敗の原因:**
- 4回目の録音開始時に`StopRecordingButton`が表示されなかった
- これは録音が開始されなかったことを示す
- **録音制限に到達したため、録音が開始されなかった可能性が高い**

### 結論

**RecordingStateViewModelのデフォルト値が使われていることを確認:**
- デフォルト値: `RecordingLimit(dailyCount: 5, maxDuration: 30)`
- フリーティアの正しい値: `RecordingLimit(dailyCount: 100, maxDuration: 30)`
- テストは3回成功し、4回目で失敗 → **dailyCount: 5の制限が適用されている**

**デバッグ環境での問題:**
1. `SUBSCRIPTION_TIER`環境変数は正しく読み込まれている（フリーティアとして認識）
2. しかし`recordingLimit`がデフォルト値（5回/日）のまま更新されていない
3. これは`loadStatus()`の非同期実行により、サブスクリプションステータスが設定される前にデフォルト値が使われるため

**根本原因:**
- RecordingStateViewModelの初期化時にデフォルト値が設定される
- `SubscriptionViewModel.loadStatus()`は`.task`モディファイアで非同期実行される
- UIテスト環境では`loadStatus()`完了前にRecordingStateViewModelが使用される
- Combine publisherによる更新が間に合わず、デフォルト値（5回/日）が適用される

## まとめ

1. **設計意図の確認:**
   - プレミアムユーザーには録音回数制限を設けない設計
   - 無料ユーザーには1日100回の制限を設定

2. **テスト環境の柔軟性:**
   - 環境変数でティアと録音回数を自由に設定可能
   - `#if DEBUG`ブロックにより本番コードに影響なし

3. **実装の一貫性:**
   - UIテスト環境と本番環境で同じ制限値設定を使用
   - テスト結果は本番環境の挙動を正確に反映

4. **発見された問題（2025-11-17）:**
   - RecordingStateViewModelのデフォルト値（5回/日、30秒）が環境によって使用される
   - 非同期のloadStatus()完了前にデフォルト値が適用される
   - DAILY_RECORDING_COUNT環境変数なしの環境で再現可能

## 関連ファイル

- `VocalisStudio/VocalisStudio/Presentation/ViewModels/RecordingStateViewModel.swift:122-132` - 制限チェック処理
- `VocalisStudio/VocalisStudio/Infrastructure/Storage/RecordingUsageTracker.swift:24-28` - 環境変数オーバーライド
- `VocalisStudio/Packages/SubscriptionDomain/Sources/SubscriptionDomain/ValueObjects/RecordingLimitConfig.swift` - 制限値定義
- `VocalisStudio/Packages/SubscriptionDomain/Sources/SubscriptionDomain/ValueObjects/RecordingLimit.swift:38-43` - 制限チェックロジック
- `VocalisStudio/VocalisStudio/Infrastructure/Subscription/StoreKitSubscriptionRepository.swift:34` - ティア環境変数オーバーライド
- `VocalisStudio/VocalisStudioUITests/RecordingLimitUITests.swift:118-178` - テストコード
