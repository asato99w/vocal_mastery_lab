# 失敗しているユニットテストの調査

**作成日**: 2025-11-07
**ステータス**: 調査中
**対象**: VocalisStudioTests (ユニットテスト)

## 概要

Phase 6のログによると、VocalisStudioTests全体で266テスト中13テストが失敗していることが確認されました。これらは今回の録音リストUI再設計とは無関係の既存の失敗テストです。

## 失敗テストの分類

### 1. SubscriptionViewModelDebugTests (3件失敗)

**カテゴリ**: Subscription/課金関連
**影響範囲**: デバッグ機能のテスト

**失敗テスト**:
- 詳細調査中

---

### 2. PurchaseSubscriptionUseCaseTests (2件失敗)

**カテゴリ**: Subscription/課金関連
**影響範囲**: 購入フローのユースケース

**失敗テスト**:
- 詳細調査中

---

### 3. SubscriptionViewModelTests (2件失敗)

**カテゴリ**: Subscription/課金関連
**影響範囲**: サブスクリプションViewModel

**失敗テスト**:
- 詳細調査中

---

### 4. SyntheticScaleEvaluationTests (6件失敗)

**カテゴリ**: 音階評価機能
**影響範囲**: スケール評価ロジック

**失敗テスト**:
- 詳細調査中

---

## 調査計画

### Phase 1: 失敗テストの特定
1. ✅ 全ユニットテストを実行
2. ⏳ 失敗テストの具体的なテスト名を取得
3. ⏳ 失敗理由（アサーションエラー）を抽出

### Phase 2: 詳細分析
各失敗テストグループについて:
1. テストコードの確認
2. テスト対象コードの確認
3. 失敗理由の分析
4. 影響範囲の特定

### Phase 3: 優先度付け
- 🔴 Critical: プロダクション機能に影響
- 🟡 Important: デバッグ/開発機能に影響
- 🟢 Low: 孤立した機能/実験的機能

### Phase 4: 修正アプローチ
各テストについて修正方針を決定

---

## 進捗状況

### 現在の調査状況
- [x] Phase 6ログの確認
- [x] テストファイルの構造確認
- [x] 全4テストファイルのコードレビュー完了
  - SubscriptionViewModelDebugTests: 9テスト（2つはprint文のみ）
  - PurchaseSubscriptionUseCaseTests: 9テスト
  - SubscriptionViewModelTests: 12テスト
  - SyntheticScaleEvaluationTests: 7テスト（1つは無効化）
- [x] 潜在的な失敗理由の仮説を立案
- [ ] 実テスト実行による具体的な失敗箇所の特定（シミュレータの問題により保留中）
- [ ] 修正優先度の決定
- [ ] 修正方針の決定

### 次のステップ

**優先度1: 実テスト実行環境の確立**
1. シミュレータの問題を解決（クローン作成ループの停止）
2. Xcode IDEでの直接実行を試行
3. 具体的なエラーメッセージとスタックトレースを取得

**優先度2: 各カテゴリの詳細調査**
1. **SyntheticScaleEvaluationTests** (6件失敗) - 最多失敗
   - `RealtimePitchDetector`の実装確認
   - ピッチ検出アルゴリズムのデバッグ
   - 非同期処理のタイミング検証
2. **SubscriptionViewModelDebugTests** (3件失敗)
   - デバッグティア設定ロジックの確認
   - DependencyContainerのシングルトン動作検証
3. **PurchaseSubscriptionUseCaseTests** (2件失敗)
   - エラーハンドリングロジックの確認
   - モックオブジェクトの動作検証
4. **SubscriptionViewModelTests** (2件失敗)
   - 非同期処理とCombineの連携確認
   - ViewModelの状態管理検証

**優先度3: 修正実施**
- 各失敗テストに対してTDDサイクル（Red → Green → Refactor）で修正
- 既存テストが全てパスすることを確認しながら進める

---

## 注意事項

- **今回の実装への影響**: RecordingListViewModel関連の17テストはすべてパス。今回の実装は既存の失敗テストに影響していない。
- **既存の失敗**: これらのテストは今回の実装以前から失敗している可能性が高い。
- **修正範囲**: 各テストカテゴリは独立しているため、個別に修正可能。

---

## 調査結果サマリー

### 失敗テスト全体像

| テストスイート | 失敗数 | 総数 | 失敗率 | 調査状況 | 重要度 |
|--------------|--------|------|--------|----------|--------|
| SyntheticScaleEvaluationTests | 6件 | 7件 | 85.7% | ✅ 完了 | 🟡 Important |
| SubscriptionViewModelDebugTests | 3件 | 9件 | 33.3% | ⏳ 保留 | 🟢 Low |
| PurchaseSubscriptionUseCaseTests | 2件 | 9件 | 22.2% | ✅ 完了 | 🟡 Important |
| SubscriptionViewModelTests | 2件 | 12件 | 16.7% | ⏳ 保留 | 🟡 Important |
| **合計** | **13件** | **37件** | **35.1%** | **2/4完了** | - |

### 調査完了項目 (2025-11-07 17:40)

#### ✅ PurchaseSubscriptionUseCaseTests (2件)
- **根本原因**: Product IDのドメイン名不一致
- **詳細**: `com.kazuasato.VocalisStudio.*` vs `com.vocalisstudio.*`
- **修正方針**: 修正保留（ユーザー指示）
- **修正箇所**: `SubscriptionTier.swift` の productId プロパティ

#### ✅ SyntheticScaleEvaluationTests (6件)
- **根本原因**: `RealtimePitchDetector.analyzePitchFromFile` の MainActor コンテキスト問題
- **詳細**: メソッドレベルで `@MainActor` 指定がなく、内部 Task が MainActor コンテキストを離脱
- **修正方針**: 修正保留（ユーザー指示）
- **修正箇所**: `RealtimePitchDetector.swift:492` の `analyzePitchFromFile` メソッド

### 調査保留項目

#### ⏳ SubscriptionViewModelTests (2件)
- 詳細なテスト実行が必要
- 非同期処理またはアクセス権限テストの失敗と推測

#### ⏳ SubscriptionViewModelDebugTests (3件)
- 詳細なテスト実行が必要
- デバッグティア設定ロジックの問題と推測

### 推定される主要な問題領域

1. **ピッチ検出機能** (SyntheticScaleEvaluationTests)
   - `RealtimePitchDetector`の実装問題の可能性
   - 合成オーディオからのピッチ検出が85.7%失敗
   - アプリの中核機能に直結するため優先度高

2. **サブスクリプション機能** (3つのテストスイート)
   - デバッグ機能、購入処理、ViewModel状態管理の各レイヤーで失敗
   - 課金機能全体の健全性に疑問
   - ビジネスクリティカルな機能

3. **非同期処理とMainActor**
   - 複数のテストで`@MainActor`や`async/await`を使用
   - タイミング問題や同期処理の問題の可能性

### 推奨される調査順序

**Phase 1: 環境問題の解決** (最優先)
- シミュレータクローン問題の解決
- Xcode IDEでのテスト実行環境確立
- 具体的なエラーメッセージとスタックトレースの取得

**Phase 2: ピッチ検出機能** (中核機能)
1. SyntheticScaleEvaluationTests (6件) - 最多失敗
2. RealtimePitchDetectorの実装確認

**Phase 3: サブスクリプション機能** (ビジネスクリティカル)
1. PurchaseSubscriptionUseCaseTests (2件) - Use Caseレイヤー
2. SubscriptionViewModelTests (2件) - Presentationレイヤー
3. SubscriptionViewModelDebugTests (3件) - デバッグ機能

---

## 詳細調査結果 (順次更新)

### SubscriptionViewModelDebugTests

**ファイルパス**: `VocalisStudioTests/Presentation/ViewModels/SubscriptionViewModelDebugTests.swift`

**テスト構成** (全9テスト):
1. `testSetDebugTier_Free_SetsCorrectStatus()` - Freeティアの設定テスト
2. `testSetDebugTier_Premium_SetsCorrectStatus()` - Premiumティアの設定テスト
3. `testSetDebugTier_PremiumPlus_SetsCorrectStatus()` - PremiumPlusティアの設定テスト
4. `testSetDebugTier_MultipleTimes_RetainsLatestValue()` - 複数回設定時の動作テスト
5. `testSetDebugTier_PublishesChanges()` - Combine publisherの動作テスト
6. `testLoadStatus_OverridesDebugTier()` - loadStatus()がデバッグティアを上書きするかテスト
7. `testDependencyContainer_ReturnsSameInstance()` - DependencyContainerのシングルトン動作テスト
8. `testDependencyContainer_DebugTierPersistsAcrossAccess()` - デバッグティアの永続性テスト
9. `testPrintDebugInfo_ViewModelState()` - デバッグ情報出力（print文のみ、アサーションなし）
10. `testPrintDebugInfo_DependencyContainerSingleton()` - シングルトンデバッグ情報出力（print文のみ、アサーションなし）

**Phase 6での失敗数**: 3件

**推定される失敗テスト**:
- テスト9,10はprint文のみでアサーションがないため失敗しないはず
- 残り8テスト中3テストが失敗 → 詳細は実行結果待ち

### PurchaseSubscriptionUseCaseTests

**ファイルパス**: `VocalisStudioTests/Application/UseCases/PurchaseSubscriptionUseCaseTests.swift`

**テスト構成** (全9テスト):
1. `testExecutePurchasesPremiumSubscription()` - Premiumティアの購入テスト
2. `testExecutePurchasesPremiumPlusSubscription()` - PremiumPlusティアの購入テスト
3. `testExecuteCannotPurchaseFreeTier()` - Freeティアの購入エラーテスト（購入不可）
4. `testExecuteThrowsErrorWhenProductNotFound()` - 商品が見つからない場合のエラーハンドリング
5. `testExecuteThrowsErrorWhenPurchaseFails()` - 購入失敗時のエラーハンドリング
6. `testExecuteThrowsErrorWhenNetworkFails()` - ネットワークエラー時のエラーハンドリング
7. ❌ `testExecutePassesCorrectProductIdForPremium()` - Premium購入時の正しいproductId渡しテスト
8. ❌ `testExecutePassesCorrectProductIdForPremiumPlus()` - PremiumPlus購入時の正しいproductId渡しテスト
9. `testExecuteUpdatesSubscriptionStatusAfterPurchase()` - 購入後のサブスクリプションステータス更新テスト

**Phase 6での失敗数**: 2件

**調査結果 (2025-11-07 17:35)**: ✅ 根本原因特定完了

**失敗テスト**:
- テスト7: `testExecutePassesCorrectProductIdForPremium()` (0.002秒で失敗)
- テスト8: `testExecutePassesCorrectProductIdForPremiumPlus()` (0.750秒で失敗)

**根本原因**: **Product IDのドメイン名不一致**

**実装側** (`SubscriptionTier.swift:22, 24`):
```swift
case .premium:
    return "com.kazuasato.VocalisStudio.premium.monthly"
case .premiumPlus:
    return "com.kazuasato.VocalisStudio.premiumplus.monthly"
```

**テスト期待値** (`PurchaseSubscriptionUseCaseTests.swift:135, 149`):
```swift
// Line 135: Premium
XCTAssertEqual(mockRepository.lastPurchasedProductId, "com.vocalisstudio.premium.monthly")

// Line 149: PremiumPlus
XCTAssertEqual(mockRepository.lastPurchasedProductId, "com.vocalisstudio.premiumplus.monthly")
```

**不一致の詳細**:
- 実装: `com.kazuasato.VocalisStudio.***` (大文字V, 開発者名kazuasato)
- テスト: `com.vocalisstudio.***` (小文字、開発者名なし)

**影響範囲**:
- 月額Product ID: `.premium.monthly`, `.premiumplus.monthly`
- 年額Product ID: `.premium.yearly`, `.premiumplus.yearly` も同様の不一致がある可能性

**修正方針**: 🔴 修正保留 (ユーザー指示)
- Product IDは `vocalisstudio` (小文字、開発者名なし) に統一すべき
- 実装側の `SubscriptionTier.swift` を修正する必要がある
- **ただし、ユーザーの指示により修正は保留し、ドキュメント記載のみとする**

**修正が必要な箇所**:
1. `SubscriptionTier.swift:22` - Premium月額
2. `SubscriptionTier.swift:24` - PremiumPlus月額
3. `SubscriptionTier.swift:70` - Premium年額 (yearlyProductIdも要確認)
4. `SubscriptionTier.swift:72` - PremiumPlus年額 (yearlyProductIdも要確認)

**テストの構造**:
```swift
final class PurchaseSubscriptionUseCaseTests: XCTestCase {
    var useCase: PurchaseSubscriptionUseCase!
    var mockRepository: MockSubscriptionRepository!

    // 成功ケース
    func testExecutePurchasesPremiumSubscription() async throws {
        let tier = SubscriptionTier.premium
        mockRepository.mockPurchaseResult = .success(())
        try await useCase.execute(tier: tier)
        XCTAssertEqual(mockRepository.lastPurchasedProductId, tier.productId)
    }

    // エラーケース
    func testExecuteCannotPurchaseFreeTier() async {
        do {
            try await useCase.execute(tier: .free)
            XCTFail("Should have thrown error")
        } catch let error as PurchaseError {
            XCTAssertEqual(error, PurchaseError.cannotPurchaseFreeTier)
        }
    }
}
```

**潜在的な失敗理由**:
1. **`PurchaseSubscriptionUseCase`の実装問題**: エラーハンドリングが不完全
2. **`MockSubscriptionRepository`の問題**: モックの動作が期待と異なる
3. **エラー型の不一致**: `PurchaseError`の定義や使用方法の問題
4. **async/await処理**: 非同期処理のエラーハンドリングの問題
5. **ステータス更新ロジック**: 購入後のステータス反映が正しく動作していない

**重要度**: 🟡 Important
- 課金機能はビジネスクリティカル
- ユーザーの購入体験に直結
- エラーハンドリングの正確性が重要

### SubscriptionViewModelTests

**ファイルパス**: `VocalisStudioTests/Presentation/ViewModels/SubscriptionViewModelTests.swift`

**テスト構成** (全12テスト):
1. `testLoadStatusSuccessUpdatesCurrentStatus()` - ステータス読み込み成功時の更新テスト (line 46)
2. `testLoadStatusFailureSetsErrorMessage()` - ステータス読み込み失敗時のエラーメッセージ設定テスト (line 61)
3. `testLoadStatusSetsLoadingStateDuringExecution()` - ステータス読み込み中のローディング状態テスト (line 74)
4. `testPurchaseSuccessRefreshesStatus()` - 購入成功時のステータス更新テスト (line 93)
5. `testPurchaseFailureSetsErrorMessage()` - 購入失敗時のエラーメッセージ設定テスト (line 120)
6. `testPurchaseSetsLoadingStateDuringExecution()` - 購入中のローディング状態テスト (line 132)
7. `testRestoreSuccessRefreshesStatus()` - 復元成功時のステータス更新テスト (line 151)
8. `testRestoreFailureSetsErrorMessage()` - 復元失敗時のエラーメッセージ設定テスト (line 178)
9. `testHasAccessToFeatureReturnsTrueForAllowedFeature()` - 許可された機能へのアクセステスト (line 192)
10. `testHasAccessToFeatureReturnsFalseForRestrictedFeature()` - 制限された機能へのアクセステスト (line 212)
11. `testHasAccessToFeatureReturnsTrueForGrandfatherUser()` - グランドファザーユーザーのアクセステスト (line 225)
12. `testClearErrorMessageResetsError()` - エラーメッセージクリアテスト (line 242)

**Phase 6での失敗数**: 2件

**推定される失敗テスト**:
- 12テスト中2テストが失敗
- 非同期処理のテスト（1-8）かアクセス権限テスト（9-11）の可能性
- 詳細は実行結果待ち

**テストの構造**:
```swift
@MainActor
final class SubscriptionViewModelTests: XCTestCase {
    var sut: SubscriptionViewModel!
    var mockGetStatusUseCase: MockGetSubscriptionStatusUseCase!
    var mockPurchaseUseCase: MockPurchaseSubscriptionUseCase!
    var mockRestoreUseCase: MockRestorePurchasesUseCase!

    // ステータス読み込みテスト
    func testLoadStatusSuccessUpdatesCurrentStatus() async {
        // Given: 成功するモックステータス
        let expectedStatus = SubscriptionStatus(...)
        mockGetStatusUseCase.mockResult = .success(expectedStatus)

        // When: ステータス読み込み
        await sut.loadStatus()

        // Then: currentStatusが更新されているはず
        XCTAssertEqual(sut.currentStatus, expectedStatus)
        XCTAssertNil(sut.errorMessage)
    }

    // 機能アクセステスト
    func testHasAccessToFeatureReturnsTrueForAllowedFeature() {
        // Given: Premium status
        sut.currentStatus = SubscriptionStatus(tier: .premium, ...)

        // When/Then: Premium機能にアクセス可能
        XCTAssertTrue(sut.hasAccessToFeature(.unlimitedRecordings))
    }
}
```

**潜在的な失敗理由**:
1. **`SubscriptionViewModel`の状態管理問題**: Published propertiesの更新タイミング
2. **非同期処理の同期問題**: async/awaitとCombineの連携
3. **モックの不一致**: UseCase mockの返り値とViewModelの期待値の不整合
4. **@MainActor関連**: メインスレッド実行要件の問題
5. **機能アクセスロジック**: `hasAccessToFeature`メソッドの実装問題

**重要度**: 🟡 Important
- ViewModelはUIとビジネスロジックの橋渡し
- 課金機能のUI表示ロジックに直結
- エラー処理とローディング状態の正確性が重要

### SyntheticScaleEvaluationTests

**ファイルパス**: `VocalisStudioTests/Infrastructure/Audio/SyntheticScaleEvaluationTests.swift`

**テスト構成** (全7テスト):
1. `testSimple()` - テストクラス動作確認（常にパス）
2. `testC4Single()` - C4音(261.63 Hz)の単音テスト
3. `testC4_BaselineEvaluation()` - C4 MIDI 60のベースライン評価
4. `testD4_BaselineEvaluation()` - D4 MIDI 62のベースライン評価
5. `testE4_BaselineEvaluation()` - E4 MIDI 64のベースライン評価
6. `testF4_BaselineEvaluation()` - F4 MIDI 65のベースライン評価
7. `testG4_BaselineEvaluation()` - G4 MIDI 67のベースライン評価
8. `DISABLED_testBaselineEvaluation_CurrentParameters()` - 無効化されたテスト（実行されない）

**Phase 6での失敗数**: 6件

**推定される失敗テスト**:
- `testSimple()`は常にパス（アサーションがtrue）
- `DISABLED_`プレフィックスのテストは実行されない
- 残り6テスト（testC4Single + 5つのベースライン評価）が全て失敗している可能性が高い

**テストの目的**:
- 合成オーディオ（sine wave + harmonics）を生成してピッチ検出精度を評価
- `RealtimePitchDetector`の精度検証
- 目標: 50セント（半音の半分）以内の精度、0.5以上の信頼度

**テストの構造**:
```swift
// 1. RealtimePitchDetectorを初期化（MainActor）
let pitchDetector = await MainActor.run {
    return RealtimePitchDetector()
}

// 2. MIDI音階から期待周波数を計算
let note = try MIDINote(60) // C4
let expectedFreq = note.frequency // 261.63 Hz

// 3. 合成オーディオファイルを生成（1秒、sine + harmonics）
let fileURL = try createTestAudioFile(duration: 1.0, frequency: expectedFreq)

// 4. ピッチ検出を実行（0.1秒位置で解析）
pitchDetector.analyzePitchFromFile(fileURL, atTime: 0.1) { pitch in
    detectedPitch = pitch
}

// 5. 結果を検証
XCTAssertNotNil(detectedPitch)
XCTAssertLessThan(errorCents, 50.0) // 50セント以内
XCTAssertGreaterThan(detected.confidence, 0.5) // 信頼度0.5以上
```

**潜在的な失敗理由**:
1. **`RealtimePitchDetector`の実装問題**: ピッチ検出アルゴリズムが正しく動作していない
2. **`analyzePitchFromFile`メソッドの問題**: ファイルからの読み取りやコールバックが機能していない
3. **非同期処理のタイミング問題**: 10秒タイムアウト内に処理が完了していない
4. **AVFoundation依存**: オーディオファイル処理の問題
5. **MainActor関連**: メインスレッドでの実行要件の問題

**重要度**: 🟡 Important
- プロダクション機能（ピッチモニタリング）に直結
- 正確なピッチ検出はアプリの中核機能
- ただし、合成オーディオテストなので実音声での動作は別途検証が必要
