# StoreKit Testing ガイド

このディレクトリには、ローカルでのサブスクリプション購入テスト用のファイルが含まれています。

## セットアップ

### 1. StoreKit Configuration ファイル

`Configuration.storekit` がプロジェクトに追加されています。このファイルには以下が定義されています:

- **月額プラン**: ¥480/月 (`com.kazuasato.VocalisStudio.premium.monthly`)
- **年額プラン**: ¥4,800/年 (`com.kazuasato.VocalisStudio.premium.yearly`)

### 2. Xcodeスキーム設定

StoreKit Testingを有効にするには:

1. Xcode で **Product → Scheme → Edit Scheme...** を選択
2. **Run** を選択
3. **Options** タブを開く
4. **StoreKit Configuration** で `Configuration.storekit` を選択
5. Close

これにより、アプリ実行時にStoreKitテスト環境が使用されます。

## テストの実行

### Unit Tests (StoreKitTest使用)

```bash
# すべてのStoreKitテストを実行
xcodebuild test -project VocalisStudio.xcodeproj -scheme VocalisStudio \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  -only-testing:VocalisStudioTests/StoreKitPurchaseTests

# 特定のテストを実行
xcodebuild test -project VocalisStudio.xcodeproj -scheme VocalisStudio \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  -only-testing:VocalisStudioTests/StoreKitPurchaseTests/testPurchaseMonthlySubscription_shouldUnlockPremium
```

### UI Tests

```bash
# PaywallのUIテストを実行
xcodebuild test -project VocalisStudio.xcodeproj -scheme VocalisStudio \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  -only-testing:VocalisStudioUITests/PaywallUITests
```

## 手動テスト (Transaction Inspector使用)

### アプリでの手動テスト手順

1. **アプリを起動** (StoreKit Configurationが設定されたスキームで)
2. **Xcodeメニュー**: Debug → StoreKit → Manage Transactions
3. **Transaction Inspector** が開く

#### Transaction Inspectorでできること:

- **購入の承認/却下**: テスト購入をシミュレート
- **サブスクリプションの更新**: 即座に次の請求期間に進める
- **サブスクリプションの期限切れ**: 有効期限を即座に切らす
- **更新間隔の変更**: 実時間での更新速度を調整
  - 1秒 = 1ヶ月（デフォルト）
  - 1分 = 1ヶ月
  - 1時間 = 1ヶ月
- **返金**: 購入の返金をシミュレート
- **自動更新の無効化**: 次回更新をキャンセル

### テストシナリオ例

#### 1. 購入フロー
```
1. アプリを起動
2. ホーム → 設定 → サブスクリプション管理
3. 「購入する」をタップ
4. Transaction Inspector で購入を承認
5. UIがプレミアムに切り替わることを確認
```

#### 2. 復元フロー
```
1. Transaction Inspector で既存の購入を削除
2. 「購入の復元」をタップ
3. プレミアムアクセスが復元されることを確認
```

#### 3. 期限切れテスト
```
1. サブスクリプションを購入
2. Transaction Inspector で「Expire Subscription」を実行
3. UIが無料版に戻ることを確認
```

#### 4. 自動更新テスト
```
1. サブスクリプションを購入
2. Transaction Inspector で時間速度を「1秒 = 1ヶ月」に設定
3. 2秒待つ
4. 自動的に更新されることを確認
```

#### 5. キャンセルテスト
```
1. Transaction Inspector で自動更新を無効化
2. 期限まではアクセス可能なことを確認
3. 期限後に無料版に戻ることを確認
```

## テストケース一覧

### StoreKitPurchaseTests.swift

| テスト | 内容 | 確認項目 |
|--------|------|----------|
| `testPurchaseMonthlySubscription_shouldUnlockPremium` | 月額購入 | プレミアムへのアップグレード |
| `testPurchase_shouldUpdateUIImmediately` | UI即時更新 | 購入後のUI反映速度 |
| `testRestorePurchases_withExistingPurchase_shouldRestoreAccess` | 購入復元 | 再インストール後の復元 |
| `testRestorePurchases_withoutPurchase_shouldRemainFree` | 復元失敗 | 未購入時の復元動作 |
| `testSubscriptionExpiration_shouldRevertToFree` | 期限切れ | 無料版への復帰 |
| `testCancelSubscription_beforeRenewal_shouldContinueUntilExpiration` | キャンセル | 期限までのアクセス維持 |
| `testPurchase_whenNetworkError_shouldShowError` | ネットワークエラー | エラーメッセージ表示 |
| `testPurchase_whenUserCancels_shouldNotShowError` | ユーザーキャンセル | キャンセル時の動作 |
| `testRecordingLimit_afterPurchase_shouldBeUnlimited` | 録音制限解除 | プレミアム特典の確認 |
| `testUpgrade_fromMonthlyToYearly_shouldReplaceSubscription` | プラン変更 | 月額→年額への切り替え |
| `testAutoRenewal_whenEnabled_shouldRenewAutomatically` | 自動更新 | 自動更新の動作確認 |

### PaywallUITests.swift

| テスト | 内容 | 確認項目 |
|--------|------|----------|
| `testPaywallDisplay_showsCorrectPricing` | 価格表示 | 無料版・プレミアム版の価格 |
| `testPaywallDisplay_showsTermsAndPrivacy` | 規約リンク | 利用規約・プライバシーポリシー |
| `testRecordingLimitReached_showsPaywall` | 制限到達 | Paywall表示の導線 |
| `testPurchaseButton_isAccessible` | 購入ボタン | ボタンの存在と有効性 |
| `testRestoreButton_isAccessible` | 復元ボタン | ボタンの存在 |
| `testSettings_hasSubscriptionLink` | 設定画面 | サブスク管理への導線 |
| `testSettings_hasTermsAndPrivacyLinks` | 規約リンク（設定） | 設定画面からのアクセス |
| `testSubscriptionManagement_showsCurrentPlan` | プラン表示 | 現在のプラン状態 |
| `testSubscriptionManagement_hasCancelLink` | 解約リンク | 解約導線の存在 |
| `testPaywall_isAccessible` | アクセシビリティ | UI要素の識別子 |

## トラブルシューティング

### StoreKitが有効にならない

1. **Scheme設定を確認**: Edit Scheme → Options → StoreKit Configuration
2. **クリーンビルド**: Product → Clean Build Folder (Cmd+Shift+K)
3. **Simulatorをリセット**: Device → Erase All Content and Settings

### テストが失敗する

1. **Transaction Inspectorでトランザクションをクリア**
2. **テスト前に `session.clearTransactions()` が呼ばれているか確認**
3. **StoreKit Configurationファイルのproduct IDが正しいか確認**

### 購入が完了しない

1. **Transaction Inspectorを確認**: 購入が保留中になっていないか
2. **ログを確認**: Xcodeコンソールでエラーメッセージを確認
3. **ネットワークエラーシミュレーションが有効になっていないか確認**

## 本番環境との違い

| 項目 | StoreKit Testing | 本番 (App Store) |
|------|------------------|------------------|
| 決済 | シミュレート（無料） | 実際の決済 |
| 更新速度 | 高速（1秒=1ヶ月など） | 実時間 |
| 返金 | 即座 | Appleの審査が必要 |
| トランザクション履歴 | ローカルのみ | Appleサーバー |
| レシート検証 | ローカル検証 | Appleサーバー検証 |

## 次のステップ

### Sandbox Testing
本番に近い環境でのテストには、App Store Connect の Sandbox環境を使用します:

1. App Store Connect でテストユーザーを作成
2. XcodeスキームでStoreKit Configurationを無効化
3. Sandboxユーザーでサインイン
4. 実際の購入フローをテスト

### TestFlight
ベータテスターによる実機テスト:

1. TestFlightでアプリを配信
2. ベータテスターに購入フローをテストしてもらう
3. フィードバックを収集

## 参考資料

- [Apple Developer - Testing In-App Purchases](https://developer.apple.com/documentation/storekit/in-app_purchase/testing_in-app_purchases)
- [Apple Developer - StoreKitTest](https://developer.apple.com/documentation/storekittest)
- [Apple Developer - Transaction Inspector](https://developer.apple.com/documentation/xcode/setting-up-storekit-testing-in-xcode)
