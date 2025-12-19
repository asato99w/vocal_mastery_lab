# SubscriptionDomain

VocalisStudio アプリケーションのサブスクリプションドメインを管理する独立したSwiftパッケージ。

## 概要

このパッケージは、VocalisStudioのサブスクリプション・課金機能に関連するビジネスロジックを提供します。Clean Architectureの依存関係ルールに従い、外部フレームワークやインフラストラクチャ層から完全に独立しています。

## ドメインオブジェクト

### Entities
- **`SubscriptionStatus`**: サブスクリプションステータスエンティティ

### ValueObjects
- **`SubscriptionTier`**: サブスクリプションティア (.free, .premium, .premiumPlus)
- **`RecordingLimit`**: 録音制限 (dailyCount, maxDuration)
- **`RecordingLimitConfig`**: 録音制限設定インターフェース
- **`Feature`**: 機能フラグ
- **`AdPolicy`**: 広告ポリシー
- **`UserCohort`**: ユーザーコホート

### RepositoryProtocols
- **`SubscriptionRepositoryProtocol`**: サブスクリプションリポジトリインターフェース

### Errors
- **`SubscriptionError`**: サブスクリプションドメインエラー定義

## 使用方法

### インポート

```swift
import SubscriptionDomain
```

### 基本的な使用例

```swift
// SubscriptionTier の使用
let tier: SubscriptionTier = .premium
print(tier.displayName)  // "Premium"
print(tier.monthlyPrice)  // 480

// RecordingLimit の取得
let limit = RecordingLimit(dailyCount: 10, maxDuration: 60)
print("1日の録音可能回数: \\(limit.dailyCount)回")
print("最大録音時間: \\(limit.maxDuration)秒")
```

## 依存関係

このパッケージは外部依存を持ちません。純粋なビジネスロジックのみを含みます。

## テスト

```bash
swift test
```

または、Xcodeプロジェクトから:

```bash
xcodebuild test -scheme SubscriptionDomain
```

## ライセンス

© 2025 VocalisStudio. All rights reserved.
