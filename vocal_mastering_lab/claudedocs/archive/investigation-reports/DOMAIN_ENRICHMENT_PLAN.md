# ドメイン充実化プラン (Domain Enrichment Plan)

## 📋 概要

このドキュメントは、SubscriptionDomainパッケージ分離完了後の次のステップとして、ドメイン層を充実させる戦略的プランを提供します。

### ユーザーからのフィードバック
> "ドメインが軽薄すぎることがあります。インフラの機能に依存したアプリなので仕方ない部分もあるかもしれませんが、もう少しドメインの方にロジックを移していくということは検討できませんか"

### 目標
1. **ビジネスロジックのドメイン層への移行**: Infrastructure層に散在するビジネスルールをドメイン層に集約
2. **ドメインサービスの導入**: 複雑なビジネスロジックをドメインサービスとしてカプセル化
3. **Rich Domain Model**: Anemic Domain Modelからリッチなドメインモデルへの進化
4. **テスタビリティの向上**: ドメインロジックの単体テストを容易にする

---

## 🎯 Phase 1: Recording Domain の充実化

### 現状分析

**現在のドメイン層 (Recording関連)**:
- `Recording` Entity: 基本的なデータコンテナ
- `RecordingId`, `Duration` Value Objects: 型安全性の提供
- `RecordingRepositoryProtocol`: データアクセスの抽象化

**問題点**:
- 録音時間制限のバリデーションがApplication層 (Use Case) に存在
- スケール設定の妥当性チェックがPresentation層やApplication層に散在
- 録音セッションのライフサイクル管理がInfrastructure層に集中

### 改善プラン

#### 1.1 Recording Entity の充実

**Before (Anemic Model)**:
```swift
struct Recording {
    let id: RecordingId
    let createdAt: Date
    let duration: Duration
    let settings: ScaleSettings?
}
```

**After (Rich Model)**:
```swift
struct Recording {
    let id: RecordingId
    let createdAt: Date
    let duration: Duration
    let settings: ScaleSettings?

    // ドメインロジック
    func validate() throws {
        guard duration.seconds > 0 else {
            throw RecordingError.invalidDuration
        }
        if let settings = settings {
            try settings.validate()
        }
    }

    func isWithinLimit(_ limit: RecordingLimit) -> Bool {
        return duration <= limit.maxDuration
    }

    func requiresPremiumTier() -> Bool {
        return duration.seconds > RecordingLimit.freeUser.maxDuration.seconds
    }
}
```

#### 1.2 新規 Value Objects の導入

**ScaleSettings Value Object** (現在はInfrastructure層に存在):
```swift
public struct ScaleSettings {
    public let scaleType: ScaleType
    public let rootNote: MIDINote
    public let tempo: Tempo

    // ドメインロジック
    public func validate() throws {
        guard tempo.bpm >= 40 && tempo.bpm <= 240 else {
            throw ScaleError.invalidTempo
        }
        guard rootNote.isValid() else {
            throw ScaleError.invalidNote
        }
    }

    public func isCompatibleWith(_ instrument: Instrument) -> Bool {
        return instrument.supportedRange.contains(rootNote)
    }
}
```

**MIDINote Value Object**:
```swift
public struct MIDINote: Equatable {
    public let number: UInt8 // 0-127

    public init?(number: UInt8) {
        guard number <= 127 else { return nil }
        self.number = number
    }

    public func isValid() -> Bool {
        return number <= 127
    }

    public var frequency: Double {
        return 440.0 * pow(2.0, Double(number - 69) / 12.0)
    }
}
```

#### 1.3 Recording Domain Services の導入

**RecordingPolicyService**: 録音ポリシーのビジネスルールを集約
```swift
public protocol RecordingPolicyService {
    func canStartRecording(
        user: User,
        settings: ScaleSettings?
    ) async throws -> RecordingPermission

    func validateDuration(
        _ duration: Duration,
        for status: SubscriptionStatus
    ) throws
}

public struct RecordingPermission {
    public let allowed: Bool
    public let reason: DenialReason?
    public let suggestedAction: UserAction?
}

public enum DenialReason {
    case dailyLimitReached(limit: Int)
    case requiresPremiumTier(requiredTier: SubscriptionTier)
    case invalidSettings(error: Error)
}
```

**RecordingPolicyServiceImpl** (Application層で実装):
```swift
public final class RecordingPolicyServiceImpl: RecordingPolicyService {
    private let subscriptionRepository: SubscriptionRepositoryProtocol
    private let usageTracker: RecordingUsageTrackerProtocol

    public func canStartRecording(
        user: User,
        settings: ScaleSettings?
    ) async throws -> RecordingPermission {
        let status = try await subscriptionRepository.getCurrentStatus()
        let usageCount = try await usageTracker.getTodayCount()

        // ドメインロジック: RecordingLimitConfig を使用
        let config = status.tier.recordingLimitConfig(cohort: status.cohort)

        guard usageCount < config.dailyLimit else {
            return RecordingPermission(
                allowed: false,
                reason: .dailyLimitReached(limit: config.dailyLimit),
                suggestedAction: .upgradeToPremium
            )
        }

        if let settings = settings {
            do {
                try settings.validate()
            } catch {
                return RecordingPermission(
                    allowed: false,
                    reason: .invalidSettings(error: error),
                    suggestedAction: .fixSettings
                )
            }
        }

        return RecordingPermission(allowed: true, reason: nil, suggestedAction: nil)
    }
}
```

---

## 🎯 Phase 2: Subscription Domain の更なる充実化

### 改善プラン

#### 2.1 SubscriptionStatus Entity の拡張

**Before**:
```swift
public struct SubscriptionStatus {
    public let tier: SubscriptionTier
    public let cohort: UserCohort
    public let isActive: Bool
    public let expirationDate: Date?
}
```

**After**:
```swift
public struct SubscriptionStatus {
    public let tier: SubscriptionTier
    public let cohort: UserCohort
    public let isActive: Bool
    public let expirationDate: Date?
    public let purchaseDate: Date?
    public let willAutoRenew: Bool

    // ドメインロジック
    public func isExpired(at date: Date = Date()) -> Bool {
        guard let expiration = expirationDate else { return false }
        return date > expiration
    }

    public func daysUntilExpiration(from date: Date = Date()) -> Int? {
        guard let expiration = expirationDate else { return nil }
        return Calendar.current.dateComponents([.day], from: date, to: expiration).day
    }

    public func requiresRenewalWarning(at date: Date = Date()) -> Bool {
        guard let days = daysUntilExpiration(from: date) else { return false }
        return days <= 7 && days > 0
    }

    public func canAccess(_ feature: Feature) -> Bool {
        return isActive && tier.supports(feature)
    }
}
```

#### 2.2 SubscriptionValidationService の導入

```swift
public protocol SubscriptionValidationService {
    func validatePurchase(
        tier: SubscriptionTier,
        for user: User
    ) async throws -> PurchaseValidation

    func validateFeatureAccess(
        feature: Feature,
        status: SubscriptionStatus
    ) -> FeatureAccessResult
}

public struct PurchaseValidation {
    public let valid: Bool
    public let warnings: [ValidationWarning]
    public let requiredActions: [UserAction]
}

public enum ValidationWarning {
    case alreadySubscribed(currentTier: SubscriptionTier)
    case downgradeNotAllowed
    case pendingRenewal
}

public struct FeatureAccessResult {
    public let hasAccess: Bool
    public let reason: AccessDenialReason?
    public let alternativeActions: [UserAction]
}
```

---

## 🎯 Phase 3: User Domain の新規作成

### 背景
現在、ユーザー情報はPresentation層とApplication層で暗黙的に扱われていますが、ドメイン層にUser概念が存在しません。

### User Domain の設計

**User Entity**:
```swift
public struct User {
    public let id: UserId
    public let cohort: UserCohort
    public let createdAt: Date
    public let settings: UserSettings

    // ドメインロジック
    public func isNewUser(threshold: TimeInterval = 7 * 24 * 3600) -> Bool {
        return Date().timeIntervalSince(createdAt) < threshold
    }

    public func shouldShowOnboarding() -> Bool {
        return isNewUser() && !settings.hasCompletedOnboarding
    }
}

public struct UserId: Equatable, Hashable {
    public let value: UUID
}

public struct UserSettings {
    public let hasCompletedOnboarding: Bool
    public let preferredTempo: Tempo?
    public let preferredScaleType: ScaleType?
}
```

**UserRepositoryProtocol** (Ports):
```swift
public protocol UserRepositoryProtocol {
    func getCurrentUser() async throws -> User
    func updateSettings(_ settings: UserSettings) async throws
}
```

---

## 🎯 Phase 4: Domain Services アーキテクチャ

### Domain Services の分類

#### 4.1 Validation Services (バリデーション)
- **RecordingValidationService**: 録音設定・期間の妥当性チェック
- **SubscriptionValidationService**: サブスクリプション購入・状態の妥当性チェック
- **ScaleValidationService**: スケール設定の音楽理論的妥当性チェック

#### 4.2 Policy Services (ポリシー・ビジネスルール)
- **RecordingPolicyService**: 録音許可・制限ポリシー
- **FeatureAccessPolicyService**: 機能アクセスポリシー
- **UsageLimitPolicyService**: 使用量制限ポリシー

#### 4.3 Calculation Services (計算・導出)
- **PitchCalculationService**: 音高周波数計算
- **ScaleGenerationService**: スケール音列生成
- **UsageStatisticsService**: 使用統計計算

### Domain Services の実装場所

```
Domain Layer (Interface定義):
├── Services/
│   ├── RecordingPolicyService.swift (protocol)
│   ├── SubscriptionValidationService.swift (protocol)
│   └── ...

Application Layer (実装):
├── Services/
│   ├── RecordingPolicyServiceImpl.swift
│   ├── SubscriptionValidationServiceImpl.swift
│   └── ...
```

**重要**: Domain ServicesのプロトコルはDomain層に、実装はApplication層に配置します。これにより:
- Domain層は外部依存を持たない (Dependency Inversion Principle)
- テストが容易 (モック化可能)
- インフラ詳細から独立

---

## 🎯 Phase 5: 段階的移行プラン

### 移行の優先順位

| Priority | ドメイン | 理由 | 推定工数 |
|----------|---------|------|---------|
| 🔴 **High** | Recording Domain | 最もビジネスロジックが散在している | 8-12時間 |
| 🟡 **Medium** | Subscription Domain | 既にパッケージ化済み、拡張が容易 | 4-6時間 |
| 🟢 **Low** | User Domain | 新規作成、既存コードへの影響小 | 6-8時間 |

### Phase 5.1: Recording Domain Services (Week 1-2)

**Step 1: RecordingPolicyService の設計とテスト**
```bash
# TDD Approach
1. RecordingPolicyServiceTests.swift を作成 (RED)
2. RecordingPolicyService protocol を定義
3. RecordingPolicyServiceImpl を実装 (GREEN)
4. リファクタリング (REFACTOR)
```

**Step 2: 既存コードからの移行**
- `StartRecordingUseCase` から録音許可ロジックを抽出
- `RecordingViewModel` からバリデーションロジックを抽出
- `RecordingLimitConfig` を活用した統一的なポリシー実装

**Step 3: 統合テスト**
- Application層のUse Caseテストでドメインサービスを検証
- Presentation層のViewModelテストで動作確認

### Phase 5.2: Value Objects の拡充 (Week 2-3)

**Step 1: ScaleSettings Value Object**
```bash
# 現在の ScaleSettings (Infrastructure層) をドメイン層に移行
1. ScaleSettingsTests.swift を作成
2. ScaleSettings をドメイン層に移動
3. バリデーションロジックを追加
4. Infrastructure層の参照を更新
```

**Step 2: MIDINote Value Object**
```bash
# 新規作成
1. MIDINoteTests.swift を作成 (TDD)
2. MIDINote Value Object を実装
3. 周波数計算ロジックを移行
```

### Phase 5.3: Subscription Domain の拡張 (Week 3-4)

**Step 1: SubscriptionStatus の拡張**
```bash
1. 既存テストに新規ビジネスロジックのテストを追加
2. SubscriptionStatus にメソッド追加
3. 既存コードで新しいメソッドを活用
```

**Step 2: SubscriptionValidationService の導入**
```bash
1. SubscriptionValidationServiceTests.swift
2. プロトコル定義と実装
3. PurchaseSubscriptionUseCase からロジック抽出
```

### Phase 5.4: User Domain の新規作成 (Week 4-5)

**Step 1: User Entity の設計**
```bash
1. UserTests.swift を作成
2. User Entity, UserId, UserSettings を実装
3. UserRepositoryProtocol を定義
```

**Step 2: Infrastructure実装**
```bash
1. UserDefaultsUserRepository (初期実装)
2. CoreDataUserRepository (将来の拡張)
```

---

## 📊 成功指標 (Success Metrics)

### コード品質指標

| 指標 | Before | Target After | 測定方法 |
|-----|--------|-------------|---------|
| ドメイン層のテストカバレッジ | 60% | 85%+ | `xcodebuild test -enableCodeCoverage YES` |
| ドメインロジックの集約率 | 40% | 75%+ | 手動コードレビュー |
| Anemic Entity の割合 | 80% | 30%- | メソッド数/Entity数 |
| Domain Services数 | 0 | 5+ | ファイル数カウント |

### アーキテクチャ健全性

**依存関係の方向性**:
```
✅ Good (Before):
Presentation → Application → Domain ← Infrastructure
                           ↑
                    (protocol only)

✅ Better (After):
Presentation → Application → Domain Services (protocol) ← Infrastructure
                           ↑                  ↑
                    Domain Entities      (implementation)
                    Value Objects
```

**期待される効果**:
1. **テスタビリティ**: ドメインロジックの単体テスト容易性向上 (モック不要)
2. **変更容易性**: ビジネスルール変更時の影響範囲が明確化
3. **可読性**: ビジネスロジックが一箇所に集約され、理解しやすい
4. **再利用性**: ドメインサービスが複数のUse Caseから利用可能

---

## ⚠️ リスクと制約

### リスク管理

| リスク | 影響 | 軽減策 |
|-------|------|--------|
| Infrastructure依存が強い | High | ドメインサービスを介したロジック抽出 |
| 既存コードへの影響大 | High | 段階的移行、各Phase後の回帰テスト |
| 学習曲線 | Medium | ドメインモデリングのドキュメント整備 |
| パフォーマンス懸念 | Low | ドメインロジックは軽量、インライン化可能 |

### 制約と妥協点

**AVFoundation依存**:
- 音声録音・再生は本質的にInfrastructure層の責務
- 妥協案: AVFoundation操作はInfrastructure層、ビジネスルール(制限・ポリシー)はDomain層

**StoreKit依存**:
- アプリ内課金はApple API依存
- 妥協案: 購入トランザクションはInfrastructure層、購入ポリシー・状態管理はDomain層

---

## 📚 参考資料

### Domain-Driven Design (DDD)
- Eric Evans「Domain-Driven Design」
- Martin Fowler「Anemic Domain Model」
- Vaughn Vernon「Implementing Domain-Driven Design」

### Clean Architecture
- Robert C. Martin「Clean Architecture」
- Dependency Inversion Principle
- Screaming Architecture

### 実装パターン
- Repository Pattern (Ports & Adapters)
- Domain Services vs. Application Services
- Value Objects vs. Entities

---

## 🚀 次のアクション

### Immediate (今すぐ実行)
1. このプランをチームでレビュー
2. Phase 5.1 (Recording Domain Services) の詳細設計
3. RecordingPolicyService のテストファースト実装開始

### Short-term (1-2週間)
1. Recording Domain Services の完全実装
2. Value Objects (ScaleSettings, MIDINote) の移行
3. Application層のUse Caseリファクタリング

### Mid-term (1-2ヶ月)
1. Subscription Domain の拡張完了
2. User Domain の新規作成
3. ドメイン層のテストカバレッジ85%達成

### Long-term (3-6ヶ月)
1. ドメインモデルの継続的改善
2. 新機能開発時のドメインファーストアプローチ定着
3. チーム全体のDDD理解度向上

---

## 📝 まとめ

このドメイン充実化プランは、「ドメインが軽薄すぎる」という課題に対して、段階的かつ実践的なアプローチを提供します。

**キーポイント**:
- ✅ TDDによる安全な移行
- ✅ Infrastructure依存の現実的な受け入れ
- ✅ ビジネスロジックのドメイン層への集約
- ✅ テスタビリティと保守性の向上
- ✅ 段階的実装による リスク最小化

次のステップは、Phase 5.1の詳細設計とRecordingPolicyServiceの実装開始です。
