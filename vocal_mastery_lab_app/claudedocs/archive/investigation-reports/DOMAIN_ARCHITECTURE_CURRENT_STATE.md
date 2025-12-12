# ドメイン層の現状分析

## 調査日時
2025-10-24

## 問題の概要

ユーザーからの指摘:
> "あなたの認識にも問題がありますね。もともとドメイン層はパッケージとして別に切り出していたはずです。それと内部のドメインでぐちゃぐちゃになっているのですね。"

**現状**: ドメイン層が2箇所に存在し、責務が混在している

## 1. 既存パッケージ: `VocalisDomain`

### 場所
`./Packages/VocalisDomain/`

### Package.swift 設定
```swift
name: "VocalisDomain"
platforms: iOS 15+, macOS 11+
products: library(VocalisDomain)
targets: VocalisDomain, VocalisDomainTests
```

### 含まれるドメインオブジェクト (19ファイル)

#### Entities (5)
- `AnalysisResult.swift` - 録音分析結果エンティティ
- `Recording.swift` - 録音エンティティ
- `RecordingSession.swift` - 録音セッションエンティティ
- `ScaleSettings.swift` - スケール設定エンティティ
- `StopRecordingResult.swift` - 録音停止結果エンティティ

#### ValueObjects (8)
- `DetectedPitch.swift` - ピッチ検出結果値オブジェクト
- `Duration.swift` - 時間値オブジェクト
- `MIDINote.swift` - MIDIノート値オブジェクト
- `NotePattern.swift` - 音階パターン値オブジェクト
- `PitchAnalysisData.swift` - ピッチ分析データ値オブジェクト
- `RecordingId.swift` - 録音ID値オブジェクト
- `SpectrogramData.swift` - スペクトログラムデータ値オブジェクト
- `Tempo.swift` - テンポ値オブジェクト

#### RepositoryInterfaces (5)
- `AudioFileRepositoryProtocol.swift` - オーディオファイルリポジトリインターフェース
- `AudioPlayerProtocol.swift` - オーディオプレイヤーインターフェース
- `AudioRecorderProtocol.swift` - オーディオレコーダーインターフェース
- `RecordingRepositoryProtocol.swift` - 録音リポジトリインターフェース
- `ScalePlayerProtocol.swift` - スケールプレイヤーインターフェース

#### ServiceInterfaces (1)
- `LoggerProtocol.swift` - ロガーサービスインターフェース

### 責務
**VocalisStudioの中核ドメイン概念**: 録音、スケール、音声分析に関する業務ロジック

---

## 2. 内部ドメインディレクトリ: `VocalisStudio/Domain`

### 場所
`./VocalisStudio/Domain/`

### 含まれるドメインオブジェクト (9ファイル)

#### Entities (1)
- `SubscriptionStatus.swift` - サブスクリプションステータスエンティティ

#### ValueObjects (6)
- `SubscriptionTier.swift` - サブスクリプションティア値オブジェクト
- `RecordingLimit.swift` - 録音制限値オブジェクト
- `RecordingLimitConfig.swift` - 録音制限設定値オブジェクト
- `Feature.swift` - 機能フラグ値オブジェクト
- `AdPolicy.swift` - 広告ポリシー値オブジェクト
- `UserCohort.swift` - ユーザーコホート値オブジェクト

#### RepositoryProtocols (1)
- `SubscriptionRepositoryProtocol.swift` - サブスクリプションリポジトリインターフェース

#### Errors (1)
- `SubscriptionError.swift` - サブスクリプションエラー定義

### 責務
**サブスクリプションドメイン概念**: 課金、ティア、機能制限に関する業務ロジック

---

## 3. Xcodeプロジェクト設定

### 依存関係
```
VocalisStudio.xcodeproj
├─ Target: VocalisStudio
│  └─ Dependencies: VocalisDomain (local package)
├─ Target: VocalisStudioTests
└─ Target: VocalisStudioUITests

Resolved Packages:
- VocalisDomain: /path/to/Packages/VocalisDomain @ local
- swift-snapshot-testing, ViewInspector, AudioKit, etc.
```

### スキーム
- `VocalisStudio` (アプリターゲット)
- `VocalisDomain` (パッケージターゲット)

---

## 4. 問題点の分析

### 4.1 ドメイン概念の混在

**問題**: 2つの異なるドメイン概念が異なる場所に配置されている

| ドメイン概念 | 現在の配置 | 理想的な配置 |
|------------|----------|------------|
| VocalisStudio核心ドメイン (録音、音声分析) | `Packages/VocalisDomain/` ✅ | 独立パッケージ ✅ |
| Subscriptionドメイン (課金、制限) | `VocalisStudio/Domain/` ❌ | 独立パッケージ ❌ |

**影響**:
- Subscriptionドメインが独立パッケージとして管理されていない
- VocalisStudioアプリターゲット内部に直接配置されている
- 再利用性、テスト容易性が低下
- Clean Architectureの依存関係ルール違反 (ドメイン層がアプリに埋め込まれている)

### 4.2 パッケージ分離の不完全性

**現状のパッケージ構成**:
```
Packages/
└── VocalisDomain/        ← VocalisStudio核心ドメインのみ

VocalisStudio/
└── Domain/              ← Subscriptionドメインが内部に混在
```

**理想的なパッケージ構成**:
```
Packages/
├── VocalisDomain/        ← VocalisStudio核心ドメイン
└── SubscriptionDomain/   ← Subscriptionドメイン (新規作成が必要)

VocalisStudio/
└── (Domainディレクトリは不要 - すべてパッケージに移行)
```

### 4.3 依存関係の問題

**現状の依存関係**:
```
VocalisStudio (App Target)
├─ depends on → VocalisDomain (Package) ✅
└─ contains → Subscription Domain (Internal) ❌
```

**問題点**:
- Subscriptionドメインがアプリケーション層に直接含まれている
- ドメイン層の独立性が保たれていない
- テスト時にアプリターゲット全体をビルドする必要がある
- 循環依存のリスク

**理想的な依存関係**:
```
VocalisStudio (App Target)
├─ depends on → VocalisDomain (Package)
└─ depends on → SubscriptionDomain (Package)

SubscriptionDomain (Package)
└─ depends on → (なし - 完全に独立)

VocalisDomain (Package)
└─ depends on → (なし - 完全に独立)
```

---

## 5. ユーザーの要望

> "理想としては、VocalisStudioドメインと、Subscriptionドメインのどちらもパッケージとして外で管理したいです。"

### 要件
1. **SubscriptionDomainの独立パッケージ化**: `Packages/SubscriptionDomain/` として新規作成
2. **VocalisDomain**: 既存パッケージは維持 (既に正しく分離されている)
3. **内部Domainの完全移行**: `VocalisStudio/Domain/` 配下のすべてのファイルを `SubscriptionDomain` パッケージに移行
4. **依存関係の整理**: VocalisStudioアプリターゲットは両パッケージに依存

---

## 6. 次のステップ (提案)

### Phase 1: 現状詳細調査
- [ ] `VocalisStudio/Domain/` の全ファイル依存関係確認
- [ ] Subscription関連のUseCaseやInfrastructureコード調査
- [ ] 既存テストの影響範囲確認

### Phase 2: SubscriptionDomainパッケージ設計
- [ ] Package.swift 設計
- [ ] ディレクトリ構造設計 (Entities, ValueObjects, RepositoryInterfaces, Errors)
- [ ] 移行ファイルリスト作成

### Phase 3: 段階的移行実施 (TDD準拠)
- [ ] SubscriptionDomainパッケージ作成 (Package.swift)
- [ ] テストファースト: 既存Subscriptionドメインのテストをパッケージに移行
- [ ] ドメインオブジェクトを段階的に移行
- [ ] VocalisStudioプロジェクトの依存関係更新
- [ ] すべてのテストがグリーンであることを確認

### Phase 4: 内部Domainディレクトリ削除
- [ ] `VocalisStudio/Domain/` ディレクトリ完全削除
- [ ] Xcodeprojファイル参照削除
- [ ] ビルド・テスト成功確認

---

## 7. リスク評価

### 高リスク
- **既存コードの大規模な変更**: Application層、Infrastructure層、Presentation層のimport文をすべて修正
- **循環依存の可能性**: VocalisDomainとSubscriptionDomainの間に依存関係がないか慎重に確認

### 中リスク
- **テストの一時的な失敗**: 移行中にテストが失敗する可能性
- **ビルド設定の複雑化**: 2つのローカルパッケージ依存管理

### 低リスク
- **パッケージ作成自体**: Swift Package Managerの標準的な使い方で対応可能

---

## 8. 参考: Clean Architectureにおけるドメイン層

### 原則
1. **最内層**: ドメイン層は最も内側で、外層に依存してはならない
2. **独立性**: ビジネスロジックはフレームワーク、UI、DBから独立
3. **再利用性**: 他のプロジェクトでも使用可能な純粋なビジネスロジック
4. **テスト容易性**: モック不要で単体テストが可能

### 現状の違反
- Subscriptionドメインがアプリケーション層と同じターゲットに含まれている
- ドメイン層の独立性が保証されていない

### 修正後の状態
- すべてのドメインオブジェクトが独立パッケージとして管理
- VocalisStudioアプリはドメインパッケージの「利用者」として振る舞う
- Clean Architectureの依存関係ルール完全準拠

---

## まとめ

**現状**: VocalisStudioの核心ドメインは正しく `VocalisDomain` パッケージとして分離されているが、Subscriptionドメインが `VocalisStudio/Domain/` 内部に混在している。

**問題**:
- ドメイン層の独立性が不完全
- Subscriptionドメインの再利用性・テスト容易性が低い
- Clean Architectureの原則に反している

**解決策**:
1. `Packages/SubscriptionDomain/` を新規作成
2. `VocalisStudio/Domain/` 配下の全ファイルを移行
3. VocalisStudioアプリは両パッケージに依存
4. 内部Domainディレクトリを完全削除

**次のアクション**: ユーザーに現状報告し、SubscriptionDomainパッケージ作成と段階的移行の承認を得る
