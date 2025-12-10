# レコーディング機能アーキテクチャ・リファクタリング提案

## 問題の本質

### 1. **現在のドメイン層が軽薄すぎる**

現状、ドメイン層にはSubscription関連のエンティティとValue Objectsしか存在せず、レコーディングという **コアドメインのビジネスロジック**がドメイン層に存在していません。

```
現在のDomain層:
- SubscriptionStatus (Entity)
- SubscriptionTier, Feature, UserCohort, AdPolicy (Value Objects)
- RecordingLimit, RecordingLimitConfig (Value Objects)

❌ 欠落: Recordingエンティティ、RecordingSessionエンティティ、PitchDetectionドメインサービス
```

### 2. **Presentation層に過剰な責務**

`RecordingViewModel`, `PitchDetectionViewModel`, `RecordingStateViewModel`が以下の責務を全て担っています:

- **ビジネスロジック**: 録音状態管理、ピッチ検出ロジック、スケール再生と録音の同期
- **非同期処理の制御**: Task管理、レースコンディション対策
- **UIState管理**: SwiftUIビューへの状態提供
- **Infrastructure層の直接操作**: AVFoundation APIの直接呼び出し

**結果**: テストできないレースコンディション、複雑な非同期処理、責務の肥大化

### 3. **テストが困難な設計**

- Mock ObjectsはUIレベルでの同期的振る舞いしか再現できない
- 実際のAVFoundationの非同期タイミング問題を再現不可能
- Presentation層のTask管理ロジックをテストするには、Presentationレイヤー全体をモックする必要がある（本末転倒）

---

## アーキテクチャ問題の詳細分析

### 問題1: ドメインロジックの欠如

#### 現状
```swift
// RecordingViewModel.swift (Presentation層)
public func stopPlayback() async {
    await recordingStateVM.stopPlayback()
    await pitchDetectionVM.stopTargetPitchMonitoring()  // ← ここにビジネスロジック
    pitchDetectionVM.stopPlaybackPitchDetection()
}
```

**問題点**:
- 「再生停止時にピッチ検出も停止する」というビジネスルールがPresentation層に記述されている
- 「停止の順序」というドメインルールがViewModelに埋もれている
- この知識を他の画面で再利用できない

#### あるべき姿
```swift
// Domain層: RecordingSessionEntity
public class RecordingSession {
    private var state: SessionState
    private let pitchDetector: PitchDetectorService

    public func stopPlayback() async throws {
        // ドメインロジック: 再生停止時の正しい順序を保証
        try await pitchDetector.stopMonitoring()  // 必ずピッチ検出を先に停止
        try await audioPlayer.stop()              // その後オーディオを停止
        state = .stopped
    }
}
```

---

### 問題2: Infrastructure依存の逆転不足

#### 現状
```swift
// PitchDetectionViewModel.swift (Presentation層)
private var progressMonitorTask: Task<Void, Never>?  // ← Infrastructure詳細がPresentation層に

public func stopTargetPitchMonitoring() async {
    progressMonitorTask?.cancel()
    _ = await progressMonitorTask?.value  // ← 非同期処理の詳細がPresentationに露出
    progressMonitorTask = nil
    targetPitch = nil
}
```

**問題点**:
- Swift ConcurrencyのTask管理がPresentation層の責務になっている
- この非同期処理の詳細はInfrastructure層の実装の詳細であるべき
- レースコンディション対策がViewModelレベルで必要（テスト困難）

#### あるべき姿
```swift
// Domain層: PitchDetectorServiceProtocol (インターフェース)
public protocol PitchDetectorService {
    func startMonitoring(settings: ScaleSettings) async throws
    func stopMonitoring() async throws  // ← 完了を保証する契約
    var currentPitch: DetectedPitch? { get }
}

// Infrastructure層: RealtimePitchDetectorAdapter
public class RealtimePitchDetectorAdapter: PitchDetectorService {
    private var monitorTask: Task<Void, Never>?

    public func stopMonitoring() async throws {
        // Infrastructure層で非同期処理の詳細を隠蔽
        monitorTask?.cancel()
        _ = await monitorTask?.value  // レースコンディション対策はここで
        monitorTask = nil
    }
}
```

---

### 問題3: ドメインサービスの欠如

#### 現状の責務配置
```
RecordingViewModel
├─ RecordingStateViewModel (録音状態管理)
├─ PitchDetectionViewModel (ピッチ検出管理)
└─ SubscriptionViewModel (サブスクリプション管理)
```

これは「技術的な分類」であり、「ドメイン概念」ではありません。

#### あるべきドメインモデル
```
RecordingSession (Entity)
├─ sessionId: RecordingSessionId
├─ audioRecording: AudioRecording (Value Object)
├─ scaleSettings: ScaleSettings? (Value Object)
├─ pitchData: [PitchDataPoint] (Value Object Collection)
└─ state: SessionState (Value Object)

PitchDetectionService (Domain Service)
├─ startMonitoring(session: RecordingSession)
├─ stopMonitoring()
└─ getCurrentPitch() -> DetectedPitch?

RecordingOrchestrator (Domain Service)
├─ startRecording(with: ScaleSettings?) -> RecordingSession
├─ stopRecording(session: RecordingSession)
├─ playback(session: RecordingSession)
└─ stopPlayback(session: RecordingSession)
```

---

## リファクタリング提案

### フェーズ1: ドメイン層の強化 (高優先度)

#### 1.1 RecordingSession Entity の導入

```swift
// Domain/Entities/RecordingSession.swift
public class RecordingSession {
    public let id: RecordingSessionId
    public private(set) var state: SessionState
    public let audioURL: URL
    public let scaleSettings: ScaleSettings?
    public private(set) var pitchData: [PitchDataPoint]
    public let recordedAt: Date

    // ドメインルール: セッション状態遷移の制約
    public func start() throws {
        guard state.canTransitionTo(.recording) else {
            throw RecordingError.invalidStateTransition(from: state, to: .recording)
        }
        state = .recording
    }

    public func stop() throws -> RecordingResult {
        guard state.canTransitionTo(.completed) else {
            throw RecordingError.invalidStateTransition(from: state, to: .completed)
        }
        state = .completed
        return RecordingResult(session: self)
    }

    public func addPitchData(_ pitch: DetectedPitch) {
        guard state == .recording || state == .playing else { return }
        pitchData.append(PitchDataPoint(pitch: pitch, timestamp: Date()))
    }
}

// Domain/ValueObjects/SessionState.swift
public enum SessionState {
    case idle
    case countdown
    case recording
    case completed
    case playing
    case stopped

    public func canTransitionTo(_ newState: SessionState) -> Bool {
        // ドメインルール: 許可される状態遷移を定義
        switch (self, newState) {
        case (.idle, .countdown), (.idle, .recording):
            return true
        case (.countdown, .recording), (.countdown, .idle):
            return true
        case (.recording, .completed):
            return true
        case (.completed, .playing):
            return true
        case (.playing, .stopped):
            return true
        default:
            return false
        }
    }
}
```

#### 1.2 Domain Service の導入

```swift
// Domain/Services/RecordingOrchestrator.swift
public protocol RecordingOrchestrator {
    func startRecording(settings: ScaleSettings?) async throws -> RecordingSession
    func stopRecording(session: RecordingSession) async throws -> RecordingResult
    func startPlayback(session: RecordingSession) async throws
    func stopPlayback(session: RecordingSession) async throws
}

// Domain/Services/PitchDetectionService.swift
public protocol PitchDetectionService {
    func startMonitoring(for session: RecordingSession) async throws
    func stopMonitoring() async throws
    var currentPitch: DetectedPitch? { get }
    var targetPitch: DetectedPitch? { get }
}
```

#### 1.3 Repository Interface の導入

```swift
// Domain/RepositoryProtocols/RecordingSessionRepositoryProtocol.swift
public protocol RecordingSessionRepository {
    func save(_ session: RecordingSession) async throws
    func findById(_ id: RecordingSessionId) async throws -> RecordingSession?
    func findRecent(limit: Int) async throws -> [RecordingSession]
    func delete(_ id: RecordingSessionId) async throws
}
```

---

### フェーズ2: Application層の整理 (高優先度)

#### 2.1 Use Case の再定義

```swift
// Application/UseCases/StartRecordingWithScaleUseCase.swift
public class StartRecordingWithScaleUseCase {
    private let orchestrator: RecordingOrchestrator
    private let pitchDetectionService: PitchDetectionService
    private let repository: RecordingSessionRepository

    public func execute(settings: ScaleSettings) async throws -> RecordingSession {
        // 1. ドメインサービスでセッション開始
        let session = try await orchestrator.startRecording(settings: settings)

        // 2. ピッチ検出開始（settingsがある場合のみ）
        try await pitchDetectionService.startMonitoring(for: session)

        // 3. 永続化
        try await repository.save(session)

        return session
    }
}

// Application/UseCases/StopPlaybackUseCase.swift
public class StopPlaybackUseCase {
    private let orchestrator: RecordingOrchestrator
    private let pitchDetectionService: PitchDetectionService

    public func execute(session: RecordingSession) async throws {
        // ドメインルール: ピッチ検出を先に停止してから再生停止
        try await pitchDetectionService.stopMonitoring()
        try await orchestrator.stopPlayback(session: session)
    }
}
```

---

### フェーズ3: Infrastructure層の実装 (中優先度)

#### 3.1 RecordingOrchestrator の実装

```swift
// Infrastructure/Services/AVFoundationRecordingOrchestrator.swift
public class AVFoundationRecordingOrchestrator: RecordingOrchestrator {
    private let audioRecorder: AudioRecorderProtocol
    private let audioPlayer: AudioPlayerProtocol
    private let scalePlayer: ScalePlayerProtocol

    public func startRecording(settings: ScaleSettings?) async throws -> RecordingSession {
        let url = try generateRecordingURL()
        let session = RecordingSession(
            id: RecordingSessionId(),
            audioURL: url,
            scaleSettings: settings
        )

        try session.start()

        // スケール再生と録音を同時開始
        if let settings = settings {
            try await scalePlayer.loadScale(settings)
            async let _ = scalePlayer.play()
            async let _ = audioRecorder.startRecording(to: url)
            try await (_, _)  // 両方の開始を待つ
        } else {
            try await audioRecorder.startRecording(to: url)
        }

        return session
    }

    public func stopPlayback(session: RecordingSession) async throws {
        // 再生停止の実装詳細をInfrastructure層で隠蔽
        await audioPlayer.stop()
        await scalePlayer.stop()
        try session.stop()
    }
}
```

#### 3.2 PitchDetectionService の実装

```swift
// Infrastructure/Services/RealtimePitchDetectionService.swift
public class RealtimePitchDetectionService: PitchDetectionService {
    private let pitchDetector: PitchDetectorProtocol
    private let scalePlayer: ScalePlayerProtocol
    private var monitoringTask: Task<Void, Never>?

    @Published public private(set) var currentPitch: DetectedPitch?
    @Published public private(set) var targetPitch: DetectedPitch?

    public func startMonitoring(for session: RecordingSession) async throws {
        guard let settings = session.scaleSettings else { return }

        try pitchDetector.startRealtimeDetection()

        monitoringTask = Task { [weak self] in
            guard let self = self else { return }
            while !Task.isCancelled {
                // スケール要素から目標ピッチを取得
                if let element = self.scalePlayer.currentScaleElement {
                    await self.updateTargetPitch(from: element)
                }

                // 現在のピッチを取得
                if let detected = self.pitchDetector.detectedPitch {
                    await self.updateCurrentPitch(detected)
                }

                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }

    public func stopMonitoring() async throws {
        // レースコンディション対策: Taskの完了を待つ
        monitoringTask?.cancel()
        _ = await monitoringTask?.value
        monitoringTask = nil

        pitchDetector.stopRealtimeDetection()

        // 状態をクリア
        currentPitch = nil
        targetPitch = nil
    }
}
```

---

### フェーズ4: Presentation層のシンプル化 (低優先度)

#### 4.1 ViewModel の責務削減

```swift
// Presentation/ViewModels/RecordingViewModel.swift
@MainActor
public class RecordingViewModel: ObservableObject {
    // Use Casesに委譲
    private let startRecordingUseCase: StartRecordingWithScaleUseCase
    private let stopRecordingUseCase: StopRecordingUseCase
    private let startPlaybackUseCase: StartPlaybackUseCase
    private let stopPlaybackUseCase: StopPlaybackUseCase

    // Domain Serviceから状態を購読
    private let pitchDetectionService: PitchDetectionService

    // UI状態のみ管理
    @Published public var currentSession: RecordingSession?
    @Published public var errorMessage: String?

    // Domain Serviceから転送
    @Published public var targetPitch: DetectedPitch?
    @Published public var detectedPitch: DetectedPitch?

    private var cancellables = Set<AnyCancellable>()

    public init(...) {
        // Pitch検出状態の購読
        pitchDetectionService.$targetPitch
            .assign(to: &$targetPitch)

        pitchDetectionService.$currentPitch
            .assign(to: &$detectedPitch)
    }

    public func startRecording(settings: ScaleSettings? = nil) async {
        do {
            currentSession = try await startRecordingUseCase.execute(settings: settings)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    public func stopPlayback() async {
        guard let session = currentSession else { return }
        do {
            try await stopPlaybackUseCase.execute(session: session)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
```

**メリット**:
- ViewModelは「UIの状態管理」と「Use Caseの呼び出し」のみに専念
- 非同期処理の詳細、レースコンディション対策はInfrastructure層で解決
- テストが容易: Use CaseとDomain Serviceをモックすればよい

---

## テスト戦略の改善

### Before (現状)
```swift
// MockScalePlayerで同期的な振る舞いをシミュレート
// → 実際のAVFoundationの非同期タイミングを再現できない
// → レースコンディションをテストできない
```

### After (リファクタリング後)
```swift
// 1. Domain層のテスト: ビジネスロジックのみ
func testRecordingSession_stateTransition() {
    let session = RecordingSession(...)
    XCTAssertNoThrow(try session.start())
    XCTAssertEqual(session.state, .recording)
    XCTAssertNoThrow(try session.stop())
    XCTAssertEqual(session.state, .completed)
}

// 2. Application層のテスト: Use Caseロジック
func testStopPlaybackUseCase_stopsInCorrectOrder() async {
    let mockOrchestrator = MockRecordingOrchestrator()
    let mockPitchService = MockPitchDetectionService()
    let useCase = StopPlaybackUseCase(
        orchestrator: mockOrchestrator,
        pitchDetectionService: mockPitchService
    )

    await useCase.execute(session: testSession)

    // ピッチ検出が先に停止されたことを検証
    XCTAssertTrue(mockPitchService.stopMonitoringCalled)
    XCTAssertTrue(mockPitchService.stopMonitoringCallTime! < mockOrchestrator.stopPlaybackCallTime!)
}

// 3. Infrastructure層の統合テスト: 実際のAVFoundationで
func testRealtimePitchDetectionService_stopMonitoring_clearsState() async {
    let service = RealtimePitchDetectionService(...)
    await service.startMonitoring(for: testSession)

    await service.stopMonitoring()

    // レースコンディションがないことを検証
    XCTAssertNil(service.currentPitch)
    XCTAssertNil(service.targetPitch)
}
```

---

## 移行戦略

### ステップ1: ドメイン層の構築 (破壊的変更なし)
1. `RecordingSession` Entity を追加
2. `RecordingOrchestrator`, `PitchDetectionService` Protocol を追加
3. 既存コードは変更せず、並行して新しいドメインモデルを構築

### ステップ2: Infrastructure層の実装
1. `AVFoundationRecordingOrchestrator` を実装
2. `RealtimePitchDetectionService` を実装
3. 既存の`RealtimePitchDetector`をラップする形で実装

### ステップ3: Application層の移行
1. 新しいUse Caseを作成 (`StopPlaybackUseCase`など)
2. 既存Use Caseを新しいDomain Serviceに委譲するように修正
3. 段階的に移行（1つずつ）

### ステップ4: Presentation層のリファクタリング
1. `RecordingViewModel`を新しいUse Caseに接続
2. `PitchDetectionViewModel`のロジックを`PitchDetectionService`に移動
3. `RecordingStateViewModel`の責務を`RecordingSession`エンティティに移動

### ステップ5: 旧コードの削除
1. 移行完了後、`PitchDetectionViewModel`, `RecordingStateViewModel`を削除
2. 旧Use Case実装を削除

---

## 期待される効果

### 1. **テスタビリティの向上**
- Domain層のロジックは純粋関数的でテストが容易
- Infrastructure層の非同期処理は統合テストでカバー
- レースコンディションをテストで再現可能

### 2. **保守性の向上**
- ビジネスロジックがDomain層に集約
- 変更の影響範囲が明確（レイヤー境界で隔離）
- 新機能追加時の設計指針が明確

### 3. **再利用性の向上**
- Domain Serviceは他の画面でも利用可能
- Use Caseは複数のViewModelから呼び出し可能
- Infrastructure実装を交換可能（例: AVFoundation → 他のライブラリ）

### 4. **Clean Architectureの実現**
- 依存性逆転の原則が正しく適用される
- Infrastructure詳細がDomain層に漏れない
- テスト可能性と変更容易性の両立

---

## 次のアクション

1. ✅ **このドキュメントのレビュー**: アーキテクチャ方向性の合意
2. ⏳ **ステップ1の着手**: `RecordingSession` Entityの実装とテスト
3. ⏳ **レースコンディション問題の根本解決**: `PitchDetectionService`実装で対応
4. ⏳ **段階的移行**: 既存機能を壊さずに新アーキテクチャへ移行

---

## 補足: なぜ今のアーキテクチャではテストできないのか

### 現状の問題
```
PitchDetectionViewModel (Presentation)
  ↓ 直接参照
AVFoundation Task管理 (Infrastructure詳細)
```

- Presentation層がInfrastructure層の実装詳細（Taskライフサイクル）に依存
- Mock ObjectsではTaskの非同期タイミングを再現できない
- テストで「await task?.value」の効果を検証できない

### リファクタリング後
```
PitchDetectionViewModel (Presentation)
  ↓ Protocolで抽象化
PitchDetectionService (Domain Interface)
  ↓ 実装
RealtimePitchDetectionService (Infrastructure)
  ↓ 非同期処理の詳細を隠蔽
AVFoundation Task管理
```

- Presentation層はDomain Interfaceのみに依存
- Infrastructure層で非同期処理を完全にカプセル化
- テストではMock Serviceで同期的に振る舞いを検証可能
- 統合テストで実際の非同期動作を検証可能

この設計により、**レースコンディション対策をInfrastructure層で完結**させ、**Presentation層はビジネスロジックのテストに集中**できます。
