# VocalMasteringLab 実装プラン

## 概要

VocalisStudioから流用可能なコンポーネントを移植し、バックグラウンド録音機能を実装する。

## フェーズ構成

### Phase 1: プロジェクト基盤

**目標**: Xcodeプロジェクト作成とビルド確認

1. Xcodeプロジェクト作成
   - iOS App テンプレート
   - SwiftUI
   - Bundle ID: com.kazuasato.VocalMasteringLab

2. Info.plist設定
   - Background Mode: audio
   - NSMicrophoneUsageDescription

3. ディレクトリ構造作成
   - App/, Domain/, Application/, Infrastructure/, Presentation/

4. ビルド確認

### Phase 2: Domain層

**目標**: ビジネスロジックの基盤を構築

1. ValueObjects（VocalisStudioから流用）
   - RecordingId
   - Duration

2. Entities
   - Recording

3. RepositoryInterfaces
   - AudioRecorderProtocol（流用）
   - RecordingRepositoryProtocol

4. ServiceInterfaces
   - LoggerProtocol（流用）
   - VocalExtractorProtocol（新規・プレースホルダー）

### Phase 3: Infrastructure層

**目標**: 外部フレームワーク連携を実装

1. Logging（VocalisStudioから流用）
   - FileLogger
   - OSLogAdapter

2. Audio（VocalisStudioから流用・改修）
   - AudioSessionManager（mixWithOthers対応）
   - AVAudioRecorderWrapper

3. Repositories
   - FileRecordingRepository

4. VocalExtraction（プレースホルダー）
   - CoreMLVocalExtractor

### Phase 4: Application層

**目標**: ユースケースを実装

1. UseCases
   - StartRecordingUseCase
   - StopRecordingUseCase
   - ExtractVocalUseCase（プレースホルダー）

### Phase 5: Presentation層

**目標**: UI実装と動作確認

1. ViewModels
   - RecordingViewModel
   - RecordingListViewModel

2. Views
   - RecordingView
   - RecordingListView

3. DependencyContainer

### Phase 6: 動作確認

**目標**: バックグラウンド録音の動作確認

1. シミュレータでビルド・起動
2. 録音開始
3. ホームに戻る（バックグラウンド移行）
4. アプリに戻る
5. 録音停止
6. 再生確認

## 流用ファイル一覧

### Domain層

| 流用元 | 流用先 |
|-------|-------|
| `VocalisDomain/ValueObjects/RecordingId.swift` | `Domain/ValueObjects/RecordingId.swift` |
| `VocalisDomain/ValueObjects/Duration.swift` | `Domain/ValueObjects/Duration.swift` |
| `VocalisDomain/RepositoryInterfaces/AudioRecorderProtocol.swift` | `Domain/RepositoryInterfaces/AudioRecorderProtocol.swift` |
| `VocalisDomain/ServiceInterfaces/LoggerProtocol.swift` | `Domain/ServiceInterfaces/LoggerProtocol.swift` |

### Infrastructure層

| 流用元 | 流用先 | 改修 |
|-------|-------|------|
| `Infrastructure/Logging/FileLogger.swift` | `Infrastructure/Logging/FileLogger.swift` | なし |
| `Infrastructure/Logging/OSLogAdapter.swift` | `Infrastructure/Logging/OSLogAdapter.swift` | なし |
| `Infrastructure/Audio/AudioSessionManager.swift` | `Infrastructure/Audio/AudioSessionManager.swift` | mixWithOthers追加 |
| `Infrastructure/Audio/AVAudioRecorderWrapper.swift` | `Infrastructure/Audio/AVAudioRecorderWrapper.swift` | 簡略化 |
| `Infrastructure/Repositories/FileRecordingRepository.swift` | `Infrastructure/Repositories/FileRecordingRepository.swift` | 簡略化 |

## 改修ポイント

### AudioSessionManager

```swift
// 変更前（VocalisStudio）
try audioSession.setCategory(
    .playAndRecord,
    mode: mode,
    options: [.defaultToSpeaker, .allowBluetooth, .allowBluetoothA2DP]
)

// 変更後（VocalMasteringLab）
try audioSession.setCategory(
    .playAndRecord,
    mode: .default,
    options: [.mixWithOthers, .defaultToSpeaker, .allowBluetooth]
)
```

### AVAudioRecorderWrapper

VocalisStudioから以下を削除して簡略化:
- メータリング機能（不要）
- スケール再生連携（不要）
- 複雑なエラーハンドリング（簡略化）

## 成功基準

- [ ] ビルドが通る
- [ ] 録音開始できる
- [ ] バックグラウンドに移行しても録音継続
- [ ] 他のアプリの音声が中断されない
- [ ] アプリに戻って録音停止できる
- [ ] 録音ファイルを再生できる
