# VocalMasteringLab アーキテクチャ設計書

## アーキテクチャ概要

VocalMasteringLabは**クリーンアーキテクチャ**を基盤とし、VocalisStudioと同様の設計思想を採用。

### 採用アーキテクチャ・設計手法
- **Clean Architecture**: ビジネスロジックの独立性とテスタビリティを確保
- **MVVM Pattern**: Presentation層でのデータバインディング
- **TDD**: テストファーストによる堅牢な実装

## クリーンアーキテクチャ層構造

```
┌─────────────────────────────────────────────┐
│          Presentation Layer                 │
│     (SwiftUI Views, ViewModels)             │
│                    ↓                        │
├─────────────────────────────────────────────┤
│          Application Layer                  │
│        (Use Cases, Application Services)    │
│                    ↓                        │
├─────────────────────────────────────────────┤
│            Domain Layer                     │
│   (Entities, Value Objects,                 │
│         Repository Interfaces)              │
│                    ↑                        │
├─────────────────────────────────────────────┤
│         Infrastructure Layer                │
│   (Repositories, External Services,         │
│        AVFoundation, Core ML)               │
└─────────────────────────────────────────────┘

依存性の方向: 外側 → 内側（Domain Layerは他に依存しない）
```

## ディレクトリ構造

```
VocalMasteringLab/
├── VocalMasteringLab.xcodeproj
├── VocalMasteringLab/
│   ├── App/
│   │   ├── VocalMasteringLabApp.swift
│   │   └── DependencyContainer.swift
│   │
│   ├── Domain/
│   │   ├── Entities/
│   │   │   └── Recording.swift
│   │   │
│   │   ├── ValueObjects/
│   │   │   ├── RecordingId.swift
│   │   │   └── Duration.swift
│   │   │
│   │   ├── RepositoryInterfaces/
│   │   │   ├── AudioRecorderProtocol.swift
│   │   │   └── RecordingRepositoryProtocol.swift
│   │   │
│   │   └── ServiceInterfaces/
│   │       ├── VocalExtractorProtocol.swift
│   │       └── LoggerProtocol.swift
│   │
│   ├── Application/
│   │   └── UseCases/
│   │       ├── StartRecordingUseCase.swift
│   │       ├── StopRecordingUseCase.swift
│   │       └── ExtractVocalUseCase.swift
│   │
│   ├── Infrastructure/
│   │   ├── Audio/
│   │   │   ├── AudioSessionManager.swift
│   │   │   └── AVAudioRecorderWrapper.swift
│   │   │
│   │   ├── VocalExtraction/
│   │   │   └── CoreMLVocalExtractor.swift  # プレースホルダー
│   │   │
│   │   ├── Repositories/
│   │   │   └── FileRecordingRepository.swift
│   │   │
│   │   └── Logging/
│   │       ├── FileLogger.swift
│   │       └── OSLogAdapter.swift
│   │
│   ├── Presentation/
│   │   ├── Views/
│   │   │   ├── RecordingView.swift
│   │   │   └── RecordingListView.swift
│   │   │
│   │   └── ViewModels/
│   │       ├── RecordingViewModel.swift
│   │       └── RecordingListViewModel.swift
│   │
│   └── Resources/
│       └── Info.plist  # Background Mode設定含む
│
└── VocalMasteringLabTests/
    ├── Domain/
    ├── Application/
    ├── Infrastructure/
    └── Presentation/
```

## VocalisStudioからの流用コンポーネント

### そのまま流用
| コンポーネント | 元ファイル |
|---------------|-----------|
| `LoggerProtocol` | VocalisDomain/ServiceInterfaces/LoggerProtocol.swift |
| `FileLogger` | Infrastructure/Logging/FileLogger.swift |
| `OSLogAdapter` | Infrastructure/Logging/OSLogAdapter.swift |
| `RecordingId` | VocalisDomain/ValueObjects/RecordingId.swift |
| `Duration` | VocalisDomain/ValueObjects/Duration.swift |
| `AudioRecorderProtocol` | VocalisDomain/RepositoryInterfaces/AudioRecorderProtocol.swift |

### 改修して流用
| コンポーネント | 改修内容 |
|---------------|----------|
| `AudioSessionManager` | `mixWithOthers`オプション追加 |
| `AVAudioRecorderWrapper` | バックグラウンド継続対応 |

### 新規作成
| コンポーネント | 説明 |
|---------------|------|
| `VocalExtractorProtocol` | ボーカル抽出インターフェース |
| `CoreMLVocalExtractor` | Core ML実装（プレースホルダー） |
| `ExtractVocalUseCase` | ボーカル抽出ユースケース |

## Info.plist 設定

### Background Mode
```xml
<key>UIBackgroundModes</key>
<array>
    <string>audio</string>
</array>
```

### マイク使用説明
```xml
<key>NSMicrophoneUsageDescription</key>
<string>ボーカル録音のためにマイクを使用します</string>
```

## AudioSession設定

```swift
// バックグラウンド録音 + 他アプリ音声との共存
try AVAudioSession.sharedInstance().setCategory(
    .playAndRecord,
    mode: .default,
    options: [.mixWithOthers, .defaultToSpeaker, .allowBluetooth]
)
```
