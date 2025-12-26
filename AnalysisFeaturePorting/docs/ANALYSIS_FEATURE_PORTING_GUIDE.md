# Analysis Feature Porting Guide

分析機能を別のアプリに移植するためのガイドです。

## 概要

Vocalis Studio の分析機能は以下の機能を提供します:
- **ピッチ検出**: YIN/PYIN/FCPE アルゴリズムによるリアルタイム・オフラインピッチ解析
- **スペクトログラム表示**: 時間-周波数スペクトラムの可視化
- **ピッチグラフ**: ピッチ変動のタイムライン表示
- **ピッチバー**: リアルタイムピッチずれの表示
- **統計分析**: ビブラート、高周波成分、シンガーズフォルマント分析

---

## ファイル一覧と配置場所

### 1. Domain レイヤー (必須・最初に移植)

Domain パッケージとして独立しているため、そのまま SPM パッケージとして追加可能です。

#### エンティティ & 値オブジェクト

| ファイル | 移植先パス | 説明 |
|---------|-----------|------|
| `Packages/VocalisDomain/Sources/VocalisDomain/Entities/AnalysisResult.swift` | `Domain/Entities/` | 分析結果エンティティ |
| `Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/PitchFrame.swift` | `Domain/ValueObjects/` | 単一フレームのピッチデータ |
| `Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/PitchAnalysisData.swift` | `Domain/ValueObjects/` | ピッチ分析データ集合 |
| `Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/DetectedPitch.swift` | `Domain/ValueObjects/` | 検出されたピッチ値 |
| `Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/PitchDetectionAlgorithm.swift` | `Domain/ValueObjects/` | アルゴリズム列挙型 |
| `Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/SpectrogramData.swift` | `Domain/ValueObjects/` | スペクトログラムデータ |
| `Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/RecordingStatistics.swift` | `Domain/ValueObjects/` | 録音統計情報 |
| `Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/VibratoAnalysis.swift` | `Domain/ValueObjects/` | ビブラート分析データ |
| `Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/HighFrequencyAnalysis.swift` | `Domain/ValueObjects/` | 高周波分析データ |
| `Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/SingersFormantAnalysis.swift` | `Domain/ValueObjects/` | シンガーズフォルマント分析 |
| `Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/NoteSegment.swift` | `Domain/ValueObjects/` | ノートセグメント |
| `Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/MIDINote.swift` | `Domain/ValueObjects/` | MIDI ノート |

#### サービスインターフェース

| ファイル | 移植先パス | 説明 |
|---------|-----------|------|
| `Packages/VocalisDomain/Sources/VocalisDomain/ServiceInterfaces/PitchDetectionStrategy.swift` | `Domain/ServiceInterfaces/` | ピッチ検出戦略プロトコル |

#### Domain サービス

| ファイル | 移植先パス | 説明 |
|---------|-----------|------|
| `Packages/VocalisDomain/Sources/VocalisDomain/Services/RecordingStatisticsCalculator.swift` | `Domain/Services/` | 統計計算サービス |
| `Packages/VocalisDomain/Sources/VocalisDomain/Services/VibratoAnalyzer.swift` | `Domain/Services/` | ビブラート分析 |
| `Packages/VocalisDomain/Sources/VocalisDomain/Services/OctaveCorrectionService.swift` | `Domain/Services/` | オクターブ補正 |
| `Packages/VocalisDomain/Sources/VocalisDomain/Services/HighFrequencyAnalyzer.swift` | `Domain/Services/` | 高周波分析 |
| `Packages/VocalisDomain/Sources/VocalisDomain/Services/SingersFormantAnalyzer.swift` | `Domain/Services/` | シンガーズフォルマント分析 |

---

### 2. Infrastructure レイヤー (ピッチ検出エンジン)

#### 分析基盤

| ファイル | 移植先パス | 説明 |
|---------|-----------|------|
| `VocalisStudio/Infrastructure/Analysis/AudioFileAnalyzer.swift` | `Infrastructure/Analysis/` | オーディオファイル分析エンジン |
| `VocalisStudio/Infrastructure/Analysis/AudioFileAnalyzerFactory.swift` | `Infrastructure/Analysis/` | アナライザーファクトリー |
| `VocalisStudio/Infrastructure/Analysis/AnalysisCache.swift` | `Infrastructure/Analysis/` | 分析結果キャッシュ |
| `VocalisStudio/Infrastructure/Analysis/FilePitchDataCache.swift` | `Infrastructure/Analysis/` | ピッチデータキャッシュ |

#### ピッチ検出戦略

| ファイル | 移植先パス | 説明 |
|---------|-----------|------|
| `VocalisStudio/Infrastructure/Analysis/PitchStrategyFactory.swift` | `Infrastructure/Analysis/` | 戦略ファクトリー |
| `VocalisStudio/Infrastructure/Analysis/Strategies/YINStrategy.swift` | `Infrastructure/Analysis/Strategies/` | YIN アルゴリズム |
| `VocalisStudio/Infrastructure/Analysis/Strategies/PYINStrategy.swift` | `Infrastructure/Analysis/Strategies/` | PYIN アルゴリズム |
| `VocalisStudio/Infrastructure/Analysis/Strategies/FCPEStrategy.swift` | `Infrastructure/Analysis/Strategies/` | FCPE アルゴリズム |

---

### 3. Application レイヤー (UseCase)

| ファイル | 移植先パス | 説明 |
|---------|-----------|------|
| `VocalisStudio/Application/UseCases/AnalyzeRecordingUseCase.swift` | `Application/UseCases/` | 録音分析ユースケース |

---

### 4. Presentation レイヤー (UI コンポーネント)

#### メインビュー & ビューモデル

| ファイル | 移植先パス | 説明 |
|---------|-----------|------|
| `VocalisStudio/Presentation/Views/AnalysisView.swift` | `Presentation/Views/` | 分析画面メインビュー |
| `VocalisStudio/Presentation/ViewModels/AnalysisViewModel.swift` | `Presentation/ViewModels/` | 分析画面 ViewModel |
| `VocalisStudio/Presentation/ViewModels/PitchDetectionViewModel.swift` | `Presentation/ViewModels/` | ピッチ検出 ViewModel |

#### 分析画面サブコンポーネント

| ファイル | 移植先パス | 説明 |
|---------|-----------|------|
| `VocalisStudio/Presentation/Views/Analysis/RecordingInfoComponents.swift` | `Presentation/Views/Analysis/` | 録音情報表示 |
| `VocalisStudio/Presentation/Views/Analysis/StatisticsComponents.swift` | `Presentation/Views/Analysis/` | 統計情報表示 |
| `VocalisStudio/Presentation/Views/Analysis/PlaybackComponents.swift` | `Presentation/Views/Analysis/` | 再生制御 |
| `VocalisStudio/Presentation/Views/Analysis/VisualizationComponents.swift` | `Presentation/Views/Analysis/` | 可視化コンポーネント |

#### ピッチバーコンポーネント

| ファイル | 移植先パス | 説明 |
|---------|-----------|------|
| `VocalisStudio/Presentation/Components/PitchBar/PitchBarView.swift` | `Presentation/Components/PitchBar/` | ピッチバー表示 |
| `VocalisStudio/Presentation/Components/PitchBar/PitchBarConstants.swift` | `Presentation/Components/PitchBar/` | 定数 |
| `VocalisStudio/Presentation/Components/PitchBar/PitchDeviationPath.swift` | `Presentation/Components/PitchBar/` | ずれパス描画 |
| `VocalisStudio/Presentation/Components/PitchBar/TargetNoteBarView.swift` | `Presentation/Components/PitchBar/` | ターゲットノートバー |
| `VocalisStudio/Presentation/Components/PitchBar/DeviationScoreView.swift` | `Presentation/Components/PitchBar/` | ずれスコア表示 |

#### ピッチグラフコンポーネント

| ファイル | 移植先パス | 説明 |
|---------|-----------|------|
| `VocalisStudio/Presentation/Components/PitchGraph/PitchGraphRenderer.swift` | `Presentation/Components/PitchGraph/` | グラフ描画 |
| `VocalisStudio/Presentation/Components/PitchGraph/PitchGraphConstants.swift` | `Presentation/Components/PitchGraph/` | 定数 |
| `VocalisStudio/Presentation/Components/PitchGraph/PitchGraphCoordinateSystem.swift` | `Presentation/Components/PitchGraph/` | 座標系 |
| `VocalisStudio/Presentation/Components/PitchGraph/TargetFrequencyCalculator.swift` | `Presentation/Components/PitchGraph/` | ターゲット周波数計算 |

#### スペクトログラムコンポーネント

| ファイル | 移植先パス | 説明 |
|---------|-----------|------|
| `VocalisStudio/Presentation/Components/Spectrogram/SpectrogramRenderer.swift` | `Presentation/Components/Spectrogram/` | スペクトログラム描画 |
| `VocalisStudio/Presentation/Components/Spectrogram/SpectrogramConstants.swift` | `Presentation/Components/Spectrogram/` | 定数 |
| `VocalisStudio/Presentation/Components/Spectrogram/SpectrogramCoordinateSystem.swift` | `Presentation/Components/Spectrogram/` | 座標系 |
| `VocalisStudio/Presentation/Components/Spectrogram/SpectrogramScrollManager.swift` | `Presentation/Components/Spectrogram/` | スクロール管理 |

#### ヘルパー

| ファイル | 移植先パス | 説明 |
|---------|-----------|------|
| `VocalisStudio/Presentation/Helpers/PitchNameHelper.swift` | `Presentation/Helpers/` | ピッチ名変換 |

---

## 依存関係図

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐   │
│  │AnalysisView │→ │AnalysisVM   │→ │PitchDetectionVM    │   │
│  └─────────────┘  └─────────────┘  └────────────────────┘   │
│         │                │                    │              │
│  ┌──────┴──────────────────────────────────────┐            │
│  │ Components: PitchBar, PitchGraph, Spectrogram│            │
│  └──────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌─────────────────────────┐                                │
│  │AnalyzeRecordingUseCase  │                                │
│  └─────────────────────────┘                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Domain Layer (VocalisDomain Package)      │
│  ┌────────────────┐  ┌───────────────────────────────────┐  │
│  │AnalysisResult  │  │ ValueObjects:                     │  │
│  │    Entity      │  │ PitchFrame, SpectrogramData, etc. │  │
│  └────────────────┘  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Services: VibratoAnalyzer, StatisticsCalculator, etc. │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ ServiceInterfaces: PitchDetectionStrategy             │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                      │
│  ┌─────────────────────┐  ┌───────────────────────────────┐ │
│  │ AudioFileAnalyzer   │  │ Strategies: YIN, PYIN, FCPE   │ │
│  └─────────────────────┘  └───────────────────────────────┘ │
│  ┌─────────────────────┐  ┌───────────────────────────────┐ │
│  │ AnalysisCache       │  │ PitchStrategyFactory          │ │
│  └─────────────────────┘  └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 移植手順

### Step 1: Domain パッケージの追加

VocalisDomain パッケージをそのまま SPM 依存として追加するか、ファイルをコピーします。

```swift
// Package.swift に追加する場合
dependencies: [
    .package(path: "../VocalisDomain")
]
```

または、ファイルを直接コピー:
```
Packages/VocalisDomain/Sources/VocalisDomain/
  ├── Entities/
  ├── ValueObjects/
  ├── Services/
  └── ServiceInterfaces/
```

### Step 2: Infrastructure レイヤーのコピー

```
VocalisStudio/Infrastructure/Analysis/
  ├── AudioFileAnalyzer.swift
  ├── AudioFileAnalyzerFactory.swift
  ├── AnalysisCache.swift
  ├── FilePitchDataCache.swift
  ├── PitchStrategyFactory.swift
  └── Strategies/
      ├── YINStrategy.swift
      ├── PYINStrategy.swift
      └── FCPEStrategy.swift
```

### Step 3: Application UseCase のコピー

```
VocalisStudio/Application/UseCases/
  └── AnalyzeRecordingUseCase.swift
```

### Step 4: Presentation レイヤーのコピー

```
VocalisStudio/Presentation/
  ├── Views/
  │   ├── AnalysisView.swift
  │   └── Analysis/
  │       ├── RecordingInfoComponents.swift
  │       ├── StatisticsComponents.swift
  │       ├── PlaybackComponents.swift
  │       └── VisualizationComponents.swift
  ├── ViewModels/
  │   ├── AnalysisViewModel.swift
  │   └── PitchDetectionViewModel.swift
  ├── Components/
  │   ├── PitchBar/
  │   ├── PitchGraph/
  │   └── Spectrogram/
  └── Helpers/
      └── PitchNameHelper.swift
```

### Step 5: DI コンテナへの登録

移植先アプリの DI コンテナに以下を登録:

```swift
// 例: DependencyContainer.swift
container.register(AnalyzeRecordingUseCaseProtocol.self) { resolver in
    AnalyzeRecordingUseCase(
        audioFileAnalyzer: resolver.resolve(AudioFileAnalyzerProtocol.self)!
    )
}

container.register(AudioFileAnalyzerProtocol.self) { resolver in
    AudioFileAnalyzerFactory.create(
        pitchStrategyFactory: resolver.resolve(PitchStrategyFactory.self)!
    )
}
```

---

## 必須の外部依存

1. **AVFoundation** - オーディオファイル読み込み・再生
2. **Accelerate** - FFT/DSP 処理 (スペクトログラム、ピッチ検出)
3. **Combine** - リアクティブプログラミング (ViewModel)
4. **SwiftUI** - UI コンポーネント

---

## ファイル数サマリー

| レイヤー | ファイル数 |
|---------|-----------|
| Domain (エンティティ・値オブジェクト) | 12 |
| Domain (サービス・インターフェース) | 6 |
| Infrastructure | 8 |
| Application | 1 |
| Presentation (Views) | 5 |
| Presentation (ViewModels) | 2 |
| Presentation (Components) | 13 |
| Presentation (Helpers) | 1 |
| **合計** | **48** |

---

## 注意事項

### 1. Recording エンティティへの依存

`AnalysisView` は `Recording` エンティティを参照しています。移植先に同等のエンティティがない場合、`URL` ベースのインターフェースに変更するか、Recording エンティティもコピーする必要があります。

### 2. ローカライズ

UI テキストは日本語でハードコードされている箇所があります。多言語対応が必要な場合は `Localizable.strings` を確認してください。

### 3. スケール設定との連携

ピッチバーやグラフは `ScaleSettings` に依存しています。スケール再生機能がない場合、これらのコンポーネントを簡略化するか、デフォルト設定を使用する必要があります。

---

## 移植前の確認事項 (VocalMasteryLab 向け)

### パス名の読み替え

本ガイドは Vocalis Studio をベースに作成されています。VocalMasteryLab へ移植する際は以下のパス名を読み替えてください：

| ガイド記載 | 実際のパス |
|-----------|-----------|
| `VocalisStudio/` | `VocalMasteryLab/` |

### 作成が必要なファイル

以下のファイルは現在の VocalMasteryLab には存在しないため、移植時に新規作成が必要です：

| ファイル | 作成先パス | 説明 |
|---------|-----------|------|
| `PitchNameHelper.swift` | `Presentation/Helpers/` | ピッチ名変換ヘルパー（周波数→音名変換など） |
| `TargetFrequencyCalculator.swift` | `Presentation/Components/PitchGraph/` | ターゲット周波数計算（ピッチグラフ用） |

### 移植前チェックリスト

移植作業を開始する前に、以下を実行してください：

- [ ] 全テストの実行（`./scripts/test-runner.sh` または Xcode）
- [ ] 現在のビルドが成功することを確認
- [ ] Git でクリーンな状態であることを確認（未コミットの変更がないこと）

---

## 更新履歴

- **2025-12-26**: VocalMasteryLab 向け移植前確認事項を追加
- **2025-12-26**: 初版作成 (v1.6.2 ベース)
