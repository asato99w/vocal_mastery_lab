# スケール再生音選択機能 - 仕様と実装プラン

## 更新履歴

| 日付 | バージョン | 更新内容 |
|-----|----------|---------|
| 2025-11-08 | 2.0 | iPhone標準搭載のGeneral MIDI音源を追加（8種類）、AVAudioUnitSampler統合、プロトタイプ検証セクション追加 |
| 2025-11-08 | 1.0 | 初版作成（ピアノ + サイン波の2種類） |

## 概要

設定画面でスケール再生時の音源（楽器音）を選択できる機能を実装します。ユーザーは自分の好みや練習目的に応じて、**iPhone標準搭載のGeneral MIDI音源（7種類）+ サイン波**の計8種類から選択できます。

## 機能仕様

### ユーザー要求

- 設定画面からスケール再生音の種類を選択できる
- 選択した音源はアプリ全体で永続化され、次回起動時も保持される
- スケール再生中に選択した音源が使用される

### 音源の種類（iPhone標準搭載MIDI音源）

iOSには**General MIDI (GM)規格の128種類の楽器音**が標準搭載されており、AVAudioUnitSamplerを使用してアクセスできます。ボーカルトレーニングに適した音源を以下の通り選定しました。

**MVP Phase**: ボーカルトレーニングに適した8種類の音源を提供

1. **アコースティック・グランド・ピアノ** (デフォルト)
   - General MIDI Program: 0
   - 最も一般的な音色、親しみやすい
   - 全音域で明瞭なピッチ
   - アイコン: 🎹

2. **エレクトリック・ピアノ**
   - General MIDI Program: 4
   - 明るく華やかな音色
   - ポップス・ジャズに適している
   - アイコン: 🎹✨

3. **アコースティック・ギター（ナイロン弦）**
   - General MIDI Program: 24
   - 柔らかく温かみのある音色
   - 中低音域が豊か
   - アイコン: 🎸

4. **ヴィブラフォン**
   - General MIDI Program: 11
   - 倍音が少なく聞き取りやすい
   - ピッチの確認に適している
   - アイコン: 🎵

5. **マリンバ**
   - General MIDI Program: 12
   - 温かみのある柔らかい音色
   - 中低音域の練習に適している
   - アイコン: 🥁

6. **フルート**
   - General MIDI Program: 73
   - 明瞭で澄んだ音色
   - 高音域の練習に最適
   - アイコン: 🎺

7. **クラリネット**
   - General MIDI Program: 71
   - 中音域が豊かで柔らかい
   - 声楽の音域に近い
   - アイコン: 🎷

8. **サイン波**
   - General MIDI Program: なし（プログラム生成）
   - 純音（純粋な周波数の音）
   - ピッチの正確性を確認しやすい
   - 音楽理論の学習に適している
   - アイコン: 〜

**Phase 2** (将来的な拡張):
- 他のGeneral MIDI音源の追加（ストリングス、ブラスなど）
- カスタム音源のインポート
- 音源プレビュー機能

### UI/UX設計

#### 設定画面の配置

```
設定画面 (SettingsView)
├── サブスクリプション
├── オーディオ設定
│   ├── 音量・検出設定 (既存)
│   └── スケール再生音 (NEW) ← ここに追加
├── 言語設定
├── アプリ情報
└── 規約・ポリシー
```

#### オーディオ設定詳細画面

```
オーディオ設定 (AudioSettingsView)
├── 音量設定
│   ├── スケール再生音量 (既存)
│   └── 録音再生音量 (既存)
├── スケール再生音 (NEW)
│   └── Picker: 8種類の楽器音から選択
├── ピッチ検出感度 (既存)
└── ピッチ検出精度 (既存)
```

#### UI要素詳細

**セクション**: "スケール再生音"

**Picker**:
- スタイル: `.menu` (8種類なのでメニュー形式)
- 表示: アイコン + 楽器名（例: 🎹 アコースティック・グランド・ピアノ）
- 選択中の音源をHStackで表示

**セクション構造**:
```swift
Section {
    HStack {
        Text("再生音")
        Spacer()
        Picker("再生音", selection: $viewModel.scaleSoundType) {
            ForEach(ScaleSoundType.allCases, id: \.self) { type in
                HStack {
                    Text(type.icon)
                    Text(type.displayName)
                }
                .tag(type)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
    }
} header: {
    Text("スケール再生音")
} footer: {
    Text(viewModel.scaleSoundType.description)
}
```

**フッターテキスト**:
選択した音源に応じて動的に説明を表示
```
アコースティック・グランド・ピアノ: 最も一般的な音色、親しみやすい
エレクトリック・ピアノ: 明るく華やかな音色
アコースティック・ギター: 柔らかく温かみのある音色
...
```

## アーキテクチャ設計

### Domain層の設計

#### 1. 音源タイプの定義 (Value Object)

**ファイル**: `VocalisStudio/Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/ScaleSoundType.swift`

```swift
/// Scale playback sound type
public enum ScaleSoundType: String, Codable, CaseIterable, Hashable {
    case acousticGrandPiano     // Acoustic Grand Piano (GM Program 0)
    case electricPiano          // Electric Piano 1 (GM Program 4)
    case acousticGuitar         // Acoustic Guitar (nylon) (GM Program 24)
    case vibraphone             // Vibraphone (GM Program 11)
    case marimba                // Marimba (GM Program 12)
    case flute                  // Flute (GM Program 73)
    case clarinet               // Clarinet (GM Program 71)
    case sineWave               // Pure sine wave (programmatic)

    /// Default sound type
    public static let `default` = ScaleSoundType.acousticGrandPiano

    /// General MIDI Program Number (nil for sine wave)
    public var midiProgram: UInt8? {
        switch self {
        case .acousticGrandPiano:
            return 0
        case .electricPiano:
            return 4
        case .acousticGuitar:
            return 24
        case .vibraphone:
            return 11
        case .marimba:
            return 12
        case .flute:
            return 73
        case .clarinet:
            return 71
        case .sineWave:
            return nil  // Programmatically generated
        }
    }

    /// Human-readable display name
    public var displayName: String {
        switch self {
        case .acousticGrandPiano:
            return "アコースティック・グランド・ピアノ"
        case .electricPiano:
            return "エレクトリック・ピアノ"
        case .acousticGuitar:
            return "アコースティック・ギター"
        case .vibraphone:
            return "ヴィブラフォン"
        case .marimba:
            return "マリンバ"
        case .flute:
            return "フルート"
        case .clarinet:
            return "クラリネット"
        case .sineWave:
            return "サイン波"
        }
    }

    /// Icon for UI display
    public var icon: String {
        switch self {
        case .acousticGrandPiano:
            return "🎹"
        case .electricPiano:
            return "🎹✨"
        case .acousticGuitar:
            return "🎸"
        case .vibraphone:
            return "🎵"
        case .marimba:
            return "🥁"
        case .flute:
            return "🎺"
        case .clarinet:
            return "🎷"
        case .sineWave:
            return "〜"
        }
    }

    /// Description for UI footer
    public var description: String {
        switch self {
        case .acousticGrandPiano:
            return "最も一般的な音色、親しみやすく全音域で明瞭なピッチ"
        case .electricPiano:
            return "明るく華やかな音色、ポップス・ジャズに適している"
        case .acousticGuitar:
            return "柔らかく温かみのある音色、中低音域が豊か"
        case .vibraphone:
            return "倍音が少なく聞き取りやすい、ピッチの確認に適している"
        case .marimba:
            return "温かみのある柔らかい音色、中低音域の練習に適している"
        case .flute:
            return "明瞭で澄んだ音色、高音域の練習に最適"
        case .clarinet:
            return "中音域が豊かで柔らかい、声楽の音域に近い"
        case .sineWave:
            return "純音でピッチを正確に確認、音楽理論の学習に適している"
        }
    }
}
```

#### 2. AudioDetectionSettings の拡張

**ファイル**: `VocalisStudio/Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/AudioDetectionSettings.swift`

```swift
public struct AudioDetectionSettings: Equatable, Codable {
    // 既存のプロパティ
    public let scalePlaybackVolume: Float
    public let recordingPlaybackVolume: Float
    public let rmsSilenceThreshold: Float
    public let confidenceThreshold: Float

    // NEW: スケール再生音タイプ
    public let scaleSoundType: ScaleSoundType

    public init(
        scalePlaybackVolume: Float = 0.5,
        recordingPlaybackVolume: Float = 0.5,
        rmsSilenceThreshold: Float = -40.0,
        confidenceThreshold: Float = 0.8,
        scaleSoundType: ScaleSoundType = .default  // NEW
    ) {
        // ... 既存のバリデーション
        self.scaleSoundType = scaleSoundType  // NEW
    }

    // デフォルト設定も更新
    public static let `default` = AudioDetectionSettings()
}
```

### Application層の設計

特に変更なし。既存のUseCase（StartRecordingWithScaleUseCase）は設定を参照するだけなので、新しいプロパティが自動的に伝播されます。

### Infrastructure層の設計

#### ScalePlayer の実装方針変更

**既存実装**: AVAudioPlayerNodeでPCMバッファを事前生成して再生
**新実装**: AVAudioUnitSamplerでMIDI音源をリアルタイム再生

**理由**:
1. General MIDI音源はAVAudioUnitSamplerでアクセス可能
2. 音源切り替えが容易（MIDI Program Changeで切り替え）
3. メモリ効率が良い（PCMバッファを事前生成する必要がない）
4. 複数音同時再生（和音）が容易

#### ⚠️ 実機の重要な制限事項

iOS実機では以下のリソースが**利用できません**：
- **DLSファイル**: `/System/Library/Components/CoreAudio.component/Contents/Resources/gs_instruments.dls`は存在しない
- **AVAudioUnitSamplerのファクトリープリセット**: `auAudioUnit.factoryPresets`は空（nil）
- **システムGeneral MIDI音源**: 実機にはデフォルトのMIDI音源が含まれていない

シミュレータではこれらのリソースが利用可能ですが、**実機では別の方法が必要です**。

#### 推奨実装方法: SF2 (SoundFont) ファイルのバンドル

**1. GeneralUser GS v1.471（推奨音源）**
- サイズ: 約30MB
- ライセンス: フリーウェア（商用利用可能）
- 品質: General MIDI完全対応、高品質な音源
- ダウンロード: http://www.schristiancollins.com/generaluser.php

**2. 実装方法**
- SF2ファイルをXcodeプロジェクトにBundle Resourceとして追加
- AVAudioUnitSamplerの`loadSoundBankInstrument(at:program:bankMSB:bankLSB:)`でロード
- すべてのプラットフォーム（実機・シミュレータ・macOS）で動作

**3. ライセンス考慮事項**
- GeneralUser GS: フリーウェア、商用利用可能
- アプリにクレジット表記を推奨: 設定画面の「ライセンス」セクション
- クレジット例: "GeneralUser GS SoundFont by S. Christian Collins"
- 詳細: GeneralUser GS配布物のREADME.txtを確認

**ファイル**: `VocalisStudio/VocalisStudio/Infrastructure/Audio/AVAudioEngineScalePlayer.swift`

```swift
final class AVAudioScalePlayer: ScalePlayerProtocol {
    // 既存のプロパティ
    private let engine: AVAudioEngine
    private let samplerNode: AVAudioUnitSampler  // NEW: PlayerNode → Sampler
    private var currentSoundType: ScaleSoundType = .default

    // Sine wave用のPlayerNode（サイン波のみPCMバッファ生成が必要）
    private let sinePlayerNode: AVAudioPlayerNode

    init() {
        engine = AVAudioEngine()
        samplerNode = AVAudioUnitSampler()
        sinePlayerNode = AVAudioPlayerNode()

        // Audio Engineにノードを接続
        engine.attach(samplerNode)
        engine.attach(sinePlayerNode)

        // Output mixerに接続
        engine.connect(samplerNode, to: engine.mainMixerNode, format: nil)
        engine.connect(sinePlayerNode, to: engine.mainMixerNode, format: nil)

        // General MIDI音源をロード
        loadGeneralMIDISoundBank()
    }

    // NEW: SF2サウンドバンクのロード
    private func loadSoundBank() async throws {
        let settings = settingsRepository.get()
        let soundType = settings.scaleSoundType
        let program = soundType.midiProgram ?? 0

        // バンドルされたSF2ファイルをロード（実機・シミュレータ両対応）
        guard let sf2URL = Bundle.main.url(forResource: "GeneralUserGS", withExtension: "sf2") else {
            Logger.scalePlayer.error("[loadSoundBank] SF2 file not found in bundle")
            throw ScalePlayerError.soundBankNotFound
        }

        do {
            try sampler.loadSoundBankInstrument(
                at: sf2URL,
                program: program,
                bankMSB: UInt8(kAUSampler_DefaultMelodicBankMSB),
                bankLSB: UInt8(kAUSampler_DefaultBankLSB)
            )
            Logger.scalePlayer.info("[loadSoundBank] Loaded SF2 program \(program) for \(soundType.displayName)")
        } catch {
            Logger.scalePlayer.error("[loadSoundBank] Failed to load SF2: \(error.localizedDescription)")
            throw ScalePlayerError.soundBankLoadFailed(error.localizedDescription)
        }
    }

    // NEW: 音源を切り替えるメソッド
    func setSoundType(_ soundType: ScaleSoundType) {
        self.currentSoundType = soundType

        // MIDI Program Changeで音源を切り替え（サイン波以外）
        if let midiProgram = soundType.midiProgram {
            try? samplerNode.loadInstrument(
                at: AVAudioUnitSampler.InstrumentType.instrument(
                    patch: midiProgram,
                    bank: AVAudioUnitSampler.GenericGMInstrument
                )
            )
        }
    }

    // 既存のloadScaleElements メソッドを修正
    func loadScaleElements(_ elements: [ScaleElement], tempo: Tempo) async throws {
        // サイン波の場合はPCMバッファ生成が必要
        if currentSoundType == .sineWave {
            let buffers = try await generateSineWaveBuffers(
                for: elements,
                tempo: tempo
            )
            // ... 既存の処理（PlayerNodeでバッファ再生）
        } else {
            // MIDI音源の場合はMIDIイベントとして記録
            self.scaleElements = elements
            self.tempo = tempo
        }
    }

    // NEW: MIDI音源での再生
    func play(muted: Bool) async throws {
        guard currentSoundType != .sineWave else {
            // サイン波は既存のPlayerNode再生を使用
            try await playWithPlayerNode(muted: muted)
            return
        }

        // MIDI音源での再生
        for element in scaleElements {
            switch element {
            case .chordShort(let notes), .chordLong(let notes):
                // 和音を同時に開始
                for note in notes {
                    samplerNode.startNote(
                        note.value,
                        withVelocity: muted ? 0 : 64,  // Velocity 64 (標準)
                        onChannel: 0
                    )
                }
                // 指定時間待機
                try await Task.sleep(nanoseconds: UInt64(element.duration * 1_000_000_000))
                // 和音を同時に停止
                for note in notes {
                    samplerNode.stopNote(note.value, onChannel: 0)
                }

            case .scaleNote(let note):
                // 単音を再生
                samplerNode.startNote(
                    note.value,
                    withVelocity: muted ? 0 : 64,
                    onChannel: 0
                )
                try await Task.sleep(nanoseconds: UInt64(tempo.secondsPerNote * 1_000_000_000))
                samplerNode.stopNote(note.value, onChannel: 0)

            case .silence(let duration):
                // 無音期間
                try await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
            }
        }
    }
}
```

#### 技術的な詳細

**AVAudioUnitSamplerの主要メソッド**:
1. `loadInstrument(at:)`: General MIDI音源をロード
   - `patch`: MIDI Program Number (0-127)
   - `bank`: `AVAudioUnitSampler.GenericGMInstrument`（General MIDI）
2. `startNote(_:withVelocity:onChannel:)`: MIDI Note Onイベント
3. `stopNote(_:onChannel:)`: MIDI Note Offイベント

**音源の2つの再生方式**:
1. **MIDI音源** (7種類): AVAudioUnitSamplerでリアルタイム再生
2. **サイン波** (1種類): 既存のAVAudioPlayerNodeでPCMバッファ再生

**メリット**:
- メモリ効率: 音源データを事前生成する必要がない
- 和音再生: 複数のMIDI Noteを同時に開始/停止できる
- 切り替え容易: MIDI Program Changeで即座に切り替え

### Presentation層の設計

#### 1. AudioSettingsViewModel の更新

**ファイル**: `VocalisStudio/VocalisStudio/Presentation/ViewModels/AudioSettingsViewModel.swift`

```swift
@MainActor
final class AudioSettingsViewModel: ObservableObject {
    // 既存のプロパティ
    @Published var scalePlaybackVolume: Float
    @Published var recordingPlaybackVolume: Float
    @Published var detectionSensitivity: AudioDetectionSettings.DetectionSensitivity
    @Published var confidenceThreshold: Float

    // NEW: スケール再生音タイプ
    @Published var scaleSoundType: ScaleSoundType

    init(repository: AudioSettingsRepositoryProtocol) {
        // ... 既存の初期化
        let settings = repository.get()

        // NEW
        self.scaleSoundType = settings.scaleSoundType
    }

    private func buildCurrentSettings() -> AudioDetectionSettings {
        AudioDetectionSettings(
            scalePlaybackVolume: scalePlaybackVolume,
            recordingPlaybackVolume: recordingPlaybackVolume,
            rmsSilenceThreshold: detectionSensitivity.rmsThreshold,
            confidenceThreshold: confidenceThreshold,
            scaleSoundType: scaleSoundType  // NEW
        )
    }
}
```

#### 2. AudioSettingsView の更新

**ファイル**: `VocalisStudio/VocalisStudio/Presentation/Views/AudioSettingsView.swift`

```swift
struct AudioSettingsView: View {
    var body: some View {
        Form {
            // 既存のセクション...

            // NEW: スケール再生音セクション
            Section {
                HStack {
                    Text("再生音")
                        .font(.body)
                    Spacer()
                    Picker("再生音", selection: $viewModel.scaleSoundType) {
                        ForEach(ScaleSoundType.allCases, id: \.self) { type in
                            HStack {
                                Text(type.icon)
                                Text(type.displayName)
                            }
                            .tag(type)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                }
            } header: {
                Text("スケール再生音")
            } footer: {
                Text(viewModel.scaleSoundType.description)
            }

            // 既存のセクション...
        }
    }
}
```

**UI挙動**:
- セクション内にHStackで「再生音」ラベルとPickerを配置
- Pickerは`.menu`スタイルで、タップすると8種類の音源リストを表示
- 選択中の音源はアイコン + 名前で表示（例: 🎹 アコースティック・グランド・ピアノ）
- フッターには選択中の音源の詳細説明を動的に表示
```

## 実装手順（TDD）

### Phase 1: Domain層の実装

#### Step 1: ScaleSoundType の実装

1. **🔴 Red**: テストを作成
   - ファイル: `ScaleSoundTypeTests.swift`
   - テスト内容:
     - すべてのケース（8種類）が定義されていること
     - `displayName`, `icon`, `description`が各音源で正しいこと
     - `midiProgram`がGeneral MIDI仕様に準拠していること
     - デフォルトが`.acousticGrandPiano`であること
     - `Codable`, `CaseIterable`, `Hashable`に準拠していること
     - サイン波のみ`midiProgram`が`nil`であること

2. **🟢 Green**: 実装
   - ファイル: `ScaleSoundType.swift`
   - 上記仕様通りに実装

3. **🔵 Refactor**: コードの整理

#### Step 2: AudioDetectionSettings の拡張

1. **🔴 Red**: 既存テストの更新 + 新規テスト
   - ファイル: `AudioDetectionSettingsTests.swift`
   - テスト内容:
     - デフォルト設定に`scaleSoundType`が含まれること
     - `scaleSoundType`のエンコード/デコードが正しいこと
     - カスタム`scaleSoundType`での初期化が正しいこと

2. **🟢 Green**: 実装
   - ファイル: `AudioDetectionSettings.swift`
   - プロパティ追加と初期化の更新

3. **🔵 Refactor**: コードの整理

### Phase 2: Presentation層の実装

#### Step 3: AudioSettingsViewModel の更新

1. **🔴 Red**: テストを更新
   - ファイル: `AudioSettingsViewModelTests.swift`
   - テスト内容:
     - `scaleSoundType`が正しく初期化されること
     - `scaleSoundType`の変更が`hasChanges`に反映されること
     - 保存時に`scaleSoundType`が永続化されること
     - リセット時にデフォルト値に戻ること

2. **🟢 Green**: 実装
   - ファイル: `AudioSettingsViewModel.swift`
   - `@Published var scaleSoundType`を追加
   - `buildCurrentSettings()`を更新

3. **🔵 Refactor**: コードの整理

#### Step 4: AudioSettingsView の更新

1. **🔴 Red**: UIテストを作成
   - ファイル: `AudioSettingsUITests.swift`（新規）
   - テスト内容:
     - スケール再生音セクションが表示されること
     - Pickerで音源を選択できること
     - 選択した音源が保存されること
     - リセット時にデフォルトに戻ること

2. **🟢 Green**: 実装
   - ファイル: `AudioSettingsView.swift`
   - スケール再生音セクションを追加

3. **🔵 Refactor**: UIコードの整理

### Phase 3: Infrastructure層の実装

#### Step 5: SF2ファイルの準備（前提条件）

1. **GeneralUser GS SF2ファイルのダウンロードと追加**
   - ダウンロード: http://www.schristiancollins.com/generaluser.php
   - ファイル名: `GeneralUserGS.sf2`
   - サイズ: 約30MB
   - Xcodeプロジェクトへの追加:
     1. Xcodeでプロジェクトを開く
     2. `VocalisStudio/Resources/` ディレクトリにSF2ファイルを追加
     3. "Copy items if needed" をチェック
     4. Target: VocalisStudio を選択
     5. "Add to targets" で VocalisStudio にチェック

2. **Bundle Resourceとしての確認**
   - Build Phases → Copy Bundle Resources にSF2ファイルが含まれていることを確認
   - Build Settings → "Excluded Source File Names" にSF2ファイルが含まれていないことを確認

#### Step 6: ScalePlayer のSF2統合

1. **🔴 Red**: テストを作成
   - ファイル: `AVAudioEngineScalePlayerTests.swift`
   - テスト内容:
     - AVAudioUnitSamplerの初期化が正しいこと
     - SF2ファイルのロードが成功すること（実機・シミュレータ両方）
     - 各MIDI音源での再生が正しいこと（7種類）
     - `loadScale()`で音源設定が反映されること
     - 和音再生が正しく動作すること（chordShort, chordLong）
     - ミュート再生が正しく動作すること（velocity=0）

2. **🟢 Green**: 実装
   - ファイル: `AVAudioEngineScalePlayer.swift`
   - `loadSoundBank()` メソッドの実装:
     - SF2ファイルのBundleからの読み込み
     - General MIDI Program番号での楽器ロード
     - エラーハンドリング（SF2ファイルが見つからない場合など）
   - シミュレータと実機の両方で動作することを確認

3. **🔵 Refactor**: コードの整理
   - エラーハンドリングの強化
   - ロギングの追加
   - 不要なフォールバックロジックの削除

### Phase 4: 統合テスト

#### Step 7: End-to-End テスト

1. **🔴 Red**: UIテストを作成
   - ファイル: `ScaleSoundSelectionE2ETests.swift`
   - テスト内容:
     - 設定画面で音源を変更
     - 録音画面でスケール再生を開始
     - 選択した音源で再生されることを確認

2. **🟢 Green**: 既存コードの接続
   - `StartRecordingWithScaleUseCase`が設定を正しく読み込むことを確認
   - ScalePlayerに音源設定が渡されることを確認

3. **🔵 Refactor**: 統合部分の最適化

## データ永続化

### UserDefaults キー

既存の`AudioSettingsRepository`を使用します。`AudioDetectionSettings`が`Codable`なので、自動的に新しいプロパティも永続化されます。

**保存場所**: `UserDefaults.standard`

**キー**: `"audio_settings"` (既存)

**マイグレーション**:
既存ユーザーのデータには`scaleSoundType`が含まれていないため、デコード時にデフォルト値（`.acousticGrandPiano`）が使用されます。

**実装方法**:
```swift
// AudioDetectionSettings.swift
extension AudioDetectionSettings {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // 既存のプロパティをデコード
        scalePlaybackVolume = try container.decode(Float.self, forKey: .scalePlaybackVolume)
        recordingPlaybackVolume = try container.decode(Float.self, forKey: .recordingPlaybackVolume)
        rmsSilenceThreshold = try container.decode(Float.self, forKey: .rmsSilenceThreshold)
        confidenceThreshold = try container.decode(Float.self, forKey: .confidenceThreshold)

        // NEW: scaleSoundTypeをオプショナルでデコード（既存データ対応）
        scaleSoundType = try container.decodeIfPresent(
            ScaleSoundType.self,
            forKey: .scaleSoundType
        ) ?? .default  // nilの場合はデフォルト値を使用
    }
}
```

**動作確認**:
- 既存ユーザー: 初回起動時にアコースティック・グランド・ピアノが自動設定される
- 新規ユーザー: デフォルトでアコースティック・グランド・ピアノが設定される
- 設定変更後: 選択した音源が永続化される

## テストカバレッジ目標

- **Domain層**: 100%（Value Objectは完全にテスト可能）
- **Presentation層**: 90%（ViewModelのロジック）
- **Infrastructure層**: 80%（AVFoundationとの統合部分）
- **UI層**: 60%（重要なユーザーフロー）

## 実装見積もり

**合計**: 約12-16時間

- Phase 1 (Domain): 2-3時間
  - ScaleSoundType実装（8種類）: 1.5時間
  - AudioDetectionSettings更新: 0.5-1時間
  - MIDI Program番号のテスト: 0.5時間
- Phase 2 (Presentation): 3-4時間
  - ViewModel更新: 1時間
  - View更新（menu Picker対応）: 1.5-2時間
  - UIテスト作成: 0.5-1時間
- Phase 3 (Infrastructure): 5-7時間
  - AVAudioUnitSampler統合: 2-3時間
  - General MIDI音源ロード実装: 1-2時間
  - MIDI音源での再生実装: 1.5-2時間
  - サイン波フォールバック処理: 0.5時間
- Phase 4 (統合テスト): 2時間
  - E2Eテスト作成: 1時間
  - 音源別の動作確認: 1時間

**内訳の詳細**:
- **Domain層**: 8種類の音源定義とMIDI Program番号のマッピング
- **Presentation層**: menu形式のPickerと動的なフッター表示
- **Infrastructure層**: AVAudioUnitSamplerの学習と統合（最も時間がかかる部分）
- **統合テスト**: 各音源での動作確認（実機テストを含む）

## マイルストーン

1. **M1: Domain層完成** ✅ - ScaleSoundType + AudioDetectionSettings更新
2. **M2: UI実装完成** ✅ - 設定画面で音源選択可能
3. **M3: SF2ファイル準備** 🔄 - GeneralUser GS SF2ファイルのダウンロードとプロジェクトへの追加
4. **M4: Infrastructure完成** 🔄 - SF2ファイルを使った音源切り替え実装
5. **M5: E2Eテスト完成** ⏳ - 実機とシミュレータでの全体フローテスト

## リスクと対策

### リスク1: 実機でのシステム音源の利用不可（解決済み）

**リスク**: iOS実機にはDLSファイルやAVAudioUnitSamplerのファクトリープリセットが存在しない
**対策**（✅ 解決済み）:
- GeneralUser GS SF2ファイル（30MB）をアプリにバンドル
- すべてのプラットフォーム（実機・シミュレータ・macOS）で同じSF2ファイルを使用
- シミュレータでも実機と同じ音源を使用することで、実機テストの信頼性を向上

### リスク2: SF2ファイルのアプリサイズへの影響

**リスク**: 30MBのSF2ファイルがアプリサイズを増加させる
**対策**:
- GeneralUser GSは高品質かつ比較的軽量（他のSF2ファイルは100MB以上が多い）
- App Thinningにより、ダウンロードサイズは最適化される
- 必要に応じて、将来的にオンデマンドリソースとして配信することも検討可能

### リスク3: SF2ファイルのライセンス管理

**リスク**: SF2ファイルのライセンス違反
**対策**:
- GeneralUser GSはフリーウェアで商用利用可能
- アプリ内の「ライセンス」セクションにクレジット表記を追加
- SF2ファイルの配布物に含まれるREADME.txtを確認し、ライセンス要件を遵守

### リスク4: 既存データの互換性

**リスク**: 既存ユーザーのデータに`scaleSoundType`がない
**対策**: `Codable`のデフォルト値機能を使用。既存データは自動的にデフォルト（`.acousticGrandPiano`）が設定される

### リスク5: SF2ファイルのロード失敗

**リスク**: 何らかの理由でSF2ファイルのロードが失敗する可能性
**対策**:
- 適切なエラーハンドリングを実装
- ロード失敗時のエラーメッセージをログに記録
- Bundle Resourceとして正しく追加されているか、Build Phasesで確認

## 将来の拡張性

### Phase 2での追加機能案

1. **音源の追加**
   - ギター、バイオリン、フルートなど
   - カスタム音源のインポート機能

2. **音源プレビュー**
   - 設定画面で音源を選択時に試聴できる

3. **音源ごとの詳細設定**
   - リバーブ、エコーなどのエフェクト
   - 音色の微調整

4. **音源のダウンロード**
   - 追加音源をオンデマンドでダウンロード
   - ストレージ管理

## 参考資料

- [AVFoundation Programming Guide](https://developer.apple.com/av-foundation/)
- [Audio Unit Programming Guide](https://developer.apple.com/library/archive/documentation/MusicAudio/Conceptual/AudioUnitProgrammingGuide/)
- Clean Architecture in Swift (既存アーキテクチャ参考)

## 実装前の検証（推奨）

本実装に入る前に、**AVAudioUnitSamplerの動作確認プロトタイプ**を作成することを強く推奨します。

### プロトタイプの目的

1. **技術的実現可能性の確認**
   - iOSでGeneral MIDI音源が実際に利用可能か確認
   - AVAudioUnitSamplerのAPI使用方法を理解
   - 音源の品質が期待に満たすか確認

2. **リスク軽減**
   - 本実装前に技術的な問題を発見
   - 見積もり時間の精度向上
   - 代替案の検討時間を確保

### プロトタイプの実装内容

**所要時間**: 2-3時間

**ファイル**: `VocalisStudio/Prototypes/MIDISamplerPrototype.swift`（新規）

```swift
import AVFoundation
import UIKit

/// AVAudioUnitSampler動作確認プロトタイプ
class MIDISamplerPrototype {
    private let engine = AVAudioEngine()
    private let sampler = AVAudioUnitSampler()

    func setup() {
        // Engine設定
        engine.attach(sampler)
        engine.connect(sampler, to: engine.mainMixerNode, format: nil)

        // General MIDI音源をロード（Program 0: Acoustic Grand Piano）
        do {
            try sampler.loadInstrument(
                at: AVAudioUnitSampler.InstrumentType.instrument(
                    patch: 0,  // Acoustic Grand Piano
                    bank: AVAudioUnitSampler.GenericGMInstrument
                )
            )
            print("✅ MIDI音源ロード成功")
        } catch {
            print("❌ MIDI音源ロード失敗: \(error)")
        }

        // Engineを起動
        do {
            try engine.start()
            print("✅ Audio Engine起動成功")
        } catch {
            print("❌ Audio Engine起動失敗: \(error)")
        }
    }

    func testAllInstruments() {
        let instruments: [(name: String, program: UInt8)] = [
            ("Piano", 0),
            ("Electric Piano", 4),
            ("Acoustic Guitar", 24),
            ("Vibraphone", 11),
            ("Marimba", 12),
            ("Flute", 73),
            ("Clarinet", 71)
        ]

        for instrument in instruments {
            print("\n🎵 Testing: \(instrument.name)")
            testInstrument(program: instrument.program)
            Thread.sleep(forTimeInterval: 2)  // 2秒待機
        }
    }

    private func testInstrument(program: UInt8) {
        // 音源を切り替え
        try? sampler.loadInstrument(
            at: AVAudioUnitSampler.InstrumentType.instrument(
                patch: program,
                bank: AVAudioUnitSampler.GenericGMInstrument
            )
        )

        // C4（Middle C）を再生
        let midiNote: UInt8 = 60
        sampler.startNote(midiNote, withVelocity: 64, onChannel: 0)
        Thread.sleep(forTimeInterval: 1)
        sampler.stopNote(midiNote, onChannel: 0)
    }
}

// 使用方法:
// let prototype = MIDISamplerPrototype()
// prototype.setup()
// prototype.testAllInstruments()
```

### プロトタイプの検証項目

- [ ] シミュレータでGeneral MIDI音源がロードできるか
- [ ] 実機でGeneral MIDI音源がロードできるか
- [ ] 各楽器（7種類）の音質が許容範囲か
- [ ] 和音（複数音同時再生）が正しく動作するか
- [ ] 音源切り替えがスムーズか（遅延なし）
- [ ] メモリ使用量が許容範囲か

### プロトタイプの結果判定

**✅ 成功基準**:
- すべての音源がロードできる
- 音質が実用レベル
- 和音再生が問題なく動作
- 本実装へ進む

**❌ 失敗時の対応**:
- 音源数を減らす（ピアノ + サイン波のみ）
- カスタムサウンドフォント（SF2）の追加を検討
- PCMバッファ生成方式に戻す（元の仕様）

## 承認と開始

**実装開始前の確認事項**:
- [ ] 仕様の承認
- [ ] UI/UXデザインの承認
- [ ] テスト計画の承認
- [ ] 実装スケジュールの承認
- [ ] **プロトタイプ検証の完了**（推奨）

**承認者**: ___________
**承認日**: ___________
**プロトタイプ検証日**: ___________
**実装開始予定日**: ___________
