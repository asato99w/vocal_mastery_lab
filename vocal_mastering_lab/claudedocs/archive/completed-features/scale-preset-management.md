# スケールプリセット管理機能 設計ドキュメント

## 概要

ユーザーがスケール設定を保存・管理できるプリセット機能を追加する。
練習目的に応じた設定（ウォームアップ、高音域練習など）を素早く切り替えられるようにする。

---

## UI設計

### 1. 録音設定パネルへの統合

```
┌─────────────────────────────────┐
│ スケール設定                      │
├─────────────────────────────────┤
│ プリセット: [マイ設定 ▼] [保存] [管理]│
├─────────────────────────────────┤
│ スケール: 5トーン                  │
│ スタートピッチ: C3                 │
│ テンポ: 120 BPM                   │
│ 進行パターン: 上昇→下降             │
│ ...                              │
└─────────────────────────────────┘
```

録音設定パネルの上部にプリセット選択UIを配置。
- ドロップダウン: 保存済みプリセットから選択
- 保存ボタン: 現在の設定を新規プリセットとして保存
- 管理ボタン: プリセット一覧・編集・削除画面へ遷移

### 2. プリセット選択（ドロップダウン）

```
┌─────────────────────┐
│ プリセットを選択       │
├─────────────────────┤
│ ● 現在の設定         │
│ ○ ウォームアップ      │
│ ○ 高音域練習         │
│ ○ 全音階トレーニング   │
│ ○ デフォルト         │
└─────────────────────┘
```

- 「現在の設定」は未保存の状態を示す
- プリセット選択時、即座に設定が反映される
- 設定変更後は「現在の設定（変更あり）」などの表示

### 3. プリセット保存シート

```
┌─────────────────────────────────┐
│      プリセットを保存              │
├─────────────────────────────────┤
│                                 │
│ 名前: [ウォームアップ________]     │
│                                 │
│ 現在の設定:                      │
│ • 5トーン、C3、120 BPM           │
│ • 上昇→下降、3回/3回             │
│ • 半音/半音                      │
│                                 │
├─────────────────────────────────┤
│     [キャンセル]  [保存]          │
└─────────────────────────────────┘
```

- モーダルシートとして表示
- 名前入力フィールド
- 保存する設定の概要を表示
- 既存名と重複時は上書き確認

### 4. プリセット管理画面

```
┌─────────────────────────────────┐
│ ←  プリセット管理                 │
├─────────────────────────────────┤
│                                 │
│ ┌─────────────────────────────┐ │
│ │ ウォームアップ          [編集]│ │
│ │ 5トーン・C3・120BPM          │ │
│ └─────────────────────────────┘ │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ 高音域練習            ← スワイプで削除
│ │ オクターブ・C4・100BPM       │ │
│ └─────────────────────────────┘ │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ 全音階トレーニング      [編集]│ │
│ │ 5トーン・C3・80BPM           │ │
│ └─────────────────────────────┘ │
│                                 │
│         [+ 新規プリセット]        │
└─────────────────────────────────┘
```

- NavigationViewで遷移
- カード形式でプリセット一覧表示
- 各プリセットに設定概要を表示
- スワイプ削除対応
- 編集ボタンで名前変更

---

## 操作フロー

### プリセット保存
1. 録音設定パネルで設定を調整
2. 「保存」ボタンをタップ
3. プリセット名を入力
4. 「保存」で確定

### プリセット読み込み
1. ドロップダウンをタップ
2. プリセット一覧から選択
3. 設定が即座に反映される

### プリセット削除
1. 「管理」ボタンをタップ
2. プリセット管理画面に遷移
3. 削除したいプリセットをスワイプ
4. 「削除」をタップ

### プリセット名変更
1. 「管理」ボタンをタップ
2. 編集ボタンをタップ
3. 新しい名前を入力
4. 「保存」で確定

---

## 技術設計

### アーキテクチャ構成

```
Domain層:
├── Entities/
│   └── ScalePreset.swift          # プリセットエンティティ
├── Repositories/
│   └── ScalePresetRepositoryProtocol.swift  # リポジトリインターフェース

Application層:
├── UseCases/
│   ├── SaveScalePresetUseCase.swift
│   ├── LoadScalePresetsUseCase.swift
│   └── DeleteScalePresetUseCase.swift

Infrastructure層:
├── Repositories/
│   └── UserDefaultsScalePresetRepository.swift  # UserDefaults実装

Presentation層:
├── ViewModels/
│   └── ScalePresetViewModel.swift
├── Views/
│   ├── PresetSelectorView.swift      # ドロップダウン
│   ├── PresetSaveSheet.swift         # 保存シート
│   └── PresetManagementView.swift    # 管理画面
```

### ScalePreset エンティティ

```swift
public struct ScalePreset: Identifiable, Codable, Equatable {
    public let id: UUID
    public var name: String
    public let settings: ScalePresetSettings
    public let createdAt: Date
    public var updatedAt: Date
}

public struct ScalePresetSettings: Codable, Equatable {
    public let scaleType: ScaleType
    public let startPitchIndex: Int
    public let tempo: Int
    public let keyProgressionPattern: KeyProgressionPattern
    public let ascendingKeyCount: Int
    public let descendingKeyCount: Int
    public let ascendingKeyStepInterval: Int
    public let descendingKeyStepInterval: Int
}
```

### 永続化

UserDefaultsを使用してJSON形式で保存。
- キー: `scale_presets`
- 形式: `[ScalePreset]` の JSON配列

将来的にCore Dataへの移行も検討可能。

---

## ローカライズキー

```
// プリセット関連
"preset.title" = "プリセット"
"preset.save" = "保存"
"preset.manage" = "管理"
"preset.current_settings" = "現在の設定"
"preset.save_title" = "プリセットを保存"
"preset.name_placeholder" = "プリセット名を入力"
"preset.management_title" = "プリセット管理"
"preset.new" = "新規プリセット"
"preset.delete_confirmation" = "このプリセットを削除しますか？"
"preset.edit_name" = "名前を編集"
```

---

## 将来の拡張案

1. **デフォルトプリセット**: アプリ初回起動時に基本プリセットを提供
2. **プリセット共有**: 他ユーザーとプリセットを共有
3. **カテゴリ分け**: 練習目的別にプリセットを整理
4. **お気に入り**: よく使うプリセットを上位表示
5. **iCloud同期**: デバイス間でプリセットを同期

---

## 実装優先度

### Phase 1（MVP）
- [ ] ScalePreset エンティティ
- [ ] UserDefaults永続化
- [ ] 基本的な保存・読み込み・削除
- [ ] シンプルなUI

### Phase 2（改善）
- [ ] プリセット名編集
- [ ] 設定概要表示の改善
- [ ] デフォルトプリセット

### Phase 3（拡張）
- [ ] iCloud同期
- [ ] カテゴリ分け
- [ ] 共有機能

---

## 作成日

2024-11-22

## ステータス

設計完了・実装待ち
