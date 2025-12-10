# エクササイズ機能 設計ドキュメント

## 概要

プリセットを組み合わせた練習メニュー（エクササイズ）を定義・実行できる機能。
プリセット機能を基盤として構築される上位機能。

---

## コンセプト

### 階層構造

```
エクササイズ（定義）
├── プリセット1 (ウォームアップ: 5トーン、C3、100BPM)
├── プリセット2 (中音域: 5トーン、E3、120BPM)
└── プリセット3 (高音域: オクターブ、G3、140BPM)

エクササイズセッション（実施結果）
├── 録音1 ← プリセット1で録音
├── 録音2 ← プリセット2で録音
└── 録音3 ← プリセット3で録音
```

### 関連性

| エンティティ | 役割 | 関連 |
|------------|------|------|
| `Exercise` | 練習メニューの定義 | `[ScalePreset]` を保持 |
| `ExerciseSession` | 1回の実施結果 | `[Recording]` をグルーピング |
| `ScalePreset` | スケール設定の再利用単位 | Exercise から参照される |
| `Recording` | 個々の録音 | ScaleSettings の値を保持 |

**重要**: Recording と Preset は直接紐づかない。セッション単位で録音がグルーピングされる。

---

## 技術設計

### エンティティ定義

```swift
public struct Exercise: Identifiable, Codable, Equatable {
    public let id: UUID
    public var name: String
    public var presets: [ScalePresetSettings]  // プリセット設定の配列
    public let createdAt: Date
    public var updatedAt: Date
}

public struct ExerciseSession: Identifiable, Codable, Equatable {
    public let id: UUID
    public let exerciseId: UUID
    public let recordings: [RecordingId]  // 録音IDの配列
    public let startedAt: Date
    public let completedAt: Date?
}
```

### ID の考え方

- **ID は端末固有**: UUID はローカルで生成、エクスポートしない
- **インポート時は新規ID**: 値のみインポートし、新しいIDを割り当て

```swift
// エクスポート: 値のみ
{
  "name": "朝のウォームアップ",
  "presets": [
    { "scaleType": "fiveTone", "startPitchIndex": 12, ... },
    { "scaleType": "octaveRepeat", "startPitchIndex": 16, ... }
  ]
}

// インポート: 新規IDで作成
Exercise(
    id: UUID(),           // 端末で新規生成
    name: importedName,
    presets: importedPresets,
    createdAt: Date()
)
```

---

## 操作フロー

### エクササイズ作成

1. 「新規エクササイズ」をタップ
2. エクササイズ名を入力
3. プリセットを順番に追加
   - 既存プリセットから選択
   - または新規プリセットを作成
4. 保存

### エクササイズ実行

1. エクササイズを選択
2. 「開始」をタップ
3. 各プリセットを順番に実行
   - プリセット1で録音 → 完了
   - プリセット2で録音 → 完了
   - ...
4. すべて完了 → セッション保存

### エクササイズ分析

1. 過去のセッション一覧から選択
2. セッション内の録音一覧を表示
3. 各録音の分析を閲覧
4. セッション全体のサマリー（将来）

---

## インポート/エクスポート

### エクスポート形式

```json
{
  "type": "exercise",
  "version": "1.0",
  "data": {
    "name": "朝のウォームアップルーティン",
    "presets": [
      {
        "scaleType": "fiveTone",
        "startPitchIndex": 12,
        "tempo": 100,
        "keyProgressionPattern": "ascendingThenDescending",
        "ascendingKeyCount": 3,
        "descendingKeyCount": 3,
        "ascendingKeyStepInterval": 1,
        "descendingKeyStepInterval": 1
      },
      {
        "scaleType": "octaveRepeat",
        "startPitchIndex": 16,
        "tempo": 120,
        ...
      }
    ]
  }
}
```

### インポート処理

1. JSONをパース
2. 各プリセット設定を検証
3. 新しいExerciseを作成（新規ID）
4. ローカルに保存

---

## プリセット機能との関係

### 共通点

- 同じ `ScalePresetSettings` を使用
- 同じ永続化パターン（UserDefaults → 将来Core Data）
- 同じインポート/エクスポートの考え方

### 違い

| 観点 | プリセット | エクササイズ |
|------|-----------|-------------|
| 単位 | 単一のスケール設定 | プリセットの集合 |
| 使用場面 | 単発の録音 | 一連の練習メニュー |
| 結果 | 1つの録音 | 録音のグループ（セッション） |

### 実装の流れ

1. **Phase 1**: プリセット機能（単体での保存・読み込み）
2. **Phase 2**: エクササイズ機能（プリセットの集合）
3. **Phase 3**: セッション管理（録音のグルーピング）
4. **Phase 4**: インポート/エクスポート

---

## 将来の拡張案

1. **エクササイズテンプレート**: アプリ提供の標準エクササイズ
2. **プリセット間の休憩設定**: 自動休憩タイマー
3. **進捗トラッキング**: セッション履歴の分析
4. **共有機能**: 他ユーザーとエクササイズを共有
5. **カレンダー連携**: 練習スケジュールの管理

---

## 作成日

2024-11-22

## ステータス

設計中・プリセット機能の実装後に着手予定
