# CSD (Children's Song Dataset) Test Resources

このディレクトリには、RealtimePitchDetectorの精度検証のためのCSDデータセットサンプルを配置します。

## データセット情報

- **名称**: CSD (Children's Song Dataset)
- **出典**: https://zenodo.org/record/4785016
- **サイズ**: 1.9GB (フルセット)
- **ライセンス**: CC BY-NC-SA 4.0 (研究利用可、商用利用不可)
- **内容**:
  - 100曲 (韓国語50曲 + 英語50曲)
  - 各曲2キーで録音 = 計200録音
  - 44.1kHz 16bit WAVファイル
  - MIDI transcription (手動調整済み)
  - CSV onset/offset timing データ

## ディレクトリ構造

```
CSD/
├── audio/   # WAVファイル配置用
├── midi/    # MIDIファイル配置用
├── csv/     # onset/offset timing CSVファイル配置用
└── README.md
```

## 使用方法

### 1. サンプル曲のダウンロード

テスト用に1-2曲をダウンロードして配置します：

```bash
# Zenodoから CSD.zip をダウンロード
# https://zenodo.org/record/4785016

# 解凍後、任意の曲（例: Korean/001.wav）を以下にコピー:
# audio/001.wav
# midi/001.mid
# csv/001.csv
```

### 2. F0変換

MIDIノートからF0 (Hz)への変換は`MIDINote.frequency`プロパティを使用：

```swift
let midiNote = try MIDINote(60)  // C4
let f0 = midiNote.frequency      // 261.63 Hz
```

### 3. CSV構造

CSVファイルには以下の情報が含まれます：
- ノートのonset/offset時刻
- 音節のタイミング
- 歌詞情報

### 4. テスト実装

`CSDAccuracyEvaluationTests.swift`で以下の精度指標を検証：
- **GPE (Gross Pitch Error)**: < 5%
- **FPE (Fine Pitch Error)**: < 10 cent
- **Octave Error Rate**: < 2%

## 注意事項

- このディレクトリのファイルは`.gitignore`に追加し、リポジトリにコミットしない
- ライセンス条件（CC BY-NC-SA 4.0）を遵守すること
- 商用利用不可
- フルデータセットのダウンロードは任意（テストには1-2曲で十分）
