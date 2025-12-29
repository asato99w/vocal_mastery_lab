# RecordingListUITests シナリオ設計書

## 概要

録音一覧画面のUIテストを2つの包括的なテストケースに統合し、実際のユーザーフローに沿った堅牢な検証を行う。

---

## テスト実行方法

テストの実行には専用スクリプトを使用してください。

### 全UIテスト実行
```bash
cd vocal_mastery_lab_app
./scripts/test-runner.sh ui
```

### RecordingListUITestsのみ実行
```bash
cd vocal_mastery_lab_app
./scripts/test-runner.sh ui RecordingListUITests
```

### 個別テスト実行
```bash
# 包括的フローテスト
./scripts/test-runner.sh ui RecordingListUITests/testRecordingListComprehensiveFlow

# 音源切り替えテスト
./scripts/test-runner.sh ui RecordingListUITests/testAudioSourceSwitchingWithExtraction
```

### 注意事項
- シミュレータ名は `iPhone 16` を使用（スクリプト内で設定済み）
- 並列テストは無効化されています（安定性向上のため）
- テスト実行前にシミュレータが起動していることを確認

---

## テスト構成

| テスト名 | 目的 | 想定時間 |
|---------|------|---------|
| `testRecordingListComprehensiveFlow` | 録音・再生・操作・削除の完全フロー | ~40秒 |
| `testAudioSourceSwitchingWithExtraction` | 抽出済み/未抽出での音源切り替え | ~60秒 |

---

## 1. testRecordingListComprehensiveFlow

### ユーザーストーリー
複数の録音を作成し、再生・切り替え・削除を行い、残った録音も正常に動作することを確認

### フェーズ詳細

#### フェーズ1: 準備
- 録音A作成（1秒）
- 録音B作成（1秒）
- 一覧画面へ遷移

#### フェーズ2: リスト表示検証
| アサート | 内容 |
|---------|------|
| ✓ | `cells.count >= 2` |
| ✓ | `menuButtons.count >= 2` |
| ✓ | `PlaybackControlPanel exists` |
| ✓ | `PlayPauseButton exists` |
| ✓ | `PreviousButton exists && !isEnabled` |
| ✓ | `NextButton exists && !isEnabled` |
| ✓ | `Slider exists` |

#### フェーズ3: 再生開始と状態変化
- 録音Aタップ → 再生開始

| アサート | 内容 |
|---------|------|
| ✓ | `PlayPauseButton` → pause アイコン |
| ✓ | `Slider isEnabled` |
| ✓ | `CurrentTime` 表示 |
| ✓ | `TotalTime` 表示 |
| ✓ | 0.5秒後 `CurrentTime` 進行 |
| ✓ | `PreviousButton !isEnabled` |
| ✓ | `NextButton isEnabled` |

#### フェーズ4: トグル再生
- 同じ録音Aタップ → 一時停止

| アサート | 内容 |
|---------|------|
| ✓ | `PlayPauseButton` → play アイコン |

- 同じ録音Aタップ → 再生再開

| アサート | 内容 |
|---------|------|
| ✓ | `PlayPauseButton` → pause アイコン |

#### フェーズ5: 録音切り替え（次へ/前へ）
- 次へボタンタップ → 録音Bへ

| アサート | 内容 |
|---------|------|
| ✓ | `CurrentTime` リセット |
| ✓ | `PlayPauseButton` → pause アイコン |
| ✓ | `PreviousButton isEnabled` |
| ✓ | `NextButton !isEnabled` |

- 前へボタンタップ → 録音Aへ

| アサート | 内容 |
|---------|------|
| ✓ | `CurrentTime` リセット |
| ✓ | `PlayPauseButton` → pause アイコン |
| ✓ | `PreviousButton !isEnabled` |
| ✓ | `NextButton isEnabled` |

#### フェーズ6: 別録音タップで切り替え
- 録音Bを直接タップ

| アサート | 内容 |
|---------|------|
| ✓ | `CurrentTime` リセット |
| ✓ | `PlayPauseButton` → pause アイコン |
| ✓ | `PreviousButton isEnabled` |
| ✓ | `NextButton !isEnabled` |

#### フェーズ7: 再生中の削除
- 録音Bを再生中に削除

| アサート | 内容 |
|---------|------|
| ✓ | `DeleteConfirmButton exists` |
| ✓ | 削除後 `cells.count == initialCount - 1` |

#### フェーズ8: 残り録音の再生確認（堅牢性）
- 録音Aタップ → 再生開始

| アサート | 内容 |
|---------|------|
| ✓ | `PlayPauseButton` → pause アイコン |
| ✓ | `Slider isEnabled` |
| ✓ | `CurrentTime` 進行 |
| ✓ | `PreviousButton !isEnabled` |
| ✓ | `NextButton !isEnabled` |
| ✓ | `AudioSourcePicker exists` |
| ✓ | `AudioSourceButton_original isEnabled` |
| ✓ | `AudioSourceButton_vocal !isEnabled` |
| ✓ | `AudioSourceButton_instrumental !isEnabled` |

---

## 2. testAudioSourceSwitchingWithExtraction

### ユーザーストーリー
抽出済み/未抽出の録音を切り替えながら、音源ピッカーの状態と再生動作を確認

### フェーズ詳細

#### フェーズ1: 準備（2件録音）
- 録音A作成（1秒）
- 録音B作成（1秒）
- 一覧画面へ遷移

#### フェーズ2: 録音Aのみ抽出
- 録音Aのメニュー → ボーカル抽出
- 抽出完了待機
- 一覧に戻る

| アサート | 内容 |
|---------|------|
| ✓ | 録音A: `ExtractionIndicators` 存在 |
| ✓ | 録音B: `ExtractionIndicators` 不在 |

#### フェーズ3: 抽出済み録音Aを再生
- 録音Aタップ

| アサート | 内容 |
|---------|------|
| ✓ | `PlayPauseButton` → pause アイコン |
| ✓ | `Slider isEnabled` |
| ✓ | `CurrentTime == "0:00"` |
| ✓ | `AudioSourcePicker` 表示 |
| ✓ | `Original` 選択状態 & 有効 |
| ✓ | `Vocal` 非選択 & 有効 |
| ✓ | `Instrumental` 非選択 & 有効 |
| ✓ | `PreviousButton !isEnabled` |
| ✓ | `NextButton isEnabled` |

#### フェーズ4: 音源切り替え（Vocal）
- Vocal ボタンタップ

| アサート | 内容 |
|---------|------|
| ✓ | `Original` 非選択 |
| ✓ | `Vocal` 選択状態 |
| ✓ | `Instrumental` 非選択 |
| ✓ | `CurrentTime` リセット |
| ✓ | `PlayPauseButton` → pause アイコン |
| ✓ | 0.3秒後 `CurrentTime` 進行 |

#### フェーズ5: 音源切り替え（Instrumental）
- Instrumental ボタンタップ

| アサート | 内容 |
|---------|------|
| ✓ | `Original` 非選択 |
| ✓ | `Vocal` 非選択 |
| ✓ | `Instrumental` 選択状態 |
| ✓ | `CurrentTime` リセット |
| ✓ | `PlayPauseButton` → pause アイコン |
| ✓ | 0.3秒後 `CurrentTime` 進行 |

#### フェーズ6: 未抽出録音Bへ切り替え（次へボタン）
- NextButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | `CurrentTime` リセット |
| ✓ | `PlayPauseButton` → pause アイコン |
| ✓ | `AudioSourcePicker` 表示 |
| ✓ | `Original` 選択状態 & 有効 |
| ✓ | `Vocal` 非選択 & 無効 |
| ✓ | `Instrumental` 非選択 & 無効 |
| ✓ | `PreviousButton isEnabled` |
| ✓ | `NextButton !isEnabled` |
| ✓ | 0.3秒後 `CurrentTime` 進行 |

#### フェーズ7: 抽出済み録音Aへ戻る（前へボタン）
- PreviousButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | `CurrentTime` リセット |
| ✓ | `PlayPauseButton` → pause アイコン |
| ✓ | `Original` 選択状態 & 有効（デフォルトに戻る） |
| ✓ | `Vocal` 非選択 & 有効 |
| ✓ | `Instrumental` 非選択 & 有効 |
| ✓ | `PreviousButton !isEnabled` |
| ✓ | `NextButton isEnabled` |

#### フェーズ8: 録音B直接タップで切り替え
- 録音Bセルをタップ

| アサート | 内容 |
|---------|------|
| ✓ | `CurrentTime` リセット |
| ✓ | `PlayPauseButton` → pause アイコン |
| ✓ | `Original` 選択状態 & 有効 |
| ✓ | `Vocal` 無効 |
| ✓ | `Instrumental` 無効 |
| ✓ | `PreviousButton isEnabled` |
| ✓ | `NextButton !isEnabled` |

#### フェーズ9: 録音A直接タップで切り替え
- 録音Aセルをタップ

| アサート | 内容 |
|---------|------|
| ✓ | `CurrentTime` リセット |
| ✓ | `PlayPauseButton` → pause アイコン |
| ✓ | `Original` 選択状態 & 有効 |
| ✓ | `Vocal` 有効 |
| ✓ | `Instrumental` 有効 |
| ✓ | `PreviousButton !isEnabled` |
| ✓ | `NextButton isEnabled` |

#### フェーズ10: Vocal選択状態で録音B→Aへ戻った時
- Vocal ボタンタップ（録音AでVocal選択）

| アサート | 内容 |
|---------|------|
| ✓ | `Vocal` 選択状態 |

- 録音Bセルをタップ

| アサート | 内容 |
|---------|------|
| ✓ | `Original` 選択状態（リセット） |
| ✓ | `Vocal` 無効 |

- 録音Aセルをタップ

| アサート | 内容 |
|---------|------|
| ✓ | `Original` 選択状態（リセット、Vocalではない） |
| ✓ | `Vocal` 有効（再度選択可能） |

#### フェーズ11: 一時停止中の音源切り替え
- 録音A再生中 → タップで一時停止

| アサート | 内容 |
|---------|------|
| ✓ | `PlayPauseButton` → play アイコン |

- Vocal ボタンタップ

| アサート | 内容 |
|---------|------|
| ✓ | `Vocal` 選択状態 |
| ✓ | `CurrentTime` リセット |
| ✓ | `PlayPauseButton` → pause アイコン（再生開始） |
| ✓ | 0.3秒後 `CurrentTime` 進行 |

---

## アサート総数

| テスト | フェーズ数 | アサート数（概算） |
|-------|-----------|------------------|
| testRecordingListComprehensiveFlow | 8 | ~45 |
| testAudioSourceSwitchingWithExtraction | 11 | ~75 |
| **合計** | 19 | **~120** |

---

## 検証カバレッジ

- ✅ 複数録音の表示
- ✅ 行タップでの再生開始
- ✅ トグル再生（一時停止/再開）
- ✅ 前へ/次へナビゲーション
- ✅ ボタン状態変化
- ✅ スライダー有効/無効
- ✅ 時間表示と進行
- ✅ 削除確認ダイアログ
- ✅ 再生中録音の削除
- ✅ 削除後の残り録音再生
- ✅ 音源ピッカーの状態
- ✅ 音源切り替え
- ✅ 抽出インジケータ
- ✅ 抽出済み/未抽出の切り替え
- ✅ 音源選択状態のリセット
