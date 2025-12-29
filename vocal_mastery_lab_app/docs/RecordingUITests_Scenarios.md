# RecordingUITests シナリオ設計書

## 概要

録音画面のUIテストを2つの包括的なテストケースに統合し、実際のユーザーフローに沿った堅牢な検証を行う。

---

## テスト実行方法

```bash
cd vocal_mastery_lab_app

# 全RecordingUITests実行
./scripts/test-runner.sh ui RecordingUITests

# 個別テスト実行
./scripts/test-runner.sh ui RecordingUITests/testRecordingComprehensiveFlow
./scripts/test-runner.sh ui RecordingUITests/testBackingTrackPlayback
```

---

## テスト構成

| テスト名 | 目的 | 想定時間 |
|---------|------|---------|
| `testRecordingComprehensiveFlow` | 録音・カウントダウン・タイマー・停止・最終録音表示の完全フロー | ~30秒 |
| `testBackingTrackPlayback` | バッキングトラック選択・再生・シーク・録音との連携 | ~40秒 |

---

## 1. testRecordingComprehensiveFlow

### ユーザーストーリー
ユーザーが録音を開始し、カウントダウンを経て録音を行い、停止後に最終録音情報を確認する

### 前提条件
- アプリ起動、録音カウントリセット済み
- 録音制限に達していない状態

### フェーズ詳細

#### フェーズ1: 初期状態検証
- 録音画面に遷移

| アサート | 内容 |
|---------|------|
| ✓ | `StartRecordingButton exists && isEnabled` |
| ✓ | `RecordingTimerLabel exists` |
| ✓ | `RecordingTimerLabel == "0:00"` |
| ✓ | `StopRecordingButton !exists` |
| ✓ | `CountdownNumber !exists` |
| ✓ | `RecordingListButton exists` |

#### フェーズ2: 録音開始とカウントダウン
- StartRecordingButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | `RecordingLoadingIndicator exists` (preparing状態) |

- カウントダウン開始

| アサート | 内容 |
|---------|------|
| ✓ | `CountdownNumber exists` |
| ✓ | `CountdownNumber` が 3 → 2 → 1 と変化 |
| ✓ | `StartRecordingButton !exists` |

#### フェーズ3: 録音中の状態検証
- カウントダウン完了後

| アサート | 内容 |
|---------|------|
| ✓ | `StopRecordingButton exists && isEnabled` |
| ✓ | `CountdownNumber !exists` |
| ✓ | `StartRecordingButton !exists` |
| ✓ | `RecordingTimerLabel` が進行（0:00 → 0:01以上） |

- 1秒待機後

| アサート | 内容 |
|---------|------|
| ✓ | `RecordingTimerLabel >= "0:01"` |

#### フェーズ4: 録音停止
- StopRecordingButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | `StopRecordingButton !exists` |
| ✓ | `StartRecordingButton exists && isEnabled` |
| ✓ | `LastRecordingSection exists` |
| ✓ | `LastRecordingDateLabel exists` |
| ✓ | `LastRecordingDurationLabel exists` |
| ✓ | `VocalExtractionButton exists` |

#### フェーズ5: 最終録音からの遷移
- VocalExtractionButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | VocalExtractionView に遷移 |
| ✓ | `抽出開始` ボタン存在 |

- 戻るボタンタップ

| アサート | 内容 |
|---------|------|
| ✓ | 録音画面に戻る |
| ✓ | `LastRecordingSection exists` |

#### フェーズ6: 録音リストへのナビゲーション
- RecordingListButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | RecordingListView に遷移 |
| ✓ | 録音セルが1件以上存在 |

---

## 2. testBackingTrackPlayback

### ユーザーストーリー
複数の録音からバッキングトラックを選択・切り替え、ソース切り替え、再生・シーク操作を行い、バッキングトラック再生中に録音を行う

### 前提条件
- 抽出済み録音が2件以上存在
- 録音画面に遷移済み

### 想定ユースケース

| フェーズ | 操作 | ユーザーの意図・場面 |
|---------|------|---------------------|
| 3 | トラック選択 | 練習したい曲の伴奏を選ぶ |
| 4 | ソース切り替え（停止中） | 「元音源で確認」→「ボーカルだけ聴く」→「伴奏で歌う」と事前確認 |
| 5 | 再生・一時停止 | 伴奏を流してみて、途中で止めてフレーズ確認 |
| 6 | 再生中のソース切り替え | 歌っている途中で「やっぱりボーカル入りで聴きたい」と切り替え |
| 7 | シーク操作 | サビから練習したい、イントロを飛ばしたい |
| 8 | 別トラックへの切り替え | 曲Aの練習を終えて曲Bに切り替え |
| 9 | トラック選択解除 | 伴奏なしで歌いたい、アカペラで録音したい |
| 10 | 再生中の録音開始 | 伴奏を聴きながら実際に歌って録音する |
| 11 | カウントダウン中の状態 | 録音開始直前、伴奏が流れ続けることを確認 |
| 12 | 録音中のプレーヤー操作 | 録音中でも伴奏のシーク・停止ができるか確認 |

### 録音状態ごとのバッキングトラック操作（検証ポイント）

| 状態 | 検証ポイント |
|------|-------------|
| idle（録音前） | トラック選択・ソース切り替え・再生操作すべて可能 |
| countdown（カウントダウン中） | 伴奏再生が継続するか、プレーヤー操作が可能か |
| recording（録音中） | 伴奏再生が継続するか、シーク・停止操作が可能か |

### フェーズ詳細

#### フェーズ1: 準備（抽出済み録音2件の作成）
- 録音A作成（1秒）
- ボーカル抽出実行 → 保存
- 録音画面に戻る
- 録音B作成（1秒）
- ボーカル抽出実行 → 保存
- 録音画面に戻る

| アサート | 内容 |
|---------|------|
| ✓ | 録音A・Bが正常に完了 |
| ✓ | 抽出A・Bが正常に完了 |
| ✓ | `BackingTrackCount` が「2件」を表示 |

#### フェーズ2: バッキングトラックセクションの検証
- 録音画面で初期状態を確認

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackSection exists` |
| ✓ | `BackingTrackLabel exists` |
| ✓ | `BackingTrackPicker exists` |
| ✓ | `BackingTrackCount exists` |
| ✓ | `BackingTrackPlayerView !exists`（トラック未選択時） |
| ✓ | `BackingSourcePicker !exists`（トラック未選択時） |

#### フェーズ3: 最初のトラック選択（録音A）
- BackingTrackPicker タップ → 録音Aを選択

| アサート | 内容 |
|---------|------|
| ✓ | `BackingSourcePicker exists`（ソース選択メニュー表示） |
| ✓ | `BackingTrackPlayerView exists` |
| ✓ | `BackingTrackInfoLabel exists` |
| ✓ | `BackingTrackInfoLabel` に録音Aの情報が表示 |
| ✓ | `BackingTrackPlayPauseButton exists` |
| ✓ | `BackingTrackStopButton exists` |
| ✓ | `BackingTrackSeekSlider exists` |
| ✓ | `BackingTrackCurrentTimeLabel == "0:00"` |
| ✓ | `BackingTrackDurationLabel exists` |

#### フェーズ4: ソース切り替え（停止中）
- BackingSourcePicker → ボーカル選択

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackInfoLabel` に「ボーカル」表示 |
| ✓ | `BackingTrackCurrentTimeLabel == "0:00"`（位置維持） |

- BackingSourcePicker → 伴奏選択

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackInfoLabel` に「伴奏」表示 |
| ✓ | `BackingTrackCurrentTimeLabel == "0:00"`（位置維持） |

- BackingSourcePicker → 元音源選択

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackInfoLabel` に「元音源」表示 |
| ✓ | `BackingTrackCurrentTimeLabel == "0:00"`（位置維持） |

#### フェーズ5: バッキングトラック再生・一時停止
- BackingTrackPlayPauseButton タップ（再生）

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackPlayingIndicator exists` |
| ✓ | `BackingTrackCurrentTimeLabel` が進行 |

- 0.5秒待機

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackCurrentTimeLabel != "0:00"` |

- BackingTrackPlayPauseButton タップ（一時停止）

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackPlayingIndicator !exists` |
| ✓ | 時間進行が停止（一時停止位置を記録） |

- BackingTrackPlayPauseButton タップ（再開）

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackPlayingIndicator exists` |
| ✓ | 一時停止位置から再開 |

#### フェーズ6: 再生中のソース切り替え
- 再生継続中に BackingSourcePicker → ボーカル選択

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackInfoLabel` に「ボーカル」表示 |
| ✓ | `BackingTrackCurrentTimeLabel == "0:00"`（位置リセット） |
| ✓ | `BackingTrackPlayingIndicator exists`（再生継続） |

- 0.3秒待機

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackCurrentTimeLabel != "0:00"`（再生進行） |

- 再生継続中に BackingSourcePicker → 伴奏選択

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackInfoLabel` に「伴奏」表示 |
| ✓ | `BackingTrackCurrentTimeLabel == "0:00"`（位置リセット） |
| ✓ | `BackingTrackPlayingIndicator exists`（再生継続） |

#### フェーズ7: シーク操作
- BackingTrackSeekSlider を50%位置にドラッグ

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackCurrentTimeLabel` が変化 |
| ✓ | `BackingTrackPlayingIndicator exists`（再生継続） |

- BackingTrackStopButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackCurrentTimeLabel == "0:00"` |
| ✓ | `BackingTrackPlayingIndicator !exists` |

#### フェーズ8: 別トラックへの切り替え
- BackingTrackPlayPauseButton タップ（再生開始）
- 0.3秒待機後、BackingTrackPicker タップ → 録音Bを選択

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackInfoLabel` に録音Bの情報が表示 |
| ✓ | `BackingTrackCurrentTimeLabel == "0:00"`（位置リセット） |
| ✓ | `BackingSourcePicker exists` |
| ✓ | ソース選択がデフォルト（元音源）にリセット |

#### フェーズ9: トラック選択解除
- BackingTrackPicker タップ → 「なし」を選択

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackPlayerView !exists` |
| ✓ | `BackingSourcePicker !exists` |
| ✓ | `BackingTrackPlayPauseButton !exists` |

- BackingTrackPicker タップ → 録音Aを再選択

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackPlayerView exists` |
| ✓ | `BackingSourcePicker exists` |
| ✓ | `BackingTrackCurrentTimeLabel == "0:00"` |

#### フェーズ10: バッキングトラック再生中の録音（カウントダウン中・録音中の状態検証含む）
- BackingTrackPlayPauseButton タップ（再生開始）
- 0.3秒待機後、StartRecordingButton タップ

**カウントダウン中の状態検証**

| アサート | 内容 |
|---------|------|
| ✓ | `CountdownNumber exists` |
| ✓ | `BackingTrackPlayerView exists`（プレーヤー表示継続） |
| ✓ | `BackingTrackPlayingIndicator exists`（伴奏再生継続） |

**録音中の状態検証**

- カウントダウン完了後

| アサート | 内容 |
|---------|------|
| ✓ | `StopRecordingButton exists` |
| ✓ | `BackingTrackPlayerView exists`（プレーヤー表示継続） |
| ✓ | `BackingTrackPlayingIndicator exists`（伴奏再生継続） |

**録音中のプレーヤー操作**

- BackingTrackSeekSlider を操作（録音中のシーク確認）

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackCurrentTimeLabel` が変化 |
| ✓ | `StopRecordingButton exists`（録音継続） |

- BackingTrackStopButton タップ（録音中の伴奏停止）

| アサート | 内容 |
|---------|------|
| ✓ | `BackingTrackPlayingIndicator !exists`（伴奏停止） |
| ✓ | `StopRecordingButton exists`（録音は継続） |

**録音停止**

- StopRecordingButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | 録音停止 |
| ✓ | `LastRecordingSection exists` |
| ✓ | `BackingTrackPlayerView exists`（プレーヤー継続表示） |

---

## アサート総数

| テスト | フェーズ数 | アサート数（概算） |
|-------|-----------|------------------|
| testRecordingComprehensiveFlow | 6 | ~30 |
| testBackingTrackPlayback | 10 | ~65 |
| **合計** | 16 | **~95** |

---

## 検証カバレッジ

### 録音機能
- ✅ 録音開始・停止
- ✅ カウントダウン表示
- ✅ タイマー進行
- ✅ 最終録音情報表示
- ✅ ボーカル抽出画面遷移
- ✅ 録音リスト画面遷移

### バッキングトラック機能
- ✅ トラック選択（複数トラック対応）
- ✅ トラック切り替え（再生中）
- ✅ トラック選択解除
- ✅ ソース切り替え（停止中）：元音源/ボーカル/伴奏
- ✅ ソース切り替え（再生中）：位置リセット確認
- ✅ 再生・一時停止・再開
- ✅ シーク操作（スライダー）
- ✅ 停止ボタン（位置リセット）

### 録音 × バッキングトラック連携
- ✅ バッキングトラック再生中の録音開始
- ✅ カウントダウン中の伴奏再生継続
- ✅ 録音中の伴奏再生継続
- ✅ 録音中のプレーヤー操作（シーク・停止）

---

## 主要なaccessibilityIdentifier

| Identifier | 要素 |
|-----------|------|
| `StartRecordingButton` | 録音開始ボタン |
| `StopRecordingButton` | 録音停止ボタン |
| `RecordingTimerLabel` | タイマー表示 |
| `CountdownNumber` | カウントダウン数字 |
| `RecordingLoadingIndicator` | 準備中インジケータ |
| `LastRecordingSection` | 最終録音セクション |
| `LastRecordingDateLabel` | 録音日時 |
| `LastRecordingDurationLabel` | 録音時間 |
| `VocalExtractionButton` | ボーカル抽出ボタン |
| `RecordingListButton` | 録音リストボタン |
| `BackingTrackSection` | バッキングトラックセクション |
| `BackingTrackPicker` | トラック選択メニュー |
| `BackingTrackSourcePicker` | ソース選択メニュー |
| `BackingTrackPlayerView` | プレーヤービュー |
| `BackingTrackPlayPauseButton` | 再生/一時停止ボタン |
| `BackingTrackStopButton` | 停止ボタン |
| `BackingTrackSeekSlider` | シークスライダー |
| `BackingTrackCurrentTimeLabel` | 現在時間 |
| `BackingTrackDurationLabel` | 総時間 |
