# 音源分離モデル → CoreML 変換の技術的課題と修正報告

## ⚠️ 重要な修正事項（2025-10-21更新）

**当初の報告における誤解**:
本レポートは **Demucs系モデル（HTDemucs）** に特化した検証結果であり、
**「すべての音源分離モデルがCoreML変換不可能」という結論は誤り**です。

**修正後の正確な理解**:
- **Demucs/HTDemucs**: STFT/iSTFTを内部統合 → 複素数演算により **CoreML変換不可** ❌
- **UVR/MDX-Net系**: 振幅スペクトログラム入力（実数のみ） → **CoreML変換可能** ✅
- **Wave-U-Net系**: 時間領域処理（実数のみ） → **CoreML変換可能** ✅

本レポートの内容は「Demucs特有の課題」であり、UVR系モデルには適用されません。

---

## 実行サマリー（Demucs限定）

**結論**: Hybrid Demucs (HTDemucs) の直接的なCoreML変換は**技術的に困難**

**主要課題**:
- Complex64型（複素数）がCoreMLで非サポート
- STFT/iSTFT処理がモデル内部に組み込まれている
- モデルサイズ: 319MB（パラメータ数が大規模）
- 動的形状テンソルの扱い

---

## 1. エラー分析

### 1.1 発生したエラー

```
ValueError: Op "135" (op_type: slice_by_index) Input x="130"
expects tensor or scalar of dtype from type domain
['fp16', 'fp32', 'int32', 'bool']
but got tensor[1,2,2049,220,complex64]
```

### 1.2 エラーの意味

- **発生箇所**: PyTorch → CoreML変換プロセスの88/1774オペレーション時点
- **根本原因**: `complex64`型（複素数float32）の扱い
- **CoreML対応型**: `fp16`, `fp32`, `int32`, `bool` のみ
- **テンソル形状**: `[1, 2, 2049, 220, complex64]`
  - バッチサイズ: 1
  - チャンネル: 2（ステレオ）
  - 周波数ビン: 2049（nFFT=4096想定）
  - 時間フレーム: 220
  - データ型: 複素数64bit

---

## 2. Demucsアーキテクチャと複素数の必要性

### 2.1 Hybrid Demucsの処理フロー

```
入力波形（時間領域）
    ↓
1. STFT（Short-Time Fourier Transform）
    ↓ 複素数スペクトログラム生成
2. スペクトル処理（U-Net等）
    ↓ 複素数演算
3. iSTFT（Inverse STFT）
    ↓
出力波形（時間領域）
```

### 2.2 なぜ複素数が必要か

**STFT出力の性質**:
- **振幅（Magnitude）**: 各周波数成分の強度
- **位相（Phase）**: 各周波数成分のタイミング情報

複素数 = 実部 + 虚部 = 振幅 × e^(i×位相)

**位相情報の重要性**:
- 音声の時間的構造を保持
- 位相を失うと音質が著しく劣化
- 特にボーカル分離では位相の一貫性が品質を左右

### 2.3 Demucsの特徴

HTDemucs (Hybrid Transformer Demucs) の構造:
- **Waveform分岐**: 時間領域処理（位相保持）
- **Spectrogram分岐**: 周波数領域処理（複素数処理）
- **Transformer**: 長距離依存関係のモデリング

モデル内で**STFT/iSTFTを含む**ため、複素数演算が不可避。

---

## 3. CoreMLの制約

### 3.1 サポートされるデータ型

| データ型 | サポート | 用途 |
|---------|---------|------|
| fp32 | ✅ | 標準浮動小数点 |
| fp16 | ✅ | 軽量化・高速化 |
| int32 | ✅ | 整数演算 |
| int8 | ✅ | 量子化 |
| bool | ✅ | 論理演算 |
| **complex64** | ❌ | **複素数（非対応）** |
| **complex128** | ❌ | **倍精度複素数（非対応）** |

### 3.2 なぜCoreMLは複素数をサポートしないか

**設計思想**:
- モバイル推論に最適化
- GPU/Neural Engineでの高速演算
- 一般的なCV/NLPタスクに焦点

**技術的制約**:
- Metal Performance Shaders (MPS) が複素数演算を限定的にサポート
- Neural Engine は実数演算のみ
- モバイルGPUの複素数命令セットが不完全

---

## 4. 変換の試み詳細

### 4.1 実行環境

```
- Python: 3.11.14
- PyTorch: 2.5.0
- TorchAudio: 2.5.0
- CoreMLTools: 8.3.0
- モデル: HDEMUCS_HIGH_MUSDB_PLUS (319MB)
```

### 4.2 変換プロセス

```python
# 1. モデル読み込み（成功）
bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()  # 319MB ダウンロード完了

# 2. モデルトレース（成功）
example_input = torch.randn(1, 2, 220500)  # 5秒, 44.1kHz
traced_model = torch.jit.trace(model, example_input)

# 3. CoreML変換（失敗）
mlmodel = ct.convert(traced_model, ...)
# → 88/1774オペレーション時点でcomplex64エラー
```

### 4.3 失敗箇所の特定

変換プロセスの5%（88/1774）で停止:
- おそらくSTFT直後のスペクトログラム処理
- `slice_by_index`オペレーションで複素数テンソルを扱おうとした時点

---

## 5. 代替アプローチの検討と正確な評価

### 5.1 モデル系統別のCoreML変換可能性

| モデル系統 | 構造 | CoreML変換 | 備考 |
|-----------|------|-----------|------|
| **Demucs/HTDemucs** | STFT/iSTFTを内部統合<br>（複素数演算） | ❌ 不可 | 複素テンソルが変換エラー<br>**本レポートの対象** |
| **UVR（MDX-Net系）** | 振幅スペクトログラム入力<br>（実数のみ） | ✅ 可能 | CoreMLに適合<br>STFTは外部処理<br>**推奨ルート** |
| **Wave-U-Net系** | 時間領域処理<br>（実数波形入力） | ✅ 可能 | 複素数不要だが<br>精度はDemucsより劣る |
| **Spleeter** | U-Netベース<br>（複素数スペクトログラム） | ❌ 不可 | Demucsと同様の問題 |
| **Open-Unmix** | LSTMベース<br>（スペクトログラム） | ⚠️ 要検証 | 実装による |

### 5.2 オプションA: UVR/MDX-Net系への移行（推奨）

**アイデア**:
```
iOS側で STFT実行（vDSP/Accelerate）
    ↓ 振幅スペクトログラム生成（実数のみ）
CoreMLモデル（MDX-Net）で処理
    ↓ ボーカル/伴奏の振幅マスク予測（実数）
iOS側で iSTFT実行（位相は元音源から）
```

**メリット**:
- UVR既存モデルを利用可能（学習済み）
- 複素数演算を完全回避
- CoreML変換の技術的障壁が低い
- 実績あるアーキテクチャ

**実装ステータス**:
- ✅ STFTライブラリ選定済み
- ✅ 変換スクリプト作成済み
- 🔄 検証フェーズ進行中

### 5.3 オプションB: Wave-U-Net（軽量代替）

**候補**:
1. **Wave-U-Net**
   - 完全時間領域処理
   - 複素数不使用
   - 精度はDemucsより劣る

**評価**:
- CoreML変換は容易
- モデルサイズが小さい
- 分離品質のトレードオフあり

### 5.4 オプションC: Demucsの実数版再学習（長期）

**アプローチ**:
- Demucsアーキテクチャを実数テンソル（real/imag 2ch形式）で再実装
- MUSDB18等のデータセットで再学習

**評価**:
- 技術的には可能
- 学習コスト・時間が膨大
- 現時点では非推奨

### 5.5 オプションD: サーバーサイド処理

**構成**:
- iOS: 録音・再生・UI
- サーバー: Demucs処理（PyTorch）
- 通信: 音声アップロード/ダウンロード

**課題**:
- 仕様書「完全オフライン」に違反
- 著作権・プライバシー問題
- サーバーコスト・レイテンシ

### 5.6 オプションE: 2段階変換（ONNX経由）

**試行価値**:
```
PyTorch → ONNX → CoreML
```

**期待**:
- ONNX Runtime は複素数サポート
- 中間形式で最適化

**現実**:
- 最終的にCoreMLの制約に直面
- coremltools公式が直接変換推奨（ONNX非推奨）

---

## 6. 技術的深堀り: 複素数演算の回避可能性

### 6.1 振幅・位相分離アプローチ

**理論**:
```python
# 複素数スペクトログラム
X_complex = STFT(waveform)

# 分解
magnitude = abs(X_complex)      # 振幅（実数）
phase = angle(X_complex)        # 位相（実数）

# モデル入力: [magnitude, phase] を実数テンソルとして処理
# モデル出力: [magnitude_out, phase_out]

# 再合成
X_out = magnitude_out * exp(1j * phase_out)
waveform_out = iSTFT(X_out)
```

**課題**:
- 既存Demucsモデルはこの形式で学習されていない
- 振幅・位相を独立処理すると位相の一貫性が崩れる
- **再学習が必須**

### 6.2 実部・虚部分離アプローチ

```python
# 複素数 = real + imag * 1j
real = X_complex.real  # 実部
imag = X_complex.imag  # 虚部

# 2チャンネル実数テンソルとして処理
input_tensor = torch.stack([real, imag], dim=1)

# モデル出力も2チャンネル
output_real, output_imag = model(input_tensor)

# 複素数再構成
X_out = output_real + 1j * output_imag
```

**評価**:
- 数学的に等価
- **CoreMLで表現可能**
- Demucsを改造して再学習すれば実現可能

---

## 7. PoC進行の推奨戦略

### 戦略1: F0抽出先行（即時実行可能）

**スコープ**:
- ボーカル分離をスキップ
- 既存音源から直接F0抽出
- CREPE/Aubio/AudioKitで実装

**メリット**:
- 即座に動作デモ可能
- CoreML変換不要（既存ライブラリ使用）
- アプリのコア価値（歌唱分析）を検証

**実装**:
```
入力: ユーザー提供の音声ファイル
処理: F0抽出 → ビブラート解析 → 可視化
出力: ピッチグラフ + KPI
```

### 戦略2: 軽量分離モデル探索（並行調査）

**候補調査**:
1. **Spleeter Mobile版** の有無確認
2. **Open-Unmix lightweight** 調査
3. **カスタムWave-U-Net** 実装検討

**評価軸**:
- モデルサイズ（< 50MB目標）
- 複素数依存度
- CoreML互換性
- 分離品質（Vocal/Accompaniment SDR）

### 戦略3: Hybrid構成（長期）

**フェーズ1（PoC）**:
- F0抽出のみ（分離なし）
- ユーザーがボーカルトラック提供

**フェーズ2（β版）**:
- 軽量分離モデル統合
- 品質 vs パフォーマンストレードオフ

**フェーズ3（製品版）**:
- 精度版: サーバーサイドDemucs（オプション機能）
- オフライン版: 軽量モデル（基本機能）

---

## 8. 技術調査の追加項目

### 8.1 最新の音源分離モデル動向

**調査対象**:
- **BS-Roformer** (2024): 最新SOTA、複素数依存は？
- **Mel-Roformer**: モバイル最適化版の可能性
- **SCNet XL**: 軽量版の有無

### 8.2 CoreML代替フレームワーク

**候補**:
- **ONNX Runtime Mobile**: 複素数サポート状況
- **TensorFlow Lite**: 複素数演算サポート（limited）
- **PyTorch Mobile**: 直接PyTorchモデル実行

**トレードオフ**:
- CoreML: Neural Engine利用可能、最高効率
- 代替: 柔軟性高いが電力効率劣る

### 8.3 iOS Metal Compute Shader

**カスタムSTFT実装**:
- Metalで複素数FFTを実装
- モデルは実数処理のみ
- iOSネイティブで最適化

**難易度**: 高（低レベルプログラミング）

---

## 9. 結論と次ステップ（修正版）

### 9.1 Demucs直接変換の結論（変更なし）

**現状**: **実現不可能**（CoreMLの複素数非対応）

**将来性**:
- CoreMLTools更新待ち（可能性低い）
- Appleが複素数サポート追加（時期不明）
- Demucsの実数版リリース（Meta次第）

### 9.2 修正後の推奨実装パス

**✅ 現在進行中（UVR/MDX-Net系）**:
```
1. UVR MDX-Netモデルの選定（完了）
2. STFTライブラリ実装（完了）
3. CoreML変換スクリプト作成（完了）
4. 品質評価・検証（進行中）
```

**短期（1-2週間）**:
```
1. UVR MDX-NetのCoreML変換完了
2. iOS統合とパフォーマンス評価
3. F0抽出機能との統合テスト
```

**中期（1-2ヶ月）**:
```
1. 分離品質の最適化（SDR測定）
2. リアルタイム処理の実現可能性検証
3. β版機能統合
```

### 9.3 重要な技術的修正事項

**誤った理解**:
- 「音源分離モデル全般がCoreML変換困難」

**正確な理解**:
- **Demucs系のみ**が複素数演算により変換困難
- **UVR/MDX-Net系は変換可能**であり、実際に進行中
- 技術的障壁は当初想定より大幅に低い

**今後の対応**:
- Demucs関連の議論は別扱い
- UVR（MDX-Net）系を基準ルートとして推進
- 将来的なDemucs利用は実数形式での再学習を検討

### 9.4 技術的学習事項（更新版）

**得られた知見**:
- DemucsのSTFT統合アーキテクチャ理解
- CoreMLの型システム制約把握
- 音源分離の複素数処理の重要性認識
- モバイル推論の現実的制約
- **モデル系統による変換可能性の違い**（重要）
- **UVR/MDX-Net系の実用性の高さ**（新規）

**今後の調査**:
- ✅ UVR MDX-Netの品質評価（進行中）
- ✅ STFT外部化の実装完了
- 🔄 モバイル向け最適化手法

---

## 10. 参考資料

### 論文・技術資料
- Défossez et al. "Hybrid Spectrogram and Waveform Source Separation" (2021)
- Apple CoreML Documentation: Supported Data Types
- TorchAudio HDEMUCS Tutorial

### 関連Issue・ディスカッション
- coremltools GitHub: Complex number support requests
- PyTorch Forums: Mobile deployment discussions

### 代替モデル
- Spleeter: https://github.com/deezer/spleeter
- Open-Unmix: https://github.com/sigsep/open-unmix-pytorch
- Wave-U-Net: https://github.com/f90/Wave-U-Net

---

---

## 11. 修正履歴

### 2025-10-21: 重要な修正
- **タイトル変更**: Demucs特化 → 音源分離モデル全般の比較
- **誤解の修正**: すべての音源分離モデルが変換困難 → Demucs系のみが困難
- **UVR/MDX-Net系の追加**: CoreML変換可能な代替手法として明記
- **推奨実装パスの更新**: UVR系を基準ルートとして設定
- **モデル系統別の比較表追加**: 変換可能性を一覧化

### 2025-10-20: 初版作成
- Demucs → CoreML変換の技術的課題を分析
- 複素数非対応による変換不可を報告
- 代替アプローチを検討

---

**文書作成日**: 2025-10-20
**最終更新**: 2025-10-21
**ステータス**: Demucs直接変換は技術的困難、**UVR/MDX-Net系で実装進行中**
