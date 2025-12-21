# ボーカル抽出比較テスト計画

## 目的
POCとAppのボーカル抽出結果を同一音源で比較し、精度差の原因を特定する。

---

## 前提条件

### 必要なファイル
| ファイル | 場所 | 用途 |
|----------|------|------|
| テスト音源 | POC側で使用した音源 | 入力データ |
| POC抽出結果 | POCで抽出済みのボーカル音声 | 比較基準 |

### 環境
- iOS Simulator: iPhone 16 Pro (iOS 18.4)
- Xcode: 16.3+
- アプリ: VocalMasteryLab (Debug build)

---

## テスト手順

### Step 1: POCでテスト音源を抽出

```bash
cd /Users/kazuasato/Documents/dev/music/vocal_mastery_lab/poc/uvr_coreml

# POCでボーカル抽出を実行
swift run CoreMLTest <テスト音源.wav>

# 出力ファイルを確認
ls -la tests/swift_output/
```

**記録する情報:**
- 入力ファイル名とパス
- 出力ファイル名 (例: `*_vocals.wav`)
- 処理時間
- コンソール出力 (統計情報)

### Step 2: テスト音源をシミュレータに配置

```bash
# シミュレータを起動
xcrun simctl boot "iPhone 16 Pro"

# 音源ファイルをシミュレータの写真ライブラリに追加
xcrun simctl addmedia "iPhone 16 Pro" <テスト音源.wav>

# または、シミュレータのDocumentsフォルダに直接配置
# (アプリのサンドボックス内に配置する場合)
APP_CONTAINER=$(xcrun simctl get_app_container "iPhone 16 Pro" com.kazuasato.VocalMasteryLab data)
cp <テスト音源.wav> "$APP_CONTAINER/Documents/"
```

### Step 3: アプリでボーカル抽出を実行

1. シミュレータでVocalMasteryLabを起動
2. 録音リストから音源を選択
3. 「ボーカル抽出」を実行
4. Xcodeコンソールで診断ログを確認

**確認するログ:**
```
✅ [VocalExtractor] Found mlmodelc at: ...
🎵 [SEPARATION_START] Starting vocal separation
📊 [LEFT_AUDIO_STATS] min=X, max=X, mean=X, rms=X
🔢 [FIRST_10_SAMPLES] [...]
📈 [STFT] freqBins=X, timeFrames=X
🤖 [COREML_START] timeFrames=X, freqBins=X, chunkSize=X, numChunks=X
🔢 [MODEL_INPUT] chunk=X, shape=X, min=X, max=X, mean=X
🔢 [MODEL_OUTPUT] chunk=X, shape=X, min=X, max=X, mean=X
🎭 [VOCAL_MASK_STATS] min=X, max=X, mean=X
🎤 [VOCAL_SPEC_STATS] min=X, max=X, mean=X
🎵 [OUTPUT_AUDIO_STATS] min=X, max=X, mean=X, rms=X
```

### Step 4: 抽出結果をシミュレータから取得

```bash
# アプリのDocumentsフォルダから抽出結果を取得
APP_CONTAINER=$(xcrun simctl get_app_container "iPhone 16 Pro" com.kazuasato.VocalMasteryLab data)
cp "$APP_CONTAINER/Documents/ExtractedAudio/"*.wav /tmp/app_extracted/
```

### Step 5: POCとAppの結果を比較

```bash
# 波形比較 (Pythonスクリプト)
python3 << 'EOF'
import numpy as np
import scipy.io.wavfile as wav

# ファイル読み込み
poc_rate, poc_data = wav.read("poc_vocals.wav")
app_rate, app_data = wav.read("app_vocals.wav")

# 正規化
poc_norm = poc_data / np.max(np.abs(poc_data))
app_norm = app_data / np.max(np.abs(app_data))

# 長さを揃える
min_len = min(len(poc_norm), len(app_norm))
poc_norm = poc_norm[:min_len]
app_norm = app_norm[:min_len]

# 相関係数
if len(poc_norm.shape) > 1:
    correlation = np.corrcoef(poc_norm[:,0], app_norm[:,0])[0,1]
else:
    correlation = np.corrcoef(poc_norm, app_norm)[0,1]

# RMS差
rms_diff = np.sqrt(np.mean((poc_norm - app_norm) ** 2))

# 結果出力
print(f"相関係数: {correlation:.6f}")
print(f"RMS差: {rms_diff:.6f}")
print(f"POC RMS: {np.sqrt(np.mean(poc_norm**2)):.6f}")
print(f"App RMS: {np.sqrt(np.mean(app_norm**2)):.6f}")

# 判定
if correlation > 0.99:
    print("✅ 結果: ほぼ同一")
elif correlation > 0.90:
    print("⚠️ 結果: 軽微な差異あり")
else:
    print("❌ 結果: 大きな差異あり - 詳細調査必要")
EOF
```

---

## 比較項目

### 数値比較
| 項目 | POC | App | 差異 |
|------|-----|-----|------|
| 入力RMS | | | |
| 出力RMS | | | |
| 相関係数 | - | - | |
| MODEL_INPUT mean | | | |
| MODEL_OUTPUT mean | | | |
| VOCAL_MASK mean | | | |

### 波形比較
- [ ] スペクトログラムの視覚的比較
- [ ] 差分波形の確認
- [ ] 特定周波数帯での差異

---

## 想定される原因と対策

| 原因 | 確認方法 | 対策 |
|------|----------|------|
| 入力音声の前処理差異 | FIRST_10_SAMPLESを比較 | AudioProcessor統一 |
| STFT計算の差異 | MODEL_INPUT統計を比較 | STFTProcessorV2確認 |
| モデル入力形式の差異 | INPUT_CH0_SAMPLEを比較 | extractChunk確認 |
| モデル出力の解釈差異 | MODEL_OUTPUT統計を比較 | extractChannelMask確認 |
| マスク適用の差異 | VOCAL_MASK統計を比較 | applyComplexMask確認 |
| iSTFT計算の差異 | OUTPUT_AUDIO統計を比較 | createAudioData確認 |

---

## 成功基準

- 相関係数 > 0.95: 許容範囲
- 相関係数 > 0.99: 同等品質
- RMS差 < 0.05: 許容範囲

---

## 備考

- POCとAppで同一の音源ファイルを使用すること
- サンプルレートは44100Hzに統一
- ステレオ音源を使用
