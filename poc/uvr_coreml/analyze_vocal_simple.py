#!/usr/bin/env python3
"""
Swift実装で生成されたボーカル音源の品質解析（matplotlib不要版）
"""

import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

# ファイルパス
swift_vocals = "tests/swift_output/hollow_crown_vocals_proper.wav"

print("=" * 80)
print("🎤 Swift実装ボーカル音源解析")
print("=" * 80)

# 音声読み込み
print(f"\n📂 音声読み込み: {swift_vocals}")
y, sr = librosa.load(swift_vocals, sr=44100, mono=False)

if y.ndim == 1:
    y = np.stack([y, y])  # モノラルをステレオに変換
    
print(f"   サンプルレート: {sr} Hz")
print(f"   チャンネル数: {y.shape[0]}")
print(f"   サンプル数: {y.shape[1]}")
print(f"   長さ: {y.shape[1] / sr:.2f} 秒")

# 左チャンネルで解析
y_left = y[0]

# 基本統計
print(f"\n📊 音声統計 (左チャンネル):")
print(f"   最小値: {np.min(y_left):.6f}")
print(f"   最大値: {np.max(y_left):.6f}")
print(f"   平均値: {np.mean(y_left):.6f}")
print(f"   RMS: {np.sqrt(np.mean(y_left**2)):.6f}")
print(f"   標準偏差: {np.std(y_left):.6f}")

# ゼロサンプルのチェック
zero_samples = np.sum(y_left == 0)
zero_ratio = zero_samples / len(y_left) * 100
print(f"\n   ゼロサンプル: {zero_samples} ({zero_ratio:.2f}%)")

# クリッピングのチェック
clipped_samples = np.sum(np.abs(y_left) >= 0.99)
clip_ratio = clipped_samples / len(y_left) * 100
print(f"   クリッピング: {clipped_samples} ({clip_ratio:.2f}%)")

# 最初と最後の数サンプルを表示
print(f"\n   最初の10サンプル:")
for i in range(10):
    print(f"     [{i}] {y_left[i]:.6f}")
    
print(f"\n   最後の10サンプル:")
for i in range(-10, 0):
    print(f"     [{len(y_left) + i}] {y_left[i]:.6f}")

# スペクトログラム解析
print(f"\n🔊 スペクトログラム解析:")
D = librosa.stft(y_left, n_fft=2048, hop_length=512)
magnitude = np.abs(D)
print(f"   周波数ビン数: {magnitude.shape[0]}")
print(f"   時間フレーム数: {magnitude.shape[1]}")
print(f"   最大振幅: {np.max(magnitude):.2f}")
print(f"   平均振幅: {np.mean(magnitude):.6f}")

# 周波数帯域ごとのエネルギー
low_band = magnitude[0:50, :].mean()    # 0-500Hz
mid_band = magnitude[50:200, :].mean()   # 500-2000Hz
high_band = magnitude[200:, :].mean()    # 2000Hz+

print(f"\n   周波数帯域エネルギー:")
print(f"     低域 (0-500Hz): {low_band:.6f}")
print(f"     中域 (500-2000Hz): {mid_band:.6f}")
print(f"     高域 (2000Hz+): {high_band:.6f}")

# 10秒間のセグメントをチェック
print(f"\n🎵 時間セグメント解析 (各10秒):")
segment_length = 10 * sr
num_segments = min(3, int(len(y_left) / segment_length))

for i in range(num_segments):
    start = i * segment_length
    end = start + segment_length
    segment = y_left[start:end]
    rms = np.sqrt(np.mean(segment**2))
    print(f"   セグメント {i+1} ({i*10}-{(i+1)*10}秒): RMS = {rms:.6f}")

# 異常値検出
print(f"\n⚠️  異常値チェック:")
nan_count = np.sum(np.isnan(y_left))
inf_count = np.sum(np.isinf(y_left))

if nan_count > 0:
    print(f"   NaN検出: {nan_count} サンプル")
else:
    print(f"   ✅ NaNなし")
    
if inf_count > 0:
    print(f"   Inf検出: {inf_count} サンプル")
else:
    print(f"   ✅ Infなし")

# 音声品質評価
print(f"\n📈 品質評価:")
if np.max(np.abs(y_left)) < 0.001:
    print("   ⚠️  音量が極端に小さい（ほぼ無音）")
elif np.max(np.abs(y_left)) > 0.99:
    print("   ⚠️  クリッピングの可能性")
else:
    print("   ✅ 音量レベル正常")

if zero_ratio > 50:
    print(f"   ⚠️  ゼロサンプルが多い ({zero_ratio:.1f}%)")
elif zero_ratio > 10:
    print(f"   ⚠️  ゼロサンプルがやや多い ({zero_ratio:.1f}%)")
else:
    print(f"   ✅ ゼロサンプル正常 ({zero_ratio:.1f}%)")

if nan_count == 0 and inf_count == 0:
    print("   ✅ 数値的に正常")

print("\n" + "=" * 80)
print("✅ 解析完了")
print("=" * 80)
