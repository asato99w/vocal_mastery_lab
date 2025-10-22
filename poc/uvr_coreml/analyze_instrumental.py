#!/usr/bin/env python3
"""
修正後の伴奏ファイル解析
"""
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

# ファイルパス
instrumental = "tests/swift_output/hollow_crown_instrumental_proper.wav"
vocals = "tests/swift_output/hollow_crown_vocals_proper.wav"

print("=" * 80)
print("🎸 伴奏ファイル解析")
print("=" * 80)

# 伴奏読み込み
print(f"\n📂 伴奏読み込み: {instrumental}")
y_inst, sr = librosa.load(instrumental, sr=44100, mono=False)
if y_inst.ndim == 1:
    y_inst = np.stack([y_inst, y_inst])

print(f"   サンプルレート: {sr} Hz")
print(f"   チャンネル数: {y_inst.shape[0]}")
print(f"   サンプル数: {y_inst.shape[1]}")

# ボーカル読み込み (比較用)
print(f"\n📂 ボーカル読み込み (比較用): {vocals}")
y_voc, sr = librosa.load(vocals, sr=44100, mono=False)
if y_voc.ndim == 1:
    y_voc = np.stack([y_voc, y_voc])

# 左チャンネルで解析
inst_left = y_inst[0]
voc_left = y_voc[0]

print(f"\n📊 伴奏統計 (左チャンネル):")
print(f"   最小値: {np.min(inst_left):.6f}")
print(f"   最大値: {np.max(inst_left):.6f}")
print(f"   平均値: {np.mean(inst_left):.6f}")
print(f"   RMS: {np.sqrt(np.mean(inst_left**2)):.6f}")

print(f"\n📊 ボーカル統計 (左チャンネル):")
print(f"   最小値: {np.min(voc_left):.6f}")
print(f"   最大値: {np.max(voc_left):.6f}")
print(f"   平均値: {np.mean(voc_left):.6f}")
print(f"   RMS: {np.sqrt(np.mean(voc_left**2)):.6f}")

# 相関チェック
correlation = np.corrcoef(inst_left, voc_left)[0, 1]
print(f"\n🔍 伴奏とボーカルの相関係数: {correlation:.6f}")
if abs(correlation) > 0.95:
    print("   ⚠️  両ファイルがほぼ同じ内容です!")
elif abs(correlation) < 0.3:
    print("   ✅ 両ファイルは異なる内容です (正常)")
else:
    print(f"   ⚠️  中程度の相関 ({correlation:.3f})")

# 最初の10サンプルを比較
print(f"\n📋 最初の10サンプル比較:")
print("   インデックス    伴奏          ボーカル       差")
for i in range(10):
    diff = inst_left[i] - voc_left[i]
    print(f"   [{i:2d}]     {inst_left[i]:10.6f}  {voc_left[i]:10.6f}  {diff:10.6f}")

# スペクトログラム比較
print(f"\n🔊 スペクトログラム比較:")
D_inst = librosa.stft(inst_left, n_fft=2048, hop_length=512)
D_voc = librosa.stft(voc_left, n_fft=2048, hop_length=512)

mag_inst = np.abs(D_inst)
mag_voc = np.abs(D_voc)

print(f"   伴奏 - 最大振幅: {np.max(mag_inst):.2f}, 平均振幅: {np.mean(mag_inst):.6f}")
print(f"   ボーカル - 最大振幅: {np.max(mag_voc):.2f}, 平均振幅: {np.mean(mag_voc):.6f}")

# 周波数帯域ごとのエネルギー比較
low_inst = mag_inst[0:50, :].mean()
mid_inst = mag_inst[50:200, :].mean()
high_inst = mag_inst[200:, :].mean()

low_voc = mag_voc[0:50, :].mean()
mid_voc = mag_voc[50:200, :].mean()
high_voc = mag_voc[200:, :].mean()

print(f"\n   周波数帯域エネルギー比較:")
print(f"     低域 (0-500Hz):    伴奏={low_inst:.6f}   ボーカル={low_voc:.6f}")
print(f"     中域 (500-2000Hz): 伴奏={mid_inst:.6f}   ボーカル={mid_voc:.6f}")
print(f"     高域 (2000Hz+):    伴奏={high_inst:.6f}   ボーカル={high_voc:.6f}")

print("\n" + "=" * 80)
print("✅ 解析完了")
print("=" * 80)
