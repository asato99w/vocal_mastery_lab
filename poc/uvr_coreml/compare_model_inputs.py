#!/usr/bin/env python3
"""
ONNX (Python) vs CoreML (Swift) のモデル入力を比較
"""
import numpy as np

print("=" * 80)
print("🔬 ONNX vs CoreML モデル入力比較")
print("=" * 80)

# Python (ONNX) の入力を読み込み
python_input = np.loadtxt('tests/python_output/model_input_ch0_frame0.txt')
print(f"\n📊 Python ONNX入力 (Ch0 Real):")
print(f"  要素数: {len(python_input)}")
print(f"  値の範囲: {python_input.min():.6f} ~ {python_input.max():.6f}")
print(f"  平均: {python_input.mean():.6f}")
print(f"  標準偏差: {python_input.std():.6f}")

# Swift (CoreML) の入力を生成（STFTから）
# まずSwiftのSTFT結果を読み込み
swift_magnitude = np.loadtxt('tests/swift_output/magnitude_frame0_swift.txt')
swift_phase = np.loadtxt('tests/swift_output/phase_frame0_swift.txt')

# 実部を計算 (magnitude * cos(phase))
swift_input = swift_magnitude[:2048] * np.cos(swift_phase[:2048])

print(f"\n📊 Swift CoreML入力 (Ch0 Real):")
print(f"  要素数: {len(swift_input)}")
print(f"  値の範囲: {swift_input.min():.6f} ~ {swift_input.max():.6f}")
print(f"  平均: {swift_input.mean():.6f}")
print(f"  標準偏差: {swift_input.std():.6f}")

# 差分統計
diff = swift_input - python_input
abs_diff = np.abs(diff)

print(f"\n📏 差分統計:")
print(f"  最大絶対差: {abs_diff.max():.6f}")
print(f"  平均絶対差: {abs_diff.mean():.6f}")
print(f"  RMS差: {np.sqrt(np.mean(diff**2)):.6f}")
print(f"  相対誤差 (RMS): {np.sqrt(np.mean(diff**2)) / np.abs(python_input).max():.6f}")

# 相関係数
correlation = np.corrcoef(python_input, swift_input)[0, 1]
print(f"\n📈 相関分析:")
print(f"  相関係数: {correlation:.6f}")

# スケーリング比率
scale_ratio = np.abs(python_input).max() / np.abs(swift_input).max()
print(f"\n📏 スケーリング比率:")
print(f"  Python Max / Swift Max = {scale_ratio:.6f}")

# 一致度判定
if correlation > 0.99 and abs_diff.max() < 10:
    print(f"\n✅ モデル入力は**ほぼ完全に一致**しています！")
    print(f"   → 問題はCoreMLモデル推論にあります")
elif correlation > 0.9:
    print(f"\n⚠️ モデル入力に小さな差異があります")
    print(f"   → STFTの実装差による可能性")
else:
    print(f"\n❌ モデル入力に大きな差異があります！")
    print(f"   → STFT実装に根本的な問題があります")

# 最大差分のインデックス
max_diff_idx = abs_diff.argmax()
print(f"\n🔍 最大差分の位置:")
print(f"  周波数bin: {max_diff_idx}")
print(f"  Python値: {python_input[max_diff_idx]:.6f}")
print(f"  Swift値: {swift_input[max_diff_idx]:.6f}")
print(f"  差: {diff[max_diff_idx]:.6f}")

# 440Hz bin (bin 40) の比較
freq_440_bin = 40
print(f"\n🎵 440Hz bin [{freq_440_bin}] の比較:")
print(f"  Python: {python_input[freq_440_bin]:.6f}")
print(f"  Swift: {swift_input[freq_440_bin]:.6f}")
print(f"  差: {diff[freq_440_bin]:.6f}")
print(f"  比率: {python_input[freq_440_bin] / swift_input[freq_440_bin]:.6f}")

# 最初の10要素の詳細比較
print(f"\n📋 最初の10要素の詳細比較:")
print(f"{'Bin':>4} {'Python':>12} {'Swift':>12} {'差':>12} {'比率':>8}")
print("-" * 56)
for i in range(10):
    ratio = python_input[i] / swift_input[i] if swift_input[i] != 0 else float('inf')
    print(f"{i:4d} {python_input[i]:12.6f} {swift_input[i]:12.6f} {diff[i]:12.6f} {ratio:8.3f}")

print("\n" + "=" * 80)
