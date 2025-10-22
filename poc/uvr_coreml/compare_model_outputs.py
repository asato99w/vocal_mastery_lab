#!/usr/bin/env python3
"""
ONNX (Python) vs CoreML (Swift) のモデル出力を比較
"""
import numpy as np

print("=" * 80)
print("🔬 ONNX vs CoreML モデル出力比較")
print("=" * 80)

# Python (ONNX) の出力を読み込み
python_output = np.loadtxt('tests/python_output/model_output_ch0_frame0.txt')
print(f"\n📊 Python ONNX出力:")
print(f"  要素数: {len(python_output)}")
print(f"  値の範囲: {python_output.min():.6f} ~ {python_output.max():.6f}")
print(f"  平均: {python_output.mean():.6f}")
print(f"  標準偏差: {python_output.std():.6f}")

# Swift (CoreML) の出力を読み込み
swift_output = np.loadtxt('tests/swift_output/model_output_ch0_frame0_swift.txt')
print(f"\n📊 Swift CoreML出力:")
print(f"  要素数: {len(swift_output)}")
print(f"  値の範囲: {swift_output.min():.6f} ~ {swift_output.max():.6f}")
print(f"  平均: {swift_output.mean():.6f}")
print(f"  標準偏差: {swift_output.std():.6f}")

# 差分統計
diff = swift_output - python_output
abs_diff = np.abs(diff)

print(f"\n📏 差分統計:")
print(f"  最大絶対差: {abs_diff.max():.6f}")
print(f"  平均絶対差: {abs_diff.mean():.6f}")
print(f"  RMS差: {np.sqrt(np.mean(diff**2)):.6f}")
print(f"  相対誤差 (RMS): {np.sqrt(np.mean(diff**2)) / np.abs(python_output).max():.6f}")

# 相関係数
correlation = np.corrcoef(python_output, swift_output)[0, 1]
print(f"\n📈 相関分析:")
print(f"  相関係数: {correlation:.6f}")

# 一致度判定
if abs_diff.max() < 0.01:
    print(f"\n✅ モデル出力は**ほぼ完全に一致**しています！")
    print(f"   → 問題はSTFT/iSTFT実装にあります")
elif abs_diff.max() < 1.0:
    print(f"\n⚠️ モデル出力に小さな差異があります")
    print(f"   → CoreMLとONNXの実装差による可能性")
else:
    print(f"\n❌ モデル出力に大きな差異があります！")
    print(f"   → モデル変換またはCoreML推論に問題があります")

# 最大差分のインデックス
max_diff_idx = abs_diff.argmax()
print(f"\n🔍 最大差分の位置:")
print(f"  周波数bin: {max_diff_idx}")
print(f"  Python値: {python_output[max_diff_idx]:.6f}")
print(f"  Swift値: {swift_output[max_diff_idx]:.6f}")
print(f"  差: {diff[max_diff_idx]:.6f}")

# 440Hz bin (bin 40) の比較
freq_440_bin = 40
print(f"\n🎵 440Hz bin [{freq_440_bin}] の比較:")
print(f"  Python: {python_output[freq_440_bin]:.6f}")
print(f"  Swift: {swift_output[freq_440_bin]:.6f}")
print(f"  差: {diff[freq_440_bin]:.6f}")

# 最初の10要素の詳細比較
print(f"\n📋 最初の10要素の詳細比較:")
print(f"{'Bin':>4} {'Python':>12} {'Swift':>12} {'差':>12}")
print("-" * 44)
for i in range(10):
    print(f"{i:4d} {python_output[i]:12.6f} {swift_output[i]:12.6f} {diff[i]:12.6f}")

print("\n" + "=" * 80)
