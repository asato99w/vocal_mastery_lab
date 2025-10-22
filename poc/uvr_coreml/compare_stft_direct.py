#!/usr/bin/env python3
"""
Python librosa STFT vs Swift vDSP STFTの直接比較
"""
import numpy as np

print("=" * 80)
print("🔬 Python librosa vs Swift vDSP STFT比較")
print("=" * 80)

# Python STFT結果を読み込み
python_mag = np.loadtxt('tests/python_output/magnitude_frame0.txt')
python_phase = np.loadtxt('tests/python_output/phase_frame0.txt')

print(f"\n📊 Python librosa STFT (Frame 0):")
print(f"  要素数: {len(python_mag)}")
print(f"  Magnitude範囲: {python_mag.min():.6f} ~ {python_mag.max():.6f}")
print(f"  Phase範囲: {python_phase.min():.6f} ~ {python_phase.max():.6f}")

# Swift STFT結果を読み込み
swift_mag = np.loadtxt('tests/swift_output/magnitude_frame0_swift.txt')
swift_phase = np.loadtxt('tests/swift_output/phase_frame0_swift.txt')

print(f"\n📊 Swift vDSP STFT (Frame 0):")
print(f"  要素数: {len(swift_mag)}")
print(f"  Magnitude範囲: {swift_mag.min():.6f} ~ {swift_mag.max():.6f}")
print(f"  Phase範囲: {swift_phase.min():.6f} ~ {swift_phase.max():.6f}")

# Magnitude比較
mag_diff = swift_mag - python_mag[:len(swift_mag)]
mag_abs_diff = np.abs(mag_diff)

print(f"\n📏 Magnitude差分統計:")
print(f"  最大絶対差: {mag_abs_diff.max():.6f}")
print(f"  平均絶対差: {mag_abs_diff.mean():.6f}")
print(f"  RMS差: {np.sqrt(np.mean(mag_diff**2)):.6f}")

# 相関係数
mag_correlation = np.corrcoef(python_mag[:len(swift_mag)], swift_mag)[0, 1]
print(f"\n📈 Magnitude相関:")
print(f"  相関係数: {mag_correlation:.6f}")

# スケーリング比率
mag_scale_ratio = python_mag.max() / swift_mag.max()
print(f"\n📏 Magnitudeスケーリング:")
print(f"  Python Max / Swift Max = {mag_scale_ratio:.6f}")

# 440Hz bin (bin 40) の比較
freq_440_bin = 40
print(f"\n🎵 440Hz bin [{freq_440_bin}] の比較:")
print(f"  Python magnitude: {python_mag[freq_440_bin]:.6f}")
print(f"  Swift magnitude: {swift_mag[freq_440_bin]:.6f}")
print(f"  比率: {python_mag[freq_440_bin] / swift_mag[freq_440_bin]:.6f}")
print(f"\n  Python phase: {python_phase[freq_440_bin]:.6f}")
print(f"  Swift phase: {swift_phase[freq_440_bin]:.6f}")
print(f"  差: {swift_phase[freq_440_bin] - python_phase[freq_440_bin]:.6f}")

# DC binの比較
print(f"\n🔍 DC bin [0] の比較:")
print(f"  Python magnitude: {python_mag[0]:.6f}")
print(f"  Swift magnitude: {swift_mag[0]:.6f}")
print(f"  Python phase: {python_phase[0]:.6f}")
print(f"  Swift phase: {swift_phase[0]:.6f}")

# 実部・虚部に変換
python_real = python_mag[:len(swift_mag)] * np.cos(python_phase[:len(swift_mag)])
python_imag = python_mag[:len(swift_mag)] * np.sin(python_phase[:len(swift_mag)])

swift_real = swift_mag * np.cos(swift_phase)
swift_imag = swift_mag * np.sin(swift_phase)

print(f"\n📊 実部の比較:")
print(f"  Python範囲: {python_real.min():.6f} ~ {python_real.max():.6f}")
print(f"  Swift範囲: {swift_real.min():.6f} ~ {swift_real.max():.6f}")

real_correlation = np.corrcoef(python_real, swift_real)[0, 1]
print(f"  相関係数: {real_correlation:.6f}")

print(f"\n📊 虚部の比較:")
print(f"  Python範囲: {python_imag.min():.6f} ~ {python_imag.max():.6f}")
print(f"  Swift範囲: {swift_imag.min():.6f} ~ {swift_imag.max():.6f}")

imag_correlation = np.corrcoef(python_imag, swift_imag)[0, 1]
print(f"  相関係数: {imag_correlation:.6f}")

# 最初の20要素の詳細比較（magnitudeのみ）
print(f"\n📋 Magnitude 最初の20要素の詳細比較:")
print(f"{'Bin':>4} {'Python':>12} {'Swift':>12} {'比率':>8}")
print("-" * 40)
for i in range(20):
    ratio = python_mag[i] / swift_mag[i] if swift_mag[i] > 0.001 else 0
    print(f"{i:4d} {python_mag[i]:12.6f} {swift_mag[i]:12.6f} {ratio:8.3f}")

print("\n" + "=" * 80)
