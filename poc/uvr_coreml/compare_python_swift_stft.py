#!/usr/bin/env python3
"""
PythonとSwiftのSTFT中間データを比較
"""
import numpy as np

print("=" * 80)
print("🔍 Python vs Swift STFT中間データ比較")
print("=" * 80)

# Python データ読み込み
print("\n📂 Pythonデータ読み込み:")
py_test_signal = np.loadtxt("tests/python_output/test_signal.txt")
py_magnitude = np.loadtxt("tests/python_output/magnitude_frame0.txt")
py_phase = np.loadtxt("tests/python_output/phase_frame0.txt")
py_reconstructed = np.loadtxt("tests/python_output/reconstructed.txt")

print(f"  テスト信号: {len(py_test_signal)} サンプル")
print(f"  Magnitude (frame 0): {len(py_magnitude)} bins")
print(f"  Phase (frame 0): {len(py_phase)} bins")
print(f"  再構成: {len(py_reconstructed)} サンプル")

# Swift データ読み込み
print("\n📂 Swiftデータ読み込み:")
swift_test_signal = np.loadtxt("tests/swift_output/test_signal_swift.txt")
swift_magnitude = np.loadtxt("tests/swift_output/magnitude_frame0_swift.txt")
swift_phase = np.loadtxt("tests/swift_output/phase_frame0_swift.txt")
swift_reconstructed = np.loadtxt("tests/swift_output/reconstructed_swift.txt")

print(f"  テスト信号: {len(swift_test_signal)} サンプル")
print(f"  Magnitude (frame 0): {len(swift_magnitude)} bins")
print(f"  Phase (frame 0): {len(swift_phase)} bins")
print(f"  再構成: {len(swift_reconstructed)} サンプル")

# テスト信号比較
print("\n📊 テスト信号比較:")
signal_diff = np.abs(py_test_signal - swift_test_signal)
print(f"  Max差: {signal_diff.max():.10f}")
print(f"  RMS差: {np.sqrt(np.mean(signal_diff**2)):.10f}")

if signal_diff.max() < 1e-6:
    print("  ✅ テスト信号は一致")
else:
    print(f"  ⚠️ テスト信号に差異あり")

# Magnitude比較
print("\n📊 Magnitude (frame 0) 比較:")
min_len = min(len(py_magnitude), len(swift_magnitude))
mag_diff = np.abs(py_magnitude[:min_len] - swift_magnitude[:min_len])

print(f"  Python Max: {py_magnitude.max():.6f}")
print(f"  Swift Max: {swift_magnitude.max():.6f}")
print(f"  Max差: {mag_diff.max():.6f}")
print(f"  RMS差: {np.sqrt(np.mean(mag_diff**2)):.6f}")
print(f"  相対誤差: {mag_diff.max() / py_magnitude.max():.6f}")

if mag_diff.max() / py_magnitude.max() < 0.01:
    print("  ✅ Magnitudeはおおむね一致 (<1%)")
elif mag_diff.max() / py_magnitude.max() < 0.1:
    print("  ⚠️ Magnitudeに小さな差異 (<10%)")
else:
    print("  ❌ Magnitudeに大きな差異 (>10%)")

# 440Hz binの詳細比較
freq_440_bin = 40
print(f"\n🎵 440Hz bin [{freq_440_bin}] 詳細比較:")
print(f"  Python: mag={py_magnitude[freq_440_bin]:.6f}, phase={py_phase[freq_440_bin]:.6f}")
print(f"  Swift:  mag={swift_magnitude[freq_440_bin]:.6f}, phase={swift_phase[freq_440_bin]:.6f}")
print(f"  差: mag={abs(py_magnitude[freq_440_bin] - swift_magnitude[freq_440_bin]):.6f}, phase={abs(py_phase[freq_440_bin] - swift_phase[freq_440_bin]):.6f}")

# Phase比較
print("\n📊 Phase (frame 0) 比較:")
phase_diff = np.abs(py_phase[:min_len] - swift_phase[:min_len])
print(f"  Max差: {phase_diff.max():.6f}")
print(f"  RMS差: {np.sqrt(np.mean(phase_diff**2)):.6f}")

if phase_diff.max() < 0.01:
    print("  ✅ Phaseはおおむね一致 (<0.01 rad)")
elif phase_diff.max() < 0.1:
    print("  ⚠️ Phaseに小さな差異 (<0.1 rad)")
else:
    print("  ❌ Phaseに大きな差異 (>0.1 rad)")

# 再構成比較
print("\n📊 再構成信号比較:")
recon_diff = np.abs(py_reconstructed - swift_reconstructed)
print(f"  Python Max: {np.abs(py_reconstructed).max():.6f}")
print(f"  Swift Max: {np.abs(swift_reconstructed).max():.6f}")
print(f"  Max差: {recon_diff.max():.6f}")
print(f"  RMS差: {np.sqrt(np.mean(recon_diff**2)):.6f}")

# 振幅のスケーリング比
scale_ratio = np.abs(py_reconstructed).max() / np.abs(swift_reconstructed).max()
print(f"  振幅スケーリング比 (Python/Swift): {scale_ratio:.6f}")

print("\n" + "=" * 80)
print("💡 結論:")
print("=" * 80)

if mag_diff.max() / py_magnitude.max() > 0.1:
    print("❌ STFT (FFT) の実装に差異があります")
    print("   → vDSPのFFT出力がlibrosaと異なる可能性")
elif phase_diff.max() > 0.1:
    print("❌ 位相計算に差異があります")
    print("   → atan2の実装または入力データに違い")
elif scale_ratio > 2.0 or scale_ratio < 0.5:
    print("❌ iSTFT (iFFT) のスケーリングに差異があります")
    print(f"   → Swiftの出力は{scale_ratio:.2f}倍のスケーリングが必要")
else:
    print("✅ STFT/iSTFTの実装はおおむね一致しています")
    print("   → 問題は別の箇所にある可能性")

print("=" * 80)
