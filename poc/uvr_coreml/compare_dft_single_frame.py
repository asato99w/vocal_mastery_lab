#!/usr/bin/env python3
"""
vDSP_DFTとPython librosaの1フレーム比較
440Hzの単一周波数信号でFFT結果を比較
"""
import numpy as np
import librosa

# パラメータ (Swiftと同じ)
sample_rate = 44100.0
frequency = 440.0  # A4音
n_fft = 4096
duration = n_fft / sample_rate

print("=" * 80)
print("🔍 Python librosa - Single Frame FFT")
print("=" * 80)

print("\n📊 テストパラメータ:")
print(f"  サンプルレート: {sample_rate} Hz")
print(f"  周波数: {frequency} Hz")
print(f"  FFTサイズ: {n_fft}")
print(f"  信号長: {n_fft} samples ({duration:.4f}秒)")

# 440Hzの正弦波を生成
t = np.arange(n_fft) / sample_rate
signal = np.sin(2.0 * np.pi * frequency * t).astype(np.float32)

print("\n🎵 テスト信号 (440Hz正弦波):")
print(f"  最初の10サンプル: {signal[:10]}")

# Hann窓を適用 (librosaのデフォルト)
window = librosa.filters.get_window('hann', n_fft, fftbins=True)
windowed_signal = signal * window

# FFT実行 (複素数FFT)
# librosaのstftは内部的にnp.fft.fftを使用
fft_result = np.fft.fft(windowed_signal)

# 周波数ビン数 (0からNyquistまで)
frequency_bins = n_fft // 2 + 1

# 複素数結果を実部・虚部に分割
real_part = np.real(fft_result[:frequency_bins])
imag_part = np.imag(fft_result[:frequency_bins])

# 振幅と位相を計算
magnitude = np.abs(fft_result[:frequency_bins])
phase = np.angle(fft_result[:frequency_bins])

# 440Hz binを見つける
expected_bin = int(np.round(frequency * n_fft / sample_rate))

print(f"\n🎯 440Hz bin [位置 {expected_bin}]:")
print(f"  Magnitude: {magnitude[expected_bin]:.6f}")
print(f"  Phase: {phase[expected_bin]:.6f}")

# DC成分
print(f"\n📊 DC bin [0]:")
print(f"  Magnitude: {magnitude[0]:.6f}")
print(f"  Phase: {phase[0]:.6f}")

# Nyquist成分
print(f"\n📊 Nyquist bin [{n_fft//2}]:")
print(f"  Magnitude: {magnitude[n_fft//2]:.6f}")
print(f"  Phase: {phase[n_fft//2]:.6f}")

# 最初の10ビンの詳細
print("\n🔍 最初の10ビンの詳細:")
for i in range(min(10, frequency_bins)):
    freq = i * sample_rate / n_fft
    print(f"  Bin {i} ({freq:.1f}Hz): mag={magnitude[i]:.6f}, phase={phase[i]:.6f}")

# 440Hz周辺の詳細 (±5 bins)
print("\n🎵 440Hz周辺のビン詳細:")
start_bin = max(0, expected_bin - 5)
end_bin = min(frequency_bins - 1, expected_bin + 5)
for i in range(start_bin, end_bin + 1):
    freq = i * sample_rate / n_fft
    marker = " ← 440Hz" if i == expected_bin else ""
    print(f"  Bin {i} ({freq:.1f}Hz): mag={magnitude[i]:.6f}, phase={phase[i]:.6f}{marker}")

# Swift比較用にデータを保存
import os
output_dir = "tests/swift_output"
os.makedirs(output_dir, exist_ok=True)

# 実部・虚部・振幅・位相を保存
np.savetxt(f"{output_dir}/dft_real_python.txt", real_part, fmt="%.10f")
np.savetxt(f"{output_dir}/dft_imag_python.txt", imag_part, fmt="%.10f")
np.savetxt(f"{output_dir}/dft_magnitude_python.txt", magnitude, fmt="%.10f")
np.savetxt(f"{output_dir}/dft_phase_python.txt", phase, fmt="%.10f")

print(f"\n💾 比較用データ保存:")
print(f"  {output_dir}/dft_real_python.txt")
print(f"  {output_dir}/dft_imag_python.txt")
print(f"  {output_dir}/dft_magnitude_python.txt")
print(f"  {output_dir}/dft_phase_python.txt")

# Swift出力との比較 (もし存在すれば)
swift_real_path = f"{output_dir}/dft_real_swift.txt"
if os.path.exists(swift_real_path):
    print("\n" + "=" * 80)
    print("📊 Swift vs Python 比較")
    print("=" * 80)

    swift_real = np.loadtxt(f"{output_dir}/dft_real_swift.txt", dtype=np.float32)
    swift_imag = np.loadtxt(f"{output_dir}/dft_imag_swift.txt", dtype=np.float32)
    swift_mag = np.loadtxt(f"{output_dir}/dft_magnitude_swift.txt", dtype=np.float32)
    swift_phase = np.loadtxt(f"{output_dir}/dft_phase_swift.txt", dtype=np.float32)

    # 相関係数を計算
    real_corr = np.corrcoef(real_part, swift_real)[0, 1]
    imag_corr = np.corrcoef(imag_part, swift_imag)[0, 1]
    mag_corr = np.corrcoef(magnitude, swift_mag)[0, 1]
    phase_corr = np.corrcoef(phase, swift_phase)[0, 1]

    print(f"\n📈 相関係数:")
    print(f"  実部: {real_corr:.10f}")
    print(f"  虚部: {imag_corr:.10f}")
    print(f"  振幅: {mag_corr:.10f}")
    print(f"  位相: {phase_corr:.10f}")

    # 440Hz binの比較
    print(f"\n🎯 440Hz bin [{expected_bin}] の比較:")
    print(f"  Python magnitude: {magnitude[expected_bin]:.6f}")
    print(f"  Swift magnitude: {swift_mag[expected_bin]:.6f}")
    print(f"  比率: {magnitude[expected_bin] / swift_mag[expected_bin]:.6f}")
    print(f"  差分: {abs(magnitude[expected_bin] - swift_mag[expected_bin]):.6f}")

    print(f"\n  Python phase: {phase[expected_bin]:.6f}")
    print(f"  Swift phase: {swift_phase[expected_bin]:.6f}")
    print(f"  差分: {abs(phase[expected_bin] - swift_phase[expected_bin]):.6f}")

    # DC binの比較
    print(f"\n📊 DC bin [0] の比較:")
    print(f"  Python magnitude: {magnitude[0]:.6f}")
    print(f"  Swift magnitude: {swift_mag[0]:.6f}")
    print(f"  差分: {abs(magnitude[0] - swift_mag[0]):.10f}")

    # 最大誤差
    max_mag_error = np.max(np.abs(magnitude - swift_mag))
    max_phase_error = np.max(np.abs(phase - swift_phase))

    print(f"\n📏 最大誤差:")
    print(f"  振幅: {max_mag_error:.10f}")
    print(f"  位相: {max_phase_error:.10f}")

    # RMS誤差
    rms_mag_error = np.sqrt(np.mean((magnitude - swift_mag) ** 2))
    rms_phase_error = np.sqrt(np.mean((phase - swift_phase) ** 2))

    print(f"\n📊 RMS誤差:")
    print(f"  振幅: {rms_mag_error:.10f}")
    print(f"  位相: {rms_phase_error:.10f}")

    # 判定
    print(f"\n✅ 判定:")
    if real_corr > 0.9999 and imag_corr > 0.9999:
        print("  ✅ 優秀！ 相関係数 > 0.9999")
    elif real_corr > 0.999 and imag_corr > 0.999:
        print("  ✅ 良好！ 相関係数 > 0.999")
    elif real_corr > 0.99 and imag_corr > 0.99:
        print("  ⚠️  許容範囲。相関係数 > 0.99")
    else:
        print(f"  ❌ 要改善。相関係数が低い (real={real_corr:.6f}, imag={imag_corr:.6f})")

print("\n" + "=" * 80)
