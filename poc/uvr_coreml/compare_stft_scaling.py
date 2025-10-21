#!/usr/bin/env python3
"""
LibrosaとvDSPのSTFT/iSTFTスケーリング比較
"""
import numpy as np
import librosa

# テスト信号生成
sr = 44100
duration = 1.0
t = np.linspace(0, duration, int(sr * duration))
test_signal = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave

print("=" * 80)
print("🔬 STFT/iSTFT スケーリング比較")
print("=" * 80)

print(f"\n📊 入力信号:")
print(f"  Max: {np.abs(test_signal).max():.6f}")
print(f"  RMS: {np.sqrt(np.mean(test_signal**2)):.6f}")

# Librosa STFT → iSTFT
n_fft = 4096
hop_length = 1024

stft_result = librosa.stft(test_signal, n_fft=n_fft, hop_length=hop_length)
print(f"\n🔄 STFT結果:")
print(f"  形状: {stft_result.shape}")
print(f"  Max magnitude: {np.abs(stft_result).max():.6f}")

# iSTFT
reconstructed = librosa.istft(stft_result, hop_length=hop_length)
print(f"\n🔄 iSTFT結果:")
print(f"  Max: {np.abs(reconstructed).max():.6f}")
print(f"  RMS: {np.sqrt(np.mean(reconstructed**2)):.6f}")

# 再構成誤差
error = test_signal[:len(reconstructed)] - reconstructed
print(f"\n📏 再構成誤差:")
print(f"  Max error: {np.abs(error).max():.6f}")
print(f"  RMS error: {np.sqrt(np.mean(error**2)):.6f}")
print(f"  SNR: {20 * np.log10(np.sqrt(np.mean(test_signal[:len(reconstructed)]**2)) / np.sqrt(np.mean(error**2))):.2f} dB")

# LibrosaのiSTFT正規化を確認
print(f"\n🔍 Librosa iSTFT正規化:")
print(f"  FFT size: {n_fft}")
print(f"  Hop length: {hop_length}")
print(f"  Window: hann (librosaデフォルト)")

# 手動でFFT/iFFTスケーリングテスト
print(f"\n🧪 手動FFT/iFFTスケーリングテスト:")
test_frame = test_signal[:n_fft]

# NumPy FFT (スケーリングなし)
fft_result = np.fft.rfft(test_frame)
ifft_no_scale = np.fft.irfft(fft_result)
print(f"  NumPy iFFT (スケーリングなし) Max: {np.abs(ifft_no_scale).max():.6f}")

# 1/N スケーリング
ifft_scaled = ifft_no_scale / n_fft
print(f"  NumPy iFFT (1/N) Max: {np.abs(ifft_scaled).max():.6f}")

# Librosaと同等の処理
window = librosa.filters.get_window('hann', n_fft)
windowed_frame = test_frame * window
fft_windowed = np.fft.rfft(windowed_frame)
ifft_windowed = np.fft.irfft(fft_windowed)
print(f"  Window適用後 iFFT Max: {np.abs(ifft_windowed).max():.6f}")

print("\n" + "=" * 80)
print("💡 結論:")
print("=" * 80)
print("Librosaは window^2 の合計で正規化を行います。")
print("vDSPの1/N スケーリングとは異なる可能性があります。")
print("=" * 80)
