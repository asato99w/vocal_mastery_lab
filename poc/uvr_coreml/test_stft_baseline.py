#!/usr/bin/env python3
"""
STFTベースライン確認 - マスク適用前の音声復元テスト
"""
import numpy as np
import librosa
import soundfile as sf

input_file = "tests/output/hollow_crown_from_flac.wav"

print("=" * 80)
print("🔍 STFTベースライン確認 - マスク適用前のiSTFT復元テスト")
print("=" * 80)

# 音声読み込み
print(f"\n📂 音声読み込み: {input_file}")
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

print(f"   サンプルレート: {sr} Hz")
print(f"   チャンネル数: {y.shape[0]}")
print(f"   サンプル数: {y.shape[1]}")
print(f"   元音声RMS: {np.sqrt(np.mean(y[0]**2)):.6f}")

# STFT
n_fft = 4096
hop_length = 1024

print(f"\n🔄 STFT実行 (n_fft={n_fft}, hop_length={hop_length})")
stft_left = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length)

print(f"   STFT shape: {stft_left.shape}")
print(f"   STFT magnitude range: {np.abs(stft_left).min():.6f} - {np.abs(stft_left).max():.6f}")
print(f"   STFT mean magnitude: {np.abs(stft_left).mean():.6f}")

# iSTFT復元 (マスクなし)
print(f"\n🔄 iSTFT復元 (マスクなし)")
audio_reconstructed = librosa.istft(stft_left, hop_length=hop_length, length=y.shape[1])

print(f"   復元音声 shape: {audio_reconstructed.shape}")
print(f"   復元音声 RMS: {np.sqrt(np.mean(audio_reconstructed**2)):.6f}")
print(f"   復元音声 Max: {np.max(np.abs(audio_reconstructed)):.6f}")

# 保存
audio_stereo = np.stack([audio_reconstructed, audio_reconstructed])
output_file = "tests/python_output/baseline_no_mask.wav"
sf.write(output_file, audio_stereo.T, sr)
print(f"   保存: {output_file}")

# 最初の2048ビンだけでテスト
freq_bins = 2048
print(f"\n🔄 iSTFT復元 (最初の{freq_bins}ビンのみ)")
stft_trimmed = stft_left[:freq_bins, :]

print(f"   Trimmed STFT shape: {stft_trimmed.shape}")
print(f"   Trimmed STFT magnitude range: {np.abs(stft_trimmed).min():.6f} - {np.abs(stft_trimmed).max():.6f}")

audio_trimmed = librosa.istft(stft_trimmed, hop_length=hop_length, length=y.shape[1])

print(f"   復元音声 RMS: {np.sqrt(np.mean(audio_trimmed**2)):.6f}")
print(f"   復元音声 Max: {np.max(np.abs(audio_trimmed)):.6f}")

# 保存
audio_stereo = np.stack([audio_trimmed, audio_trimmed])
output_file = "tests/python_output/baseline_2048bins.wav"
sf.write(output_file, audio_stereo.T, sr)
print(f"   保存: {output_file}")

# 値1.0のマスクをかけてテスト
print(f"\n🎭 マスク適用テスト (全て1.0)")
mask_ones = np.ones((freq_bins, stft_trimmed.shape[1]), dtype=np.float32)
stft_masked = stft_trimmed * mask_ones

audio_masked = librosa.istft(stft_masked, hop_length=hop_length, length=y.shape[1])

print(f"   マスク適用後 RMS: {np.sqrt(np.mean(audio_masked**2)):.6f}")
print(f"   マスク適用後 Max: {np.max(np.abs(audio_masked)):.6f}")

# 保存
audio_stereo = np.stack([audio_masked, audio_masked])
output_file = "tests/python_output/baseline_mask_ones.wav"
sf.write(output_file, audio_stereo.T, sr)
print(f"   保存: {output_file}")

print("\n✅ ベースライン確認完了")
print("\n次のファイルを聴いて確認してください:")
print("  1. tests/python_output/baseline_no_mask.wav - マスクなし復元")
print("  2. tests/python_output/baseline_2048bins.wav - 2048ビンのみ復元")
print("  3. tests/python_output/baseline_mask_ones.wav - 1.0マスク適用")

print("\n" + "=" * 80)
