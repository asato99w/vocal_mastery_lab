#!/usr/bin/env python3
"""
Pythonリファレンス実装の中間データをダンプ
Swift実装との比較用
"""
import numpy as np
import librosa
import soundfile as sf
import pickle

print("=" * 80)
print("🔍 Python STFT/iSTFT 中間データダンプ")
print("=" * 80)

# 簡単なテスト信号で検証
sr = 44100
duration = 1.0  # 1秒
t = np.linspace(0, duration, int(sr * duration))
test_signal = np.sin(2 * np.pi * 440 * t)  # 440Hz

print(f"\n📊 テスト信号 (440Hz, 1秒):")
print(f"  Max: {np.abs(test_signal).max():.6f}")
print(f"  RMS: {np.sqrt(np.mean(test_signal**2)):.6f}")
print(f"  サンプル数: {len(test_signal)}")

# ステレオに変換
test_stereo = np.stack([test_signal, test_signal])
print(f"  ステレオ形状: {test_stereo.shape}")

# STFT
n_fft = 4096
hop_length = 1024

print(f"\n🔄 STFT実行 (n_fft={n_fft}, hop={hop_length}):")
left_stft = librosa.stft(test_stereo[0], n_fft=n_fft, hop_length=hop_length)
right_stft = librosa.stft(test_stereo[1], n_fft=n_fft, hop_length=hop_length)

print(f"  Left STFT形状: {left_stft.shape}")
print(f"  周波数ビン数: {left_stft.shape[0]}")
print(f"  時間フレーム数: {left_stft.shape[1]}")

# 振幅と位相に分解
left_magnitude = np.abs(left_stft)
left_phase = np.angle(left_stft)

print(f"\n📊 STFT統計 (Left):")
print(f"  Magnitude Max: {left_magnitude.max():.6f}")
print(f"  Magnitude Mean: {left_magnitude.mean():.6f}")
print(f"  Phase range: {left_phase.min():.6f} ~ {left_phase.max():.6f}")

# 440Hz binの確認
freq_440_bin = int(440 * n_fft / sr)
print(f"\n🎵 440Hz bin [{freq_440_bin}]:")
print(f"  Magnitude: {left_magnitude[freq_440_bin, :].mean():.6f}")
print(f"  Phase: {left_phase[freq_440_bin, 0]:.6f}")

# 最初の3フレームの詳細
print(f"\n🔬 最初の3フレームの詳細 (440Hz bin):")
for i in range(min(3, left_stft.shape[1])):
    print(f"  Frame {i}: mag={left_magnitude[freq_440_bin, i]:.6f}, phase={left_phase[freq_440_bin, i]:.6f}")

# iSTFT
reconstructed_left = librosa.istft(left_stft, hop_length=hop_length)

print(f"\n🔄 iSTFT結果:")
print(f"  Max: {np.abs(reconstructed_left).max():.6f}")
print(f"  RMS: {np.sqrt(np.mean(reconstructed_left**2)):.6f}")
print(f"  長さ: {len(reconstructed_left)}")

# 再構成誤差
min_len = min(len(test_signal), len(reconstructed_left))
error = test_signal[:min_len] - reconstructed_left[:min_len]
print(f"\n📏 再構成誤差:")
print(f"  Max error: {np.abs(error).max():.6f}")
print(f"  RMS error: {np.sqrt(np.mean(error**2)):.6f}")

# 中間データを保存
debug_data = {
    'test_signal': test_signal,
    'stft_complex': left_stft,
    'magnitude': left_magnitude,
    'phase': left_phase,
    'reconstructed': reconstructed_left,
    'n_fft': n_fft,
    'hop_length': hop_length,
    'sr': sr
}

with open('tests/python_output/debug_stft_data.pkl', 'wb') as f:
    pickle.dump(debug_data, f)

# テキスト形式でも保存（Swift から読める形式）
np.savetxt('tests/python_output/test_signal.txt', test_signal[:100])
np.savetxt('tests/python_output/magnitude_frame0.txt', left_magnitude[:, 0])
np.savetxt('tests/python_output/phase_frame0.txt', left_phase[:, 0])
np.savetxt('tests/python_output/reconstructed.txt', reconstructed_left[:100])

print(f"\n💾 デバッグデータ保存完了:")
print(f"  tests/python_output/debug_stft_data.pkl")
print(f"  tests/python_output/test_signal.txt (最初の100サンプル)")
print(f"  tests/python_output/magnitude_frame0.txt (フレーム0)")
print(f"  tests/python_output/phase_frame0.txt (フレーム0)")

print("\n" + "=" * 80)
