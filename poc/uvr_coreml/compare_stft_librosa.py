#!/usr/bin/env python3
"""
Compare STFT/iSTFT round-trip behavior between librosa and our implementation.
This helps us understand the correct window normalization.
"""

import numpy as np
import librosa

# Test parameters
n_fft = 8
hop_length = 2
win_length = n_fft

# Simple DC signal
signal = np.ones(16, dtype=np.float32)

print("=" * 80)
print("Python librosa STFT/iSTFT Round-Trip Test")
print("=" * 80)

print("\n📊 Input signal (DC):")
print(f"   {signal}")

# Get window
window = librosa.filters.get_window('hann', n_fft, fftbins=True)
print(f"\n📊 Hann window:")
print(f"   {window}")

# STFT
stft_result = librosa.stft(
    signal,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window='hann',
    center=False  # No padding
)

print(f"\n📊 STFT output shape: {stft_result.shape}")
print(f"   Frequency bins: {stft_result.shape[0]}")
print(f"   Time frames: {stft_result.shape[1]}")

for frame_idx in range(stft_result.shape[1]):
    print(f"\n   Frame {frame_idx}:")
    print(f"     Real: {np.real(stft_result[:, frame_idx])}")
    print(f"     Imag: {np.imag(stft_result[:, frame_idx])}")
    print(f"     Magnitude: {np.abs(stft_result[:, frame_idx])}")

# iSTFT
reconstructed = librosa.istft(
    stft_result,
    hop_length=hop_length,
    win_length=win_length,
    window='hann',
    center=False,
    length=len(signal)
)

print(f"\n📊 Reconstructed signal:")
print(f"   Length: {len(reconstructed)}")
print(f"   Values: {reconstructed}")

# Error
error = signal - reconstructed
print(f"\n📊 Reconstruction errors:")
print(f"   {error}")

rms_error = np.sqrt(np.mean(error ** 2))
print(f"   RMS Error: {rms_error}")

print("\n" + "=" * 80)
