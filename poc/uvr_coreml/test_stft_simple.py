#!/usr/bin/env python3
"""
Simple STFT/iSTFT test to understand the correct implementation.
"""

import numpy as np

def manual_stft_istft():
    """Manual STFT/iSTFT implementation for understanding."""

    # Parameters
    n_fft = 8
    hop_length = 2

    # Simple DC signal
    signal = np.ones(16, dtype=np.float32)

    print("=" * 80)
    print("Manual STFT/iSTFT Implementation")
    print("=" * 80)

    print(f"\n📊 Input signal: {signal}")

    # Create Hann window
    window = np.hanning(n_fft).astype(np.float32)
    print(f"\n📊 Hann window: {window}")
    print(f"   Window sum: {window.sum()}")
    print(f"   Window^2 sum: {(window**2).sum()}")

    # STFT - Forward
    num_frames = (len(signal) - n_fft) // hop_length + 1
    stft_result = []

    print(f"\n📊 STFT Forward (frames: {num_frames}):")

    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        frame = signal[start:start + n_fft]

        # Apply window
        windowed = frame * window

        # FFT
        fft_result = np.fft.fft(windowed)
        stft_result.append(fft_result)

        print(f"\n   Frame {frame_idx}:")
        print(f"     Input: {frame}")
        print(f"     Windowed: {windowed}")
        print(f"     FFT result (first 5): {fft_result[:5]}")

    # iSTFT - Inverse
    output_length = (num_frames - 1) * hop_length + n_fft
    output = np.zeros(output_length, dtype=np.float32)
    window_sum = np.zeros(output_length, dtype=np.float32)

    print(f"\n📊 iSTFT Inverse:")

    for frame_idx in range(num_frames):
        # IFFT
        ifft_result = np.fft.ifft(stft_result[frame_idx]).real.astype(np.float32)

        # Apply window for synthesis
        windowed_output = ifft_result * window

        # Overlap-add
        start = frame_idx * hop_length
        output[start:start + n_fft] += windowed_output
        window_sum[start:start + n_fft] += window

        print(f"\n   Frame {frame_idx}:")
        print(f"     IFFT result: {ifft_result}")
        print(f"     Windowed: {windowed_output}")

    # Normalize
    # Avoid division by zero
    window_sum = np.where(window_sum > 1e-8, window_sum, 1.0)
    reconstructed = output / window_sum

    print(f"\n📊 Reconstruction:")
    print(f"   Window sum: {window_sum}")
    print(f"   Output before normalization: {output}")
    print(f"   Reconstructed: {reconstructed}")

    # Error
    error = signal - reconstructed
    rms_error = np.sqrt(np.mean(error**2))

    print(f"\n📊 Error:")
    print(f"   Difference: {error}")
    print(f"   RMS Error: {rms_error}")
    print(f"   Max absolute error: {np.abs(error).max()}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    manual_stft_istft()
