#!/usr/bin/env python3
"""
torch.istftの窓適用動作を確認
UVRがなぜiSTFT後にさらに窓を掛けるのかを理解する
"""
import numpy as np
import torch

# パラメータ
n_fft = 4096
hop_length = 1024
chunk_size = 261120
overlap = 0.25

print("=" * 80)
print("🔬 torch.istftの窓適用動作テスト")
print("=" * 80)

# テスト信号生成（単純なサイン波）
sr = 44100
duration = chunk_size / sr  # ~5.93秒
t = np.linspace(0, duration, chunk_size)
freq = 440  # A4
test_signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
test_signal_stereo = np.stack([test_signal, test_signal])

print(f"\n📊 テスト信号:")
print(f"   長さ: {chunk_size} サンプル")
print(f"   RMS: {np.sqrt(np.mean(test_signal**2)):.6f}")

# PyTorch変換
signal_torch = torch.from_numpy(test_signal_stereo).float()
window = torch.hann_window(window_length=n_fft, periodic=True)

# STFT
print(f"\n🔧 STFT実行")
stft_result = torch.stft(
    signal_torch.reshape([-1, signal_torch.shape[-1]]),
    n_fft=n_fft,
    hop_length=hop_length,
    window=window,
    center=True,
    return_complex=True
)
print(f"   STFT shape: {stft_result.shape}")

# iSTFT (デフォルト動作)
print(f"\n🔄 iSTFT実行 (デフォルト)")
reconstructed_default = torch.istft(
    stft_result,
    n_fft=n_fft,
    hop_length=hop_length,
    window=window,
    center=True
)
reconstructed_default_np = reconstructed_default.numpy()

print(f"   出力shape: {reconstructed_default_np.shape}")
print(f"   出力RMS: {np.sqrt(np.mean(reconstructed_default_np[0]**2)):.6f}")

# 元信号との比較
min_len = min(test_signal.shape[0], reconstructed_default_np.shape[1])
original_trimmed = test_signal[:min_len]
reconstructed_trimmed = reconstructed_default_np[0, :min_len]

reconstruction_error = np.sqrt(np.mean((original_trimmed - reconstructed_trimmed)**2))
print(f"\n✅ 再構成誤差 (STFT→iSTFT):")
print(f"   RMS誤差: {reconstruction_error:.8f}")
print(f"   相対誤差: {reconstruction_error / np.sqrt(np.mean(original_trimmed**2)) * 100:.2f}%")

# スケール比較
scale_ratio = np.sqrt(np.mean(reconstructed_trimmed**2)) / np.sqrt(np.mean(original_trimmed**2))
print(f"   スケール比: {scale_ratio:.6f}")

# UVR方式：iSTFT後にさらに窓を掛ける
print(f"\n🔄 UVR方式：iSTFT後にさらに窓を適用")
actual_len = reconstructed_default_np.shape[1]
window_np = np.hanning(actual_len)
window_2d = np.tile(window_np[None, :], (2, 1))

reconstructed_uvr = reconstructed_default_np * window_2d

print(f"   窓適用後RMS: {np.sqrt(np.mean(reconstructed_uvr[0]**2)):.6f}")

# 正規化（仮定：window^2で割る）
# COLA条件: Σ w[n-kH]^2 = 1 for all n
# overlap=0.25, hop=1024, chunk=261120の場合
step = int((1 - overlap) * chunk_size)  # 195840
print(f"\n🔍 OLA正規化シミュレーション:")
print(f"   chunk_size: {chunk_size}")
print(f"   step: {step}")
print(f"   overlap: {overlap}")

# 簡単なOLAシミュレーション（2チャンク）
total_length = chunk_size + step
result = np.zeros((2, total_length), dtype=np.float32)
divider = np.zeros((2, total_length), dtype=np.float32)

# チャンク1
result[:, :chunk_size] += reconstructed_uvr[:, :chunk_size]
divider[:, :chunk_size] += window_2d[:, :chunk_size]

# チャンク2（オーバーラップ）
if actual_len >= chunk_size:
    result[:, step:step+chunk_size] += reconstructed_uvr[:, :chunk_size]
    divider[:, step:step+chunk_size] += window_2d[:, :chunk_size]

# OLA正規化
# ゼロ除算を避ける
divider_safe = np.where(divider > 1e-8, divider, 1.0)
ola_normalized = result / divider_safe

print(f"   OLA正規化後RMS: {np.sqrt(np.mean(ola_normalized[0]**2)):.6f}")

# dividerの統計
print(f"\n📊 Divider統計:")
print(f"   最小値: {np.min(divider):.6f}")
print(f"   最大値: {np.max(divider):.6f}")
print(f"   平均値: {np.mean(divider):.6f}")
print(f"   ゼロの数: {np.sum(divider < 1e-8)}")

# 中央部のdivider値（オーバーラップ領域）
if step < chunk_size:
    overlap_region = divider[:, step:chunk_size]
    print(f"   オーバーラップ領域平均: {np.mean(overlap_region):.6f}")

print(f"\n💡 結論:")
print(f"   1. torch.istftは窓を適用して正しく再構成: RMS誤差 {reconstruction_error:.8f}")
print(f"   2. UVRはさらに窓を掛けてOLA正規化を実施")
print(f"   3. これはチャンク境界でのスムージングのため")

print("=" * 80)
