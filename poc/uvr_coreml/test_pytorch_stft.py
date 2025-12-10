#!/usr/bin/env python3
"""
PyTorchのSTFT/iSTFTを使用した正しいアプローチ
UVRと同じSTFT実装を使用する
"""
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
import librosa

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"

print("=" * 80)
print("🔬 PyTorch STFT/iSTFTを使用したテスト")
print("=" * 80)

# 音声読み込み
print(f"\n📂 音声読み込み: {input_file}")
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

print(f"   元音声shape: {y.shape}")
print(f"   元音声RMS: {np.sqrt(np.mean(y[0]**2)):.6f}")

# PyTorchテンソルに変換
mix = torch.from_numpy(y).float()

# PyTorch STFTパラメータ (UVRと同じ)
n_fft = 4096
hop_length = 1024
window = torch.hann_window(window_length=n_fft, periodic=True)

print(f"\n🔧 PyTorch STFT実行")
# STFT
stft_result = torch.stft(
    mix.reshape([-1, mix.shape[-1]]),
    n_fft=n_fft,
    hop_length=hop_length,
    window=window,
    center=True,
    return_complex=False
)

# UVRと同じ形式に変換: [batch, channel, 2(real/imag), freq, time]
stft_result = stft_result.permute([0, 3, 1, 2])  # [batch, 2, freq, time]
c = 2  # stereo
stft_result = stft_result.reshape([1, c, 2, -1, stft_result.shape[-1]])
stft_result = stft_result.reshape([1, c * 2, -1, stft_result.shape[-1]])

print(f"   STFT shape: {stft_result.shape}")
print(f"   STFT mean: {stft_result.mean():.6f}")
print(f"   STFT max: {stft_result.abs().max():.6f}")

# ONNXモデル読み込み
print(f"\n🤖 ONNXモデル読み込み")
session = ort.InferenceSession(model_path)

# 最初のチャンクでテスト
freq_bins = 2048
time_frames = 256

chunk = stft_result[:, :, :freq_bins, :time_frames]

# 入力データ (既に[batch, 4, freq, time]の形式)
input_data = chunk.numpy().astype(np.float32)

# UVRの前処理: 最初の3周波数ビンをゼロにする
input_data[:, :, :3, :] = 0

print(f"\n🧠 推論実行")
print(f"   入力shape: {input_data.shape}")
print(f"   入力 mean: {np.mean(input_data):.6f}")
print(f"   入力 max: {np.max(np.abs(input_data)):.6f}")

# 推論
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: input_data})
output_array = output[0]

print(f"   出力shape: {output_array.shape}")
print(f"   出力 mean: {np.mean(output_array):.6f}")
print(f"   出力 max: {np.max(np.abs(output_array)):.6f}")

# 出力をPyTorchテンソルに変換
output_tensor = torch.from_numpy(output_array).float()

# UVRのinverse STFTと同じ処理
batch_dims = output_tensor.shape[:-3]
c, f, t = output_tensor.shape[-3:]
n = n_fft // 2 + 1

# 周波数ビンをパディング (2048 -> 2049)
f_pad = torch.zeros([*batch_dims, c, n - f, t])
output_padded = torch.cat([output_tensor, f_pad], -2)

print(f"\n🔧 iSTFT準備:")
print(f"   パディング後shape: {output_padded.shape}")

# [batch, 4, 2049, time] -> [batch, 2, 2, 2049, time]
output_reshaped = output_padded.reshape([*batch_dims, c // 2, 2, n, t])
# -> [2, 2, 2049, time] (flatten batch)
output_reshaped = output_reshaped.reshape([-1, 2, n, t])
# -> [2, 2049, time, 2] (for complex conversion)
output_reshaped = output_reshaped.permute([0, 2, 3, 1])

# 複素数に変換
output_complex = output_reshaped[..., 0] + output_reshaped[..., 1] * 1.j

print(f"   複素数shape: {output_complex.shape}")
print(f"   複素数 mean magnitude: {torch.abs(output_complex).mean():.6f}")

# PyTorch iSTFT
audio_tensor = torch.istft(
    output_complex,
    n_fft=n_fft,
    hop_length=hop_length,
    window=window,
    center=True
)

# [2, time] -> [batch, 2, time]
audio_tensor = audio_tensor.reshape([*batch_dims, 2, -1])

print(f"\n📊 iSTFT結果:")
print(f"   出力shape: {audio_tensor.shape}")
print(f"   出力 RMS: {torch.sqrt(torch.mean(audio_tensor[0, 0]**2)):.6f}")
print(f"   出力 max: {torch.abs(audio_tensor).max():.6f}")

# Numpyに変換して保存
audio_numpy = audio_tensor.squeeze(0).numpy()

# 元音声との比較
original_rms = np.sqrt(np.mean(y[0]**2))
result_rms = np.sqrt(np.mean(audio_numpy[0]**2))

print(f"\n📊 元音声との比較:")
print(f"   元音声RMS: {original_rms:.6f}")
print(f"   結果RMS: {result_rms:.6f}")
print(f"   比率: {result_rms/original_rms*100:.1f}%")

# 保存
output_path = "tests/python_output/pytorch_stft_approach.wav"
sf.write(output_path, audio_numpy.T, sr)

print(f"\n✅ 完了")
print(f"   出力: {output_path}")
print(f"\nこのファイルを聴いて、正しく動作しているか確認してください。")
print("=" * 80)
