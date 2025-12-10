#!/usr/bin/env python3
"""
UVRの完全な実装に基づくテスト
1. モデルでボーカル抽出
2. 元音声 - ボーカル = インストゥルメンタル
"""
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
import librosa

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"

print("=" * 80)
print("🔬 UVR完全実装テスト: ボーカル抽出 → インストゥルメンタル計算")
print("=" * 80)

# 音声読み込み
print(f"\n📂 音声読み込み: {input_file}")
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

original_audio = y.copy()
print(f"   元音声shape: {y.shape}")
print(f"   元音声RMS: {np.sqrt(np.mean(y[0]**2)):.6f}")

# PyTorch STFT (UVRと同じ)
mix = torch.from_numpy(y).float()
n_fft = 4096
hop_length = 1024
window = torch.hann_window(window_length=n_fft, periodic=True)

print(f"\n🔧 STFT実行 (PyTorch)")
stft_result = torch.stft(
    mix.reshape([-1, mix.shape[-1]]),
    n_fft=n_fft,
    hop_length=hop_length,
    window=window,
    center=True,
    return_complex=False
)

# UVRのフォーマットに変換
stft_result = stft_result.permute([0, 3, 1, 2])
c = 2
stft_result = stft_result.reshape([1, c, 2, -1, stft_result.shape[-1]])
stft_result = stft_result.reshape([1, c * 2, -1, stft_result.shape[-1]])

print(f"   STFT shape: {stft_result.shape}")

# ONNXモデル
session = ort.InferenceSession(model_path)

# チャンク処理 (最初のチャンクのみテスト)
freq_bins = 2048
time_frames = 256
chunk = stft_result[:, :, :freq_bins, :time_frames]
input_data = chunk.numpy().astype(np.float32)

# UVRの前処理
input_data[:, :, :3, :] = 0

print(f"\n🧠 モデル推論実行")
output = session.run([session.get_outputs()[0].name],
                     {session.get_inputs()[0].name: input_data})
output_array = output[0]

print(f"   出力shape: {output_array.shape}")

# iSTFT (UVRと同じ)
output_tensor = torch.from_numpy(output_array).float()
batch_dims = output_tensor.shape[:-3]
c, f, t = output_tensor.shape[-3:]
n = n_fft // 2 + 1

f_pad = torch.zeros([*batch_dims, c, n - f, t])
output_padded = torch.cat([output_tensor, f_pad], -2)
output_reshaped = output_padded.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
output_reshaped = output_reshaped.permute([0, 2, 3, 1])
output_complex = output_reshaped[..., 0] + output_reshaped[..., 1] * 1.j

print(f"\n🔄 iSTFT実行")
audio_vocal = torch.istft(
    output_complex,
    n_fft=n_fft,
    hop_length=hop_length,
    window=window,
    center=True
)
audio_vocal = audio_vocal.reshape([*batch_dims, 2, -1])
vocal_numpy = audio_vocal.squeeze(0).numpy()

# compensate係数を適用 (UVR separate.py:615)
compensate = 1.035  # デフォルト値
vocal_numpy = vocal_numpy * compensate

print(f"   ボーカルshape: {vocal_numpy.shape}")
print(f"   ボーカルRMS: {np.sqrt(np.mean(vocal_numpy[0]**2)):.6f}")

# インストゥルメンタル計算 (UVR separate.py:514)
print(f"\n➖ インストゥルメンタル計算: Original - Vocal")

# 長さを揃える
min_len = min(original_audio.shape[1], vocal_numpy.shape[1])
original_trimmed = original_audio[:, :min_len]
vocal_trimmed = vocal_numpy[:, :min_len]

instrumental_numpy = original_trimmed - vocal_trimmed

print(f"   インストゥルメンタルshape: {instrumental_numpy.shape}")
print(f"   インストゥルメンタルRMS: {np.sqrt(np.mean(instrumental_numpy[0]**2)):.6f}")

# 統計
print(f"\n📊 結果統計:")
print(f"   元音声RMS:          {np.sqrt(np.mean(original_trimmed[0]**2)):.6f}")
print(f"   ボーカルRMS:        {np.sqrt(np.mean(vocal_trimmed[0]**2)):.6f}")
print(f"   インストゥルメンタルRMS: {np.sqrt(np.mean(instrumental_numpy[0]**2)):.6f}")

# エネルギー保存の確認
total_energy = np.sqrt(np.mean(vocal_trimmed[0]**2)**2 + np.mean(instrumental_numpy[0]**2)**2)
original_energy = np.sqrt(np.mean(original_trimmed[0]**2))
print(f"\n   エネルギー保存:")
print(f"     ボーカル² + インストゥルメンタル² ≈ 元音声²")
print(f"     {total_energy:.6f} ≈ {original_energy:.6f}")

# 保存
print(f"\n💾 結果保存:")
sf.write("tests/python_output/uvr_vocal.wav", vocal_trimmed.T, sr)
sf.write("tests/python_output/uvr_instrumental.wav", instrumental_numpy.T, sr)
print(f"   ボーカル: uvr_vocal.wav")
print(f"   インストゥルメンタル: uvr_instrumental.wav")

print(f"\n✅ 完了")
print(f"\n次のステップ:")
print(f"  1. uvr_vocal.wav を聴いてボーカルが正しく抽出できているか確認")
print(f"  2. uvr_instrumental.wav を聴いて伴奏が正しく抽出できているか確認")
print(f"  3. 両方が正しければ、Swiftで同じロジックを実装")
print("=" * 80)
