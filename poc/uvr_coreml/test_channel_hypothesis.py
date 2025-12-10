#!/usr/bin/env python3
"""
チャンネル解釈の検証
仮説: Ch0=Vocal, Ch1=Instrumental かもしれない
"""
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
import librosa

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"

print("=" * 80)
print("🔬 チャンネル解釈の検証")
print("=" * 80)

# 音声読み込み
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

print(f"\n📂 元音声RMS: {np.sqrt(np.mean(y[0]**2)):.6f}")

# PyTorch STFT
mix = torch.from_numpy(y).float()
n_fft = 4096
hop_length = 1024
window = torch.hann_window(window_length=n_fft, periodic=True)

stft_result = torch.stft(
    mix.reshape([-1, mix.shape[-1]]),
    n_fft=n_fft,
    hop_length=hop_length,
    window=window,
    center=True,
    return_complex=False
)

stft_result = stft_result.permute([0, 3, 1, 2])
c = 2
stft_result = stft_result.reshape([1, c, 2, -1, stft_result.shape[-1]])
stft_result = stft_result.reshape([1, c * 2, -1, stft_result.shape[-1]])

# ONNXモデル
session = ort.InferenceSession(model_path)

# チャンク
freq_bins = 2048
time_frames = 256
chunk = stft_result[:, :, :freq_bins, :time_frames]
input_data = chunk.numpy().astype(np.float32)
input_data[:, :, :3, :] = 0

# 推論
output = session.run([session.get_outputs()[0].name],
                     {session.get_inputs()[0].name: input_data})
output_array = output[0]

print(f"\n🧠 モデル出力shape: {output_array.shape}")

# 仮説1: Ch0/Ch1 = ステレオL/R (これまでの解釈)
print(f"\n📊 仮説1: Ch0=左, Ch1=右 (ステレオ)")
output_tensor = torch.from_numpy(output_array).float()
batch_dims = output_tensor.shape[:-3]
c, f, t = output_tensor.shape[-3:]
n = n_fft // 2 + 1
f_pad = torch.zeros([*batch_dims, c, n - f, t])
output_padded = torch.cat([output_tensor, f_pad], -2)
output_reshaped = output_padded.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
output_reshaped = output_reshaped.permute([0, 2, 3, 1])
output_complex = output_reshaped[..., 0] + output_reshaped[..., 1] * 1.j

audio_h1 = torch.istft(output_complex, n_fft=n_fft, hop_length=hop_length, window=window, center=True)
audio_h1 = audio_h1.reshape([*batch_dims, 2, -1])
audio_h1_numpy = audio_h1.squeeze(0).numpy()

rms_h1_left = np.sqrt(np.mean(audio_h1_numpy[0]**2))
rms_h1_right = np.sqrt(np.mean(audio_h1_numpy[1]**2))
print(f"   左チャンネルRMS: {rms_h1_left:.6f}")
print(f"   右チャンネルRMS: {rms_h1_right:.6f}")

sf.write("tests/python_output/h1_stereo_left.wav", audio_h1_numpy[0], sr)
sf.write("tests/python_output/h1_stereo_right.wav", audio_h1_numpy[1], sr)

# 仮説2: Ch0 (両チャンネル) = Vocal, Ch1 (両チャンネル) = Instrumental
print(f"\n📊 仮説2: Ch0=Vocal, Ch1=Instrumental (ソース分離)")
# Ch0を取り出し (real: index 0, imag: index 1)
ch0_stft = output_array[0, 0:2, :, :]  # [2, freq, time]
# Ch1を取り出し (real: index 2, imag: index 3)
ch1_stft = output_array[0, 2:4, :, :]  # [2, freq, time]

# Ch0: Vocal
ch0_tensor = torch.from_numpy(ch0_stft).float()
f_pad_single = torch.zeros([n - freq_bins, time_frames])
ch0_padded = torch.cat([ch0_tensor, f_pad_single.unsqueeze(0).repeat(2, 1, 1)], 1)
ch0_complex = ch0_padded[0] + 1.j * ch0_padded[1]
audio_vocal = torch.istft(ch0_complex, n_fft=n_fft, hop_length=hop_length, window=window, center=True)

# Ch1: Instrumental
ch1_tensor = torch.from_numpy(ch1_stft).float()
ch1_padded = torch.cat([ch1_tensor, f_pad_single.unsqueeze(0).repeat(2, 1, 1)], 1)
ch1_complex = ch1_padded[0] + 1.j * ch1_padded[1]
audio_inst = torch.istft(ch1_complex, n_fft=n_fft, hop_length=hop_length, window=window, center=True)

audio_vocal_numpy = audio_vocal.numpy()
audio_inst_numpy = audio_inst.numpy()

rms_vocal = np.sqrt(np.mean(audio_vocal_numpy**2))
rms_inst = np.sqrt(np.mean(audio_inst_numpy**2))

print(f"   Vocal RMS: {rms_vocal:.6f}")
print(f"   Instrumental RMS: {rms_inst:.6f}")

sf.write("tests/python_output/h2_vocal.wav", audio_vocal_numpy, sr)
sf.write("tests/python_output/h2_instrumental.wav", audio_inst_numpy, sr)

print(f"\n✅ 検証完了")
print(f"\n次のファイルを聴き比べてください:")
print(f"  仮説1 (ステレオ):")
print(f"    - h1_stereo_left.wav")
print(f"    - h1_stereo_right.wav")
print(f"  仮説2 (ソース分離):")
print(f"    - h2_vocal.wav")
print(f"    - h2_instrumental.wav")
print("=" * 80)
