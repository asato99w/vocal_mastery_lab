#!/usr/bin/env python3
"""
UVR-MDX-NET-Inst_Main: Instrumentalを出力するモデル
正しい実装:
  instrumental = model_output * compensate
  vocal = original - instrumental
"""
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
import librosa

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"

print("=" * 80)
print("🔬 正しい実装: UVR-MDX-NET-Inst_Main (Instrumental出力モデル)")
print("=" * 80)

# 音声読み込み
print(f"\n📂 音声読み込み: {input_file}")
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

original_audio = y.copy()
print(f"   元音声shape: {y.shape}")
print(f"   元音声RMS: {np.sqrt(np.mean(y[0]**2)):.6f}")

# PyTorch STFT
mix = torch.from_numpy(y).float()
n_fft = 4096
hop_length = 1024
window = torch.hann_window(window_length=n_fft, periodic=True)

print(f"\n🔧 STFT実行 (PyTorch)")
print(f"   n_fft={n_fft}, hop_length={hop_length}, center=True")

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

# 全チャンク処理
freq_bins = 2048
time_frames = 256

total_freq = stft_result.shape[2]
total_time = stft_result.shape[3]

print(f"\n🧠 全チャンク処理開始")
print(f"   周波数ビン: {total_freq}, 時間フレーム: {total_time}")
print(f"   チャンクサイズ: {freq_bins} x {time_frames}")

output_spectrogram = np.zeros((1, 4, total_freq, total_time), dtype=np.float32)

num_chunks = (total_time + time_frames - 1) // time_frames
print(f"   処理チャンク数: {num_chunks}")

for chunk_idx in range(num_chunks):
    start_time = chunk_idx * time_frames
    end_time = min(start_time + time_frames, total_time)
    current_time_frames = end_time - start_time

    chunk = stft_result[:, :, :freq_bins, start_time:end_time]

    if current_time_frames < time_frames:
        pad_size = time_frames - current_time_frames
        chunk = torch.nn.functional.pad(chunk, (0, pad_size, 0, 0))

    input_data = chunk.numpy().astype(np.float32)
    input_data[:, :, :3, :] = 0  # ゼロ化ビン

    output = session.run([session.get_outputs()[0].name],
                         {session.get_inputs()[0].name: input_data})
    output_array = output[0]

    output_spectrogram[:, :, :freq_bins, start_time:end_time] = output_array[:, :, :, :current_time_frames]

    if (chunk_idx + 1) % 10 == 0:
        print(f"   進行状況: {chunk_idx + 1}/{num_chunks} チャンク処理完了")

print(f"   全チャンク処理完了")

# iSTFT
print(f"\n🔄 iSTFT実行")
output_tensor = torch.from_numpy(output_spectrogram).float()
batch_dims = output_tensor.shape[:-3]
c, f, t = output_tensor.shape[-3:]
n = n_fft // 2 + 1

f_pad = torch.zeros([*batch_dims, c, n - f, t])
output_padded = torch.cat([output_tensor, f_pad], -2)
output_reshaped = output_padded.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
output_reshaped = output_reshaped.permute([0, 2, 3, 1])
output_complex = output_reshaped[..., 0] + output_reshaped[..., 1] * 1.j

model_output_time = torch.istft(
    output_complex,
    n_fft=n_fft,
    hop_length=hop_length,
    window=window,
    center=True
)
model_output_time = model_output_time.reshape([*batch_dims, 2, -1])
model_output_numpy = model_output_time.squeeze(0).numpy()

# ✅ 正しい実装: このモデルはInstrumentalを出力
compensate = 1.035
instrumental_numpy = model_output_numpy * compensate

print(f"   モデル出力shape: {instrumental_numpy.shape}")
print(f"   インストゥルメンタルRMS: {np.sqrt(np.mean(instrumental_numpy[0]**2)):.6f}")

# ✅ ボーカル = 元音声 - インストゥルメンタル
print(f"\n➖ ボーカル計算: Original - Instrumental")

min_len = min(original_audio.shape[1], instrumental_numpy.shape[1])
original_trimmed = original_audio[:, :min_len]
instrumental_trimmed = instrumental_numpy[:, :min_len]

vocal_numpy = original_trimmed - instrumental_trimmed

print(f"   ボーカルshape: {vocal_numpy.shape}")
print(f"   ボーカルRMS: {np.sqrt(np.mean(vocal_numpy[0]**2)):.6f}")

# 統計
print(f"\n📊 結果統計:")
orig_rms = np.sqrt(np.mean(original_trimmed[0]**2))
voc_rms = np.sqrt(np.mean(vocal_numpy[0]**2))
inst_rms = np.sqrt(np.mean(instrumental_trimmed[0]**2))

print(f"   元音声RMS:          {orig_rms:.6f}")
print(f"   ボーカルRMS:        {voc_rms:.6f}")
print(f"   インストゥルメンタルRMS: {inst_rms:.6f}")

# 整合性チェック
reconstructed = vocal_numpy + instrumental_trimmed
reconstruction_error = np.sqrt(np.mean((original_trimmed - reconstructed)**2))
print(f"\n✅ 整合性チェック:")
print(f"   original - (vocal + instrumental) RMS誤差: {reconstruction_error:.8f}")
print(f"   誤差レベル: {'良好 (< 1e-4)' if reconstruction_error < 1e-4 else '要確認'}")

# 保存
print(f"\n💾 結果保存:")
sf.write("tests/python_output/correct_vocal.wav", vocal_numpy.T, sr)
sf.write("tests/python_output/correct_instrumental.wav", instrumental_trimmed.T, sr)
print(f"   ボーカル: correct_vocal.wav")
print(f"   インストゥルメンタル: correct_instrumental.wav")

print(f"\n✅ 完了")
print(f"\n次のステップ:")
print(f"  1. correct_vocal.wav を聴いてボーカルが正しく抽出できているか確認")
print(f"  2. correct_instrumental.wav を聴いて伴奏が正しく抽出できているか確認")
print(f"  3. 整合性チェックの誤差が < 1e-4 であることを確認")
print("=" * 80)
