#!/usr/bin/env python3
"""
UVR完全実装テスト - 全チャンク処理版
1. 全STFTチャンクを処理
2. モデルでボーカル抽出
3. 元音声 - ボーカル = インストゥルメンタル
"""
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
import librosa

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"

print("=" * 80)
print("🔬 UVR完全実装テスト: 全チャンク処理版")
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

# 全チャンク処理
freq_bins = 2048
time_frames = 256

total_freq = stft_result.shape[2]
total_time = stft_result.shape[3]

print(f"\n🧠 全チャンク処理開始")
print(f"   周波数ビン: {total_freq}, 時間フレーム: {total_time}")
print(f"   チャンクサイズ: {freq_bins} x {time_frames}")

# 出力用の配列を準備
output_spectrogram = np.zeros((1, 4, total_freq, total_time), dtype=np.float32)

# 時間方向にチャンク分割
num_chunks = (total_time + time_frames - 1) // time_frames
print(f"   処理チャンク数: {num_chunks}")

for chunk_idx in range(num_chunks):
    start_time = chunk_idx * time_frames
    end_time = min(start_time + time_frames, total_time)
    current_time_frames = end_time - start_time

    # チャンク抽出
    chunk = stft_result[:, :, :freq_bins, start_time:end_time]

    # パディングが必要な場合
    if current_time_frames < time_frames:
        pad_size = time_frames - current_time_frames
        chunk = torch.nn.functional.pad(chunk, (0, pad_size, 0, 0))

    input_data = chunk.numpy().astype(np.float32)

    # UVRの前処理
    input_data[:, :, :3, :] = 0

    # 推論
    output = session.run([session.get_outputs()[0].name],
                         {session.get_inputs()[0].name: input_data})
    output_array = output[0]

    # 結果を出力配列に格納
    output_spectrogram[:, :, :freq_bins, start_time:end_time] = output_array[:, :, :, :current_time_frames]

    if (chunk_idx + 1) % 10 == 0:
        print(f"   進行状況: {chunk_idx + 1}/{num_chunks} チャンク処理完了")

print(f"   全チャンク処理完了")

# iSTFT (UVRと同じ)
print(f"\n🔄 iSTFT実行")
output_tensor = torch.from_numpy(output_spectrogram).float()
batch_dims = output_tensor.shape[:-3]
c, f, t = output_tensor.shape[-3:]
n = n_fft // 2 + 1

# 周波数ビンをパディング
f_pad = torch.zeros([*batch_dims, c, n - f, t])
output_padded = torch.cat([output_tensor, f_pad], -2)
output_reshaped = output_padded.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
output_reshaped = output_reshaped.permute([0, 2, 3, 1])
output_complex = output_reshaped[..., 0] + output_reshaped[..., 1] * 1.j

audio_vocal = torch.istft(
    output_complex,
    n_fft=n_fft,
    hop_length=hop_length,
    window=window,
    center=True
)
audio_vocal = audio_vocal.reshape([*batch_dims, 2, -1])
vocal_numpy = audio_vocal.squeeze(0).numpy()

# compensate係数を適用
compensate = 1.035
vocal_numpy = vocal_numpy * compensate

print(f"   ボーカルshape: {vocal_numpy.shape}")
print(f"   ボーカルRMS: {np.sqrt(np.mean(vocal_numpy[0]**2)):.6f}")

# インストゥルメンタル計算
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

# エネルギー確認
vocal_energy = np.mean(vocal_trimmed[0]**2)
inst_energy = np.mean(instrumental_numpy[0]**2)
original_energy = np.mean(original_trimmed[0]**2)
print(f"\n   エネルギー分析:")
print(f"     元音声エネルギー: {original_energy:.9f}")
print(f"     ボーカルエネルギー: {vocal_energy:.9f}")
print(f"     インストゥルメンタルエネルギー: {inst_energy:.9f}")
print(f"     合計エネルギー: {vocal_energy + inst_energy:.9f}")

# 保存
print(f"\n💾 結果保存:")
sf.write("tests/python_output/uvr_vocal_full.wav", vocal_trimmed.T, sr)
sf.write("tests/python_output/uvr_instrumental_full.wav", instrumental_numpy.T, sr)
print(f"   ボーカル: uvr_vocal_full.wav")
print(f"   インストゥルメンタル: uvr_instrumental_full.wav")

print(f"\n✅ 完了")
print(f"\n次のステップ:")
print(f"  1. uvr_vocal_full.wav を聴いてボーカルが正しく抽出できているか確認")
print(f"  2. uvr_instrumental_full.wav を聴いて伴奏が正しく抽出できているか確認")
print(f"  3. 両方が正しければ、Swiftで同じロジックを実装")
print("=" * 80)
