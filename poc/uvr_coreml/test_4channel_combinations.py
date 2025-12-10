#!/usr/bin/env python3
"""
4チャンネル全ての組み合わせをテスト
"""
import numpy as np
import librosa
import soundfile as sf
import coremltools as ct

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/coreml/UVR-MDX-NET-Inst_Main.mlpackage"

print("=" * 80)
print("🔬 4チャンネル全組み合わせテスト")
print("=" * 80)

# 音声読み込み
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

# STFT
n_fft = 4096
hop_length = 1024
stft_left = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length)

# モデル読み込み
model = ct.models.MLModel(model_path)

# 最初のチャンクのみ
freq_bins = 2048
time_frames = 256

chunk_left = stft_left[:freq_bins, :time_frames]

input_data = np.zeros((1, 4, freq_bins, time_frames), dtype=np.float32)
input_data[0, 0] = np.real(chunk_left)
input_data[0, 1] = np.imag(chunk_left)
input_data[0, 2] = np.real(chunk_left)
input_data[0, 3] = np.imag(chunk_left)

output = model.predict({"input_1": input_data})
output_array = output["var_992"]

print(f"\n📊 モデル出力 shape: {output_array.shape}")

# magnitude と phase
magnitude = np.abs(chunk_left)
phase = np.angle(chunk_left)

results = []

# 組み合わせ1: Ch0単独
print(f"\n🧪 テスト1: Channel 0 単独")
mask = output_array[0, 0, :, :]
masked_magnitude = magnitude * mask
masked_stft = masked_magnitude * np.exp(1j * phase)
audio = librosa.istft(masked_stft, hop_length=hop_length, length=y.shape[1])
rms = np.sqrt(np.mean(audio**2))
print(f"   RMS: {rms:.6f}")
results.append(("ch0_solo", audio, rms))

# 組み合わせ2: Ch1単独
print(f"\n🧪 テスト2: Channel 1 単独")
mask = output_array[0, 1, :, :]
masked_magnitude = magnitude * mask
masked_stft = masked_magnitude * np.exp(1j * phase)
audio = librosa.istft(masked_stft, hop_length=hop_length, length=y.shape[1])
rms = np.sqrt(np.mean(audio**2))
print(f"   RMS: {rms:.6f}")
results.append(("ch1_solo", audio, rms))

# 組み合わせ3: Ch0+Ch1の和
print(f"\n🧪 テスト3: Channel 0 + Channel 1 の和")
mask = output_array[0, 0, :, :] + output_array[0, 1, :, :]
masked_magnitude = magnitude * mask
masked_stft = masked_magnitude * np.exp(1j * phase)
audio = librosa.istft(masked_stft, hop_length=hop_length, length=y.shape[1])
rms = np.sqrt(np.mean(audio**2))
print(f"   RMS: {rms:.6f}")
results.append(("ch0_plus_ch1", audio, rms))

# 組み合わせ4: Ch0-Ch1の差
print(f"\n🧪 テスト4: Channel 0 - Channel 1 の差")
mask = output_array[0, 0, :, :] - output_array[0, 1, :, :]
masked_magnitude = magnitude * mask
masked_stft = masked_magnitude * np.exp(1j * phase)
audio = librosa.istft(masked_stft, hop_length=hop_length, length=y.shape[1])
rms = np.sqrt(np.mean(audio**2))
print(f"   RMS: {rms:.6f}")
results.append(("ch0_minus_ch1", audio, rms))

# 組み合わせ5: Ch0/(Ch0+Ch1+eps) 比率マスク
print(f"\n🧪 テスト5: Ch0/(Ch0+Ch1+eps) 比率マスク")
ch0 = output_array[0, 0, :, :]
ch1 = output_array[0, 1, :, :]
mask = ch0 / (np.abs(ch0) + np.abs(ch1) + 1e-10)
masked_magnitude = magnitude * mask
masked_stft = masked_magnitude * np.exp(1j * phase)
audio = librosa.istft(masked_stft, hop_length=hop_length, length=y.shape[1])
rms = np.sqrt(np.mean(audio**2))
print(f"   RMS: {rms:.6f}")
results.append(("ratio_ch0", audio, rms))

# 組み合わせ6: sigmoid(Ch0)
print(f"\n🧪 テスト6: sigmoid(Ch0)")
mask = 1 / (1 + np.exp(-output_array[0, 0, :, :]))
masked_magnitude = magnitude * mask
masked_stft = masked_magnitude * np.exp(1j * phase)
audio = librosa.istft(masked_stft, hop_length=hop_length, length=y.shape[1])
rms = np.sqrt(np.mean(audio**2))
print(f"   RMS: {rms:.6f}")
results.append(("sigmoid_ch0", audio, rms))

# 保存
print(f"\n💾 保存中...")
for name, audio, rms in results:
    # RMS正規化
    target_rms = 0.092343  # 元音声RMS
    if rms > 1e-6:
        audio_normalized = audio * (target_rms / rms)
    else:
        audio_normalized = audio

    output_file = f"tests/python_output/combo_{name}.wav"
    sf.write(output_file, np.stack([audio_normalized, audio_normalized]).T, sr)
    print(f"   {output_file}")

print(f"\n✅ 完了")
print(f"\n各ファイルを聴いて、伴奏/ボーカルを確認してください")
print("=" * 80)
