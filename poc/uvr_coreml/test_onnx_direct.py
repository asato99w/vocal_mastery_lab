#!/usr/bin/env python3
"""
ONNXモデルを直接実行して正しい動作を確認
"""
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"

print("=" * 80)
print("🔬 ONNX モデル直接実行テスト")
print("=" * 80)

# 音声読み込み
print(f"\n📂 音声読み込み: {input_file}")
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

print(f"   元音声RMS: {np.sqrt(np.mean(y[0]**2)):.6f}")

# STFT
n_fft = 4096
hop_length = 1024
stft_left = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length)
stft_right = librosa.stft(y[1], n_fft=n_fft, hop_length=hop_length)

print(f"   STFT shape: {stft_left.shape}")

# ONNXモデル読み込み
print(f"\n🤖 ONNXモデル読み込み: {model_path}")
session = ort.InferenceSession(model_path)

# モデル情報確認
print(f"\n📋 モデル情報:")
for input_meta in session.get_inputs():
    print(f"   入力: {input_meta.name}, shape: {input_meta.shape}, type: {input_meta.type}")
for output_meta in session.get_outputs():
    print(f"   出力: {output_meta.name}, shape: {output_meta.shape}, type: {output_meta.type}")

# 最初のチャンクでテスト
freq_bins = 2048
time_frames = 256

chunk_left = stft_left[:freq_bins, :time_frames]
chunk_right = stft_right[:freq_bins, :time_frames]

# 入力データ作成
input_data = np.zeros((1, 4, freq_bins, time_frames), dtype=np.float32)
input_data[0, 0] = np.real(chunk_left)
input_data[0, 1] = np.imag(chunk_left)
input_data[0, 2] = np.real(chunk_right)
input_data[0, 3] = np.imag(chunk_right)

print(f"\n🧠 推論実行")
print(f"   入力 shape: {input_data.shape}")

# 推論
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: input_data})
output_array = output[0]

print(f"   出力 shape: {output_array.shape}")

# 各チャンネルの統計
print(f"\n📊 出力統計:")
for ch in range(output_array.shape[1]):
    ch_data = output_array[0, ch]
    print(f"   Channel {ch}: min={ch_data.min():.6f}, max={ch_data.max():.6f}, mean={ch_data.mean():.6f}")

# Channel 0をマスクとして使用
print(f"\n🧪 テスト: Channel 0をマスクとして使用")
magnitude = np.abs(chunk_left)
phase = np.angle(chunk_left)

mask = output_array[0, 0, :, :]
masked_magnitude = magnitude * mask
masked_stft = masked_magnitude * np.exp(1j * phase)

audio = librosa.istft(masked_stft, hop_length=hop_length, length=y.shape[1])
rms = np.sqrt(np.mean(audio**2))

print(f"   結果 RMS: {rms:.6f}")

# レベル調整して保存
target_rms = np.sqrt(np.mean(y[0]**2))
if rms > 1e-6:
    audio_normalized = audio * (target_rms / rms)
else:
    audio_normalized = audio

sf.write("tests/python_output/onnx_direct_ch0.wav", np.stack([audio_normalized, audio_normalized]).T, sr)

print(f"\n✅ 完了")
print(f"   tests/python_output/onnx_direct_ch0.wav を確認してください")
print("=" * 80)
