#!/usr/bin/env python3
"""
レベル調整版: マスクをそのまま使い、出力レベルだけを調整
"""
import numpy as np
import librosa
import soundfile as sf
import coremltools as ct

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/coreml/UVR-MDX-NET-Inst_Main.mlpackage"

print("=" * 80)
print("🔬 レベル調整版テスト")
print("=" * 80)

# 音声読み込み
print(f"\n📂 音声読み込み: {input_file}")
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

# STFT
n_fft = 4096
hop_length = 1024
stft_left = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length)

print(f"   元音声RMS: {np.sqrt(np.mean(y[0]**2)):.6f}")

# モデル読み込み
model = ct.models.MLModel(model_path)

# 全体処理
freq_bins = 2048
time_frames = 256
num_chunks = (stft_left.shape[1] + time_frames - 1) // time_frames

inst_masks = []

for chunk_idx in range(num_chunks):
    start_t = chunk_idx * time_frames
    end_t = min((chunk_idx + 1) * time_frames, stft_left.shape[1])
    actual_size = end_t - start_t

    chunk_left = stft_left[:freq_bins, start_t:end_t]

    if chunk_left.shape[1] < time_frames:
        pad_width = time_frames - chunk_left.shape[1]
        chunk_left = np.pad(chunk_left, ((0, 0), (0, pad_width)), mode='constant')

    input_data = np.zeros((1, 4, freq_bins, time_frames), dtype=np.float32)
    input_data[0, 0] = np.real(chunk_left)
    input_data[0, 1] = np.imag(chunk_left)
    input_data[0, 2] = np.real(chunk_left)
    input_data[0, 3] = np.imag(chunk_left)

    output = model.predict({"input_1": input_data})
    output_array = output["var_992"]

    inst_chunk = output_array[0, 0, :, :actual_size]
    inst_masks.append(inst_chunk.T)

    if (chunk_idx + 1) % 10 == 0:
        print(f"   進捗: {chunk_idx + 1}/{num_chunks}")

inst_mask = np.vstack(inst_masks).T

print(f"\n📊 マスク統計:")
print(f"   Range: {inst_mask.min():.6f} - {inst_mask.max():.6f}")
print(f"   Mean: {inst_mask.mean():.6f}")

# magnitude と phase
magnitude = np.abs(stft_left[:freq_bins, :inst_mask.shape[1]])
phase = np.angle(stft_left[:freq_bins, :inst_mask.shape[1]])

# マスク適用（そのまま）
inst_magnitude = magnitude * inst_mask
inst_stft = inst_magnitude * np.exp(1j * phase)

# iSTFT
audio_inst_raw = librosa.istft(inst_stft, hop_length=hop_length, length=y.shape[1])

# レベル調整係数を計算（RMSベース）
original_rms = np.sqrt(np.mean(y[0]**2))
inst_raw_rms = np.sqrt(np.mean(audio_inst_raw**2))
level_correction = original_rms / inst_raw_rms

print(f"\n🔧 レベル調整:")
print(f"   元音声 RMS: {original_rms:.6f}")
print(f"   Inst未調整 RMS: {inst_raw_rms:.6f}")
print(f"   調整係数: {level_correction:.6f} ({1/level_correction:.2f}倍に縮小)")

# レベル調整適用
audio_inst = audio_inst_raw * level_correction

print(f"\n🎸 Instrumental出力 (レベル調整後):")
print(f"   RMS: {np.sqrt(np.mean(audio_inst**2)):.6f}")
print(f"   Max: {np.max(np.abs(audio_inst)):.6f}")

# ボーカル = 元音声 - Instrumental
audio_vocal = y[0] - audio_inst

print(f"\n🎤 Vocal出力 (元音声 - Instrumental):")
print(f"   RMS: {np.sqrt(np.mean(audio_vocal**2)):.6f}")
print(f"   Max: {np.max(np.abs(audio_vocal)):.6f}")

# 保存
print(f"\n💾 結果を保存中...")
sf.write("tests/python_output/level_adjusted_inst.wav", np.stack([audio_inst, audio_inst]).T, sr)
sf.write("tests/python_output/level_adjusted_vocal.wav", np.stack([audio_vocal, audio_vocal]).T, sr)

# 検証
audio_sum = audio_inst + audio_vocal
correlation = np.corrcoef(y[0], audio_sum)[0, 1]
print(f"\n🔍 検証:")
print(f"   Inst + Vocal と元音声の相関: {correlation:.6f}")

print(f"\n✅ 完了")
print(f"\n次のファイルを聴いて確認してください:")
print(f"  tests/python_output/level_adjusted_inst.wav - 伴奏（RMSレベル調整）")
print(f"  tests/python_output/level_adjusted_vocal.wav - ボーカル（元音声 - 伴奏）")

print("\n" + "=" * 80)
