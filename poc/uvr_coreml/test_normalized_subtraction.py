#!/usr/bin/env python3
"""
正規化した引き算アプローチ: マスクを適切なレベルに正規化
"""
import numpy as np
import librosa
import soundfile as sf
import coremltools as ct

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/coreml/UVR-MDX-NET-Inst_Main.mlpackage"

print("=" * 80)
print("🔬 正規化した引き算アプローチ")
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
print(f"\n🤖 モデル読み込み")
model = ct.models.MLModel(model_path)

# 全体処理
freq_bins = 2048
time_frames = 256
num_chunks = (stft_left.shape[1] + time_frames - 1) // time_frames

print(f"   チャンク数: {num_chunks}")

inst_masks = []

for chunk_idx in range(num_chunks):
    start_t = chunk_idx * time_frames
    end_t = min((chunk_idx + 1) * time_frames, stft_left.shape[1])
    actual_size = end_t - start_t

    chunk_left = stft_left[:freq_bins, start_t:end_t]

    # パディング
    if chunk_left.shape[1] < time_frames:
        pad_width = time_frames - chunk_left.shape[1]
        chunk_left = np.pad(chunk_left, ((0, 0), (0, pad_width)), mode='constant')

    # 入力データ作成
    input_data = np.zeros((1, 4, freq_bins, time_frames), dtype=np.float32)
    input_data[0, 0] = np.real(chunk_left)
    input_data[0, 1] = np.imag(chunk_left)
    input_data[0, 2] = np.real(chunk_left)
    input_data[0, 3] = np.imag(chunk_left)

    # 推論
    output = model.predict({"input_1": input_data})
    output_array = output["var_992"]

    # Channel 0をInstrumental maskとして使用
    inst_chunk = output_array[0, 0, :, :actual_size]
    inst_masks.append(inst_chunk.T)

    if (chunk_idx + 1) % 10 == 0:
        print(f"   進捗: {chunk_idx + 1}/{num_chunks}")

# マスク結合
inst_mask = np.vstack(inst_masks).T

print(f"\n📊 マスク統計:")
print(f"   Shape: {inst_mask.shape}")
print(f"   Range: {inst_mask.min():.6f} - {inst_mask.max():.6f}")
print(f"   Mean: {inst_mask.mean():.6f}")

# magnitude と phase
magnitude = np.abs(stft_left[:freq_bins, :inst_mask.shape[1]])
phase = np.angle(stft_left[:freq_bins, :inst_mask.shape[1]])

# マスクを0-1範囲に正規化
inst_mask_normalized = np.clip(inst_mask / inst_mask.max(), 0, 1)

print(f"\n🔄 正規化後のマスク統計:")
print(f"   Range: {inst_mask_normalized.min():.6f} - {inst_mask_normalized.max():.6f}")
print(f"   Mean: {inst_mask_normalized.mean():.6f}")

# マスク適用
inst_magnitude = magnitude * inst_mask_normalized
inst_stft = inst_magnitude * np.exp(1j * phase)

# iSTFT
print(f"\n🔄 iSTFT実行")
audio_inst = librosa.istft(inst_stft, hop_length=hop_length, length=y.shape[1])

print(f"\n🎸 Instrumental出力 (正規化版):")
print(f"   RMS: {np.sqrt(np.mean(audio_inst**2)):.6f}")
print(f"   Max: {np.max(np.abs(audio_inst)):.6f}")

# ボーカル = 元音声 - Instrumental
audio_vocal = y[0] - audio_inst

print(f"\n🎤 Vocal出力 (元音声 - Instrumental):")
print(f"   RMS: {np.sqrt(np.mean(audio_vocal**2)):.6f}")
print(f"   Max: {np.max(np.abs(audio_vocal)):.6f}")

# 保存
print(f"\n💾 結果を保存中...")
sf.write("tests/python_output/normalized_inst.wav", np.stack([audio_inst, audio_inst]).T, sr)
sf.write("tests/python_output/normalized_vocal.wav", np.stack([audio_vocal, audio_vocal]).T, sr)

# 検証
print(f"\n🔍 検証:")
print(f"   元音声 RMS: {np.sqrt(np.mean(y[0]**2)):.6f}")
print(f"   Inst / 元音声: {np.sqrt(np.mean(audio_inst**2)) / np.sqrt(np.mean(y[0]**2)) * 100:.1f}%")
print(f"   Vocal / 元音声: {np.sqrt(np.mean(audio_vocal**2)) / np.sqrt(np.mean(y[0]**2)) * 100:.1f}%")

# Inst + Vocal との相関
audio_sum = audio_inst + audio_vocal
correlation = np.corrcoef(y[0], audio_sum)[0, 1]
print(f"   Inst + Vocal と元音声の相関: {correlation:.6f}")

if correlation > 0.99:
    print(f"   ✅ ほぼ完全に元音声に戻る")
elif correlation > 0.95:
    print(f"   ⚠️  高い相関あり")
else:
    print(f"   ❌ 相関が低い")

print(f"\n✅ 完了")
print(f"\n次のファイルを聴いて確認してください:")
print(f"  tests/python_output/normalized_inst.wav - 伴奏（正規化マスク）")
print(f"  tests/python_output/normalized_vocal.wav - ボーカル（元音声 - 伴奏）")

print("\n" + "=" * 80)
