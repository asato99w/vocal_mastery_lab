#!/usr/bin/env python3
"""
比率マスク仮説の検証: モデル出力が分離比率を表現している可能性
"""
import numpy as np
import librosa
import soundfile as sf
import coremltools as ct

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/coreml/UVR-MDX-NET-Inst_Main.mlpackage"

print("=" * 80)
print("🔬 比率マスク仮説の検証")
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
print(f"   STFT magnitude range: {np.abs(stft_left).min():.6f} - {np.abs(stft_left).max():.6f}")

# モデル読み込み
print(f"\n🤖 モデル読み込み")
model = ct.models.MLModel(model_path)

# 最初のチャンクで推論
freq_bins = 2048
time_frames = 256

chunk_left = stft_left[:freq_bins, :time_frames]

# 入力データ作成
input_data = np.zeros((1, 4, freq_bins, time_frames), dtype=np.float32)
input_data[0, 0] = np.real(chunk_left)
input_data[0, 1] = np.imag(chunk_left)
input_data[0, 2] = np.real(chunk_left)
input_data[0, 3] = np.imag(chunk_left)

print(f"\n🧠 推論実行")
output = model.predict({"input_1": input_data})
output_array = output["var_992"]

print(f"   モデル出力shape: {output_array.shape}")

# マスクの和を確認
mask_ch0 = output_array[0, 0, :, :]
mask_ch1 = output_array[0, 1, :, :]
mask_sum = mask_ch0 + mask_ch1

print(f"\n📊 マスクの和を確認:")
print(f"   Channel 0 range: {mask_ch0.min():.6f} - {mask_ch0.max():.6f}, mean: {mask_ch0.mean():.6f}")
print(f"   Channel 1 range: {mask_ch1.min():.6f} - {mask_ch1.max():.6f}, mean: {mask_ch1.mean():.6f}")
print(f"   Sum range: {mask_sum.min():.6f} - {mask_sum.max():.6f}, mean: {mask_sum.mean():.6f}")
print(f"   Sum std: {mask_sum.std():.6f}")

# 仮説: Ch0とCh1の和で正規化して比率マスクにする
print(f"\n🧪 仮説: Ch0とCh1の和で正規化して比率マスクにする")
mask_inst_ratio = mask_ch0 / (mask_sum + 1e-10)
mask_vocal_ratio = mask_ch1 / (mask_sum + 1e-10)

print(f"   Inst ratio range: {mask_inst_ratio.min():.6f} - {mask_inst_ratio.max():.6f}")
print(f"   Vocal ratio range: {mask_vocal_ratio.min():.6f} - {mask_vocal_ratio.max():.6f}")
print(f"   Ratio sum: {(mask_inst_ratio + mask_vocal_ratio).mean():.6f} (should be ~1.0)")

# magnitude と phase を分離
magnitude = np.abs(chunk_left)
phase = np.angle(chunk_left)

# 比率マスク適用
inst_magnitude = magnitude * mask_inst_ratio
vocal_magnitude = magnitude * mask_vocal_ratio

# 複素数に戻す
inst_stft = inst_magnitude * np.exp(1j * phase)
vocal_stft = vocal_magnitude * np.exp(1j * phase)

# iSTFT
audio_inst = librosa.istft(inst_stft, hop_length=hop_length, length=y.shape[1])
audio_vocal = librosa.istft(vocal_stft, hop_length=hop_length, length=y.shape[1])

print(f"\n   Inst RMS: {np.sqrt(np.mean(audio_inst**2)):.6f}, Max: {np.max(np.abs(audio_inst)):.6f}")
print(f"   Vocal RMS: {np.sqrt(np.mean(audio_vocal**2)):.6f}, Max: {np.max(np.abs(audio_vocal)):.6f}")
print(f"   元音声 RMS: {np.sqrt(np.mean(y[0]**2)):.6f}")

# 保存
print(f"\n💾 結果を保存中...")
sf.write("tests/python_output/ratio_mask_inst.wav", np.stack([audio_inst, audio_inst]).T, sr)
sf.write("tests/python_output/ratio_mask_vocal.wav", np.stack([audio_vocal, audio_vocal]).T, sr)

# 検証: 和が元音声に戻るか
audio_sum = audio_inst + audio_vocal
correlation = np.corrcoef(y[0][:len(audio_sum)], audio_sum)[0, 1]

print(f"\n🔍 検証:")
print(f"   Inst + Vocal RMS: {np.sqrt(np.mean(audio_sum**2)):.6f}")
print(f"   元音声との相関: {correlation:.6f}")
if correlation > 0.95:
    print(f"   ✅ 元音声とほぼ一致！比率マスクが正しい可能性が高い")
elif correlation > 0.8:
    print(f"   ⚠️  高い相関あり。比率マスクが部分的に正しい可能性")
else:
    print(f"   ❌ 相関が低い。比率マスクは正しくない")

print(f"\n✅ 完了")
print(f"\n次のファイルを聴いて確認してください:")
print(f"  tests/python_output/ratio_mask_inst.wav - 比率マスクでの伴奏")
print(f"  tests/python_output/ratio_mask_vocal.wav - 比率マスクでのボーカル")

print("\n" + "=" * 80)
