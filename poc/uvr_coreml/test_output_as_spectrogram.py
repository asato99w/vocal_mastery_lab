#!/usr/bin/env python3
"""
新仮説: モデル出力がマスクではなくSTFT自体を表現している可能性を検証
"""
import numpy as np
import librosa
import soundfile as sf
import coremltools as ct

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/coreml/UVR-MDX-NET-Inst_Main.mlpackage"

print("=" * 80)
print("🔬 新仮説: モデル出力がSTFTそのものか検証")
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
print(f"   STFT shape: {stft_left.shape}")
print(f"   STFT magnitude range: {np.abs(stft_left).min():.6f} - {np.abs(stft_left).max():.6f}")

# モデル読み込み
print(f"\n🤖 モデル読み込み")
model = ct.models.MLModel(model_path)

# 最初のチャンクで推論
freq_bins = 2048
time_frames = 256

chunk_left = stft_left[:freq_bins, :time_frames]

# 入力データ作成 [1, 4, 2048, 256]
input_data = np.zeros((1, 4, freq_bins, time_frames), dtype=np.float32)
input_data[0, 0] = np.real(chunk_left)
input_data[0, 1] = np.imag(chunk_left)
input_data[0, 2] = np.real(chunk_left)
input_data[0, 3] = np.imag(chunk_left)

print(f"\n🧠 推論実行")
output = model.predict({"input_1": input_data})
output_array = output["var_992"]

print(f"   モデル出力shape: {output_array.shape}")

# 仮説1: Channel 0 + 1j * Channel 1 = Instrumental STFT
print(f"\n\n🧪 仮説1: Ch0=Inst Real, Ch1=Inst Imag として複素数STFT構築")
inst_real = output_array[0, 0, :, :]
inst_imag = output_array[0, 1, :, :]
inst_stft = inst_real + 1j * inst_imag

print(f"   Instrumental STFT magnitude range: {np.abs(inst_stft).min():.6f} - {np.abs(inst_stft).max():.6f}")
print(f"   Instrumental STFT mean magnitude: {np.abs(inst_stft).mean():.6f}")

audio_inst = librosa.istft(inst_stft, hop_length=hop_length, length=y.shape[1])
print(f"   結果 RMS: {np.sqrt(np.mean(audio_inst**2)):.6f}, Max: {np.max(np.abs(audio_inst)):.6f}")

# 仮説2: Channel 2 + 1j * Channel 3 = Vocal STFT
print(f"\n🧪 仮説2: Ch2=Vocal Real, Ch3=Vocal Imag として複素数STFT構築")
vocal_real = output_array[0, 2, :, :]
vocal_imag = output_array[0, 3, :, :]
vocal_stft = vocal_real + 1j * vocal_imag

print(f"   Vocal STFT magnitude range: {np.abs(vocal_stft).min():.6f} - {np.abs(vocal_stft).max():.6f}")
print(f"   Vocal STFT mean magnitude: {np.abs(vocal_stft).mean():.6f}")

audio_vocal = librosa.istft(vocal_stft, hop_length=hop_length, length=y.shape[1])
print(f"   結果 RMS: {np.sqrt(np.mean(audio_vocal**2)):.6f}, Max: {np.max(np.abs(audio_vocal)):.6f}")

# 仮説3: 逆のチャンネル配置
print(f"\n🧪 仮説3: Ch0=Vocal Real, Ch1=Vocal Imag, Ch2=Inst Real, Ch3=Inst Imag")
vocal_stft_alt = output_array[0, 0, :, :] + 1j * output_array[0, 1, :, :]
inst_stft_alt = output_array[0, 2, :, :] + 1j * output_array[0, 3, :, :]

audio_vocal_alt = librosa.istft(vocal_stft_alt, hop_length=hop_length, length=y.shape[1])
audio_inst_alt = librosa.istft(inst_stft_alt, hop_length=hop_length, length=y.shape[1])

print(f"   Vocal RMS: {np.sqrt(np.mean(audio_vocal_alt**2)):.6f}, Max: {np.max(np.abs(audio_vocal_alt)):.6f}")
print(f"   Inst RMS: {np.sqrt(np.mean(audio_inst_alt**2)):.6f}, Max: {np.max(np.abs(audio_inst_alt)):.6f}")

# 入力と比較
print(f"\n📊 元音声との比較:")
print(f"   元音声: RMS={np.sqrt(np.mean(y[0]**2)):.6f}")
print(f"   仮説1 Inst: RMS={np.sqrt(np.mean(audio_inst**2)):.6f} ({np.sqrt(np.mean(audio_inst**2))/np.sqrt(np.mean(y[0]**2))*100:.1f}%)")
print(f"   仮説2 Vocal: RMS={np.sqrt(np.mean(audio_vocal**2)):.6f} ({np.sqrt(np.mean(audio_vocal**2))/np.sqrt(np.mean(y[0]**2))*100:.1f}%)")
print(f"   仮説3 Vocal: RMS={np.sqrt(np.mean(audio_vocal_alt**2)):.6f} ({np.sqrt(np.mean(audio_vocal_alt**2))/np.sqrt(np.mean(y[0]**2))*100:.1f}%)")
print(f"   仮説3 Inst: RMS={np.sqrt(np.mean(audio_inst_alt**2)):.6f} ({np.sqrt(np.mean(audio_inst_alt**2))/np.sqrt(np.mean(y[0]**2))*100:.1f}%)")

# 保存
print(f"\n💾 結果を保存中...")
sf.write("tests/python_output/hypothesis_1_inst.wav", np.stack([audio_inst, audio_inst]).T, sr)
sf.write("tests/python_output/hypothesis_2_vocal.wav", np.stack([audio_vocal, audio_vocal]).T, sr)
sf.write("tests/python_output/hypothesis_3_vocal.wav", np.stack([audio_vocal_alt, audio_vocal_alt]).T, sr)
sf.write("tests/python_output/hypothesis_3_inst.wav", np.stack([audio_inst_alt, audio_inst_alt]).T, sr)

print(f"\n✅ 完了")
print(f"\n次のファイルを聴いて確認してください:")
print(f"  仮説1: tests/python_output/hypothesis_1_inst.wav - Ch0+1j*Ch1 = Inst")
print(f"  仮説2: tests/python_output/hypothesis_2_vocal.wav - Ch2+1j*Ch3 = Vocal")
print(f"  仮説3: tests/python_output/hypothesis_3_vocal.wav - Ch0+1j*Ch1 = Vocal")
print(f"         tests/python_output/hypothesis_3_inst.wav - Ch2+1j*Ch3 = Inst")

print("\n" + "=" * 80)
