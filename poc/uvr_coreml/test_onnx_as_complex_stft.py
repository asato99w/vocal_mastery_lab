#!/usr/bin/env python3
"""
ONNXモデル出力を複素数STFTとして解釈するテスト
仮説: 4チャンネル = [Inst_Real, Inst_Imag, Vocal_Real, Vocal_Imag]
"""
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"

print("=" * 80)
print("🔬 ONNX出力を複素数STFTとして解釈")
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

# ONNXモデル読み込み
print(f"\n🤖 ONNXモデル読み込み")
session = ort.InferenceSession(model_path)

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

# 推論
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: input_data})
output_array = output[0]

print(f"\n📊 出力統計:")
for ch in range(4):
    ch_data = output_array[0, ch]
    print(f"   Ch{ch}: min={ch_data.min():.6f}, max={ch_data.max():.6f}, mean={ch_data.mean():.6f}")

# 仮説1: Ch0+1j*Ch1 = Inst, Ch2+1j*Ch3 = Vocal
print(f"\n🧪 仮説1: Ch0+1j*Ch1 = Inst, Ch2+1j*Ch3 = Vocal")
inst_stft_h1 = output_array[0, 0, :, :] + 1j * output_array[0, 1, :, :]
vocal_stft_h1 = output_array[0, 2, :, :] + 1j * output_array[0, 3, :, :]

audio_inst_h1 = librosa.istft(inst_stft_h1, hop_length=hop_length, length=y.shape[1])
audio_vocal_h1 = librosa.istft(vocal_stft_h1, hop_length=hop_length, length=y.shape[1])

rms_inst_h1 = np.sqrt(np.mean(audio_inst_h1**2))
rms_vocal_h1 = np.sqrt(np.mean(audio_vocal_h1**2))
print(f"   Inst RMS: {rms_inst_h1:.6f}, Max: {np.max(np.abs(audio_inst_h1)):.6f}")
print(f"   Vocal RMS: {rms_vocal_h1:.6f}, Max: {np.max(np.abs(audio_vocal_h1)):.6f}")

# 仮説2: Ch0+1j*Ch1 = Vocal, Ch2+1j*Ch3 = Inst
print(f"\n🧪 仮説2: Ch0+1j*Ch1 = Vocal, Ch2+1j*Ch3 = Inst")
vocal_stft_h2 = output_array[0, 0, :, :] + 1j * output_array[0, 1, :, :]
inst_stft_h2 = output_array[0, 2, :, :] + 1j * output_array[0, 3, :, :]

audio_vocal_h2 = librosa.istft(vocal_stft_h2, hop_length=hop_length, length=y.shape[1])
audio_inst_h2 = librosa.istft(inst_stft_h2, hop_length=hop_length, length=y.shape[1])

rms_vocal_h2 = np.sqrt(np.mean(audio_vocal_h2**2))
rms_inst_h2 = np.sqrt(np.mean(audio_inst_h2**2))
print(f"   Vocal RMS: {rms_vocal_h2:.6f}, Max: {np.max(np.abs(audio_vocal_h2)):.6f}")
print(f"   Inst RMS: {rms_inst_h2:.6f}, Max: {np.max(np.abs(audio_inst_h2)):.6f}")

# 元音声と比較
print(f"\n📊 元音声との比較:")
original_rms = np.sqrt(np.mean(y[0]**2))
print(f"   元音声: RMS={original_rms:.6f}")
print(f"\n   仮説1:")
print(f"     Inst: {rms_inst_h1:.6f} ({rms_inst_h1/original_rms*100:.1f}%)")
print(f"     Vocal: {rms_vocal_h1:.6f} ({rms_vocal_h1/original_rms*100:.1f}%)")
print(f"     合計: {np.sqrt(rms_inst_h1**2 + rms_vocal_h1**2):.6f} ({np.sqrt(rms_inst_h1**2 + rms_vocal_h1**2)/original_rms*100:.1f}%)")
print(f"\n   仮説2:")
print(f"     Vocal: {rms_vocal_h2:.6f} ({rms_vocal_h2/original_rms*100:.1f}%)")
print(f"     Inst: {rms_inst_h2:.6f} ({rms_inst_h2/original_rms*100:.1f}%)")
print(f"     合計: {np.sqrt(rms_vocal_h2**2 + rms_inst_h2**2):.6f} ({np.sqrt(rms_vocal_h2**2 + rms_inst_h2**2)/original_rms*100:.1f}%)")

# 保存
print(f"\n💾 結果を保存中...")
sf.write("tests/python_output/onnx_h1_inst.wav", np.stack([audio_inst_h1, audio_inst_h1]).T, sr)
sf.write("tests/python_output/onnx_h1_vocal.wav", np.stack([audio_vocal_h1, audio_vocal_h1]).T, sr)
sf.write("tests/python_output/onnx_h2_vocal.wav", np.stack([audio_vocal_h2, audio_vocal_h2]).T, sr)
sf.write("tests/python_output/onnx_h2_inst.wav", np.stack([audio_inst_h2, audio_inst_h2]).T, sr)

print(f"\n✅ 完了")
print(f"\n次のファイルを聴いて確認してください:")
print(f"  仮説1: onnx_h1_inst.wav (Ch0+1j*Ch1) と onnx_h1_vocal.wav (Ch2+1j*Ch3)")
print(f"  仮説2: onnx_h2_vocal.wav (Ch0+1j*Ch1) と onnx_h2_inst.wav (Ch2+1j*Ch3)")
print("=" * 80)
