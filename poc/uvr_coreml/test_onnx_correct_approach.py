#!/usr/bin/env python3
"""
UVRのソースコード解析に基づく正しいアプローチのテスト
発見: モデル出力はマスクではなく、分離されたスペクトログラムそのもの
"""
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"

print("=" * 80)
print("🔬 UVR正式アプローチに基づくテスト")
print("=" * 80)

# 音声読み込み
print(f"\n📂 音声読み込み: {input_file}")
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

print(f"   元音声shape: {y.shape}")
print(f"   元音声RMS: {np.sqrt(np.mean(y[0]**2)):.6f}")

# STFT (UVRと同じパラメータ)
n_fft = 4096
hop_length = 1024
stft_left = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length)
stft_right = librosa.stft(y[1], n_fft=n_fft, hop_length=hop_length)

print(f"\n📊 STFT情報:")
print(f"   shape: {stft_left.shape}")
print(f"   n_fft: {n_fft}, hop_length: {hop_length}")

# ONNXモデル読み込み
print(f"\n🤖 ONNXモデル読み込み")
session = ort.InferenceSession(model_path)

# 最初のチャンクでテスト
freq_bins = 2048
time_frames = 256

chunk_left = stft_left[:freq_bins, :time_frames]
chunk_right = stft_right[:freq_bins, :time_frames]

# 入力データ作成 (実数部・虚数部に分離)
input_data = np.zeros((1, 4, freq_bins, time_frames), dtype=np.float32)
input_data[0, 0] = np.real(chunk_left)
input_data[0, 1] = np.imag(chunk_left)
input_data[0, 2] = np.real(chunk_right)
input_data[0, 3] = np.imag(chunk_right)

# UVRの実装に基づく前処理: 最初の3周波数ビンをゼロにする
input_data[:, :, :3, :] = 0

print(f"\n🧠 推論実行")
print(f"   入力shape: {input_data.shape}")
print(f"   前処理: 最初の3周波数ビンをゼロ化")

# 推論
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: input_data})
output_array = output[0]

print(f"   出力shape: {output_array.shape}")

# UVRのアプローチ: モデル出力を複素数スペクトログラムとして再構成
print(f"\n🔧 UVRアプローチ適用:")
print(f"   モデル出力を分離されたスペクトログラムとして扱う")

# 左チャンネル: Ch0 (real) + 1j * Ch1 (imag)
separated_left = output_array[0, 0, :, :] + 1j * output_array[0, 1, :, :]

# 右チャンネル: Ch2 (real) + 1j * Ch3 (imag)
separated_right = output_array[0, 2, :, :] + 1j * output_array[0, 3, :, :]

print(f"   左チャンネル複素数STFT shape: {separated_left.shape}")
print(f"   右チャンネル複素数STFT shape: {separated_right.shape}")

# iSTFTで波形に変換
audio_left = librosa.istft(separated_left, hop_length=hop_length, length=y.shape[1])
audio_right = librosa.istft(separated_right, hop_length=hop_length, length=y.shape[1])

# ステレオ結合
audio_stereo = np.stack([audio_left, audio_right])

print(f"\n📊 結果統計:")
print(f"   左チャンネルRMS: {np.sqrt(np.mean(audio_left**2)):.6f}")
print(f"   右チャンネルRMS: {np.sqrt(np.mean(audio_right**2)):.6f}")
print(f"   左チャンネルMax: {np.max(np.abs(audio_left)):.6f}")
print(f"   右チャンネルMax: {np.max(np.abs(audio_right)):.6f}")

# 元音声との比較
original_rms = np.sqrt(np.mean(y[0]**2))
result_rms = np.sqrt(np.mean(audio_left**2))
print(f"\n📊 元音声との比較:")
print(f"   元音声RMS: {original_rms:.6f}")
print(f"   結果RMS: {result_rms:.6f}")
print(f"   比率: {result_rms/original_rms*100:.1f}%")

# 保存
output_path = "tests/python_output/onnx_uvr_approach.wav"
sf.write(output_path, audio_stereo.T, sr)

print(f"\n✅ 完了")
print(f"   出力: {output_path}")
print(f"\nこのファイルを聴いて、正しくインストゥルメンタルが抽出できているか確認してください。")
print("=" * 80)
