#!/usr/bin/env python3
"""
正規化を修正したUVRアプローチのテスト
Librosa STFT/iSTFTのスケーリング問題に対処
"""
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"

print("=" * 80)
print("🔬 正規化修正版UVRアプローチテスト")
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
window = 'hann'

# STFT (window引数を明示的に指定)
stft_left = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length, window=window)
stft_right = librosa.stft(y[1], n_fft=n_fft, hop_length=hop_length, window=window)

# STFTのスケールを確認
print(f"\n📊 STFT統計:")
print(f"   左チャンネル magnitude mean: {np.mean(np.abs(stft_left)):.6f}")
print(f"   左チャンネル magnitude max: {np.max(np.abs(stft_left)):.6f}")

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

# UVRの前処理: 最初の3周波数ビンをゼロにする
input_data[:, :, :3, :] = 0

print(f"\n🧠 推論実行")
print(f"   入力 real mean: {np.mean(input_data[0, 0]):.6f}")
print(f"   入力 real max: {np.max(np.abs(input_data[0, 0])):.6f}")

# 推論
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: input_data})
output_array = output[0]

print(f"\n📊 モデル出力統計:")
print(f"   出力 Ch0 mean: {np.mean(output_array[0, 0]):.6f}")
print(f"   出力 Ch0 max: {np.max(np.abs(output_array[0, 0])):.6f}")

# 複素数STFTとして再構成
separated_left = output_array[0, 0, :, :] + 1j * output_array[0, 1, :, :]
separated_right = output_array[0, 2, :, :] + 1j * output_array[0, 3, :, :]

print(f"\n📊 複素数STFT統計:")
print(f"   分離済み magnitude mean: {np.mean(np.abs(separated_left)):.6f}")
print(f"   分離済み magnitude max: {np.max(np.abs(separated_left)):.6f}")

# iSTFT (centerパラメータを確認)
audio_left = librosa.istft(separated_left, hop_length=hop_length, window=window, length=y.shape[1])
audio_right = librosa.istft(separated_right, hop_length=hop_length, window=window, length=y.shape[1])

# ステレオ結合
audio_stereo = np.stack([audio_left, audio_right])

print(f"\n📊 結果統計 (iSTFT直後):")
print(f"   左チャンネルRMS: {np.sqrt(np.mean(audio_left**2)):.6f}")
print(f"   左チャンネルMax: {np.max(np.abs(audio_left)):.6f}")

# 元音声との比較
original_rms = np.sqrt(np.mean(y[0]**2))
result_rms = np.sqrt(np.mean(audio_left**2))
scale_factor = result_rms / original_rms

print(f"\n📊 スケール分析:")
print(f"   元音声RMS: {original_rms:.6f}")
print(f"   結果RMS: {result_rms:.6f}")
print(f"   スケール係数: {scale_factor:.2f}x")

# モデルの出力スケールを元の音声レベルに正規化
if result_rms > 1e-6:
    # 単純に元の音声レベルに正規化
    audio_stereo_normalized = audio_stereo * (original_rms / result_rms)

    print(f"\n🔧 正規化適用:")
    print(f"   正規化後RMS: {np.sqrt(np.mean(audio_stereo_normalized[0]**2)):.6f}")
    print(f"   正規化後Max: {np.max(np.abs(audio_stereo_normalized[0])):.6f}")

    # 保存
    output_path = "tests/python_output/onnx_normalized.wav"
    sf.write(output_path, audio_stereo_normalized.T, sr)

    print(f"\n✅ 完了")
    print(f"   出力: {output_path}")
    print(f"\nこのファイルを聴いて確認してください。")
else:
    print(f"\n⚠️ 出力が無音です")

print("=" * 80)
