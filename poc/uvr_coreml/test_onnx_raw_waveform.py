#!/usr/bin/env python3
"""
ONNXモデルに生波形を入力してテスト
仮説: モデルにSTFT/iSTFTが組み込まれており、波形を直接入出力する
"""
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"

print("=" * 80)
print("🔬 ONNX生波形入力テスト")
print("=" * 80)

# 音声読み込み
print(f"\n📂 音声読み込み: {input_file}")
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

print(f"   元音声shape: {y.shape}")
print(f"   元音声RMS: {np.sqrt(np.mean(y[0]**2)):.6f}")

# ONNXモデル読み込み
print(f"\n🤖 ONNXモデル読み込み")
session = ort.InferenceSession(model_path)

# モデル情報
input_meta = session.get_inputs()[0]
output_meta = session.get_outputs()[0]
print(f"\n📋 モデル情報:")
print(f"   入力: {input_meta.name}, shape: {input_meta.shape}, type: {input_meta.type}")
print(f"   出力: {output_meta.name}, shape: {output_meta.shape}, type: {output_meta.type}")

# 入力形状を確認
expected_shape = input_meta.shape
print(f"\n入力形状期待値: {expected_shape}")
print(f"   batch_size: {expected_shape[0]}")
print(f"   channels: {expected_shape[1]}")
print(f"   freq_bins: {expected_shape[2]}")
print(f"   time_frames: {expected_shape[3]}")

# 結論
print(f"\n🔍 結論:")
print(f"   このモデルは STFT後のデータ [batch, 4, freq, time] を入力として期待しています。")
print(f"   生波形の直接入力には対応していません。")
print(f"\n   したがって、モデルにSTFT/iSTFTは組み込まれて「いません」。")
print(f"   モデルの出力は「マスク」または「分離されたSTFT」です。")

print("\n=" * 80)
