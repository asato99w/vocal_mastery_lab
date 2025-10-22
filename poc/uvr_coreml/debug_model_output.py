#!/usr/bin/env python3
"""
ONNX/CoreMLモデルの出力を比較するため、入力と出力をダンプ
"""
import numpy as np
import librosa
import onnxruntime as ort

print("=" * 80)
print("🔍 ONNX Model Output Debug")
print("=" * 80)

# 簡単なテスト信号でモデル出力を確認
sr = 44100
duration = 1.0
t = np.linspace(0, duration, int(sr * duration))
test_signal = np.sin(2 * np.pi * 440 * t)

print(f"\n📊 テスト信号 (440Hz, 1秒):")
print(f"  Max: {np.abs(test_signal).max():.6f}")
print(f"  RMS: {np.sqrt(np.mean(test_signal**2)):.6f}")

# ステレオに変換
test_stereo = np.stack([test_signal, test_signal])

# STFT
n_fft = 4096
hop_length = 1024

print(f"\n🔄 STFT実行:")
left_stft = librosa.stft(test_stereo[0], n_fft=n_fft, hop_length=hop_length)
print(f"  STFT形状: {left_stft.shape}")

# 実部・虚部に分解
real_part = np.real(left_stft).astype(np.float32)
imag_part = np.imag(left_stft).astype(np.float32)

print(f"  Real range: {real_part.min():.6f} ~ {real_part.max():.6f}")
print(f"  Imag range: {imag_part.min():.6f} ~ {imag_part.max():.6f}")

# 2048ビンに制限 (モデル要求)
real_part = real_part[:2048, :]
imag_part = imag_part[:2048, :]

print(f"\n📊 モデル入力準備 (2048 bins):")
print(f"  Real shape: {real_part.shape}")
print(f"  Imag shape: {imag_part.shape}")

# 最初のチャンクを抽出 (256フレーム)
chunk_size = 256
if real_part.shape[1] < chunk_size:
    pad_width = chunk_size - real_part.shape[1]
    real_chunk = np.pad(real_part, ((0, 0), (0, pad_width)))
    imag_chunk = np.pad(imag_part, ((0, 0), (0, pad_width)))
else:
    real_chunk = real_part[:, :chunk_size]
    imag_chunk = imag_part[:, :chunk_size]

print(f"\n📦 チャンク (256フレーム):")
print(f"  Real chunk shape: {real_chunk.shape}")
print(f"  Imag chunk shape: {imag_chunk.shape}")

# モデル入力形式に変換 [1, 4, 2048, 256]
# チャンネル順: [Left Real, Left Imag, Right Real, Right Imag]
input_data = np.stack([
    real_chunk,  # Left Real (ch 0)
    imag_chunk,  # Left Imag (ch 1)
    real_chunk,  # Right Real (ch 2) - 同じデータ
    imag_chunk   # Right Imag (ch 3) - 同じデータ
], axis=0)

input_data = np.expand_dims(input_data, axis=0)

print(f"\n📥 ONNX入力:")
print(f"  形状: {input_data.shape}")
print(f"  データ型: {input_data.dtype}")
print(f"  値の範囲: {input_data.min():.6f} ~ {input_data.max():.6f}")

# ONNX推論
model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"
session = ort.InferenceSession(str(model_path))
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"\n🤖 ONNX推論実行:")
print(f"  Input name: {input_name}")
print(f"  Output name: {output_name}")

outputs = session.run([output_name], {input_name: input_data})
output_array = outputs[0]

print(f"\n📤 ONNX出力:")
print(f"  形状: {output_array.shape}")
print(f"  データ型: {output_array.dtype}")
print(f"  値の範囲: {output_array.min():.6f} ~ {output_array.max():.6f}")

# チャンネル別統計
print(f"\n📊 チャンネル別出力統計:")
for ch in range(4):
    ch_data = output_array[0, ch, :, :]
    print(f"  Ch{ch}: min={ch_data.min():.6f}, max={ch_data.max():.6f}, mean={ch_data.mean():.6f}")

# 440Hz binの出力を確認
freq_440_bin = int(440 * n_fft / sr)
print(f"\n🎵 440Hz bin [{freq_440_bin}] の出力 (最初の3フレーム):")
for frame_idx in range(3):
    print(f"  Frame {frame_idx}:")
    for ch in range(4):
        val = output_array[0, ch, freq_440_bin, frame_idx]
        print(f"    Ch{ch}: {val:.6f}")

# デバッグデータ保存
np.save('tests/python_output/model_input.npy', input_data)
np.save('tests/python_output/model_output.npy', output_array)

# テキスト形式でも保存 (最初のフレーム、ch0のみ)
np.savetxt('tests/python_output/model_input_ch0_frame0.txt', input_data[0, 0, :, 0])
np.savetxt('tests/python_output/model_output_ch0_frame0.txt', output_array[0, 0, :, 0])

print(f"\n💾 デバッグデータ保存完了:")
print(f"  tests/python_output/model_input.npy")
print(f"  tests/python_output/model_output.npy")
print(f"  tests/python_output/model_input_ch0_frame0.txt")
print(f"  tests/python_output/model_output_ch0_frame0.txt")

print("\n" + "=" * 80)
