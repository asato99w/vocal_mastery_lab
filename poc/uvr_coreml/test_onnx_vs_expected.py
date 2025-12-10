#!/usr/bin/env python3
"""
ONNX出力レベル診断テスト
問題: インストゥルメンタルRMSが0.001567（期待値の1/59）
目的: モデル出力の生値を確認し、スケーリング問題を特定
"""
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
import librosa

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"

print("=" * 80)
print("🔍 ONNX出力レベル診断")
print("=" * 80)

# 音声読み込み
print(f"\n📂 音声読み込み: {input_file}")
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

original_audio = y.copy()
print(f"   元音声shape: {y.shape}")
print(f"   元音声RMS: {np.sqrt(np.mean(y[0]**2)):.6f}")

# PyTorch STFT
mix = torch.from_numpy(y).float()
n_fft = 4096
hop_length = 1024
window = torch.hann_window(window_length=n_fft, periodic=True)

print(f"\n🔧 STFT実行")
stft_result = torch.stft(
    mix.reshape([-1, mix.shape[-1]]),
    n_fft=n_fft,
    hop_length=hop_length,
    window=window,
    center=True,
    return_complex=False
)

# UVRフォーマット変換
stft_result = stft_result.permute([0, 3, 1, 2])
c = 2
stft_result = stft_result.reshape([1, c, 2, -1, stft_result.shape[-1]])
stft_result = stft_result.reshape([1, c * 2, -1, stft_result.shape[-1]])

print(f"   STFT shape: {stft_result.shape}")
print(f"   STFT magnitude range: [{torch.min(torch.abs(stft_result)):.6f}, {torch.max(torch.abs(stft_result)):.6f}]")
print(f"   STFT mean absolute value: {torch.mean(torch.abs(stft_result)):.6f}")

# ONNXモデル
session = ort.InferenceSession(model_path)

# 最初のチャンクのみ処理
freq_bins = 2048
time_frames = 256
chunk = stft_result[:, :, :freq_bins, :time_frames]
input_data = chunk.numpy().astype(np.float32)

print(f"\n🧠 モデル入力分析:")
print(f"   入力shape: {input_data.shape}")
print(f"   入力 magnitude range: [{np.min(np.abs(input_data)):.6f}, {np.max(np.abs(input_data)):.6f}]")
print(f"   入力 mean absolute value: {np.mean(np.abs(input_data)):.6f}")

# ゼロ化ビン適用
input_data[:, :, :3, :] = 0

print(f"\n🎯 モデル推論実行")
output = session.run([session.get_outputs()[0].name],
                     {session.get_inputs()[0].name: input_data})
output_array = output[0]

print(f"\n📊 モデル出力分析 (RAW - compensate適用前):")
print(f"   出力shape: {output_array.shape}")
print(f"   出力 magnitude range: [{np.min(np.abs(output_array)):.6f}, {np.max(np.abs(output_array)):.6f}]")
print(f"   出力 mean absolute value: {np.mean(np.abs(output_array)):.6f}")
print(f"   出力 RMS: {np.sqrt(np.mean(output_array**2)):.6f}")

# 入力と出力のスケール比較
input_rms = np.sqrt(np.mean(input_data**2))
output_rms = np.sqrt(np.mean(output_array**2))
scale_ratio = output_rms / input_rms if input_rms > 0 else 0

print(f"\n⚖️ スケール比較:")
print(f"   入力RMS: {input_rms:.6f}")
print(f"   出力RMS: {output_rms:.6f}")
print(f"   出力/入力比: {scale_ratio:.6f}")

# iSTFT（1チャンクのみ）
print(f"\n🔄 iSTFT実行 (1チャンクのみ)")
output_tensor = torch.from_numpy(output_array).float()
batch_dims = output_tensor.shape[:-3]
c, f, t = output_tensor.shape[-3:]
n = n_fft // 2 + 1

f_pad = torch.zeros([*batch_dims, c, n - f, t])
output_padded = torch.cat([output_tensor, f_pad], -2)
output_reshaped = output_padded.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
output_reshaped = output_reshaped.permute([0, 2, 3, 1])
output_complex = output_reshaped[..., 0] + output_reshaped[..., 1] * 1.j

model_output_time = torch.istft(
    output_complex,
    n_fft=n_fft,
    hop_length=hop_length,
    window=window,
    center=True
)
model_output_time = model_output_time.reshape([*batch_dims, 2, -1])
model_output_numpy = model_output_time.squeeze(0).numpy()

print(f"   時間領域出力shape: {model_output_numpy.shape}")
print(f"   時間領域出力RMS (compensate前): {np.sqrt(np.mean(model_output_numpy[0]**2)):.6f}")

# compensate適用
compensate = 1.035
instrumental_with_compensate = model_output_numpy * compensate
print(f"   時間領域出力RMS (compensate後): {np.sqrt(np.mean(instrumental_with_compensate[0]**2)):.6f}")

# 期待値との比較
expected_inst_rms = 0.092  # ボーカルRMSと同程度を期待
actual_inst_rms = np.sqrt(np.mean(instrumental_with_compensate[0]**2))
ratio = actual_inst_rms / expected_inst_rms

print(f"\n🎯 期待値比較:")
print(f"   期待されるインストゥルメンタルRMS: ~{expected_inst_rms:.6f}")
print(f"   実際のインストゥルメンタルRMS: {actual_inst_rms:.6f}")
print(f"   実際/期待比: {ratio:.6f} ({ratio*100:.1f}%)")
print(f"   不足倍率: {1/ratio:.1f}x")

print(f"\n💡 診断結果:")
if ratio < 0.05:
    print(f"   ❌ CRITICAL: モデル出力が期待値の{ratio*100:.1f}%しかありません")
    print(f"   🔍 考えられる原因:")
    print(f"      1. モデル出力のスケールファクターが欠落")
    print(f"      2. STFT/iSTFT正規化の問題")
    print(f"      3. モデル訓練時のスケーリング設定")
elif ratio < 0.5:
    print(f"   ⚠️  WARNING: モデル出力レベルが低い ({ratio*100:.1f}%)")
else:
    print(f"   ✅ OK: モデル出力レベルは正常範囲")

print("=" * 80)
