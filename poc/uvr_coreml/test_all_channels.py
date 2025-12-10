#!/usr/bin/env python3
"""
全4チャンネルを個別にテストして、どれがボーカル/伴奏か確認
"""
import numpy as np
import librosa
import soundfile as sf
import coremltools as ct

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/coreml/UVR-MDX-NET-Inst_Main.mlpackage"

print("=" * 80)
print("🔍 全チャンネルテスト - どのチャンネルがボーカル/伴奏か確認")
print("=" * 80)

# 音声読み込み
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

# STFT
n_fft = 4096
hop_length = 1024
stft_left = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length)

# モデル読み込み
model = ct.models.MLModel(model_path)

# 最初のチャンクだけテスト
freq_bins = 2048
time_frames = 256

chunk_left = stft_left[:freq_bins, :time_frames]
if chunk_left.shape[1] < time_frames:
    pad_width = time_frames - chunk_left.shape[1]
    chunk_left = np.pad(chunk_left, ((0, 0), (0, pad_width)), mode='constant')

# 入力データ作成 [1, 4, 2048, 256]
input_data = np.zeros((1, 4, freq_bins, time_frames), dtype=np.float32)
input_data[0, 0] = np.real(chunk_left)
input_data[0, 1] = np.imag(chunk_left)
input_data[0, 2] = np.real(chunk_left)  # 右チャンネル（簡易版: 左と同じ）
input_data[0, 3] = np.imag(chunk_left)

# 推論
output = model.predict({"input_1": input_data})
output_array = output["var_992"]

print(f"\n📊 モデル出力形状: {output_array.shape}")
print(f"\n各チャンネルの統計:")

for ch in range(output_array.shape[1]):
    ch_data = output_array[0, ch]
    print(f"\n  Channel {ch}:")
    print(f"    平均値: {ch_data.mean():.6f}")
    print(f"    最大値: {ch_data.max():.6f}")
    print(f"    最小値: {ch_data.min():.6f}")
    print(f"    RMS: {np.sqrt(np.mean(ch_data**2)):.6f}")

    # サンプル値
    print(f"    最初の5値: {ch_data[0, :5]}")

# 各チャンネルで音声を再構成して保存
print(f"\n🎵 各チャンネルから音声を再構成...")

for ch in range(4):
    # マスクとして使用
    mask = output_array[0, ch, :, :stft_left.shape[1]]
    stft_masked = stft_left[:freq_bins, :mask.shape[1]] * mask

    # iSTFT
    audio = librosa.istft(stft_masked, hop_length=hop_length, length=y.shape[1])
    audio_stereo = np.stack([audio, audio])

    # 保存
    output_file = f"tests/python_output/channel_{ch}_output.wav"
    sf.write(output_file, audio_stereo.T, sr)

    # 統計
    rms = np.sqrt(np.mean(audio**2))
    max_val = np.max(np.abs(audio))
    print(f"  Channel {ch}: RMS={rms:.6f}, Max={max_val:.6f} → {output_file}")

print(f"\n✅ 完了")
print(f"\n🎧 各ファイルを聴いて、どれがボーカル/伴奏か確認してください:")
for ch in range(4):
    print(f"  Channel {ch}: tests/python_output/channel_{ch}_output.wav")

print("\n" + "=" * 80)
