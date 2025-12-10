#!/usr/bin/env python3
"""
Python版音源分離 - ボーカルと伴奏の両方を出力
"""
import numpy as np
import librosa
import soundfile as sf
import coremltools as ct

# ファイルパス
input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/coreml/UVR-MDX-NET-Inst_Main.mlpackage"
vocal_output = "tests/python_output/hollow_crown_vocals_python.wav"
instrumental_output = "tests/python_output/hollow_crown_instrumental_python.wav"

print("=" * 80)
print("🎵 Python CoreML音源分離 (ボーカル + 伴奏)")
print("=" * 80)

# 音声読み込み
print(f"\n📂 音声読み込み: {input_file}")
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

print(f"   サンプルレート: {sr} Hz")
print(f"   チャンネル数: {y.shape[0]}")
print(f"   サンプル数: {y.shape[1]}")

# STFT
n_fft = 4096
hop_length = 1024

print(f"\n🔄 STFT実行 (n_fft={n_fft}, hop_length={hop_length})")
stft_left = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length)
stft_right = librosa.stft(y[1], n_fft=n_fft, hop_length=hop_length)

print(f"   STFT shape: {stft_left.shape}")

# モデル読み込み
print(f"\n🤖 モデル読み込み: {model_path}")
model = ct.models.MLModel(model_path)

# 推論用データ準備
freq_bins = 2048
time_frames = 256
num_chunks = (stft_left.shape[1] + time_frames - 1) // time_frames

print(f"   チャンク数: {num_chunks}")

# マスク格納
vocal_masks = []
instrumental_masks = []

for chunk_idx in range(num_chunks):
    start_t = chunk_idx * time_frames
    end_t = min((chunk_idx + 1) * time_frames, stft_left.shape[1])
    actual_size = end_t - start_t

    # チャンク抽出（ゼロパディング）
    chunk_left = stft_left[:freq_bins, start_t:end_t]
    chunk_right = stft_right[:freq_bins, start_t:end_t]

    # パディング
    if chunk_left.shape[1] < time_frames:
        pad_width = time_frames - chunk_left.shape[1]
        chunk_left = np.pad(chunk_left, ((0, 0), (0, pad_width)), mode='constant')
        chunk_right = np.pad(chunk_right, ((0, 0), (0, pad_width)), mode='constant')

    # 入力データ作成 [1, 4, 2048, 256]
    # Channel 0: Left Real, Channel 1: Left Imag, Channel 2: Right Real, Channel 3: Right Imag
    input_data = np.zeros((1, 4, freq_bins, time_frames), dtype=np.float32)
    input_data[0, 0] = np.real(chunk_left)
    input_data[0, 1] = np.imag(chunk_left)
    input_data[0, 2] = np.real(chunk_right)
    input_data[0, 3] = np.imag(chunk_right)

    # 推論
    output = model.predict({"input_1": input_data})
    output_array = output["var_992"]  # Shape: [1, 4, 2048, 256]

    # 出力は [1, 4, 2048, 256] のはず
    # どのチャンネルがボーカル/伴奏かを確認
    if chunk_idx == 0:
        print(f"\n   出力shape: {output_array.shape}")
        for ch in range(output_array.shape[1]):
            ch_mean = output_array[0, ch].mean()
            ch_max = output_array[0, ch].max()
            print(f"     Channel {ch}: mean={ch_mean:.6f}, max={ch_max:.6f}")

    # 仮定: Channel 0-1 が Left (Inst/Vocal), Channel 2-3 が Right (Inst/Vocal)
    # または Channel 0 = Inst, Channel 1 = Vocal, Channel 2 = Inst, Channel 3 = Vocal

    # まずは Channel 0 と Channel 1 を取得
    inst_chunk = output_array[0, 0, :, :actual_size]  # Channel 0
    vocal_chunk = output_array[0, 1, :, :actual_size]  # Channel 1

    instrumental_masks.append(inst_chunk.T)  # [time, freq]
    vocal_masks.append(vocal_chunk.T)

    if (chunk_idx + 1) % 10 == 0:
        print(f"   進捗: {chunk_idx + 1}/{num_chunks}")

# マスク結合
instrumental_mask = np.vstack(instrumental_masks).T  # [freq, time]
vocal_mask = np.vstack(vocal_masks).T

print(f"\n   Instrumental mask shape: {instrumental_mask.shape}")
print(f"   Vocal mask shape: {vocal_mask.shape}")

# マスク適用
print(f"\n🎭 マスク適用")
stft_inst = stft_left[:freq_bins, :instrumental_mask.shape[1]] * instrumental_mask
stft_vocal = stft_left[:freq_bins, :vocal_mask.shape[1]] * vocal_mask

# iSTFT
print(f"\n🔄 iSTFT実行")
audio_inst = librosa.istft(stft_inst, hop_length=hop_length, length=y.shape[1])
audio_vocal = librosa.istft(stft_vocal, hop_length=hop_length, length=y.shape[1])

# ステレオ化（簡易版: モノラルを複製）
audio_inst_stereo = np.stack([audio_inst, audio_inst])
audio_vocal_stereo = np.stack([audio_vocal, audio_vocal])

# 保存
print(f"\n💾 保存中...")
sf.write(instrumental_output, audio_inst_stereo.T, sr)
sf.write(vocal_output, audio_vocal_stereo.T, sr)

print(f"\n✅ 完了")
print(f"   伴奏: {instrumental_output}")
print(f"   ボーカル: {vocal_output}")

# 統計
print(f"\n📊 統計:")
print(f"   伴奏 - RMS: {np.sqrt(np.mean(audio_inst**2)):.6f}, Max: {np.max(np.abs(audio_inst)):.6f}")
print(f"   ボーカル - RMS: {np.sqrt(np.mean(audio_vocal**2)):.6f}, Max: {np.max(np.abs(audio_vocal)):.6f}")

print("\n" + "=" * 80)
