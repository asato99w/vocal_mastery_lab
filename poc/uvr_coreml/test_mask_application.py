#!/usr/bin/env python3
"""
マスク適用方法の検証 - モデル出力の正しい使い方を探る
"""
import numpy as np
import librosa
import soundfile as sf
import coremltools as ct

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/coreml/UVR-MDX-NET-Inst_Main.mlpackage"

print("=" * 80)
print("🔍 マスク適用方法の検証")
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

# モデル読み込み
print(f"\n🤖 モデル読み込み: {model_path}")
model = ct.models.MLModel(model_path)

# 最初のチャンクで推論
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
input_data[0, 2] = np.real(chunk_left)
input_data[0, 3] = np.imag(chunk_left)

# 推論
print(f"\n🧠 推論実行")
output = model.predict({"input_1": input_data})
output_array = output["var_992"]

print(f"   モデル出力shape: {output_array.shape}")

# 各チャンネルのマスク統計
print(f"\n📊 マスク統計:")
for ch in range(4):
    ch_data = output_array[0, ch]
    print(f"\n  Channel {ch}:")
    print(f"    最小値: {ch_data.min():.6f}")
    print(f"    最大値: {ch_data.max():.6f}")
    print(f"    平均値: {ch_data.mean():.6f}")
    print(f"    標準偏差: {ch_data.std():.6f}")
    print(f"    サンプル [0,0:5]: {ch_data[0, :5]}")

# テスト1: Channel 0のマスクを直接適用
print(f"\n\n🧪 テスト1: Channel 0マスクを直接複素数STFTに適用")
mask_ch0 = output_array[0, 0, :, :]
stft_masked = stft_left[:freq_bins, :mask_ch0.shape[1]] * mask_ch0

print(f"   元STFT magnitude range: {np.abs(stft_left[:freq_bins, :mask_ch0.shape[1]]).min():.6f} - {np.abs(stft_left[:freq_bins, :mask_ch0.shape[1]]).max():.6f}")
print(f"   Mask range: {mask_ch0.min():.6f} - {mask_ch0.max():.6f}")
print(f"   Masked STFT magnitude range: {np.abs(stft_masked).min():.6f} - {np.abs(stft_masked).max():.6f}")

audio1 = librosa.istft(stft_masked, hop_length=hop_length, length=y.shape[1])
print(f"   結果 RMS: {np.sqrt(np.mean(audio1**2)):.6f}, Max: {np.max(np.abs(audio1)):.6f}")

# テスト2: マスクを複素数として再構成 (Ch0=Real, Ch1=Imag)
print(f"\n🧪 テスト2: Channel 0+1を複素数マスクとして適用")
mask_real = output_array[0, 0, :, :]
mask_imag = output_array[0, 1, :, :]
mask_complex = mask_real + 1j * mask_imag

stft_masked2 = stft_left[:freq_bins, :mask_real.shape[1]] * mask_complex

print(f"   Complex mask magnitude range: {np.abs(mask_complex).min():.6f} - {np.abs(mask_complex).max():.6f}")
print(f"   Masked STFT magnitude range: {np.abs(stft_masked2).min():.6f} - {np.abs(stft_masked2).max():.6f}")

audio2 = librosa.istft(stft_masked2, hop_length=hop_length, length=y.shape[1])
print(f"   結果 RMS: {np.sqrt(np.mean(audio2**2)):.6f}, Max: {np.max(np.abs(audio2)):.6f}")

# テスト3: マスクを正規化してから適用
print(f"\n🧪 テスト3: Channel 0を正規化してから適用")
mask_normalized = mask_ch0 / (mask_ch0.max() + 1e-8)

stft_masked3 = stft_left[:freq_bins, :mask_normalized.shape[1]] * mask_normalized

print(f"   Normalized mask range: {mask_normalized.min():.6f} - {mask_normalized.max():.6f}")
print(f"   Masked STFT magnitude range: {np.abs(stft_masked3).min():.6f} - {np.abs(stft_masked3).max():.6f}")

audio3 = librosa.istft(stft_masked3, hop_length=hop_length, length=y.shape[1])
print(f"   結果 RMS: {np.sqrt(np.mean(audio3**2)):.6f}, Max: {np.max(np.abs(audio3)):.6f}")

# テスト4: マスクをSTFTのmagnitudeにのみ適用
print(f"\n🧪 テスト4: Channel 0をmagnitudeにのみ適用、phaseは保持")
magnitude = np.abs(stft_left[:freq_bins, :mask_ch0.shape[1]])
phase = np.angle(stft_left[:freq_bins, :mask_ch0.shape[1]])

masked_magnitude = magnitude * mask_ch0
stft_masked4 = masked_magnitude * np.exp(1j * phase)

print(f"   Masked magnitude range: {masked_magnitude.min():.6f} - {masked_magnitude.max():.6f}")
print(f"   Masked STFT magnitude range: {np.abs(stft_masked4).min():.6f} - {np.abs(stft_masked4).max():.6f}")

audio4 = librosa.istft(stft_masked4, hop_length=hop_length, length=y.shape[1])
print(f"   結果 RMS: {np.sqrt(np.mean(audio4**2)):.6f}, Max: {np.max(np.abs(audio4)):.6f}")

# 保存
print(f"\n💾 結果を保存中...")
audio_stereo = np.stack([audio1, audio1])
sf.write("tests/python_output/mask_test_1_direct.wav", audio_stereo.T, sr)

audio_stereo = np.stack([audio2, audio2])
sf.write("tests/python_output/mask_test_2_complex.wav", audio_stereo.T, sr)

audio_stereo = np.stack([audio3, audio3])
sf.write("tests/python_output/mask_test_3_normalized.wav", audio_stereo.T, sr)

audio_stereo = np.stack([audio4, audio4])
sf.write("tests/python_output/mask_test_4_magnitude_only.wav", audio_stereo.T, sr)

print(f"\n✅ 完了")
print(f"\n次のファイルを聴いて、音が聞こえるものを確認してください:")
print(f"  1. tests/python_output/mask_test_1_direct.wav - 直接適用")
print(f"  2. tests/python_output/mask_test_2_complex.wav - 複素数として適用")
print(f"  3. tests/python_output/mask_test_3_normalized.wav - 正規化してから適用")
print(f"  4. tests/python_output/mask_test_4_magnitude_only.wav - magnitudeのみ適用")

print("\n" + "=" * 80)
