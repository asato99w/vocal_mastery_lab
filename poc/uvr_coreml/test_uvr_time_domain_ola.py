#!/usr/bin/env python3
"""
UVR完全再現: 時間領域OLA (Overlap-Add) 実装
separate.py:542-625 の demix メソッドを正確に再現
"""
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
import librosa

input_file = "tests/output/hollow_crown_from_flac.wav"
model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"

print("=" * 80)
print("🎯 UVR時間領域OLA完全再現")
print("=" * 80)

# === パラメータ設定 (UVRと完全一致) ===
n_fft = 4096
hop_length = 1024
mdx_segment_size = 256
overlap_mdx = 0.25  # UVRデフォルト
compensate = 1.035
dim_f = 2048

# 計算値
trim = n_fft // 2  # 2048
chunk_size = hop_length * (mdx_segment_size - 1)  # 261120
gen_size = chunk_size - 2 * trim  # 257024
step = int((1 - overlap_mdx) * chunk_size)  # 195840

print(f"\n📊 UVRパラメータ:")
print(f"   n_fft: {n_fft}")
print(f"   hop_length: {hop_length}")
print(f"   trim: {trim}")
print(f"   chunk_size: {chunk_size} サンプル")
print(f"   gen_size: {gen_size}")
print(f"   overlap: {overlap_mdx}")
print(f"   step: {step} サンプル")

# === 音声読み込み ===
print(f"\n📂 音声読み込み: {input_file}")
y, sr = librosa.load(input_file, sr=44100, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])

original_audio = y.copy()
print(f"   元音声shape: {y.shape}")
print(f"   元音声RMS: {np.sqrt(np.mean(y[0]**2)):.6f}")

# === パディング (UVR separate.py:560-561) ===
mix = y
pad = gen_size + trim - ((mix.shape[-1]) % gen_size)
mixture = np.concatenate((np.zeros((2, trim), dtype='float32'), mix, np.zeros((2, pad), dtype='float32')), 1)

print(f"\n🔧 パディング:")
print(f"   元音声長: {mix.shape[-1]} → パディング後: {mixture.shape[-1]}")
print(f"   前パディング: {trim}, 後パディング: {pad}")

# === STFT準備 ===
window = torch.hann_window(window_length=n_fft, periodic=True)

# === 出力バッファ初期化 (OLA用) ===
result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)

# === ONNXモデル ===
session = ort.InferenceSession(model_path)

# === 時間領域OLA処理 ===
print(f"\n🔄 時間領域OLA処理開始:")
total_chunks = (mixture.shape[-1] - chunk_size) // step + 1
print(f"   総チャンク数: {total_chunks}")

chunk_count = 0
for i in range(0, mixture.shape[-1], step):
    start = i
    end = min(i + chunk_size, mixture.shape[-1])
    chunk_size_actual = end - start

    if chunk_size_actual < chunk_size:
        # 最後のチャンクはパディング
        pad_size = chunk_size - chunk_size_actual
        mix_part_ = np.concatenate((mixture[:, start:end], np.zeros((2, pad_size), dtype='float32')), axis=-1)
    else:
        mix_part_ = mixture[:, start:end]

    # Hann窓を適用 (分析窓) - UVR separate.py:579-580
    window_np = np.hanning(chunk_size_actual)
    window_2d = np.tile(window_np[None, :], (2, 1))

    # 時間領域チャンクをテンソルに変換
    mix_part = torch.tensor([mix_part_], dtype=torch.float32)

    # === STFT → モデル → iSTFT ===
    with torch.no_grad():
        # STFT
        stft_result = torch.stft(
            mix_part.reshape([-1, mix_part.shape[-1]]),
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

        # dim_fで切り取り (UVR tfc_tdf_v3.py:30)
        stft_result = stft_result[..., :dim_f, :]

        # モデル入力準備
        spek = stft_result
        spek[:, :, :3, :] = 0  # 低域ゼロ化

        # ONNX推論
        spek_np = spek.numpy().astype(np.float32)
        output = session.run([session.get_outputs()[0].name],
                           {session.get_inputs()[0].name: spek_np})
        spec_pred = output[0]

        # iSTFT
        spec_pred_tensor = torch.from_numpy(spec_pred).float()
        batch_dims = spec_pred_tensor.shape[:-3]
        c, f, t = spec_pred_tensor.shape[-3:]
        n = n_fft // 2 + 1

        # 周波数パディング
        f_pad = torch.zeros([*batch_dims, c, n - f, t])
        spec_padded = torch.cat([spec_pred_tensor, f_pad], -2)
        spec_reshaped = spec_padded.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        spec_reshaped = spec_reshaped.permute([0, 2, 3, 1])
        spec_complex = spec_reshaped[..., 0] + spec_reshaped[..., 1] * 1.j

        tar_waves = torch.istft(
            spec_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=True
        )

        tar_waves_np = tar_waves.numpy()

        # === OLA: 窓関数適用と累積 (UVR separate.py:596-602) ===
        # 窓関数を出力に適用
        tar_waves_windowed = tar_waves_np[..., :chunk_size_actual] * window_2d

        # resultに累積
        result[0, :, start:end] += tar_waves_windowed

        # dividerに窓の累積
        divider[0, :, start:end] += window_2d

    chunk_count += 1
    if chunk_count % 10 == 0:
        print(f"   進行状況: {chunk_count}/{total_chunks} チャンク完了")

print(f"   全チャンク完了 ({chunk_count}個)")

# === OLA正規化 (UVR separate.py:604) ===
print(f"\n➗ OLA正規化: result / divider")
tar_waves_final = result / divider

# === トリミング (UVR separate.py:607) ===
print(f"\n✂️  トリミング: 前後 {trim} サンプル除去")
tar_waves_trimmed = tar_waves_final[:, :, trim:-trim]

# === 元の長さに切り詰め (UVR separate.py:608) ===
tar_waves_cropped = tar_waves_trimmed[:, :, :mix.shape[-1]]

source = tar_waves_cropped[0, :, :]  # [2, samples]

print(f"   出力shape: {source.shape}")
print(f"   出力RMS (compensate前): {np.sqrt(np.mean(source[0]**2)):.6f}")

# === Compensate適用 (UVR separate.py:615) ===
instrumental_numpy = source * compensate

print(f"   インストゥルメンタルRMS (compensate後): {np.sqrt(np.mean(instrumental_numpy[0]**2)):.6f}")

# === ボーカル計算 ===
print(f"\n➖ ボーカル計算: Original - Instrumental")

min_len = min(original_audio.shape[1], instrumental_numpy.shape[1])
original_trimmed = original_audio[:, :min_len]
instrumental_trimmed = instrumental_numpy[:, :min_len]

vocal_numpy = original_trimmed - instrumental_trimmed

print(f"   ボーカルshape: {vocal_numpy.shape}")
print(f"   ボーカルRMS: {np.sqrt(np.mean(vocal_numpy[0]**2)):.6f}")

# === 統計 ===
print(f"\n📊 結果統計:")
orig_rms = np.sqrt(np.mean(original_trimmed[0]**2))
voc_rms = np.sqrt(np.mean(vocal_numpy[0]**2))
inst_rms = np.sqrt(np.mean(instrumental_trimmed[0]**2))

print(f"   元音声RMS:          {orig_rms:.6f}")
print(f"   ボーカルRMS:        {voc_rms:.6f}")
print(f"   インストゥルメンタルRMS: {inst_rms:.6f}")

# === 整合性チェック ===
reconstructed = vocal_numpy + instrumental_trimmed
reconstruction_error = np.sqrt(np.mean((original_trimmed - reconstructed)**2))
print(f"\n✅ 整合性チェック:")
print(f"   original - (vocal + instrumental) RMS誤差: {reconstruction_error:.8f}")
print(f"   誤差レベル: {'良好 (< 1e-4)' if reconstruction_error < 1e-4 else '要確認'}")

# === 期待値比較 ===
expected_inst_rms = 0.092
ratio = inst_rms / expected_inst_rms
print(f"\n🎯 期待値比較:")
print(f"   期待されるインストゥルメンタルRMS: ~{expected_inst_rms:.6f}")
print(f"   実際のインストゥルメンタルRMS: {inst_rms:.6f}")
print(f"   実際/期待比: {ratio:.6f} ({ratio*100:.1f}%)")
if ratio >= 0.8:
    print(f"   ✅ 成功！正常範囲内")
elif ratio >= 0.5:
    print(f"   ⚠️  改善されたが、まだ不足")
else:
    print(f"   ❌ まだ大幅に不足")

# === 保存 ===
print(f"\n💾 結果保存:")
sf.write("tests/python_output/ola_vocal.wav", vocal_numpy.T, sr)
sf.write("tests/python_output/ola_instrumental.wav", instrumental_trimmed.T, sr)
print(f"   ボーカル: ola_vocal.wav")
print(f"   インストゥルメンタル: ola_instrumental.wav")

print(f"\n✅ 完了")
print("=" * 80)
