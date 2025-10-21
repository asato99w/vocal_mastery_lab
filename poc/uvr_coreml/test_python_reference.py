#!/usr/bin/env python3
"""
Python参照実装でHollow Crown音声を処理
Swift実装との比較用
"""

import sys
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort
from pathlib import Path
import time

# 音声読み込み
def load_audio(audio_path, sr=44100):
    print(f"📂 音声読み込み: {audio_path}")
    audio, sample_rate = librosa.load(audio_path, sr=sr, mono=False)

    if audio.ndim == 1:
        audio = np.stack([audio, audio])

    print(f"   形状: {audio.shape}")
    print(f"   サンプルレート: {sample_rate} Hz")
    print(f"   長さ: {audio.shape[1] / sample_rate:.2f} 秒")

    return audio, sample_rate

# STFT
def stft_transform(audio, n_fft=4096, hop_length=1024):
    print(f"\n🔄 STFT実行中... (n_fft={n_fft}, hop={hop_length})")

    spectrograms = []
    for channel in audio:
        stft = librosa.stft(channel, n_fft=n_fft, hop_length=hop_length)
        spectrograms.append(stft)

    spectrograms = np.array(spectrograms)
    print(f"   スペクトログラム形状: {spectrograms.shape}")

    return spectrograms

# ONNX推論
def run_onnx_inference(model_path, spectrogram, chunk_size=256):
    print(f"\n🤖 ONNX推論実行: {model_path}")

    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    real_part = np.real(spectrogram).astype(np.float32)
    imag_part = np.imag(spectrogram).astype(np.float32)

    channels, freq_bins, time_frames = spectrogram.shape
    print(f"   入力形状: {spectrogram.shape}")

    if freq_bins > 2048:
        real_part = real_part[:, :2048, :]
        imag_part = imag_part[:, :2048, :]
        freq_bins = 2048

    num_chunks = (time_frames + chunk_size - 1) // chunk_size
    print(f"   チャンク数: {num_chunks}")

    masks = []
    start_time = time.time()

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, time_frames)

        real_chunk = real_part[:, :, start_idx:end_idx]
        imag_chunk = imag_part[:, :, start_idx:end_idx]

        if real_chunk.shape[2] < chunk_size:
            pad_width = chunk_size - real_chunk.shape[2]
            real_chunk = np.pad(real_chunk, ((0, 0), (0, 0), (0, pad_width)))
            imag_chunk = np.pad(imag_chunk, ((0, 0), (0, 0), (0, pad_width)))

        input_data = np.stack([
            real_chunk[0],
            imag_chunk[0],
            real_chunk[1] if channels > 1 else real_chunk[0],
            imag_chunk[1] if channels > 1 else imag_chunk[0]
        ], axis=0)

        input_data = np.expand_dims(input_data, axis=0)

        outputs = session.run([output_name], {input_name: input_data})
        mask_chunk = outputs[0][0]

        left_complex = mask_chunk[0] + 1j * mask_chunk[1]
        right_complex = mask_chunk[2] + 1j * mask_chunk[3]

        combined_mask = np.stack([left_complex, right_complex], axis=0)

        if end_idx - start_idx < chunk_size:
            combined_mask = combined_mask[:, :, :end_idx - start_idx]

        masks.append(combined_mask)

        if (i + 1) % 10 == 0:
            print(f"   進捗: {i + 1}/{num_chunks}")

    full_mask = np.concatenate(masks, axis=2)
    elapsed = time.time() - start_time

    print(f"   推論完了: {elapsed:.2f} 秒")
    print(f"   出力形状: {full_mask.shape}")

    return full_mask

# iSTFT
def istft_transform(spectrograms, hop_length=1024):
    print("\n🔄 iSTFT実行中...")

    audio_channels = []
    for spectrogram in spectrograms:
        audio = librosa.istft(spectrogram, hop_length=hop_length)
        audio_channels.append(audio)

    audio = np.array(audio_channels)
    print(f"   音声形状: {audio.shape}")

    return audio

# 保存
def save_audio(audio, sr, output_path):
    print(f"\n💾 音声保存: {output_path}")
    audio_t = audio.T
    sf.write(output_path, audio_t, sr)
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"   保存完了: {file_size_mb:.2f} MB")

# メイン
def main():
    print("=" * 80)
    print("🎵 Python参照実装テスト (ONNX)")
    print("=" * 80)

    # パス
    input_path = "tests/output/hollow_crown_from_flac.wav"
    model_path = "models/onnx/UVR-MDX-NET-Inst_Main.onnx"
    output_dir = Path("tests/python_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 読み込み
    audio, sr = load_audio(input_path, sr=44100)

    # STFT
    spectrogram = stft_transform(audio, n_fft=4096, hop_length=1024)

    # ONNX推論
    vocal_mask = run_onnx_inference(model_path, spectrogram, chunk_size=256)

    # スペクトログラム調整
    spectrogram_adjusted = spectrogram[:, :2048, :]

    # マスク適用
    print("\n🎭 マスク適用中...")
    vocal_spec = spectrogram_adjusted * vocal_mask

    # iSTFT
    vocals = istft_transform(vocal_spec, hop_length=1024)

    # 保存
    vocals_path = output_dir / "hollow_crown_vocals_python.wav"
    save_audio(vocals, sr, vocals_path)

    print("\n" + "=" * 80)
    print("✅ Python参照実装完了")
    print("=" * 80)
    print(f"\n📂 出力: {vocals_path}")

    # 統計情報
    print(f"\n📊 出力統計:")
    print(f"   Max: {np.abs(vocals).max():.6f}")
    print(f"   RMS: {np.sqrt(np.mean(vocals**2)):.6f}")

if __name__ == "__main__":
    main()
