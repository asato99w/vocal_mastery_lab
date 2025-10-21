#!/usr/bin/env python3
"""
音源分離テストスクリプト

ONNXモデルを使用して実際に音源分離を実行し、結果を検証します。
CoreML変換前の参照実装として使用します。
"""

import sys
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort
from pathlib import Path
import time


def load_audio(audio_path: Path, sr: int = 44100):
    """
    音声ファイル読み込み

    Args:
        audio_path: 音声ファイルパス
        sr: サンプルレート

    Returns:
        (audio_data, sample_rate)
    """
    print(f"📂 音声読み込み: {audio_path.name}")

    # librosaで読み込み（自動的にモノラル/ステレオ対応）
    audio, sample_rate = librosa.load(audio_path, sr=sr, mono=False)

    # モノラルの場合はステレオに変換
    if audio.ndim == 1:
        audio = np.stack([audio, audio])

    print(f"   形状: {audio.shape}")
    print(f"   サンプルレート: {sample_rate} Hz")
    print(f"   長さ: {audio.shape[1] / sample_rate:.2f} 秒")

    return audio, sample_rate


def stft_transform(audio: np.ndarray, n_fft: int = 4096, hop_length: int = 1024):
    """
    STFT変換

    Args:
        audio: 音声データ [channels, samples]
        n_fft: FFTサイズ
        hop_length: ホップサイズ

    Returns:
        スペクトログラム [channels, freq_bins, time_frames]
    """
    print(f"\n🔄 STFT実行中... (n_fft={n_fft}, hop={hop_length})")

    spectrograms = []
    for channel in audio:
        # STFT実行
        stft = librosa.stft(channel, n_fft=n_fft, hop_length=hop_length)
        spectrograms.append(stft)

    spectrograms = np.array(spectrograms)  # [channels, freq, time]

    print(f"   スペクトログラム形状: {spectrograms.shape}")

    return spectrograms


def istft_transform(spectrograms: np.ndarray, hop_length: int = 1024):
    """
    iSTFT変換

    Args:
        spectrograms: スペクトログラム [channels, freq_bins, time_frames]
        hop_length: ホップサイズ

    Returns:
        音声データ [channels, samples]
    """
    print("\n🔄 iSTFT実行中...")

    audio_channels = []
    for spectrogram in spectrograms:
        # iSTFT実行
        audio = librosa.istft(spectrogram, hop_length=hop_length)
        audio_channels.append(audio)

    audio = np.array(audio_channels)

    print(f"   音声形状: {audio.shape}")

    return audio


def run_onnx_inference(
    model_path: Path,
    spectrogram: np.ndarray,
    chunk_size: int = 256
):
    """
    ONNX推論実行

    Args:
        model_path: ONNXモデルパス
        spectrogram: 入力スペクトログラム [channels, freq, time]
        chunk_size: チャンクサイズ（時間フレーム）

    Returns:
        分離マスク [channels, freq, time]
    """
    print(f"\n🤖 ONNX推論実行: {model_path.name}")

    # ONNX Runtimeセッション作成
    session = ort.InferenceSession(str(model_path))

    # 入力/出力名取得
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"   入力: {input_name}")
    print(f"   出力: {output_name}")

    # スペクトログラムを実部/虚部に分解
    # モデルは [batch, 4, 2048, 256] を期待（4 = ステレオ2ch × 実部/虚部2）
    real_part = np.real(spectrogram).astype(np.float32)
    imag_part = np.imag(spectrogram).astype(np.float32)

    channels, freq_bins, time_frames = spectrogram.shape
    print(f"   入力形状: {spectrogram.shape}")

    # 周波数ビンを2048に調整（モデル期待値）
    if freq_bins > 2048:
        real_part = real_part[:, :2048, :]
        imag_part = imag_part[:, :2048, :]
        freq_bins = 2048
        print(f"   周波数ビン調整: {freq_bins}")

    # チャンク処理
    num_chunks = (time_frames + chunk_size - 1) // chunk_size
    print(f"   チャンク数: {num_chunks} (chunk_size={chunk_size})")

    masks = []

    start_time = time.time()

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, time_frames)

        # チャンク抽出
        real_chunk = real_part[:, :, start_idx:end_idx]
        imag_chunk = imag_part[:, :, start_idx:end_idx]

        # パディング（チャンクサイズに満たない場合）
        if real_chunk.shape[2] < chunk_size:
            pad_width = chunk_size - real_chunk.shape[2]
            real_chunk = np.pad(real_chunk, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
            imag_chunk = np.pad(imag_chunk, ((0, 0), (0, 0), (0, pad_width)), mode='constant')

        # [batch, 4, freq, time] 形式に変換
        # 4チャンネル = [Left_Real, Left_Imag, Right_Real, Right_Imag]
        input_data = np.stack([
            real_chunk[0],  # Left Real
            imag_chunk[0],  # Left Imag
            real_chunk[1] if channels > 1 else real_chunk[0],  # Right Real
            imag_chunk[1] if channels > 1 else imag_chunk[0]   # Right Imag
        ], axis=0)

        # バッチ次元追加
        input_data = np.expand_dims(input_data, axis=0)

        # 推論実行
        outputs = session.run([output_name], {input_name: input_data})
        mask_chunk = outputs[0][0]  # [4, freq, time]

        # 複素数マスクに変換
        left_complex = mask_chunk[0] + 1j * mask_chunk[1]
        right_complex = mask_chunk[2] + 1j * mask_chunk[3]

        # チャンネルごとにまとめる
        combined_mask = np.stack([left_complex, right_complex], axis=0)

        # 元のサイズに戻す
        if end_idx - start_idx < chunk_size:
            combined_mask = combined_mask[:, :, :end_idx - start_idx]

        masks.append(combined_mask)

        if (i + 1) % 10 == 0:
            print(f"   進捗: {i + 1}/{num_chunks} チャンク処理完了")

    elapsed = time.time() - start_time

    # マスク結合
    full_mask = np.concatenate(masks, axis=2)

    print(f"   推論完了: {elapsed:.2f} 秒")
    print(f"   出力形状: {full_mask.shape}")

    return full_mask


def apply_mask(spectrogram: np.ndarray, mask: np.ndarray):
    """
    マスク適用

    Args:
        spectrogram: 元のスペクトログラム（複素数）
        mask: 分離マスク（複素数）

    Returns:
        マスク適用後のスペクトログラム
    """
    print("\n🎭 マスク適用中...")

    # 複素数マスクを直接適用
    masked_spectrogram = spectrogram * mask

    print(f"   マスク適用完了")

    return masked_spectrogram


def save_audio(audio: np.ndarray, sr: int, output_path: Path):
    """
    音声保存

    Args:
        audio: 音声データ [channels, samples]
        sr: サンプルレート
        output_path: 出力パス
    """
    print(f"\n💾 音声保存: {output_path.name}")

    # 転置（soundfileはsamples x channels形式）
    audio_t = audio.T

    sf.write(output_path, audio_t, sr)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   保存完了: {file_size_mb:.2f} MB")


def main():
    """メイン処理"""
    print("=" * 80)
    print("🎵 音源分離テスト (ONNX)")
    print("=" * 80)

    # パス設定
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # モデルパス
    model_path = project_root / "models" / "onnx" / "UVR-MDX-NET-Inst_Main.onnx"

    if not model_path.exists():
        print(f"\n❌ エラー: モデルが見つかりません")
        print(f"   パス: {model_path}")
        sys.exit(1)

    # サンプル音声パス
    sample_dir = project_root.parent / "sample" / "AlejoGranados_RumbaChonta"

    # いくつかのトラックをミックス
    tracks = [
        "01_Bombo.wav",
        "11_Marimba.wav",
        "13_Saxophone1.wav"
    ]

    print(f"\n📂 サンプル音声ミックス:")
    for track in tracks:
        print(f"   - {track}")

    # ミックス作成
    print("\n🎚️  ミックス作成中...")
    mixed_audio = None
    sr = 44100

    for track_name in tracks:
        track_path = sample_dir / track_name
        if track_path.exists():
            audio, sr = load_audio(track_path, sr=sr)
            if mixed_audio is None:
                mixed_audio = audio
            else:
                # 長さを揃える
                min_len = min(mixed_audio.shape[1], audio.shape[1])
                mixed_audio = mixed_audio[:, :min_len] + audio[:, :min_len]
        else:
            print(f"   警告: {track_name} が見つかりません")

    if mixed_audio is None:
        print("❌ エラー: ミックス作成失敗")
        sys.exit(1)

    # 正規化
    mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))

    print(f"✅ ミックス完成: {mixed_audio.shape}")

    # 出力ディレクトリ作成
    output_dir = project_root / "tests" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ミックス保存
    mixed_path = output_dir / "mixed.wav"
    save_audio(mixed_audio, sr, mixed_path)

    # 1. STFT
    spectrogram = stft_transform(mixed_audio, n_fft=4096, hop_length=1024)

    # 2. ONNX推論
    vocal_mask = run_onnx_inference(model_path, spectrogram, chunk_size=256)

    # スペクトログラムも2048周波数ビンに調整（マスクと同じサイズ）
    spectrogram_adjusted = spectrogram[:, :2048, :]

    # 伴奏マスクは反転（vocal_maskが複素数なので共役を取る）
    # 簡易的に振幅の反転として処理
    vocal_mask_magnitude = np.abs(vocal_mask)
    instrumental_mask = (1.0 - vocal_mask_magnitude) * np.exp(1j * np.angle(vocal_mask))

    # 3. マスク適用
    vocal_spec = apply_mask(spectrogram_adjusted, vocal_mask)
    inst_spec = apply_mask(spectrogram_adjusted, instrumental_mask)

    # 4. iSTFT
    vocals = istft_transform(vocal_spec, hop_length=1024)
    instrumental = istft_transform(inst_spec, hop_length=1024)

    # 5. 保存
    vocals_path = output_dir / "vocals.wav"
    instrumental_path = output_dir / "instrumental.wav"

    save_audio(vocals, sr, vocals_path)
    save_audio(instrumental, sr, instrumental_path)

    print("\n" + "=" * 80)
    print("✅ 音源分離完了！")
    print("=" * 80)
    print(f"\n📂 出力ファイル:")
    print(f"   - ミックス: {mixed_path}")
    print(f"   - ボーカル: {vocals_path}")
    print(f"   - 伴奏: {instrumental_path}")
    print("\n次のステップ:")
    print("   音声ファイルを確認して、分離品質を評価してください。")


if __name__ == "__main__":
    main()
