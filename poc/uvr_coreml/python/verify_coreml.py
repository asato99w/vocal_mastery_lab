#!/usr/bin/env python3
"""
CoreMLモデル検証スクリプト

ONNXモデルとCoreMLモデルの出力を比較して、
変換が正しく行われたことを検証します。
"""

import sys
import numpy as np
import librosa
import coremltools as ct
import onnxruntime as ort
from pathlib import Path
import time


def load_test_audio(audio_path: Path, sr: int = 44100, duration: float = 5.0):
    """
    テスト用音声読み込み（短時間のみ）

    Args:
        audio_path: 音声ファイルパス
        sr: サンプルレート
        duration: 読み込み時間（秒）

    Returns:
        (audio_data, sample_rate)
    """
    print(f"📂 音声読み込み: {audio_path.name} ({duration}秒)")

    audio, sample_rate = librosa.load(
        audio_path,
        sr=sr,
        mono=False,
        duration=duration
    )

    # モノラルの場合はステレオに変換
    if audio.ndim == 1:
        audio = np.stack([audio, audio])

    print(f"   形状: {audio.shape}")
    print(f"   サンプルレート: {sample_rate} Hz")

    return audio, sample_rate


def stft_transform(audio: np.ndarray, n_fft: int = 4096, hop_length: int = 1024):
    """
    STFT変換
    """
    print(f"\n🔄 STFT実行中... (n_fft={n_fft}, hop={hop_length})")

    spectrograms = []
    for channel in audio:
        stft = librosa.stft(channel, n_fft=n_fft, hop_length=hop_length)
        spectrograms.append(stft)

    spectrograms = np.array(spectrograms)
    print(f"   スペクトログラム形状: {spectrograms.shape}")

    return spectrograms


def prepare_model_input(spectrogram: np.ndarray):
    """
    モデル入力データ準備

    Args:
        spectrogram: 複素数スペクトログラム [channels, freq, time]

    Returns:
        [batch, 4, 2048, 256] 形式の入力データ
    """
    # 実部/虚部分解
    real_part = np.real(spectrogram).astype(np.float32)
    imag_part = np.imag(spectrogram).astype(np.float32)

    channels, freq_bins, time_frames = spectrogram.shape

    # 周波数ビンを2048に調整
    if freq_bins > 2048:
        real_part = real_part[:, :2048, :]
        imag_part = imag_part[:, :2048, :]
        freq_bins = 2048

    # 時間フレームを256に調整（最初の256フレームのみ）
    chunk_size = 256
    if time_frames > chunk_size:
        real_part = real_part[:, :, :chunk_size]
        imag_part = imag_part[:, :, :chunk_size]
        time_frames = chunk_size
    elif time_frames < chunk_size:
        # パディング
        pad_width = chunk_size - time_frames
        real_part = np.pad(real_part, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        imag_part = np.pad(imag_part, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        time_frames = chunk_size

    # [batch, 4, freq, time] 形式に変換
    input_data = np.stack([
        real_part[0],  # Left Real
        imag_part[0],  # Left Imag
        real_part[1] if channels > 1 else real_part[0],  # Right Real
        imag_part[1] if channels > 1 else imag_part[0]   # Right Imag
    ], axis=0)

    # バッチ次元追加
    input_data = np.expand_dims(input_data, axis=0)

    return input_data


def run_onnx_inference(model_path: Path, input_data: np.ndarray):
    """
    ONNX推論実行
    """
    print(f"\n🤖 ONNX推論実行: {model_path.name}")

    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"   入力形状: {input_data.shape}")

    start_time = time.time()
    outputs = session.run([output_name], {input_name: input_data})
    elapsed = time.time() - start_time

    output = outputs[0]

    print(f"   推論完了: {elapsed:.4f} 秒")
    print(f"   出力形状: {output.shape}")

    return output


def run_coreml_inference(model_path: Path, input_data: np.ndarray):
    """
    CoreML推論実行
    """
    print(f"\n🍎 CoreML推論実行: {model_path.name}")

    mlmodel = ct.models.MLModel(str(model_path))

    print(f"   入力形状: {input_data.shape}")

    start_time = time.time()
    output = mlmodel.predict({'input_1': input_data})
    elapsed = time.time() - start_time

    output_array = output['var_992']

    print(f"   推論完了: {elapsed:.4f} 秒")
    print(f"   出力形状: {output_array.shape}")

    return output_array


def compare_outputs(onnx_output: np.ndarray, coreml_output: np.ndarray):
    """
    ONNX と CoreML の出力を比較
    """
    print("\n" + "=" * 80)
    print("📊 出力比較")
    print("=" * 80)

    # 形状確認
    print(f"\nONNX出力形状: {onnx_output.shape}")
    print(f"CoreML出力形状: {coreml_output.shape}")

    if onnx_output.shape != coreml_output.shape:
        print("⚠️  警告: 出力形状が異なります")
        return

    # 統計情報
    print("\nONNX統計:")
    print(f"  範囲: [{onnx_output.min():.6f}, {onnx_output.max():.6f}]")
    print(f"  平均: {onnx_output.mean():.6f}")
    print(f"  標準偏差: {onnx_output.std():.6f}")

    print("\nCoreML統計:")
    print(f"  範囲: [{coreml_output.min():.6f}, {coreml_output.max():.6f}]")
    print(f"  平均: {coreml_output.mean():.6f}")
    print(f"  標準偏差: {coreml_output.std():.6f}")

    # 差分計算
    abs_diff = np.abs(onnx_output - coreml_output)
    rel_diff = abs_diff / (np.abs(onnx_output) + 1e-8)

    print("\n差分統計:")
    print(f"  絶対誤差:")
    print(f"    最大: {abs_diff.max():.6f}")
    print(f"    平均: {abs_diff.mean():.6f}")
    print(f"    中央値: {np.median(abs_diff):.6f}")

    print(f"  相対誤差:")
    print(f"    最大: {rel_diff.max():.6f}")
    print(f"    平均: {rel_diff.mean():.6f}")
    print(f"    中央値: {np.median(rel_diff):.6f}")

    # 許容範囲チェック
    tolerance = 1e-3
    match_ratio = np.sum(abs_diff < tolerance) / abs_diff.size

    print(f"\n許容範囲内の要素比率 (絶対誤差 < {tolerance}):")
    print(f"  {match_ratio * 100:.2f}%")

    if match_ratio > 0.99:
        print("\n✅ 検証成功: ONNX と CoreML の出力はほぼ一致しています")
    elif match_ratio > 0.95:
        print("\n⚠️  注意: 出力は概ね一致していますが、一部に誤差があります")
    else:
        print("\n❌ 警告: 出力に大きな差異があります")


def main():
    """メイン処理"""
    print("=" * 80)
    print("🔍 CoreML検証スクリプト")
    print("=" * 80)

    # パス設定
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    onnx_model_path = project_root / "models" / "onnx" / "UVR-MDX-NET-Inst_Main.onnx"
    coreml_model_path = project_root / "models" / "coreml" / "UVR-MDX-NET-Inst_Main.mlpackage"

    # モデル存在確認
    if not onnx_model_path.exists():
        print(f"\n❌ エラー: ONNXモデルが見つかりません")
        print(f"   パス: {onnx_model_path}")
        sys.exit(1)

    if not coreml_model_path.exists():
        print(f"\n❌ エラー: CoreMLモデルが見つかりません")
        print(f"   パス: {coreml_model_path}")
        sys.exit(1)

    # テスト音声パス
    test_audio_path = project_root / "tests" / "output" / "mixed.wav"

    if not test_audio_path.exists():
        print(f"\n❌ エラー: テスト音声が見つかりません")
        print(f"   パス: {test_audio_path}")
        sys.exit(1)

    # 1. 音声読み込み（5秒のみ）
    audio, sr = load_test_audio(test_audio_path, duration=5.0)

    # 2. STFT変換
    spectrogram = stft_transform(audio, n_fft=4096, hop_length=1024)

    # 3. モデル入力準備
    print("\n🎯 モデル入力準備中...")
    model_input = prepare_model_input(spectrogram)
    print(f"   入力形状: {model_input.shape}")

    # 4. ONNX推論
    onnx_output = run_onnx_inference(onnx_model_path, model_input)

    # 5. CoreML推論
    coreml_output = run_coreml_inference(coreml_model_path, model_input)

    # 6. 出力比較
    compare_outputs(onnx_output, coreml_output)

    print("\n" + "=" * 80)
    print("✅ 検証完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
