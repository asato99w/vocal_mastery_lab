#!/usr/bin/env python3
"""
音声ファイル変換スクリプト

任意の音声形式を44.1kHz WAVに変換します。

使用方法:
    python scripts/convert_audio.py input.mp3 [output.wav] [--duration 15]

例:
    python scripts/convert_audio.py test_audio/raw/song.mp3
    python scripts/convert_audio.py song.mp3 test_audio/wav/song_15s.wav --duration 15
"""

import argparse
import sys
from pathlib import Path

import librosa
import soundfile as sf


def convert_audio(
    input_path: Path,
    output_path: Path,
    target_sr: int = 44100,
    duration: float = None
):
    """
    音声ファイルを変換

    Args:
        input_path: 入力ファイルパス
        output_path: 出力ファイルパス
        target_sr: 目標サンプルレート
        duration: 切り出す秒数（Noneで全体）
    """
    print(f"読み込み中: {input_path}")

    # 読み込み
    audio, sr = librosa.load(str(input_path), mono=False, sr=target_sr, duration=duration)

    # モノラルの場合はステレオに
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
        audio = librosa.core.resample(audio, orig_sr=sr, target_sr=target_sr)
        audio = audio.repeat(2, axis=0)

    print(f"  形状: {audio.shape}")
    print(f"  サンプルレート: {target_sr}")
    print(f"  長さ: {audio.shape[1] / target_sr:.2f}秒")

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio.T, target_sr)
    print(f"保存完了: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="音声ファイルを44.1kHz WAVに変換")
    parser.add_argument("input", help="入力ファイルパス")
    parser.add_argument("output", nargs="?", help="出力ファイルパス（省略時は自動生成）")
    parser.add_argument("--duration", "-d", type=float, help="切り出す秒数")
    parser.add_argument("--sr", type=int, default=44100, help="サンプルレート（デフォルト: 44100）")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: {input_path} が見つかりません")
        sys.exit(1)

    # 出力パス決定
    if args.output:
        output_path = Path(args.output)
    else:
        base_dir = Path(__file__).parent.parent
        suffix = f"_{int(args.duration)}s" if args.duration else ""
        output_path = base_dir / "test_audio" / "wav" / f"{input_path.stem}{suffix}.wav"

    convert_audio(input_path, output_path, args.sr, args.duration)


if __name__ == "__main__":
    main()
