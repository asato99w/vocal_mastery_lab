#!/usr/bin/env python3
"""
ボーカル分離精度評価スクリプト（評価のみ）

保存された分離結果と正解データを比較し、精度を評価します。
事前に separate_vocals.py で出力を保存しておく必要があります。

使用方法:
    python scripts/evaluate_accuracy.py <reference_dir> <estimation_dir>

例:
    python scripts/evaluate_accuracy.py test_audio output/python
"""

import sys
from pathlib import Path
import numpy as np
import librosa


def calculate_sdr(reference: np.ndarray, estimated: np.ndarray) -> float:
    """SDR (Signal-to-Distortion Ratio) を計算"""
    min_len = min(len(reference), len(estimated))
    reference = reference[:min_len]
    estimated = estimated[:min_len]

    noise = reference - estimated
    signal_power = np.sum(reference ** 2)
    noise_power = np.sum(noise ** 2)

    if noise_power < 1e-10:
        return float('inf')

    return 10 * np.log10(signal_power / noise_power)


def evaluate_sample(ref_path: Path, est_path: Path, sr: int = 44100) -> dict:
    """1つのサンプルを評価"""
    if not ref_path.exists():
        print(f"  正解ファイルなし: {ref_path}")
        return None
    if not est_path.exists():
        print(f"  分離結果なし: {est_path}")
        return None

    # 読み込み（モノラル）
    vocals_ref, _ = librosa.load(str(ref_path), mono=True, sr=sr)
    vocals_est, _ = librosa.load(str(est_path), mono=True, sr=sr)

    # 長さを揃える
    min_len = min(len(vocals_ref), len(vocals_est))
    vocals_ref = vocals_ref[:min_len]
    vocals_est = vocals_est[:min_len]

    # SDR計算
    sdr = calculate_sdr(vocals_ref, vocals_est)

    # 相関係数
    corr = np.corrcoef(vocals_ref.flatten(), vocals_est.flatten())[0, 1]

    return {
        'sdr': sdr,
        'correlation': corr,
    }


def main():
    base_dir = Path(__file__).parent.parent

    if len(sys.argv) < 3:
        print("使用方法: python scripts/evaluate_accuracy.py <reference_dir> <estimation_dir>")
        print("例: python scripts/evaluate_accuracy.py test_audio output/python")
        sys.exit(1)

    ref_dir = Path(sys.argv[1])
    est_dir = Path(sys.argv[2])

    if not ref_dir.is_absolute():
        ref_dir = base_dir / sys.argv[1]
    if not est_dir.is_absolute():
        est_dir = base_dir / sys.argv[2]

    if not ref_dir.exists():
        print(f"エラー: {ref_dir} が見つかりません")
        sys.exit(1)
    if not est_dir.exists():
        print(f"エラー: {est_dir} が見つかりません")
        print("先に separate_vocals.py を実行してください")
        sys.exit(1)

    print("=" * 70)
    print("ボーカル分離精度評価")
    print("=" * 70)
    print(f"正解データ: {ref_dir}")
    print(f"分離結果:   {est_dir}")
    print("-" * 70)

    # サンプル一覧（正解ディレクトリから取得）
    sample_names = sorted([
        d.name for d in ref_dir.iterdir()
        if d.is_dir() and d.name != "raw" and (d / "vocal.wav").exists()
    ])

    if not sample_names:
        print("評価可能なサンプルがありません")
        sys.exit(1)

    print(f"サンプル数: {len(sample_names)}\n")

    results = []
    for sample_name in sample_names:
        ref_path = ref_dir / sample_name / "vocal.wav"
        est_path = est_dir / sample_name / "vocals.wav"

        print(f"{sample_name}", end="")
        metrics = evaluate_sample(ref_path, est_path)
        if metrics:
            metrics['sample'] = sample_name
            results.append(metrics)
            print(f" → SDR: {metrics['sdr']:.2f} dB, 相関: {metrics['correlation']:.4f}")
        else:
            print(" → スキップ")

    # 集計
    if results:
        print("\n" + "=" * 70)
        print("集計結果")
        print("=" * 70)

        sdrs = [r['sdr'] for r in results]
        corrs = [r['correlation'] for r in results]

        print(f"\nSDR: 平均 {np.mean(sdrs):.2f} dB, 中央値 {np.median(sdrs):.2f} dB")
        print(f"相関: 平均 {np.mean(corrs):.4f}, 中央値 {np.median(corrs):.4f}")


if __name__ == "__main__":
    main()
