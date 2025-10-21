#!/usr/bin/env python3
"""
Python参照実装の結果を正解データと比較検証
"""
from scipy.io import wavfile
import numpy as np

print("=" * 80)
print("🎵 Python参照実装 結果検証")
print("=" * 80)

# 正解データ読み込み
print("\n📂 正解データ読み込み (Hollow Crown vocals)")
sr_gt, ground_truth = wavfile.read("tests/output/hollow_crown_from_flac.wav")
gt = ground_truth.astype(np.float32) / 32768.0

print(f"  サンプルレート: {sr_gt} Hz")
print(f"  チャンネル数: {gt.shape[1] if len(gt.shape) > 1 else 1}")
print(f"  サンプル数: {len(gt)}")
print(f"  長さ: {len(gt) / sr_gt:.2f} 秒")
print(f"  RMS: {np.sqrt(np.mean(gt**2)):.6f}")
print(f"  Max: {np.abs(gt).max():.6f}")

# Python参照実装結果読み込み
print("\n📂 Python参照実装結果読み込み")
sr_py, py_result = wavfile.read("tests/python_output/hollow_crown_vocals_python.wav")
py = py_result.astype(np.float32) / 32768.0

print(f"  サンプルレート: {sr_py} Hz")
print(f"  チャンネル数: {py.shape[1] if len(py.shape) > 1 else 1}")
print(f"  サンプル数: {len(py)}")
print(f"  長さ: {len(py) / sr_py:.2f} 秒")
print(f"  RMS: {np.sqrt(np.mean(py**2)):.6f}")
print(f"  Max: {np.abs(py).max():.6f}")

# 波形相関分析 (最初の5秒)
print("\n📊 波形相関分析 (最初の5秒)")
n_5sec = min(sr_gt * 5, len(gt), len(py))

gt_5sec = gt[:n_5sec, 0]
py_5sec = py[:n_5sec, 0]

correlation = np.corrcoef(gt_5sec, py_5sec)[0, 1]
print(f"  相関係数: {correlation:.6f}")

if correlation > 0.7:
    print("  評価: ✅ 高い相関 - 抽出成功")
elif correlation > 0.3:
    print("  評価: ⚠️ 中程度の相関 - 部分的に成功")
else:
    print("  評価: ❌ 低い相関 - 抽出失敗")

# 振幅比較
print("\n📏 振幅比較")
print(f"  正解データMax: {np.abs(gt).max():.6f}")
print(f"  Python実装Max: {np.abs(py).max():.6f}")
print(f"  振幅比: {np.abs(py).max() / np.abs(gt).max():.2f}x")

# スペクトルエネルギー比較
print("\n🎼 スペクトルエネルギー比較")
gt_fft = np.fft.rfft(gt_5sec)
py_fft = np.fft.rfft(py_5sec)

gt_energy = np.abs(gt_fft).sum()
py_energy = np.abs(py_fft).sum()

print(f"  正解データ: {gt_energy:.2f}")
print(f"  Python実装: {py_energy:.2f}")
print(f"  エネルギー比: {py_energy/gt_energy:.4f}")

print("\n" + "=" * 80)
