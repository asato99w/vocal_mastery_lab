#!/usr/bin/env python3
"""
ノイズ分析: 抽出結果のノイズ特性を調査
"""
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

print("=" * 80)
print("🔍 ノイズ分析")
print("=" * 80)

# 正解データ読み込み
print("\n📂 データ読み込み")
sr_gt, gt_data = wavfile.read("tests/output/hollow_crown_from_flac.wav")
gt = gt_data.astype(np.float32) / 32768.0

sr_result, result_data = wavfile.read("tests/swift_output/hollow_crown_vocals_proper.wav")
result = result_data.astype(np.float32) / 32768.0

print(f"正解データ - RMS: {np.sqrt(np.mean(gt**2)):.6f}, Max: {np.abs(gt).max():.6f}")
print(f"抽出結果 - RMS: {np.sqrt(np.mean(result**2)):.6f}, Max: {np.abs(result).max():.6f}")

# 無音部分の検出 (正解データで振幅が非常に小さい部分)
print("\n🔇 無音部分でのノイズ分析")
threshold = 0.001  # 無音と見なす閾値

gt_mono = gt[:, 0]
result_mono = result[:, 0]

# 長さを合わせる
min_len = min(len(gt_mono), len(result_mono))
gt_mono = gt_mono[:min_len]
result_mono = result_mono[:min_len]

# 正解データで無音の部分を検出
silent_mask = np.abs(gt_mono) < threshold
silent_samples = result_mono[silent_mask]

if len(silent_samples) > 0:
    print(f"  無音サンプル数: {len(silent_samples):,}")
    print(f"  無音部分のRMS: {np.sqrt(np.mean(silent_samples**2)):.6f}")
    print(f"  無音部分のMax: {np.abs(silent_samples).max():.6f}")
    print(f"  無音部分のMin: {silent_samples.min():.6f}")
else:
    print("  無音部分が見つかりませんでした")

# 差分信号分析 (ノイズ成分の推定)
print("\n📊 差分信号分析 (最初の5秒)")
n_5sec = min(sr_gt * 5, len(gt_mono), len(result_mono))

gt_5sec = gt_mono[:n_5sec]
result_5sec = result_mono[:n_5sec]

# 振幅を正規化して比較
gt_5sec_norm = gt_5sec / (np.abs(gt_5sec).max() + 1e-8)
result_5sec_norm = result_5sec / (np.abs(result_5sec).max() + 1e-8)

diff = result_5sec_norm - gt_5sec_norm

print(f"  差分RMS: {np.sqrt(np.mean(diff**2)):.6f}")
print(f"  差分Max: {np.abs(diff).max():.6f}")

# スペクトル分析
print("\n🎼 スペクトル分析")

# FFT
gt_fft = np.fft.rfft(gt_5sec)
result_fft = np.fft.rfft(result_5sec)
freqs = np.fft.rfftfreq(len(gt_5sec), 1/sr_gt)

# 高周波ノイズのチェック (10kHz以上)
high_freq_mask = freqs > 10000
if np.any(high_freq_mask):
    gt_high = np.abs(gt_fft[high_freq_mask]).mean()
    result_high = np.abs(result_fft[high_freq_mask]).mean()
    print(f"  高周波(>10kHz)平均:")
    print(f"    正解: {gt_high:.6f}")
    print(f"    結果: {result_high:.6f}")
    print(f"    比率: {result_high / (gt_high + 1e-8):.2f}x")

# 最初の数サンプルの詳細確認
print("\n🔬 サンプル詳細 (最初の50サンプル)")
print(f"  正解データ:")
print(f"    値の範囲: {gt_mono[:50].min():.6f} ~ {gt_mono[:50].max():.6f}")
print(f"    最初の10個: {gt_mono[:10]}")

print(f"  抽出結果:")
print(f"    値の範囲: {result_mono[:50].min():.6f} ~ {result_mono[:50].max():.6f}")
print(f"    最初の10個: {result_mono[:10]}")

# 異常値の検出
print("\n⚠️ 異常値検出")
nan_count = np.isnan(result_mono).sum()
inf_count = np.isinf(result_mono).sum()
print(f"  NaN数: {nan_count}")
print(f"  Inf数: {inf_count}")

# 急激な変化の検出
diff_signal = np.diff(result_mono)
large_jumps = np.abs(diff_signal) > 0.5  # 急激な変化
print(f"  急激な変化(>0.5): {large_jumps.sum():,} 箇所")

# クリッピングの検出
clipping_threshold = 0.99
clipped = np.abs(result_mono) > clipping_threshold
print(f"  クリッピング(>{clipping_threshold}): {clipped.sum():,} サンプル")

# DCオフセットの確認
dc_offset = np.mean(result_mono)
print(f"  DCオフセット: {dc_offset:.6f}")

# 時系列での変化を確認
print("\n📈 セグメント別SNR推定")
segment_duration = 10  # 10秒ごと
n_segments = min(5, len(gt_mono) // (sr_gt * segment_duration))

for i in range(n_segments):
    start = i * sr_gt * segment_duration
    end = start + sr_gt * segment_duration

    gt_seg = gt_mono[start:end]
    result_seg = result_mono[start:end]

    # 正規化
    gt_seg_norm = gt_seg / (np.abs(gt_seg).max() + 1e-8)
    result_seg_norm = result_seg / (np.abs(result_seg).max() + 1e-8)

    # ノイズ推定 (差分のRMS)
    noise = result_seg_norm - gt_seg_norm
    noise_rms = np.sqrt(np.mean(noise**2))
    signal_rms = np.sqrt(np.mean(result_seg_norm**2))

    if noise_rms > 0:
        snr_estimate = 20 * np.log10(signal_rms / noise_rms)
    else:
        snr_estimate = float('inf')

    print(f"  セグメント {i+1} ({i*segment_duration}-{(i+1)*segment_duration}秒):")
    print(f"    ノイズRMS: {noise_rms:.6f}")
    print(f"    推定SNR: {snr_estimate:.2f} dB")

print("\n" + "=" * 80)
print("🔍 ノイズ原因の推測")
print("=" * 80)

issues = []

if nan_count > 0 or inf_count > 0:
    issues.append("❌ NaN/Inf値が存在 → 数値計算エラー")

if clipped.sum() > 1000:
    issues.append("❌ 大量のクリッピング → ゲインが大きすぎる")

if abs(dc_offset) > 0.01:
    issues.append("⚠️ DCオフセット大 → iSTFT実装の問題")

if len(silent_samples) > 0 and np.sqrt(np.mean(silent_samples**2)) > 0.01:
    issues.append("❌ 無音部分にノイズ大 → バックグラウンドノイズ")

if large_jumps.sum() > len(result_mono) * 0.01:
    issues.append("⚠️ 急激な変化が多い → ウィンドウ処理の問題")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✅ 明確な異常は検出されませんでした")
    print("   → より詳細な周波数領域分析が必要かもしれません")

print("=" * 80)
