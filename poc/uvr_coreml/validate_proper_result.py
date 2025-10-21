#!/usr/bin/env python3
"""
VocalSeparatorCompleteの結果を正解データと比較検証
"""
from scipy.io import wavfile
import numpy as np

print("=" * 80)
print("🎵 VocalSeparatorComplete 結果検証")
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

# VocalSeparatorComplete結果読み込み
print("\n📂 VocalSeparatorComplete結果読み込み")
sr_proper, proper_result = wavfile.read("tests/swift_output/hollow_crown_vocals_proper.wav")
proper = proper_result.astype(np.float32) / 32768.0

print(f"  サンプルレート: {sr_proper} Hz")
print(f"  チャンネル数: {proper.shape[1] if len(proper.shape) > 1 else 1}")
print(f"  サンプル数: {len(proper)}")
print(f"  長さ: {len(proper) / sr_proper:.2f} 秒")
print(f"  RMS: {np.sqrt(np.mean(proper**2)):.6f}")
print(f"  Max: {np.abs(proper).max():.6f}")

# 基本的な品質チェック
print("\n🔍 基本品質チェック")
print(f"  無音チェック: {'❌ 無音' if np.abs(proper).max() < 0.001 else '✅ 音声あり'}")
print(f"  振幅範囲: {proper.min():.6f} ~ {proper.max():.6f}")

# 最初の10サンプル確認
print(f"\n  最初の10サンプル (左チャンネル):")
print(f"    正解: {gt[:10, 0]}")
print(f"    結果: {proper[:10, 0]}")

# 波形相関分析 (最初の5秒)
print("\n📊 波形相関分析 (最初の5秒)")
n_5sec = min(sr_gt * 5, len(gt), len(proper))

gt_5sec = gt[:n_5sec, 0]
proper_5sec = proper[:n_5sec, 0]

correlation = np.corrcoef(gt_5sec, proper_5sec)[0, 1]
print(f"  相関係数: {correlation:.6f}")

if correlation > 0.7:
    print("  評価: ✅ 高い相関 - 抽出成功の可能性が高い")
elif correlation > 0.3:
    print("  評価: ⚠️ 中程度の相関 - 部分的に成功")
else:
    print("  評価: ❌ 低い相関 - 抽出失敗の可能性")

# スペクトル比較
print("\n🎼 スペクトルエネルギー比較")
gt_fft = np.fft.rfft(gt_5sec)
proper_fft = np.fft.rfft(proper_5sec)

gt_energy = np.abs(gt_fft).sum()
proper_energy = np.abs(proper_fft).sum()

print(f"  正解データ: {gt_energy:.2f}")
print(f"  抽出結果: {proper_energy:.2f}")
print(f"  エネルギー比: {proper_energy/gt_energy:.4f}")

# 周波数帯域別分析
print("\n🎛️ 周波数帯域別エネルギー")
freqs = np.fft.rfftfreq(len(gt_5sec), 1/sr_gt)

bands = [
    ("低域 (0-500Hz)", 0, 500),
    ("中低域 (500-2kHz)", 500, 2000),
    ("中域 (2k-5kHz)", 2000, 5000),
    ("高域 (5k-10kHz)", 5000, 10000),
]

for name, low, high in bands:
    mask = (freqs >= low) & (freqs < high)
    gt_band_energy = np.abs(gt_fft[mask]).sum()
    proper_band_energy = np.abs(proper_fft[mask]).sum()

    if gt_band_energy > 0:
        ratio = proper_band_energy / gt_band_energy
        print(f"  {name}: {ratio:.4f}")

# セグメント別相関分析
print("\n📈 セグメント別相関分析")
segment_duration = 10  # 10秒ごと
n_segments = min(len(gt), len(proper)) // (sr_gt * segment_duration)

correlations = []
for i in range(n_segments):
    start = i * sr_gt * segment_duration
    end = start + sr_gt * segment_duration

    gt_seg = gt[start:end, 0]
    proper_seg = proper[start:end, 0]

    seg_corr = np.corrcoef(gt_seg, proper_seg)[0, 1]
    correlations.append(seg_corr)

    print(f"  セグメント {i+1} ({i*segment_duration}-{(i+1)*segment_duration}秒): {seg_corr:.6f}")

avg_correlation = np.mean(correlations)
print(f"\n  平均相関係数: {avg_correlation:.6f}")
print(f"  標準偏差: {np.std(correlations):.6f}")
print(f"  最小: {np.min(correlations):.6f}")
print(f"  最大: {np.max(correlations):.6f}")

# 総合評価
print("\n" + "=" * 80)
print("📋 総合評価")
print("=" * 80)

score = 0
max_score = 5

# 1. 無音チェック
if np.abs(proper).max() > 0.01:
    score += 1
    print("✅ [1/5] 音声出力確認")
else:
    print("❌ [0/5] 音声出力なし")

# 2. 相関係数
if avg_correlation > 0.7:
    score += 1
    print("✅ [1/5] 高い波形相関")
elif avg_correlation > 0.3:
    score += 0.5
    print("⚠️ [0.5/5] 中程度の波形相関")
else:
    print("❌ [0/5] 低い波形相関")

# 3. エネルギー比
energy_ratio = proper_energy / gt_energy
if 0.5 < energy_ratio < 2.0:
    score += 1
    print("✅ [1/5] 適切なエネルギーレベル")
elif 0.1 < energy_ratio < 5.0:
    score += 0.5
    print("⚠️ [0.5/5] エネルギーレベル許容範囲")
else:
    print("❌ [0/5] エネルギーレベル異常")

# 4. 振幅範囲
if np.abs(proper).max() > 0.1:
    score += 1
    print("✅ [1/5] 十分な振幅")
elif np.abs(proper).max() > 0.01:
    score += 0.5
    print("⚠️ [0.5/5] 低い振幅")
else:
    print("❌ [0/5] 振幅不足")

# 5. 一貫性
if np.std(correlations) < 0.2:
    score += 1
    print("✅ [1/5] 安定した品質")
elif np.std(correlations) < 0.4:
    score += 0.5
    print("⚠️ [0.5/5] 中程度の安定性")
else:
    print("❌ [0/5] 品質が不安定")

print(f"\n総合スコア: {score:.1f}/{max_score}")

if score >= 4:
    print("🎉 評価: 優秀 - 抽出は成功しています")
elif score >= 3:
    print("👍 評価: 良好 - 抽出はおおむね成功")
elif score >= 2:
    print("⚠️ 評価: 要改善 - 抽出に問題がある可能性")
else:
    print("❌ 評価: 失敗 - 抽出は失敗している可能性が高い")

print("=" * 80)
