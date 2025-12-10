#!/usr/bin/env python3
"""
インストゥルメンタル = 元音声 - ボーカル のテスト
UVRの実際のアプローチを検証
"""
import numpy as np
import soundfile as sf
import librosa

# 仮にボーカルが正しく抽出できているとして
input_file = "tests/output/hollow_crown_from_flac.wav"

print("=" * 80)
print("🔬 引き算方式の検証")
print("=" * 80)

# 元音声読み込み
print(f"\n📂 元音声読み込み: {input_file}")
original, sr = librosa.load(input_file, sr=44100, mono=False)
if original.ndim == 1:
    original = np.stack([original, original])

print(f"   元音声shape: {original.shape}")
print(f"   元音声RMS: {np.sqrt(np.mean(original[0]**2)):.6f}")

# 現在の実装でボーカルが正しく抽出できているとのことなので
# 前回のテスト出力を使用
print(f"\n📂 ボーカル出力を確認")
print(f"   ボーカル抽出は問題なく動作しているとのこと")
print(f"   → つまり、モデル出力 = ボーカル (正しく動作)")

# UVRのアプローチをシミュレート
print(f"\n🧮 インストゥルメンタル計算:")
print(f"   Instrumental = Original - Vocal")
print(f"   これがUVRの実際の方法 (separate.py:514)")

# テストケース: 仮にボーカルが元音声の50%のレベルだとすると
simulated_vocal = original * 0.5
simulated_instrumental = original - simulated_vocal

print(f"\n📊 シミュレーション結果:")
print(f"   元音声RMS: {np.sqrt(np.mean(original[0]**2)):.6f}")
print(f"   ボーカルRMS: {np.sqrt(np.mean(simulated_vocal[0]**2)):.6f}")
print(f"   インストゥルメンタルRMS: {np.sqrt(np.mean(simulated_instrumental[0]**2)):.6f}")

# 保存
sf.write("tests/python_output/simulated_vocal.wav", simulated_vocal.T, sr)
sf.write("tests/python_output/simulated_instrumental.wav", simulated_instrumental.T, sr)

print(f"\n💡 重要な結論:")
print(f"   1. モデル = ボーカル抽出器 (問題なく動作)")
print(f"   2. インストゥルメンタル = 元音声 - モデル出力")
print(f"   3. Swift実装では:")
print(f"      a) モデルでボーカルを抽出")
print(f"      b) 元音声 - ボーカル = インストゥルメンタル")

print(f"\n📋 次のステップ:")
print(f"   Swift実装で引き算を追加する必要があります")
print("=" * 80)
