#!/usr/bin/env python3
"""
カラオケ録音環境シミュレーション

実際の録音条件を模擬:
1. 残響 (部屋の反射)
2. 背景ノイズ
3. スピーカー再生の劣化 (ローパス、歪み)
4. ボーカル/伴奏の音量バランス変更
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy import signal
import sys


def add_reverb(audio: np.ndarray, sr: int, decay: float = 0.3, delay_ms: float = 30) -> np.ndarray:
    """簡易残響を追加"""
    delay_samples = int(sr * delay_ms / 1000)
    reverb = np.zeros_like(audio)
    
    # 複数の反射を追加
    for i, (d, g) in enumerate([(1.0, 0.6), (1.5, 0.4), (2.2, 0.25), (3.0, 0.15)]):
        delay = int(delay_samples * d)
        gain = decay * g
        if delay < len(audio):
            reverb[delay:] += audio[:-delay] * gain
    
    return audio + reverb


def add_noise(audio: np.ndarray, snr_db: float = 30) -> np.ndarray:
    """ホワイトノイズを追加"""
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise


def add_clicks(audio: np.ndarray, sr: int, clicks_per_sec: float = 0.5, intensity: float = 0.3) -> np.ndarray:
    """クリック音/ポップ音を追加"""
    result = audio.copy()
    duration = len(audio) / sr
    n_clicks = int(duration * clicks_per_sec)

    for _ in range(n_clicks):
        pos = np.random.randint(0, len(audio))
        # クリック音: 短いインパルス
        click_len = np.random.randint(10, 50)
        click = np.random.randn(click_len) * intensity
        # フェードイン/アウト
        fade = np.hanning(click_len)
        click = click * fade

        end_pos = min(pos + click_len, len(audio))
        result[pos:end_pos] += click[:end_pos - pos]

    return result


def add_ambient_noise(audio: np.ndarray, sr: int, snr_db: float = 25) -> np.ndarray:
    """環境ノイズを追加（空調音、低周波ハム）"""
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    # ピンクノイズ（低周波成分が多い）
    white = np.random.randn(len(audio))
    # ローパスで低周波を強調
    nyquist = sr / 2
    b, a = signal.butter(2, 500 / nyquist, btype='low')
    pink = signal.filtfilt(b, a, white)

    # 60Hz ハム音
    t = np.arange(len(audio)) / sr
    hum = np.sin(2 * np.pi * 60 * t) * 0.3
    hum += np.sin(2 * np.pi * 120 * t) * 0.15  # 倍音

    # 合成
    ambient = pink + hum * 0.5
    ambient = ambient / np.std(ambient) * np.sqrt(noise_power)

    return audio + ambient


def simulate_speaker(audio: np.ndarray, sr: int, cutoff_hz: float = 8000) -> np.ndarray:
    """スピーカー再生をシミュレート (ローパス + 軽い歪み)"""
    # ローパスフィルタ
    nyquist = sr / 2
    normalized_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='low')
    filtered = signal.filtfilt(b, a, audio)
    
    # 軽い歪み (soft clipping)
    filtered = np.tanh(filtered * 1.2) / 1.2
    
    return filtered


def apply_degradation(
    audio: np.ndarray,
    sr: int,
    reverb_amount: float = 0.3,
    noise_snr: float = 35,
    clicks_per_sec: float = 0.0,
    click_intensity: float = 0.3,
    ambient_snr: float = 0
) -> np.ndarray:
    """音声に劣化を適用"""
    result = audio.copy()

    # 残響
    if reverb_amount > 0:
        result = add_reverb(result, sr, reverb_amount)

    # ホワイトノイズ
    if noise_snr > 0:
        result = add_noise(result, noise_snr)

    # 環境ノイズ
    if ambient_snr > 0:
        result = add_ambient_noise(result, sr, ambient_snr)

    # クリック音
    if clicks_per_sec > 0:
        result = add_clicks(result, sr, clicks_per_sec, click_intensity)

    return result


def generate_noise_components(
    length: int,
    sr: int,
    noise_snr: float,
    ambient_snr: float,
    clicks_per_sec: float,
    click_intensity: float,
    reference_power: float
) -> np.ndarray:
    """共通のノイズ成分を生成（ホワイトノイズ + 環境ノイズ + クリック）"""
    noise = np.zeros(length)

    # ホワイトノイズ
    if noise_snr > 0:
        noise_power = reference_power / (10 ** (noise_snr / 10))
        noise += np.random.normal(0, np.sqrt(noise_power), length)

    # 環境ノイズ
    if ambient_snr > 0:
        ambient_power = reference_power / (10 ** (ambient_snr / 10))
        white = np.random.randn(length)
        nyquist = sr / 2
        b, a = signal.butter(2, 500 / nyquist, btype='low')
        pink = signal.filtfilt(b, a, white)
        t = np.arange(length) / sr
        hum = np.sin(2 * np.pi * 60 * t) * 0.3 + np.sin(2 * np.pi * 120 * t) * 0.15
        ambient = pink + hum * 0.5
        ambient = ambient / np.std(ambient) * np.sqrt(ambient_power)
        noise += ambient

    # クリック音
    if clicks_per_sec > 0:
        duration = length / sr
        n_clicks = int(duration * clicks_per_sec)
        for _ in range(n_clicks):
            pos = np.random.randint(0, length)
            click_len = np.random.randint(10, 50)
            click = np.random.randn(click_len) * click_intensity
            fade = np.hanning(click_len)
            click = click * fade
            end_pos = min(pos + click_len, length)
            noise[pos:end_pos] += click[:end_pos - pos]

    return noise


def create_karaoke_mix(
    vocal: np.ndarray,
    accomp: np.ndarray,
    sr: int,
    vocal_gain: float = 1.0,
    accomp_gain: float = 0.8,
    reverb_amount: float = 0.3,
    noise_snr: float = 35,
    clicks_per_sec: float = 0.0,
    click_intensity: float = 0.3,
    ambient_snr: float = 0
) -> tuple:
    """カラオケ録音をシミュレート

    物理的に正確なシミュレーション:
    1. まずクリーンな mix を作成
    2. 残響: 各信号に個別適用（音源位置に依存）
    3. ノイズ: 共通のノイズ成分を mix, vocal, accomp に追加

    Returns:
        (mix, degraded_vocal, degraded_accomp): 劣化ミックス、劣化ボーカル、劣化伴奏
    """
    # 1. ゲイン調整
    vocal_scaled = vocal * vocal_gain
    accomp_scaled = accomp * accomp_gain

    # 2. 残響を各信号に適用（音が空間で反射する物理現象）
    if reverb_amount > 0:
        vocal_reverb = add_reverb(vocal_scaled, sr, reverb_amount)
        accomp_reverb = add_reverb(accomp_scaled, sr, reverb_amount)
    else:
        vocal_reverb = vocal_scaled
        accomp_reverb = accomp_scaled

    # 3. 残響適用後に mix を作成
    mix_with_reverb = vocal_reverb + accomp_reverb

    # 4. 共通のノイズ成分を生成（録音環境からのノイズ）
    reference_power = np.mean(mix_with_reverb ** 2)
    common_noise = generate_noise_components(
        length=len(mix_with_reverb),
        sr=sr,
        noise_snr=noise_snr,
        ambient_snr=ambient_snr,
        clicks_per_sec=clicks_per_sec,
        click_intensity=click_intensity,
        reference_power=reference_power
    )

    # 5. 同じノイズを mix, vocal, accomp に追加
    mix_degraded = mix_with_reverb + common_noise
    vocal_degraded = vocal_reverb + common_noise
    accomp_degraded = accomp_reverb + common_noise

    # 正規化（mix 基準で全体を揃える）
    max_val = np.max(np.abs(mix_degraded))
    if max_val > 0:
        scale = 0.9 / max_val
        mix_degraded = mix_degraded * scale
        vocal_degraded = vocal_degraded * scale
        accomp_degraded = accomp_degraded * scale

    return mix_degraded, vocal_degraded, accomp_degraded


def process_sample(sample_dir: Path, output_dir: Path, condition: str, **params):
    """1サンプルを処理"""
    vocal, sr = librosa.load(str(sample_dir / "vocal.wav"), mono=True, sr=None)
    accomp, _ = librosa.load(str(sample_dir / "accomp.wav"), mono=True, sr=None)

    # 長さを揃える
    min_len = min(len(vocal), len(accomp))
    vocal = vocal[:min_len]
    accomp = accomp[:min_len]

    # シミュレーション（ミックス、劣化ボーカル、劣化伴奏を取得）
    mix, degraded_vocal, degraded_accomp = create_karaoke_mix(vocal, accomp, sr, **params)

    # 保存ディレクトリ作成
    sample_output = output_dir / sample_dir.name
    sample_output.mkdir(parents=True, exist_ok=True)

    # ステレオ化して保存
    # 1. 劣化ミックス（分離モデルへの入力）
    mix_stereo = np.stack([mix, mix])
    sf.write(str(sample_output / "mix.wav"), mix_stereo.T, sr)

    # 2. 劣化ボーカル（正解データ）
    vocal_stereo = np.stack([degraded_vocal, degraded_vocal])
    sf.write(str(sample_output / "vocal.wav"), vocal_stereo.T, sr)

    # 3. 劣化伴奏（正解データ）
    accomp_stereo = np.stack([degraded_accomp, degraded_accomp])
    sf.write(str(sample_output / "accomp.wav"), accomp_stereo.T, sr)

    return True


def main():
    base_dir = Path(__file__).parent.parent
    test_audio_dir = base_dir / "test_audio"
    
    # 条件設定
    conditions = {
        "mild": {  # 軽度劣化
            "vocal_gain": 1.0,
            "accomp_gain": 0.7,
            "reverb_amount": 0.15,
            "noise_snr": 40,
            "clicks_per_sec": 0.0,
            "ambient_snr": 0
        },
        "moderate": {  # 中程度劣化
            "vocal_gain": 1.0,
            "accomp_gain": 0.8,
            "reverb_amount": 0.3,
            "noise_snr": 30,
            "clicks_per_sec": 0.2,
            "click_intensity": 0.2,
            "ambient_snr": 30
        },
        "severe": {  # 重度劣化
            "vocal_gain": 1.0,
            "accomp_gain": 1.0,
            "reverb_amount": 0.5,
            "noise_snr": 20,
            "clicks_per_sec": 0.5,
            "click_intensity": 0.4,
            "ambient_snr": 20
        },
        "realistic": {  # 現実的なカラオケ環境
            "vocal_gain": 1.0,
            "accomp_gain": 0.9,
            "reverb_amount": 0.4,
            "noise_snr": 15,
            "clicks_per_sec": 1.0,
            "click_intensity": 0.5,
            "ambient_snr": 15
        }
    }
    
    # コマンドライン引数から条件を取得
    condition = sys.argv[1] if len(sys.argv) > 1 else "moderate"
    if condition not in conditions:
        print(f"条件: {', '.join(conditions.keys())}")
        sys.exit(1)
    
    params = conditions[condition]
    output_dir = base_dir / "test_audio_simulated" / condition
    
    print(f"カラオケ録音シミュレーション: {condition}")
    print("=" * 50)
    print(f"パラメータ: {params}")
    print()
    
    sample_dirs = sorted([
        d for d in test_audio_dir.iterdir()
        if d.is_dir() and d.name != "raw" and (d / "vocal.wav").exists()
    ])
    
    for sample_dir in sample_dirs:
        process_sample(sample_dir, output_dir, condition, **params)
        print(f"✓ {sample_dir.name}")
    
    print(f"\n出力: {output_dir}")


if __name__ == "__main__":
    main()
