#!/usr/bin/env python3
"""
MIR-1K データセットからテストサンプルを準備

MIR-1K形式:
  - ステレオ WAV (16kHz)
  - 左チャンネル: 伴奏
  - 右チャンネル: ボーカル

出力形式:
  - 44.1kHz モノラル/ステレオ WAV
  - mix.wav: ミックス音源
  - vocal.wav: ボーカル
  - accomp.wav: 伴奏
"""

import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf


def prepare_sample(input_wav: Path, output_dir: Path, target_sr: int = 44100):
    """MIR-1Kファイルをテストサンプルに変換"""
    # 読み込み (ステレオ)
    audio, sr = librosa.load(str(input_wav), mono=False, sr=None)
    
    if audio.ndim == 1:
        print(f"警告: {input_wav.name} はモノラルです")
        return False
    
    # 左=伴奏, 右=ボーカル
    accomp = audio[0]
    vocal = audio[1]
    
    # ミックス作成
    mix = (accomp + vocal) / 2
    
    # リサンプル
    if sr != target_sr:
        accomp = librosa.resample(accomp, orig_sr=sr, target_sr=target_sr)
        vocal = librosa.resample(vocal, orig_sr=sr, target_sr=target_sr)
        mix = librosa.resample(mix, orig_sr=sr, target_sr=target_sr)
    
    # ステレオ化 (モデル入力用)
    mix_stereo = np.stack([mix, mix])
    vocal_stereo = np.stack([vocal, vocal])
    accomp_stereo = np.stack([accomp, accomp])
    
    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_dir / "mix.wav"), mix_stereo.T, target_sr)
    sf.write(str(output_dir / "vocal.wav"), vocal_stereo.T, target_sr)
    sf.write(str(output_dir / "accomp.wav"), accomp_stereo.T, target_sr)
    
    return True


def main():
    base_dir = Path(__file__).parent.parent
    mir1k_dir = base_dir / "test_audio" / "raw" / "MIR-1K" / "Wavfile"
    output_base = base_dir / "test_audio"
    
    if not mir1k_dir.exists():
        print(f"エラー: {mir1k_dir} が見つかりません")
        sys.exit(1)
    
    # 処理するファイル
    if len(sys.argv) > 1:
        # 指定されたファイルのみ
        files = [mir1k_dir / f"{sys.argv[1]}.wav"]
    else:
        # 最初の5つをサンプルとして
        files = sorted(mir1k_dir.glob("*.wav"))[:5]
    
    print(f"MIR-1K → テストサンプル変換")
    print("=" * 50)
    
    for wav_file in files:
        if not wav_file.exists():
            print(f"スキップ: {wav_file.name} (存在しません)")
            continue
            
        sample_name = wav_file.stem
        output_dir = output_base / sample_name
        
        print(f"変換中: {wav_file.name} → {sample_name}/")
        if prepare_sample(wav_file, output_dir):
            print(f"  ✓ mix.wav, vocal.wav, accomp.wav")
    
    print("\n完了")


if __name__ == "__main__":
    main()
