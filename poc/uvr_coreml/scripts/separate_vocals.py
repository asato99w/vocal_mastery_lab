#!/usr/bin/env python3
"""
ボーカル分離スクリプト (Python/ONNX 参照実装)

使用方法:
    python scripts/separate_vocals.py <input_dir> <output_dir>

例:
    python scripts/separate_vocals.py test_audio output/python
    python scripts/separate_vocals.py test_audio_simulated/mild output/simulated_mild
"""

import sys
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf
import onnxruntime as ort


class MDXNetSeparator:
    """UVR-MDX-NET ボーカル分離器"""

    def __init__(
        self,
        model_path: str,
        n_fft: int = 6144,
        dim_f: int = 3072,
        dim_t: int = 8
    ):
        self.n_fft = n_fft
        self.dim_f = dim_f
        self.dim_t = 2 ** dim_t  # 256
        self.hop = 1024
        self.sr = 44100
        self.dim_c = 4

        self.n_bins = n_fft // 2 + 1
        self.chunk_size = self.hop * (self.dim_t - 1)
        self.window = torch.hann_window(n_fft, periodic=True)
        self.freq_pad = torch.zeros([1, self.dim_c, self.n_bins - self.dim_f, self.dim_t])

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )
        print(f"モデル読み込み完了: {Path(model_path).name}")

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """STFT: 時間領域 → 周波数領域"""
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t])
        x = x.reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, x: torch.Tensor) -> torch.Tensor:
        """iSTFT: 周波数領域 → 時間領域"""
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1])
        x = torch.cat([x, freq_pad], dim=-2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t])
        x = x.reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1]).contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True
        )
        return x.reshape([-1, 2, self.chunk_size])

    def separate(self, audio_path: str) -> tuple:
        """
        ボーカル分離実行

        Returns:
            (vocals, instrumental, sample_rate)
        """
        # 音声読み込み
        mix, sr = librosa.load(audio_path, mono=False, sr=self.sr)
        if mix.ndim == 1:
            mix = np.stack([mix, mix])
        print(f"入力: {mix.shape}, SR: {sr}")

        # パディング計算
        trim = self.n_fft // 2
        gen_size = self.chunk_size - 2 * trim
        n_sample = mix.shape[1]
        pad = gen_size - n_sample % gen_size

        mix_p = np.concatenate([
            np.zeros((2, trim)),
            mix,
            np.zeros((2, pad)),
            np.zeros((2, trim))
        ], axis=1)

        # チャンク分割
        mix_waves = []
        i = 0
        while i < n_sample + pad:
            waves = mix_p[:, i:i + self.chunk_size]
            mix_waves.append(waves)
            i += gen_size

        mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float32)

        # 推論
        with torch.no_grad():
            spek = self.stft(mix_waves)
            print(f"STFT出力: {spek.shape}")

            pred = self.session.run(None, {"input": spek.numpy()})[0]
            print(f"モデル出力: {pred.shape}")

            tar_waves = self.istft(torch.tensor(pred))
            tar_signal = tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy()[:, :-pad]

        # Voc_FT モデルはボーカルを直接出力
        vocals = tar_signal[:, :n_sample]
        instrumental = mix - vocals

        return vocals.T, instrumental.T, sr


def main():
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "models" / "onnx" / "UVR-MDX-NET-Voc_FT.onnx"

    if len(sys.argv) < 3:
        print("使用方法: python scripts/separate_vocals.py <input_dir> <output_dir>")
        print("例: python scripts/separate_vocals.py test_audio output/python")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.is_absolute():
        input_dir = base_dir / sys.argv[1]
    if not output_dir.is_absolute():
        output_dir = base_dir / sys.argv[2]

    if not input_dir.exists():
        print(f"エラー: {input_dir} が見つかりません")
        sys.exit(1)

    print("=" * 60)
    print("ボーカル分離 (Python/ONNX)")
    print("=" * 60)
    print(f"入力: {input_dir}")
    print(f"出力: {output_dir}")

    # モデル読み込み
    separator = MDXNetSeparator(str(model_path))

    # サンプル一覧
    sample_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name != "raw" and (d / "mix.wav").exists()
    ])

    if not sample_dirs:
        print("処理可能なサンプルがありません (mix.wav が必要)")
        sys.exit(1)

    print(f"\nサンプル数: {len(sample_dirs)}")
    print("-" * 60)

    for sample_dir in sample_dirs:
        mix_path = sample_dir / "mix.wav"
        sample_output = output_dir / sample_dir.name
        sample_output.mkdir(parents=True, exist_ok=True)

        print(f"\n{sample_dir.name}")
        vocals, instrumental, sr = separator.separate(str(mix_path))

        # 保存
        sf.write(str(sample_output / "vocals.wav"), vocals, sr)
        sf.write(str(sample_output / "instrumental.wav"), instrumental, sr)

    print(f"\n完了: {output_dir}")


if __name__ == "__main__":
    main()
