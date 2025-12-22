#!/usr/bin/env python3
"""
UVR-MDX-NET ボーカル抽出 (ONNX)
参照: https://github.com/seanghay/uvr-mdx-infer
"""

import numpy as np
import torch
import librosa
import soundfile as sf
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm


class MDXNet:
    """UVR-MDX-NET モデルラッパー"""

    def __init__(self, model_path: str, n_fft: int = 6144, dim_f: int = 2048, dim_t: int = 8):
        self.n_fft = n_fft
        self.dim_f = dim_f
        self.dim_t = 2 ** dim_t  # 256
        self.hop = 1024
        self.sr = 44100
        self.dim_c = 4  # ステレオ x (実部+虚部)

        self.n_bins = n_fft // 2 + 1
        self.chunk_size = self.hop * (self.dim_t - 1)
        self.window = torch.hann_window(n_fft, periodic=True)

        # 周波数パディング用
        self.freq_pad = torch.zeros([1, self.dim_c, self.n_bins - self.dim_f, self.dim_t])

        # ONNXセッション
        providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(model_path), providers=providers)

        print(f"モデル読み込み完了: {Path(model_path).name}")
        print(f"  n_fft={n_fft}, dim_f={dim_f}, dim_t={self.dim_t}, hop={self.hop}")

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """STFT: 時間領域 → 周波数領域"""
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        # 複素数 → 実部/虚部
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        # [batch*2, 2, n_bins, dim_t] → [batch, 4, n_bins, dim_t]
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t])
        x = x.reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        # 周波数ビンをdim_fに制限
        return x[:, :, :self.dim_f]

    def istft(self, x: torch.Tensor) -> torch.Tensor:
        """iSTFT: 周波数領域 → 時間領域"""
        # 周波数パディングを追加
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1])
        x = torch.cat([x, freq_pad], dim=-2)
        # [batch, 4, n_bins, dim_t] → [batch*2, 2, n_bins, dim_t]
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t])
        x = x.reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        # 実部/虚部 → 複素数
        x = torch.view_as_complex(x)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
        )
        return x.reshape([-1, 2, self.chunk_size])

    def separate(self, audio_path: str, denoise: bool = True) -> tuple:
        """
        ボーカル分離実行

        Args:
            audio_path: 入力音声パス
            denoise: デノイズ有効化

        Returns:
            (vocals, instrumental, sample_rate)
        """
        print(f"\n入力: {audio_path}")

        # 音声読み込み
        mix, sr = librosa.load(audio_path, mono=False, sr=self.sr)
        if mix.ndim == 1:
            mix = np.stack([mix, mix])
        print(f"  形状: {mix.shape}, サンプルレート: {sr}")

        # 処理
        mix = mix.T  # [samples, channels]
        sources = self._demix(mix.T, denoise=denoise)

        # 出力
        separated = sources[0].T
        vocals = mix - separated  # モデルはInstrumental用なので反転
        instrumental = separated

        return vocals, instrumental, sr

    def _demix(self, mix: np.ndarray, denoise: bool, margin: int = 44100, chunks: int = 15) -> np.ndarray:
        """チャンク処理でデミックス"""
        samples = mix.shape[-1]
        chunk_size = chunks * self.sr

        if margin > chunk_size:
            margin = chunk_size

        # セグメント分割
        segmented = {}
        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1
            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)
            start = skip - s_margin
            segmented[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        # 各セグメント処理
        chunked_sources = []
        trim = self.n_fft // 2
        gen_size = self.chunk_size - 2 * trim

        for key in tqdm(segmented, desc="処理中"):
            cmix = segmented[key]
            n_sample = cmix.shape[1]
            pad = gen_size - n_sample % gen_size

            # パディング
            mix_p = np.concatenate([
                np.zeros((2, trim)),
                cmix,
                np.zeros((2, pad)),
                np.zeros((2, trim))
            ], axis=1)

            # チャンクに分割
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

                if denoise:
                    # デノイズ: 正負両方向の推論を平均
                    pred_pos = self.session.run(None, {"input": spek.numpy()})[0]
                    pred_neg = self.session.run(None, {"input": -spek.numpy()})[0]
                    spec_pred = (-pred_neg + pred_pos) * 0.5
                else:
                    spec_pred = self.session.run(None, {"input": spek.numpy()})[0]

                tar_waves = self.istft(torch.tensor(spec_pred))
                tar_signal = tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy()[:, :-pad]

            # マージン処理
            start = 0 if key == 0 else margin
            end = None if key == list(segmented.keys())[-1] else -margin
            if margin == 0:
                end = None

            chunked_sources.append(tar_signal[:, start:end])

        return np.concatenate(chunked_sources, axis=-1)[np.newaxis, :]


def main():
    # パス設定
    base_dir = Path(__file__).parent
    model_path = base_dir / "models/onnx/UVR-MDX-NET-Inst_Main.onnx"
    input_path = base_dir / "tests/output/hollow_crown.wav"
    output_dir = base_dir / "tests/python_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UVR-MDX-NET ボーカル抽出")
    print("=" * 60)

    # モデル初期化
    mdx = MDXNet(str(model_path))

    # 分離実行
    vocals, instrumental, sr = mdx.separate(str(input_path), denoise=True)

    # 保存
    vocals_path = output_dir / "hollow_crown_vocals.wav"
    inst_path = output_dir / "hollow_crown_instrumental.wav"

    sf.write(str(vocals_path), vocals, sr)
    sf.write(str(inst_path), instrumental, sr)

    print(f"\n出力:")
    print(f"  ボーカル: {vocals_path}")
    print(f"  伴奏: {inst_path}")

    # 統計
    print(f"\n統計:")
    print(f"  ボーカル RMS: {np.sqrt(np.mean(vocals**2)):.6f}")
    print(f"  伴奏 RMS: {np.sqrt(np.mean(instrumental**2)):.6f}")


if __name__ == "__main__":
    main()
