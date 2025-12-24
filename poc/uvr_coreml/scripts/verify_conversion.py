#!/usr/bin/env python3
"""
CoreML変換検証スクリプト

Python (ONNX) と CoreML の出力を比較し、変換精度を検証します。

使用方法:
    python scripts/verify_conversion.py [input.wav]

例:
    python scripts/verify_conversion.py test_audio/hollow_crown/mix.wav
"""

import sys
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf
import onnxruntime as ort
import coremltools as ct


class MDXNetBase:
    """MDX-Net 共通処理"""

    def __init__(self, n_fft: int = 6144, dim_f: int = 3072, dim_t: int = 8):
        self.n_fft = n_fft
        self.dim_f = dim_f
        self.dim_t = 2 ** dim_t
        self.hop = 1024
        self.sr = 44100
        self.dim_c = 4
        self.n_bins = n_fft // 2 + 1
        self.chunk_size = self.hop * (self.dim_t - 1)
        self.window = torch.hann_window(n_fft, periodic=True)
        self.freq_pad = torch.zeros([1, self.dim_c, self.n_bins - self.dim_f, self.dim_t])

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                       window=self.window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t])
        x = x.reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, x):
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1])
        x = torch.cat([x, freq_pad], dim=-2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t])
        x = x.reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1]).contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop,
                        window=self.window, center=True)
        return x.reshape([-1, 2, self.chunk_size])

    def prepare_audio(self, audio_path: str):
        mix, sr = librosa.load(audio_path, mono=False, sr=self.sr)
        if mix.ndim == 1:
            mix = np.stack([mix, mix])

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

        mix_waves = []
        i = 0
        while i < n_sample + pad:
            waves = mix_p[:, i:i + self.chunk_size]
            mix_waves.append(waves)
            i += gen_size

        return mix, torch.tensor(np.array(mix_waves), dtype=torch.float32), n_sample, pad, trim


def separate_with_onnx(model_path: str, audio_path: str, output_dir: Path):
    """ONNX で分離"""
    base = MDXNetBase()
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    mix, mix_waves, n_sample, pad, trim = base.prepare_audio(audio_path)

    with torch.no_grad():
        spek = base.stft(mix_waves)
        pred = session.run(None, {"input": spek.numpy()})[0]
        tar_waves = base.istft(torch.tensor(pred))
        tar_signal = tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy()[:, :-pad]

    # Voc_FT モデルはボーカルを直接出力
    vocals = tar_signal[:, :n_sample]
    instrumental = mix - vocals

    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_dir / "vocals.wav"), vocals.T, base.sr)
    sf.write(str(output_dir / "instrumental.wav"), instrumental.T, base.sr)

    return vocals.T


def separate_with_coreml(model_path: str, audio_path: str, output_dir: Path):
    """CoreML で分離"""
    base = MDXNetBase()
    model = ct.models.MLModel(model_path)

    mix, mix_waves, n_sample, pad, trim = base.prepare_audio(audio_path)

    with torch.no_grad():
        spek = base.stft(mix_waves)

        # CoreML推論（バッチ処理）
        # 入力名を動的に取得
        spec = model.get_spec()
        input_name = spec.description.input[0].name
        output_name = spec.description.output[0].name

        preds = []
        for i in range(spek.shape[0]):
            input_data = {input_name: spek[i:i+1].numpy()}
            result = model.predict(input_data)
            preds.append(result[output_name])
        pred = np.concatenate(preds, axis=0)

        tar_waves = base.istft(torch.tensor(pred))
        tar_signal = tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy()[:, :-pad]

    # Voc_FT モデルはボーカルを直接出力
    vocals = tar_signal[:, :n_sample]
    instrumental = mix - vocals

    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_dir / "vocals.wav"), vocals.T, base.sr)
    sf.write(str(output_dir / "instrumental.wav"), instrumental.T, base.sr)

    return vocals.T


def compare_outputs(python_vocals: np.ndarray, coreml_vocals: np.ndarray):
    """出力を比較"""
    print("=" * 60)
    print("Python (ONNX) vs CoreML 比較結果")
    print("=" * 60)

    # RMS比較
    print(f"\n【RMS値】")
    print(f"  Python:  {np.sqrt(np.mean(python_vocals**2)):.8f}")
    print(f"  CoreML:  {np.sqrt(np.mean(coreml_vocals**2)):.8f}")

    # 差分分析
    diff = python_vocals - coreml_vocals
    print(f"\n【差分分析】")
    print(f"  最大絶対差: {np.max(np.abs(diff)):.10f}")
    print(f"  平均絶対差: {np.mean(np.abs(diff)):.10f}")
    print(f"  差分RMS:    {np.sqrt(np.mean(diff**2)):.10f}")

    # 相関係数
    corr = np.corrcoef(python_vocals.flatten(), coreml_vocals.flatten())[0, 1]
    print(f"\n【相関係数】")
    print(f"  {corr:.10f}")

    # SNR
    signal_power = np.mean(python_vocals**2)
    noise_power = np.mean(diff**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    print(f"\n【SNR】")
    print(f"  {snr:.2f} dB")

    # 判定
    print(f"\n【判定】")
    if corr > 0.9999 and np.max(np.abs(diff)) < 0.001:
        print("  ✅ 完全一致: CoreML変換は正確です")
        return True
    elif corr > 0.999:
        print("  ✅ ほぼ一致: float16による微小誤差のみ")
        return True
    elif corr > 0.99:
        print("  ⚠️ 軽微な差異あり")
        return False
    else:
        print("  ❌ 不一致: 変換に問題があります")
        return False


def main():
    base_dir = Path(__file__).parent.parent
    onnx_path = base_dir / "models" / "onnx" / "UVR-MDX-NET-Voc_FT.onnx"
    coreml_path = base_dir / "models" / "coreml" / "UVR-MDX-NET-Voc_FT.mlpackage"

    # 入力ファイル
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = base_dir / "test_audio" / "hollow_crown" / "mix.wav"

    if not input_path.exists():
        print(f"エラー: {input_path} が見つかりません")
        sys.exit(1)

    if not onnx_path.exists():
        print(f"エラー: {onnx_path} が見つかりません")
        sys.exit(1)

    if not coreml_path.exists():
        print(f"エラー: {coreml_path} が見つかりません")
        print("先に変換を実行してください: python scripts/convert_to_coreml.py")
        sys.exit(1)

    print("=" * 60)
    print("CoreML変換検証")
    print("=" * 60)
    print(f"入力: {input_path}")
    print()

    # 分離実行
    print("Python (ONNX) で分離中...")
    python_vocals = separate_with_onnx(
        str(onnx_path), str(input_path),
        base_dir / "output" / "python"
    )

    print("\nCoreML で分離中...")
    coreml_vocals = separate_with_coreml(
        str(coreml_path), str(input_path),
        base_dir / "output" / "coreml"
    )

    print()
    success = compare_outputs(python_vocals, coreml_vocals)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
