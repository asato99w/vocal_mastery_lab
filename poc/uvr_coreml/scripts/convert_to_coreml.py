#!/usr/bin/env python3
"""
ONNX → CoreML 変換スクリプト

使用方法:
    python scripts/convert_to_coreml.py [model_name]

例:
    python scripts/convert_to_coreml.py UVR-MDX-NET-Voc_FT
"""

import sys
from pathlib import Path

import torch
import coremltools as ct
from onnx2torch import convert as onnx2torch_convert


def convert_onnx_to_coreml(
    onnx_path: Path,
    output_path: Path,
    input_shape: tuple = (1, 4, 3072, 256)
):
    """
    ONNX → CoreML 変換

    Args:
        onnx_path: ONNXモデルパス
        output_path: CoreML出力パス
        input_shape: 入力テンソル形状 (batch, channels, freq, time)
    """
    print(f"1. ONNX → PyTorch 変換中...")
    torch_model = onnx2torch_convert(str(onnx_path))
    torch_model.eval()
    print(f"   完了")

    print(f"2. トレース用入力作成...")
    example_input = torch.randn(*input_shape)
    print(f"   入力形状: {example_input.shape}")

    print(f"3. PyTorchモデルをトレース...")
    traced_model = torch.jit.trace(torch_model, example_input)
    print(f"   完了")

    print(f"4. CoreML変換中...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape)],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram"
    )
    print(f"   完了")

    print(f"5. 保存中...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    print(f"   保存完了: {output_path}")

    # モデル情報
    spec = mlmodel.get_spec()
    print(f"\nモデル情報:")
    print(f"  入力: {spec.description.input[0].name}")
    print(f"  出力: {spec.description.output[0].name}")

    return mlmodel


def main():
    base_dir = Path(__file__).parent.parent
    onnx_dir = base_dir / "models" / "onnx"
    coreml_dir = base_dir / "models" / "coreml"

    # モデル名取得
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        if not model_name.endswith(".onnx"):
            model_name += ".onnx"
    else:
        # デフォルト
        model_name = "UVR-MDX-NET-Voc_FT.onnx"

    onnx_path = onnx_dir / model_name
    output_path = coreml_dir / (onnx_path.stem + ".mlpackage")

    if not onnx_path.exists():
        print(f"エラー: {onnx_path} が見つかりません")
        sys.exit(1)

    print("=" * 60)
    print("ONNX → CoreML 変換")
    print("=" * 60)
    print(f"入力: {onnx_path}")
    print(f"出力: {output_path}")
    print()

    convert_onnx_to_coreml(onnx_path, output_path)


if __name__ == "__main__":
    main()
