#!/usr/bin/env python3
"""
ONNX → CoreML 変換スクリプト
"""

import coremltools as ct
from pathlib import Path


def convert_onnx_to_coreml(onnx_path: str, output_path: str):
    """ONNXモデルをCoreMLに変換"""
    print(f"変換開始: {onnx_path}")

    # ONNX → CoreML変換 (ct.convert使用)
    model = ct.convert(
        model=onnx_path,
        source='onnx',
        minimum_deployment_target=ct.target.iOS17
    )

    # 保存
    model.save(output_path)
    print(f"保存完了: {output_path}")

    # モデル情報表示
    spec = model.get_spec()
    print(f"\n入力:")
    for inp in spec.description.input:
        print(f"  {inp.name}: {inp.type}")
    print(f"\n出力:")
    for out in spec.description.output:
        print(f"  {out.name}: {out.type}")


def main():
    base_dir = Path(__file__).parent
    onnx_path = base_dir / "models/onnx/UVR-MDX-NET-Inst_Main.onnx"
    coreml_path = base_dir / "models/coreml/UVR-MDX-NET-Inst_Main_new.mlpackage"

    convert_onnx_to_coreml(str(onnx_path), str(coreml_path))


if __name__ == "__main__":
    main()
