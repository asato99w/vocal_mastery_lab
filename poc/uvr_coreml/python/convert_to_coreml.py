#!/usr/bin/env python3
"""
ONNX → CoreML 変換スクリプト

UVR MDX-Net ONNXモデルをCoreMLフォーマットに変換します。
"""

import sys
import onnx
import coremltools as ct
from pathlib import Path
import numpy as np


def inspect_onnx_model(onnx_path: Path):
    """
    ONNXモデルの情報を表示

    Args:
        onnx_path: ONNXモデルパス
    """
    print(f"\n🔍 ONNXモデル情報: {onnx_path.name}")
    print("=" * 80)

    model = onnx.load(str(onnx_path))

    # 入力情報
    print("\n📥 入力:")
    for input_tensor in model.graph.input:
        name = input_tensor.name
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"   名前: {name}")
        print(f"   形状: {shape}")

    # 出力情報
    print("\n📤 出力:")
    for output_tensor in model.graph.output:
        name = output_tensor.name
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"   名前: {name}")
        print(f"   形状: {shape}")

    print()
    return model


def convert_onnx_to_coreml(
    onnx_path: Path,
    output_path: Path,
    compute_units: str = "ALL",
    minimum_deployment_target: str = "iOS17"
):
    """
    ONNX → CoreML変換

    Args:
        onnx_path: ONNXモデルパス
        output_path: CoreMLモデル保存パス
        compute_units: 計算ユニット（ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE）
        minimum_deployment_target: 最小デプロイメントターゲット
    """
    print(f"\n🔄 ONNX → CoreML 変換開始")
    print(f"   入力: {onnx_path}")
    print(f"   出力: {output_path}")
    print(f"   計算ユニット: {compute_units}")
    print(f"   最小ターゲット: {minimum_deployment_target}")

    # 計算ユニット設定
    compute_units_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE
    }

    # デプロイメントターゲット設定
    target_map = {
        "iOS15": ct.target.iOS15,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
        "iOS18": ct.target.iOS18
    }

    try:
        # ONNX読み込み
        print("\n📂 ONNXモデル読み込み中...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("✅ ONNXモデル検証完了")

        # CoreML変換
        print("\n⚙️  CoreML変換中...")
        print("   (この処理には数分かかる場合があります)")

        mlmodel = ct.convert(
            str(onnx_path),
            minimum_deployment_target=target_map.get(
                minimum_deployment_target,
                ct.target.iOS17
            ),
            compute_units=compute_units_map.get(
                compute_units,
                ct.ComputeUnit.ALL
            ),
            # 変換オプション
            convert_to="mlprogram"  # ML Program形式（iOS15+）
        )

        print("✅ CoreML変換完了")

        # モデル保存
        print(f"\n💾 CoreMLモデル保存中: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(output_path))

        # モデル情報表示
        print("\n📊 CoreMLモデル情報:")
        print(f"   入力: {mlmodel.input_description}")
        print(f"   出力: {mlmodel.output_description}")

        # ファイルサイズ
        if output_path.exists():
            # .mlpackageはディレクトリなので、サイズ計算
            total_size = sum(
                f.stat().st_size
                for f in output_path.rglob('*')
                if f.is_file()
            )
            size_mb = total_size / (1024 * 1024)
            print(f"   サイズ: {size_mb:.2f} MB")

        print("\n✅ 変換成功!")
        return mlmodel

    except Exception as e:
        print(f"\n❌ 変換エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """メイン処理"""
    print("=" * 80)
    print("🔄 UVR MDX-Net ONNX → CoreML 変換ツール")
    print("=" * 80)

    # パス設定
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    onnx_dir = project_root / "models" / "onnx"
    coreml_dir = project_root / "models" / "coreml"

    # ONNXモデル検索
    onnx_files = list(onnx_dir.glob("*.onnx"))

    if not onnx_files:
        print("\n❌ エラー: ONNXモデルが見つかりません")
        print(f"   ディレクトリ: {onnx_dir}")
        print("\n先にモデルをダウンロードしてください:")
        print("   python python/download_model.py")
        sys.exit(1)

    # 利用可能なモデルをリスト表示
    print("\n📋 変換可能なONNXモデル:")
    for idx, onnx_file in enumerate(onnx_files, 1):
        size_mb = onnx_file.stat().st_size / (1024*1024)
        print(f"{idx}. {onnx_file.name} ({size_mb:.2f} MB)")

    # モデル選択
    if len(sys.argv) > 1:
        # コマンドライン引数で指定
        model_name = sys.argv[1]
        if not model_name.endswith('.onnx'):
            model_name += '.onnx'
        onnx_path = onnx_dir / model_name
    else:
        # インタラクティブ選択
        choice = input(f"\n変換するモデルを選択 (1-{len(onnx_files)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(onnx_files):
                onnx_path = onnx_files[idx]
            else:
                print("❌ 無効な選択です")
                sys.exit(1)
        except ValueError:
            print("❌ 数値を入力してください")
            sys.exit(1)

    if not onnx_path.exists():
        print(f"❌ エラー: {onnx_path} が見つかりません")
        sys.exit(1)

    # ONNXモデル情報表示
    inspect_onnx_model(onnx_path)

    # 出力パス
    output_filename = onnx_path.stem + ".mlpackage"
    output_path = coreml_dir / output_filename

    # 変換設定
    print("\n⚙️  変換設定:")
    print("   計算ユニット: ALL (CPU + GPU + Neural Engine)")
    print("   最小ターゲット: iOS 17")

    # 確認
    if output_path.exists():
        response = input(f"\n既存のモデルを上書きしますか? (y/N): ").strip().lower()
        if response != 'y':
            print("キャンセルしました")
            sys.exit(0)

    # 変換実行
    mlmodel = convert_onnx_to_coreml(
        onnx_path,
        output_path,
        compute_units="ALL",
        minimum_deployment_target="iOS17"
    )

    if mlmodel is None:
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"📂 CoreMLモデル保存先: {output_path}")
    print("=" * 80)

    print("\n次のステップ:")
    print("   1. 量子化（モデルサイズ削減）:")
    print(f"      python python/quantize_model.py {output_filename}")
    print("   2. テスト:")
    print(f"      python python/test_conversion.py {output_filename}")


if __name__ == "__main__":
    main()
