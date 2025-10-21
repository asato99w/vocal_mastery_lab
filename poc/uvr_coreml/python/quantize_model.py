#!/usr/bin/env python3
"""
CoreML モデル量子化スクリプト

8-bit量子化を適用してモデルサイズを削減し、Neural Engine最適化を実現します。
"""

import sys
import coremltools as ct
from pathlib import Path
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights
)


def quantize_model(
    input_path: Path,
    output_path: Path,
    nbits: int = 8,
    mode: str = "kmeans",
    granularity: str = "per_channel"
):
    """
    CoreMLモデルの量子化

    Args:
        input_path: 入力CoreMLモデルパス
        output_path: 出力CoreMLモデルパス
        nbits: 量子化ビット数（4, 6, 8）
        mode: 量子化モード（kmeans, uniform, custom）
        granularity: 粒度（per_tensor, per_channel, per_block）
    """
    print(f"\n⚙️  モデル量子化開始")
    print(f"   入力: {input_path}")
    print(f"   出力: {output_path}")
    print(f"   ビット数: {nbits}-bit")
    print(f"   モード: {mode}")
    print(f"   粒度: {granularity}")

    try:
        # モデル読み込み
        print("\n📂 CoreMLモデル読み込み中...")
        mlmodel = ct.models.MLModel(str(input_path))
        print("✅ モデル読み込み完了")

        # 元のサイズ
        original_size = sum(
            f.stat().st_size
            for f in input_path.rglob('*')
            if f.is_file()
        )
        original_size_mb = original_size / (1024 * 1024)
        print(f"   元のサイズ: {original_size_mb:.2f} MB")

        # 量子化設定
        print(f"\n🔧 {nbits}-bit量子化設定:")

        config = OptimizationConfig(
            global_config=OpPalettizerConfig(
                mode=mode,
                nbits=nbits,
                granularity=granularity
            )
        )

        if nbits == 8:
            print("   ✅ 8-bit量子化: Neural Engine最適化")
            print("   ✅ per_channel粒度: 精度保持")
        elif nbits == 4:
            print("   ⚠️  4-bit量子化: per_block粒度推奨")
            if granularity != "per_block":
                print("   警告: 4-bitではper_block粒度を推奨")

        # 量子化実行
        print("\n⚙️  量子化処理中...")
        print("   (この処理には数分かかる場合があります)")

        compressed_model = palettize_weights(mlmodel, config=config)

        print("✅ 量子化完了")

        # モデル保存
        print(f"\n💾 量子化モデル保存中: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compressed_model.save(str(output_path))

        # 圧縮後のサイズ
        compressed_size = sum(
            f.stat().st_size
            for f in output_path.rglob('*')
            if f.is_file()
        )
        compressed_size_mb = compressed_size / (1024 * 1024)

        # 圧縮率計算
        compression_ratio = (1 - compressed_size / original_size) * 100

        print("\n📊 量子化結果:")
        print(f"   元のサイズ: {original_size_mb:.2f} MB")
        print(f"   量子化後: {compressed_size_mb:.2f} MB")
        print(f"   削減率: {compression_ratio:.1f}%")
        print(f"   削減量: {original_size_mb - compressed_size_mb:.2f} MB")

        # 理論値との比較
        if nbits == 8:
            expected_reduction = 75  # 8-bit = 75%削減
        elif nbits == 4:
            expected_reduction = 87.5  # 4-bit = 87.5%削減
        else:
            expected_reduction = None

        if expected_reduction:
            print(f"\n   理論削減率: ~{expected_reduction}%")
            if compression_ratio >= expected_reduction * 0.9:
                print("   ✅ 期待通りの圧縮率")
            else:
                print("   ⚠️  理論値より低い圧縮率（モデル構造による）")

        print("\n✅ 量子化成功!")
        return compressed_model

    except Exception as e:
        print(f"\n❌ 量子化エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """メイン処理"""
    print("=" * 80)
    print("🔧 CoreML モデル量子化ツール")
    print("=" * 80)

    # パス設定
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    coreml_dir = project_root / "models" / "coreml"
    quantized_dir = project_root / "models" / "quantized"

    # CoreMLモデル検索
    coreml_packages = [
        p for p in coreml_dir.glob("*.mlpackage")
        if not p.name.endswith("_quantized.mlpackage")
    ]

    if not coreml_packages:
        print("\n❌ エラー: CoreMLモデルが見つかりません")
        print(f"   ディレクトリ: {coreml_dir}")
        print("\n先にモデルを変換してください:")
        print("   python python/convert_to_coreml.py")
        sys.exit(1)

    # 利用可能なモデルをリスト表示
    print("\n📋 量子化可能なCoreMLモデル:")
    for idx, package in enumerate(coreml_packages, 1):
        size = sum(f.stat().st_size for f in package.rglob('*') if f.is_file())
        size_mb = size / (1024*1024)
        print(f"{idx}. {package.name} ({size_mb:.2f} MB)")

    # モデル選択
    if len(sys.argv) > 1:
        # コマンドライン引数で指定
        model_name = sys.argv[1]
        if not model_name.endswith('.mlpackage'):
            model_name += '.mlpackage'
        input_path = coreml_dir / model_name
    else:
        # インタラクティブ選択
        choice = input(f"\n量子化するモデルを選択 (1-{len(coreml_packages)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(coreml_packages):
                input_path = coreml_packages[idx]
            else:
                print("❌ 無効な選択です")
                sys.exit(1)
        except ValueError:
            print("❌ 数値を入力してください")
            sys.exit(1)

    if not input_path.exists():
        print(f"❌ エラー: {input_path} が見つかりません")
        sys.exit(1)

    # 量子化設定選択
    print("\n⚙️  量子化レベル選択:")
    print("1. 8-bit (推奨) - Neural Engine最適化、精度保持")
    print("2. 4-bit - 最大圧縮、精度トレードオフあり")

    quant_choice = input("選択 (1-2, デフォルト: 1): ").strip() or "1"

    if quant_choice == "1":
        nbits = 8
        granularity = "per_channel"
    elif quant_choice == "2":
        nbits = 4
        granularity = "per_block"  # 4-bitはper_block推奨
    else:
        print("❌ 無効な選択です")
        sys.exit(1)

    # 出力パス
    output_filename = input_path.stem + f"_quantized_{nbits}bit.mlpackage"
    output_path = quantized_dir / output_filename

    # 確認
    if output_path.exists():
        response = input(f"\n既存のモデルを上書きしますか? (y/N): ").strip().lower()
        if response != 'y':
            print("キャンセルしました")
            sys.exit(0)

    # 量子化実行
    compressed_model = quantize_model(
        input_path,
        output_path,
        nbits=nbits,
        mode="kmeans",
        granularity=granularity
    )

    if compressed_model is None:
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"📂 量子化モデル保存先: {output_path}")
    print("=" * 80)

    print("\n次のステップ:")
    print("   1. テスト:")
    print(f"      python python/test_conversion.py {output_filename}")
    print("   2. iOS統合:")
    print("      モデルをXcodeプロジェクトにドラッグ&ドロップ")


if __name__ == "__main__":
    main()
