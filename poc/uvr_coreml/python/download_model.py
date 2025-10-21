#!/usr/bin/env python3
"""
UVR MDX-Net ONNX モデルダウンロードスクリプト

推奨モデル:
- UVR-MDX-NET-Inst_Main.onnx (ボーカル/伴奏分離)
- Kim_Vocal_1.onnx (高品質ボーカル抽出)
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm


# モデル定義
MODELS = {
    "UVR-MDX-NET-Inst_Main": {
        "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_Main.onnx",
        "filename": "UVR-MDX-NET-Inst_Main.onnx",
        "description": "ボーカル/伴奏分離用メインモデル（推奨）",
        "size": "~30MB"
    },
    "Kim_Vocal_1": {
        "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_1.onnx",
        "filename": "Kim_Vocal_1.onnx",
        "description": "高品質ボーカル抽出モデル",
        "size": "~40MB"
    },
    "UVR-MDX-NET-Voc_FT": {
        "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx",
        "filename": "UVR-MDX-NET-Voc_FT.onnx",
        "description": "ボーカル特化モデル（高速）",
        "size": "~25MB"
    }
}


def download_file(url: str, output_path: Path, description: str = ""):
    """
    ファイルをダウンロード（進捗バー表示）

    Args:
        url: ダウンロードURL
        output_path: 保存先パス
        description: 説明文
    """
    print(f"\n📥 ダウンロード: {description}")
    print(f"   URL: {url}")
    print(f"   保存先: {output_path}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"✅ ダウンロード完了: {output_path.name}")
        print(f"   ファイルサイズ: {output_path.stat().st_size / (1024*1024):.2f} MB")
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ ダウンロードエラー: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def list_available_models():
    """利用可能なモデルをリスト表示"""
    print("\n📋 利用可能なUVR MDX-Netモデル:")
    print("=" * 80)
    for idx, (key, info) in enumerate(MODELS.items(), 1):
        print(f"{idx}. {key}")
        print(f"   説明: {info['description']}")
        print(f"   サイズ: {info['size']}")
        print()


def download_model(model_key: str, models_dir: Path) -> bool:
    """
    指定されたモデルをダウンロード

    Args:
        model_key: モデルキー（MODELS辞書のキー）
        models_dir: モデル保存ディレクトリ

    Returns:
        成功時True
    """
    if model_key not in MODELS:
        print(f"❌ エラー: モデル '{model_key}' が見つかりません")
        list_available_models()
        return False

    model_info = MODELS[model_key]
    output_path = models_dir / model_info['filename']

    # 既にダウンロード済みかチェック
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024*1024)
        print(f"\n📦 モデル既存: {output_path.name} ({file_size_mb:.2f} MB)")
        response = input("再ダウンロードしますか? (y/N): ").strip().lower()
        if response != 'y':
            print("スキップしました")
            return True

    return download_file(
        model_info['url'],
        output_path,
        model_info['description']
    )


def main():
    """メイン処理"""
    # モデル保存ディレクトリ
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models" / "onnx"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🎵 UVR MDX-Net ONNX モデルダウンロードツール")
    print("=" * 80)

    # コマンドライン引数処理
    if len(sys.argv) > 1:
        model_key = sys.argv[1]
        if model_key == "--list" or model_key == "-l":
            list_available_models()
            return

        success = download_model(model_key, models_dir)
        sys.exit(0 if success else 1)

    # インタラクティブモード
    list_available_models()

    print("ダウンロードするモデルを選択してください:")
    print("1. UVR-MDX-NET-Inst_Main (推奨)")
    print("2. Kim_Vocal_1")
    print("3. UVR-MDX-NET-Voc_FT")
    print("4. すべてダウンロード")
    print("0. 終了")

    choice = input("\n選択 (0-4): ").strip()

    model_keys = list(MODELS.keys())

    if choice == '0':
        print("終了します")
        return
    elif choice == '4':
        # すべてダウンロード
        print("\n📦 すべてのモデルをダウンロードします...")
        success_count = 0
        for key in model_keys:
            if download_model(key, models_dir):
                success_count += 1

        print(f"\n✅ {success_count}/{len(model_keys)} モデルのダウンロード完了")
    elif choice in ['1', '2', '3']:
        idx = int(choice) - 1
        model_key = model_keys[idx]
        download_model(model_key, models_dir)
    else:
        print("❌ 無効な選択です")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"📂 モデル保存先: {models_dir}")
    print("=" * 80)

    # ダウンロード済みモデルをリスト
    onnx_files = list(models_dir.glob("*.onnx"))
    if onnx_files:
        print("\n✅ ダウンロード済みモデル:")
        for onnx_file in onnx_files:
            size_mb = onnx_file.stat().st_size / (1024*1024)
            print(f"   - {onnx_file.name} ({size_mb:.2f} MB)")

    print("\n次のステップ:")
    print("   python python/convert_to_coreml.py")


if __name__ == "__main__":
    main()
