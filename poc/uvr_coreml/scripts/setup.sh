#!/bin/bash
# UVR CoreML 環境セットアップスクリプト

set -e

echo "======================================================================"
echo "🎵 UVR CoreML 環境セットアップ"
echo "======================================================================"

# プロジェクトルート
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo ""
echo "📂 プロジェクトディレクトリ: $PROJECT_ROOT"

# Python バージョンチェック
echo ""
echo "🐍 Python バージョンチェック..."
python3 --version

# 仮想環境作成
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Python仮想環境作成中..."
    python3 -m venv venv
    echo "✅ 仮想環境作成完了"
else
    echo ""
    echo "✅ 仮想環境既存"
fi

# 仮想環境アクティベート
echo ""
echo "🔄 仮想環境アクティベート..."
source venv/bin/activate

# 依存パッケージインストール
echo ""
echo "📦 依存パッケージインストール中..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ インストール完了"

# インストール済みパッケージ確認
echo ""
echo "📋 主要パッケージバージョン:"
python3 -c "import coremltools; print(f'  - coremltools: {coremltools.__version__}')"
python3 -c "import onnx; print(f'  - onnx: {onnx.__version__}')"
python3 -c "import numpy; print(f'  - numpy: {numpy.__version__}')"

# ディレクトリ構造確認
echo ""
echo "📁 ディレクトリ構造確認:"
mkdir -p models/onnx
mkdir -p models/coreml
mkdir -p models/quantized
mkdir -p tests/test_audio

echo "  ✅ models/onnx/"
echo "  ✅ models/coreml/"
echo "  ✅ models/quantized/"
echo "  ✅ tests/test_audio/"

echo ""
echo "======================================================================"
echo "✅ セットアップ完了！"
echo "======================================================================"

echo ""
echo "次のステップ:"
echo "  1. モデルダウンロード:"
echo "     python python/download_model.py"
echo ""
echo "  2. CoreML変換:"
echo "     python python/convert_to_coreml.py"
echo ""
echo "  3. モデル量子化:"
echo "     python python/quantize_model.py"
echo ""
echo "仮想環境の有効化:"
echo "  source venv/bin/activate"
echo ""
