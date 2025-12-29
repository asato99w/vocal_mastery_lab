# VocalMasteryLab プロジェクト

## テスト実行

テスト実行時は専用スクリプトを使用してください。

### 主要スクリプト

```bash
# プロジェクトディレクトリに移動
cd vocal_mastery_lab_app

# テストランナーの使用方法を確認
./scripts/test-runner.sh --help

# テストタイプ別実行
./scripts/test-runner.sh critical   # クリティカルテスト (~1分)
./scripts/test-runner.sh smoke      # スモークテスト (~3分)
./scripts/test-runner.sh unit       # ユニットテストのみ
./scripts/test-runner.sh ui         # UIテストのみ
./scripts/test-runner.sh all        # 全テスト

# 特定のテストクラスを実行
./scripts/test-runner.sh ui PaywallUITests
```

### スクリプト一覧

| スクリプト | 用途 |
|-----------|------|
| `test-runner.sh` | メインテストランナー（推奨） |
| `test_with_analysis.sh` | エラー分析付きテスト実行 |
| `run_uitest_with_logs.sh` | UIテスト + ログ収集 |
| `check_latest_log.sh` | 最新ログの確認 |

### スクリプトの改善について

テストスクリプトに不備があったり、使い勝手が悪い場合は、改善したバージョンを作成してください。改善時は以下を考慮：

- 既存スクリプトの挙動を壊さない
- 新しい要件に合わせて拡張
- エラーハンドリングの強化
- ログ出力の改善
