#!/usr/bin/env bash
# スクリプト: 最新のFileLoggerログを確認
# 使用例: ./check_latest_log.sh [grep_pattern]

UDID="508462B0-4692-4B9B-88F9-73A63F9B91F5"
BUNDLE="com.kazuasato.VocalisStudio"

# アプリコンテナのパスを取得
APP_CONTAINER=$(xcrun simctl get_app_container "$UDID" "$BUNDLE" data 2>/dev/null)

if [ -z "$APP_CONTAINER" ]; then
    echo "❌ アプリコンテナが見つかりません"
    exit 1
fi

# 最新のログファイルを取得
LATEST_LOG=$(find "$APP_CONTAINER/Documents/logs" -name 'vocalis_*.log' -type f \
    -exec stat -f '%m %N' {} + 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ ログファイルが見つかりません"
    exit 1
fi

echo "📄 最新ログ: $LATEST_LOG"
echo "📅 最終更新: $(stat -f '%Sm' "$LATEST_LOG")"
echo ""

# grepパターンが指定されていればフィルタリング、なければ最後の100行を表示
if [ -n "$1" ]; then
    echo "🔍 パターン検索: $1"
    echo "----------------------------------------"
    grep -E "$1" "$LATEST_LOG" | tail -100
else
    echo "📋 最後の100行:"
    echo "----------------------------------------"
    tail -100 "$LATEST_LOG"
fi
