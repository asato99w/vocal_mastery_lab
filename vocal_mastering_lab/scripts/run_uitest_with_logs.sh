#!/usr/bin/env bash
set -euo pipefail

# === 設定 ===
UDID="508462B0-4692-4B9B-88F9-73A63F9B91F5"
BUNDLE="com.kazuasato.VocalisStudio"
PROJECT="VocalisStudio.xcodeproj"
SCHEME="VocalisStudio"
TEST="VocalisStudioUITests/VocalisStudioUITests/testTargetPitchShouldDisappearAfterStoppingPlayback"

RESULT_BUNDLE="/tmp/TestResult.xcresult"
ARCHIVE="/tmp/vs_oslog.logarchive"
EXTRACTED="/tmp/vs_oslog_extracted.log"
XCODE_OUT="/tmp/vs_xcodebuild.out"

cd /Users/asatokazu/Documents/dev/mine/music/vocalis-studio/VocalisStudio

# 0) 残骸掃除
rm -rf "$RESULT_BUNDLE" "$ARCHIVE" "$EXTRACTED" || true

# 1) Boot & 永続設定
xcrun simctl boot "$UDID" 2>/dev/null || true
sleep 2
xcrun simctl spawn "$UDID" log config --subsystem "$BUNDLE" --mode "level:debug,persist:debug"
xcrun simctl spawn "$UDID" log config --status --subsystem "$BUNDLE" || true

# 2) 単一UIテスト(同期/クローン抑止) — 失敗しても続行
set +e
xcodebuild \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -destination "id=$UDID" \
  -resultBundlePath "$RESULT_BUNDLE" \
  -parallel-testing-enabled NO \
  -maximum-concurrent-test-simulator-destinations 1 \
  -only-testing:"$TEST" \
  test 2>&1 | tee "$XCODE_OUT"
XCODE_STATUS=${PIPESTATUS[0]}
set -e
echo "xcodebuild exit code: $XCODE_STATUS"

# 3) 事後収集(相対30分)
xcrun simctl boot "$UDID" 2>/dev/null || true
sleep 2
xcrun simctl spawn "$UDID" log collect --output "$ARCHIVE" --last 30m 2>&1 || true

# 4) オフライン抽出(広めの述語)
if [ -d "$ARCHIVE" ]; then
  /usr/bin/log show --archive "$ARCHIVE" --style syslog --info --debug \
    --last 30m \
    --predicate '(subsystem == "'"$BUNDLE"'") OR (process CONTAINS[c] "Vocalis") OR (senderImagePath CONTAINS[c] "VocalisStudio") OR (eventMessage CONTAINS[c] "UI_TEST_MARK")' \
    | tee "$EXTRACTED"
else
  echo "❌ OSLog archive not created"
fi

# 5) FileLogger フォールバック
APP_CONTAINER=$(xcrun simctl get_app_container "$UDID" "$BUNDLE" data 2>/dev/null || true)
if [[ -n "${APP_CONTAINER:-}" && -d "$APP_CONTAINER/Documents/logs" ]]; then
  LATEST=$(find "$APP_CONTAINER/Documents/logs" -name 'vocalis_*.log' -type f \
     -exec stat -f '%m %N' {} + 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [[ -n "${LATEST:-}" ]]; then
    echo "----- FileLogger latest -----"
    /bin/cat -- "$LATEST" | tail -100
  else
    echo "No FileLogger file found"
  fi
else
  echo "No container/logs dir; skipping FileLogger fallback"
fi

# 6) 結果サマリ(どちらか出ていればOK)
if [[ -f "$EXTRACTED" ]] && grep -qE "UI_TEST_MARK|$BUNDLE|Vocalis" "$EXTRACTED"; then
  echo "✅ OSLog captured (see $EXTRACTED)"
elif [[ -n "${LATEST:-}" ]]; then
  echo "✅ FileLogger captured (see $LATEST)"
else
  echo "❌ No logs captured"
fi

# (任意)終了時にシャットダウン
xcrun simctl shutdown "$UDID" 2>/dev/null || true
