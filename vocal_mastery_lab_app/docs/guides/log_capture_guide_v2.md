# iOS UITest ログ取得ガイド v2(最小・再現性重視)

**作成日**: 2025-10-31
**ステータス**: 最小手順・再現性重視版
**対象環境**: Xcode 16, iOS Simulator 18.5, macOS Sonoma 14.6
**更新履歴**: 2025-10-31 - テスト失敗時も継続するよう改善

---

## 0) 前提(変数だけ直す)

```bash
UDID="508462B0-4692-4B9B-88F9-73A63F9B91F5"   # ←あなたのSim UDID
BUNDLE="com.kazuasato.VocalisStudio"          # ←Bundle ID
PROJECT="VocalisStudio.xcodeproj"
SCHEME="VocalisStudio"
TEST="VocalisStudioUITests/VocalisStudioUITests/testTargetPitchShouldDisappearAfterStoppingPlayback"
```

---

## 1) アプリ側(1回だけ入れればOK)

```swift
import OSLog

@main
struct AppEntry: App {
    private static let boot = Logger(
        subsystem: "com.kazuasato.VocalisStudio",
        category: "boot"
    )

    init() {
        Self.boot.error("UI_TEST_MARK: APP_INIT")
        FileLogger.shared.log(level: "INFO", category: "boot", message: "APP_INIT_FILE")
    }

    var body: some Scene {
        WindowGroup { RootView() }
    }
}
```

**重要**:
- `error`レベルを使用(永続化されやすい)
- `static let`で宣言、`Self.logger`でアクセス
- FileLoggerとOSLog両方にマーカーを出力

---

## 2) 実行スクリプト(これ1本)

**保存例**: `/tmp/logcap_v2.sh` → `bash /tmp/logcap_v2.sh`

```bash
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
```

---

## 3) 成功判定(この2つのどちらか見えればOK)

✅ `/tmp/vs_oslog_extracted.log` に以下のいずれかを含む行がある:
- `UI_TEST_MARK`
- `com.kazuasato.VocalisStudio`

**または**

✅ フォールバックで FileLogger の最新ファイル末尾が出力される

---

## 4) ショート・トラブルシュート(3行で直す)

### 問題1: テスト失敗でスクリプトが停止
**原因**: `set -e`がテスト失敗で停止してログ収集に到達しない
**解決**: Step 2で`set +e` → `PIPESTATUS`取得 → `set -e`に戻す(v2で修正済み)

### 問題2: 別UDIDで実行(Clone 1/Clone 2)
**原因**: 並列テスト抑止フラグが不足
**解決**: 以下を**必ず指定**(v2で実装済み):
```bash
-parallel-testing-enabled NO \
-maximum-concurrent-test-simulator-destinations 1
```

### 問題3: ログ0件
**原因**: アプリ側にマーカーがない
**解決**: 上記「1) アプリ側」の `APP_INIT` マーカーと `UI_TEST_MARK` を入れる(`error`レベル推奨)

---

## 5) V2の重要な改善点

| 改善項目 | 内容 |
|---------|------|
| **テスト失敗時の継続** | `set +e`により、テスト失敗してもログ収集を続行 |
| **終了コード取得** | `PIPESTATUS[0]`でxcodebuildの終了コードを記録 |
| **アーカイブ存在確認** | Step 4で`if [ -d "$ARCHIVE" ]`チェック追加 |
| **結果サマリ** | Step 6でOSLog/FileLoggerのどちらが成功したか明示 |
| **完全な残骸掃除** | Step 0で`$ARCHIVE`と`$EXTRACTED`も削除 |

---

## 6) V2の設計思想

1. **テスト結果に依存しない**: テスト成功/失敗どちらでもログ取得可能
2. **毎回再現できる**: 同じ手順で必ずログが取れる
3. **最小手順**: 余計な説明を削り、コピペで動く
4. **フォールバック**: OSLog失敗時はFileLoggerで補完
5. **並列テスト完全抑止**: UDID指定を確実に効かせる
6. **残骸掃除**: xcresultの衝突を事前に防ぐ

---

## 7) 使用例

```bash
# 1. スクリプト作成
cat > /tmp/logcap_v2.sh << 'EOF'
[上記スクリプトをコピペ]
EOF

chmod +x /tmp/logcap_v2.sh

# 2. 実行
bash /tmp/logcap_v2.sh

# 3. ログ確認
grep "UI_TEST_MARK" /tmp/vs_oslog_extracted.log
# または
grep "🔴" /tmp/vs_oslog_extracted.log  # カスタムマーカーの場合
```

---

## 8) 検証済みの成功例(2025-10-31)

### 実行結果
- **OSLogアーカイブ**: 162MB作成成功
- **抽出ログ**: 9,418行作成成功
- **FileLogger**: 🔴マーカー確認成功

### FileLoggerログ例
```
2025-10-31 09:32:22.174 [INFO] [pitch_monitoring] 🔴 stopTargetPitchMonitoring START: targetPitch=Optional(...G3...), taskExists=true
2025-10-31 09:32:22.177 [INFO] [pitch_monitoring] 🔴 stopTargetPitchMonitoring END: targetPitch set to nil
```

**重要**: UITestが失敗(Line 256)してもログ取得に成功しました。

---

## 関連ドキュメント

- `log_capture_guide.md`: v1(詳細版) - 原理と複数の方法を説明
- `LOGGING_SYSTEM_ANALYSIS.md`: ログシステムの分析と失敗原因の詳細
