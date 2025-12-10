# UIテストスクリーンショット取得ガイド

## 概要

XCUITestで撮影したスクリーンショットは`.xcresult`バンドル内に保存されます。このドキュメントでは、スクリーンショットを効率的に抽出する方法と避けるべきアンチパターンをまとめます。

## ✅ 推奨される方法

### 1. スクリーンショットの保存場所

#### Xcode GUIで実行した場合
```
~/Library/Developer/Xcode/DerivedData/<ProjectName>-<RandomString>/Logs/Test/Test-<ProjectName>-YYYY.MM.DD_HH-MM-SS-+0900.xcresult
```

#### xcodebuildで実行した場合（デフォルト）
```
<ProjectDir>/DerivedData/<ProjectName>/Logs/Test/Test-<ProjectName>-YYYY.MM.DD_HH-MM-SS-+0900.xcresult
```

#### xcodebuildで`-resultBundlePath`を指定した場合
```
<指定したパス>.xcresult
```

### 2. テストコードでのスクリーンショット撮影

```swift
@MainActor
func testExample() throws {
    let app = XCUIApplication()
    app.launch()

    // スクリーンショット撮影
    let screenshot = app.screenshot()
    let attachment = XCTAttachment(screenshot: screenshot)
    attachment.name = "descriptive_screenshot_name"  // 検索しやすい名前をつける
    attachment.lifetime = .keepAlways  // 必ず保存
    add(attachment)
}
```

**重要ポイント**:
- `attachment.lifetime = .keepAlways` を指定しないとテスト成功時に削除される
- `attachment.name` に分かりやすい名前をつけると後で探しやすい

### 3. スクリーンショット抽出の推奨ワークフロー

#### ステップ1: xcresultパスを特定

最新のテスト結果を探す:
```bash
ls -lt ~/Library/Developer/Xcode/DerivedData/VocalisStudio-*/Logs/Test/*.xcresult | head -1
```

または環境変数に保存:
```bash
XCRESULT_PATH="/Users/kazuasato/Library/Developer/Xcode/DerivedData/VocalisStudio-frcxxiswixbmnpedzxgbxeyluinf/Logs/Test/Test-VocalisStudio-2025.10.28_19-08-38-+0900.xcresult"
```

#### ステップ2: データベースからスクリーンショット情報を取得

```bash
sqlite3 "$XCRESULT_PATH/database.sqlite3" \
  "SELECT xcResultKitPayloadRefId, filenameOverride, name FROM Attachments WHERE uniformTypeIdentifier = 'public.png';"
```

**出力例**:
```
0~3Ru-WZ-RZ...rg==|01_initial_recording_screen_0_D7D4C4F3-2E9C-4D13-B2D6-82C1B3E3CB70.png|01_initial_recording_screen
0~rxr8IfyAFj...xA==|02_during_recording_0_6D54A67D-4905-4BCE-BE29-318FD8F996E0.png|02_during_recording
```

#### ステップ3: 個別スクリーンショットを抽出

```bash
# 出力ディレクトリ作成
mkdir -p /tmp/screenshots

# IDを使用してエクスポート
xcrun xcresulttool export --legacy --type file \
  --path "$XCRESULT_PATH" \
  --id "0~3Ru-WZ-RZwarj-mXvaJ_lxiHN4B3-V3tRrykSNKqml7kDj2a8qao4_gDNuIZpAmtl7sYe-qi2BNnyE6yfL34rg==" \
  --output-path /tmp/screenshots/01_initial_recording_screen.png
```

### 4. 自動抽出スクリプト（推奨）

```bash
#!/bin/bash
# extract_screenshots.sh - UIテストスクリーンショット自動抽出

set -e

XCRESULT_PATH="$1"
OUTPUT_DIR="${2:-./screenshots}"

if [ -z "$XCRESULT_PATH" ]; then
    echo "Usage: $0 <path_to_xcresult> [output_dir]"
    exit 1
fi

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"

# スクリーンショット情報取得
echo "📸 Extracting screenshots from: $XCRESULT_PATH"

sqlite3 "$XCRESULT_PATH/database.sqlite3" \
  "SELECT xcResultKitPayloadRefId, name FROM Attachments WHERE uniformTypeIdentifier = 'public.png';" | \
while IFS='|' read -r id name; do
    output_file="$OUTPUT_DIR/${name}.png"
    echo "  → $output_file"

    xcrun xcresulttool export --legacy --type file \
      --path "$XCRESULT_PATH" \
      --id "$id" \
      --output-path "$output_file"
done

echo "✅ Screenshots exported to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
```

**使用方法**:
```bash
chmod +x extract_screenshots.sh
./extract_screenshots.sh "/path/to/Test-Result.xcresult" ./screenshots
```

## ❌ アンチパターン（避けるべき方法）

### 1. ❌ 非推奨: `xcresulttool export --type attachments`

**問題点**: このオプションは存在しない

```bash
# ❌ 動作しない
xcrun xcresulttool export --type attachments \
  --path Test.xcresult \
  --output-path ./screenshots
```

**エラー**:
```
Error: The value 'attachments' is invalid for '--type <type>'.
Please provide one of 'file', 'directory', 'diagnostics' or 'coverage'.
```

**正しい方法**: `--type file` を使用し、IDを個別に指定する

### 2. ❌ 非推奨: findコマンドでPNG直接検索

**問題点**: xcresultバンドルの内部構造が複雑でPNGファイルが見つからない

```bash
# ❌ 通常は何も見つからない
find Test.xcresult -name "*.png"
```

**理由**:
- xcresultは独自のデータベース構造を使用
- 画像ファイルは`Data/`ディレクトリに暗号化された名前で保存されている
- 直接ファイル名では検索できない

**正しい方法**: SQLiteデータベースを経由してIDを取得

### 3. ❌ 非推奨: JSON解析による抽出

**問題点**: JSONが複雑すぎて実用的でない

```bash
# ❌ 複雑すぎて保守困難
xcrun xcresulttool get --legacy --path Test.xcresult --format json | \
  python3 -c "import json; ..." # 複雑なJSON解析コード
```

**理由**:
- xcresultのJSON構造は非常に複雑でネストが深い
- アタッチメントの場所を見つけるのが困難
- コード保守が難しい

**正しい方法**: SQLiteデータベースを直接クエリ（シンプルで確実）

### 4. ❌ 非推奨: `--legacy`フラグなし

**問題点**: 新しいxcresulttoolはデフォルトでレガシー形式をサポートしない

```bash
# ❌ エラーになる場合がある
xcrun xcresulttool export --type file \
  --path Test.xcresult \
  --id "..." \
  --output-path output.png
```

**エラー**:
```
Error: This command is deprecated and will be removed in a future release,
--legacy flag is required to use it.
```

**正しい方法**: `--legacy`フラグを追加

### 5. ❌ 非推奨: スクリーンショット名を指定しない

**問題点**: デフォルト名では後で識別が困難

```swift
// ❌ 名前なし - 後で何のスクリーンショットか分からない
let attachment = XCTAttachment(screenshot: screenshot)
add(attachment)
```

**問題**:
- ファイル名が`Screenshot_0_<UUID>.png`のような意味のない名前になる
- 複数のスクリーンショットがある場合、どれがどの状態か不明

**正しい方法**:
```swift
// ✅ 明確な名前をつける
let attachment = XCTAttachment(screenshot: screenshot)
attachment.name = "01_login_screen_before_authentication"
attachment.lifetime = .keepAlways
add(attachment)
```

## 📊 xcresultバンドル構造

### ディレクトリ構造
```
Test-VocalisStudio-2025.10.28_19-08-38-+0900.xcresult/
├── Info.plist                  # メタデータ
├── database.sqlite3            # テスト結果データベース ← ここが重要
└── Data/                       # 実際のファイル（暗号化された名前）
    ├── 0~3Ru-WZ...            # スクリーンショット（実ファイル）
    ├── 0~rxr8If...            # スクリーンショット（実ファイル）
    └── ...
```

### データベーステーブル
```sql
-- 主要なテーブル
Attachments         -- スクリーンショット、ビデオなどの添付ファイル
TestCaseRuns        -- 個別テストケースの実行結果
Activities          -- テスト実行中のアクティビティ
TestIssues          -- テスト失敗情報
```

### Attachmentsテーブルスキーマ
```sql
CREATE TABLE Attachments (
    xcResultKitPayloadRefId TEXT,      -- ファイル抽出用のID
    uniformTypeIdentifier TEXT,         -- ファイル形式 (public.png, public.mpeg-4)
    filenameOverride TEXT,              -- 元のファイル名
    name TEXT,                          -- XCTAttachmentで指定した名前
    timestamp REAL,                     -- 撮影タイムスタンプ
    lifetime TEXT,                      -- keepAlways, deleteOnSuccess
    ...
);
```

## 🔍 トラブルシューティング

### 問題: スクリーンショットが見つからない

**原因1**: `lifetime`が`deleteOnSuccess`でテストが成功した
```swift
// 解決策: 必ず.keepAlwaysを指定
attachment.lifetime = .keepAlways
```

**原因2**: 正しいxcresultパスを見ていない
```bash
# 解決策: 最新のxcresultを確認
ls -lt ~/Library/Developer/Xcode/DerivedData/*/Logs/Test/*.xcresult | head -5
```

### 問題: xcresulttoolコマンドが失敗する

**エラー**: `This command is deprecated and will be removed in a future release`
```bash
# 解決策: --legacyフラグを追加
xcrun xcresulttool export --legacy --type file ...
```

### 問題: SQLiteクエリでスクリーンショットが0件

**原因**: テストがスクリーンショット撮影前に失敗した
```bash
# デバッグ: すべてのアタッチメントを確認
sqlite3 "$XCRESULT_PATH/database.sqlite3" \
  "SELECT uniformTypeIdentifier, name FROM Attachments;"
```

## 🎯 ベストプラクティス

### 1. スクリーンショット命名規則

```swift
// ✅ 推奨: 連番 + 状態説明
attachment.name = "01_initial_login_screen"
attachment.name = "02_after_entering_credentials"
attachment.name = "03_after_successful_login"
attachment.name = "04_user_dashboard"

// ❌ 非推奨: 意味のない名前
attachment.name = "screenshot1"
attachment.name = "test"
```

### 2. 重要なタイミングで撮影

```swift
// ✅ バグ再現に必要なタイミング
// - 初期状態
// - ユーザーアクション直前
// - ユーザーアクション直後
// - エラー発生時
// - 期待される最終状態

// 例: バグ調査用
let screenshot = app.screenshot()
let attachment = XCTAttachment(screenshot: screenshot)
attachment.name = "05_BUG_target_pitch_should_be_cleared"
attachment.lifetime = .keepAlways
add(attachment)
```

### 3. スクリーンショット抽出をCI/CDに統合

```yaml
# .github/workflows/test.yml 例
- name: Run UI Tests
  run: xcodebuild test -scheme MyApp -destination '...' -resultBundlePath TestResults.xcresult

- name: Extract Screenshots
  if: failure()  # テスト失敗時のみ
  run: |
    ./scripts/extract_screenshots.sh TestResults.xcresult ./screenshots

- name: Upload Screenshots
  if: failure()
  uses: actions/upload-artifact@v3
  with:
    name: ui-test-screenshots
    path: screenshots/
```

## 📚 参考リンク

- [Apple Developer Documentation: XCTAttachment](https://developer.apple.com/documentation/xctest/xctattachment)
- [xcresulttool man page](https://keith.github.io/xcode-man-pages/xcresulttool.1.html)
- [XCUITest Best Practices](https://developer.apple.com/documentation/xctest/user_interface_tests)

## ⚠️ Xcode 16での変更点（2025-10-29 追加）

### 新しい抽出コマンド（Xcode 16+）

Xcode 16では、より簡単にアタッチメントを一括抽出できる新しいコマンドが追加されました。

#### ✅ 推奨: `xcrun xcresulttool export attachments`

```bash
# すべてのアタッチメント（スクリーンショット、ビデオ、ログなど）を一括抽出
XCRESULT_PATH="/path/to/Test-Result.xcresult"
mkdir -p /tmp/screenshots
xcrun xcresulttool export attachments --path "$XCRESULT_PATH" --output-path /tmp/screenshots
```

**重要**: フラグは`--output-path`であり、`--output`ではない（`--output`を使うとエラーになる）

**出力例**:
```
Exported 19 attachments for: VocalisStudioUITests/testTargetPitchShouldDisappearAfterStoppingPlayback():
File: 4FC9ADF5-00B7-4A33-AB11-BB2BDFBF3B6B.png, suggested name: "04_during_playback_0_66BBF1ED-D2EC-45A4-984A-519674501D8C.png"
File: 15FF732E-8D40-4F8B-A85B-4D36167C298B.png, suggested name: "03_after_recording_stopped_0_9AB3619E-4C82-4DB5-8501-5F084F8D0074.png"
...

Generated manifest file with attachment details: /tmp/screenshots/manifest.json
```

**利点**:
- 一括抽出で簡単
- `manifest.json`が自動生成され、元のファイル名とUUIDのマッピングが分かる
- SQLiteクエリ不要

#### manifest.jsonの活用

```bash
# manifest.jsonからスクリーンショット一覧を確認
cat /tmp/screenshots/manifest.json | jq -r '.[] | .attachments[] | select(.suggestedHumanReadableName | contains(".png")) | .suggestedHumanReadableName'
```

**出力例**:
```json
{
  "exportedFileName": "4FC9ADF5-00B7-4A33-AB11-BB2BDFBF3B6B.png",
  "suggestedHumanReadableName": "04_during_playback_0_66BBF1ED-D2EC-45A4-984A-519674501D8C.png",
  "timestamp": 1761696250.727
}
```

### ❌ Xcode 16で動作しなくなった方法

#### 1. SQLiteデータベースの直接クエリ

**問題点**: Xcode 16では`Attachments`テーブルが存在しない

```bash
# ❌ Xcode 16では失敗
sqlite3 "$XCRESULT_PATH/database.sqlite3" \
  "SELECT * FROM Attachments WHERE uniformTypeIdentifier = 'public.png';"
```

**エラー**:
```
Error: in prepare, no such table: Attachments
```

**理由**: Xcode 16で`.xcresult`バンドルのデータベーススキーマが変更され、`Attachments`テーブルが削除された

**解決策**: 新しい`xcrun xcresulttool export attachments`コマンドを使用

#### 2. 個別ファイルエクスポートによる手動抽出

Xcode 16以前の方法も動作するが、新しいコマンドを使う方が簡単:

```bash
# ⚠️ 動作するが非推奨（Xcode 16以降）
xcrun xcresulttool export --legacy --type file \
  --path "$XCRESULT_PATH" \
  --id "0~3Ru-WZ-RZ..." \
  --output-path /tmp/screenshot.png
```

### 推奨ワークフロー（Xcode 16+）

```bash
#!/bin/bash
# extract_screenshots_xcode16.sh - Xcode 16対応版

set -e

XCRESULT_PATH="$1"
OUTPUT_DIR="${2:-./screenshots}"

if [ -z "$XCRESULT_PATH" ]; then
    echo "Usage: $0 <path_to_xcresult> [output_dir]"
    exit 1
fi

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"

# すべてのアタッチメントを一括抽出
echo "📸 Extracting attachments from: $XCRESULT_PATH"
xcrun xcresulttool export attachments --path "$XCRESULT_PATH" --output-path "$OUTPUT_DIR"

# スクリーンショットのみリスト表示
echo ""
echo "✅ Extracted screenshots:"
ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null || echo "No PNG files found"

# manifest.jsonの確認
if [ -f "$OUTPUT_DIR/manifest.json" ]; then
    echo ""
    echo "📋 Manifest file created: $OUTPUT_DIR/manifest.json"
fi
```

**使用例**:
```bash
chmod +x extract_screenshots_xcode16.sh
./extract_screenshots_xcode16.sh ~/Library/Developer/Xcode/DerivedData/.../Test-Result.xcresult ./screenshots
```

## 履歴

- 2025-10-29: Xcode 16の新しい`export attachments`コマンドについて追記
- 2025-10-29: SQLiteデータベーススキーマ変更（Attachmentsテーブル削除）について追記
- 2025-10-29: `--output`フラグではなく`--output-path`が正しいことを明記
- 2025-10-28: 初版作成（VocalisStudioプロジェクトでの実装経験に基づく）
