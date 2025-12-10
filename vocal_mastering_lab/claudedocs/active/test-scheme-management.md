# テストスキーマ管理ガイド

VocalisStudioプロジェクトでは、複数のテストスキーマを使用してテストを効率的に管理します。

## スキーマ一覧

### 1. VocalisStudio-All
**用途**: すべてのテスト(Unit + UI)を実行
**テスト対象**:
- ✅ VocalisStudioTests (Unit Tests)
- ✅ VocalisStudioUITests (UI Tests)

**使用場面**:
- CI/CD環境での全体テスト実行
- リリース前の最終確認
- すべてのテストが正常であることを確認したい時

### 2. VocalisStudio-UIOnly (デフォルト)
**用途**: UI Testsのみを実行
**テスト対象**:
- ❌ VocalisStudioTests (スキップ)
- ✅ VocalisStudioUITests (UI Tests)

**使用場面**:
- UI機能の開発・デバッグ中
- Unit Testにコンパイルエラーがある時
- UI動作確認が必要な時

### 3. VocalisStudio-UnitOnly
**用途**: Unit Testsのみを実行
**テスト対象**:
- ✅ VocalisStudioTests (Unit Tests)
- ❌ VocalisStudioUITests (スキップ)

**使用場面**:
- ビジネスロジックの開発・デバッグ中
- TDDサイクル実行時
- 高速なフィードバックが必要な時

## スキーマの切り替え方法

### A. コマンドライン(推奨)

プロジェクトルートで以下のスクリプトを実行:

```bash
# UI Testsのみ実行
./scripts/test-runner.sh ui

# Unit Testsのみ実行
./scripts/test-runner.sh unit

# すべてのテスト実行
./scripts/test-runner.sh all

# 特定のテストクラスのみ実行
./scripts/test-runner.sh ui PaywallUITests
./scripts/test-runner.sh unit RecordingStateViewModelTests
```

### B. Xcode GUI

1. Xcodeのツールバー左上のスキーマ選択メニューをクリック
2. 目的のスキーマを選択:
   - `VocalisStudio-All` → すべてのテスト
   - `VocalisStudio-UIOnly` → UI Testsのみ
   - `VocalisStudio-UnitOnly` → Unit Testsのみ
3. `⌘+U` でテスト実行

### C. 直接xcodebuildコマンド

```bash
# UI Testsのみ
xcodebuild test \
  -project VocalisStudio.xcodeproj \
  -scheme VocalisStudio-UIOnly \
  -destination 'platform=iOS Simulator,name=iPhone 16'

# Unit Testsのみ
xcodebuild test \
  -scheme VocalisStudio-UnitOnly \
  -destination 'platform=iOS Simulator,name=iPhone 16'

# すべてのテスト
xcodebuild test \
  -scheme VocalisStudio-All \
  -destination 'platform=iOS Simulator,name=iPhone 16'
```

## よくある質問

### Q1: スキーマの変更は毎回必要ですか?
**A**: いいえ、必要ありません。

スキーマファイルは`.xcscheme`形式でプロジェクトに保存されているため、一度選択すれば設定は永続的に保持されます。次回のテスト実行時も同じスキーマが使用されます。

### Q2: なぜ複数のスキーマが必要なのですか?
**A**: 以下の理由からです:

1. **効率性**: 必要なテストだけを実行して時間を節約
2. **エラー分離**: 一方のテストにコンパイルエラーがあっても、もう一方を実行可能
3. **開発フロー最適化**: 開発中は関連するテストのみを高速実行

### Q3: どのスキーマをデフォルトにすべきですか?
**A**: 現在の開発状況によります:

- **Unit Testにエラーがある場合**: `VocalisStudio-UIOnly`(現在のデフォルト)
- **すべてのテストが正常な場合**: `VocalisStudio-All`
- **TDD開発中**: `VocalisStudio-UnitOnly`

### Q4: CI/CDではどのスキーマを使うべきですか?
**A**: `VocalisStudio-All` を使用してください。

すべてのテストを実行することで、リグレッションを確実に検出できます。

### Q5: スキーマファイルの変更履歴を確認できますか?
**A**: はい、Gitで管理されています。

```bash
# スキーマファイルの履歴確認
git log -- VocalisStudio.xcodeproj/xcshareddata/xcschemes/

# 特定のスキーマの差分確認
git diff HEAD~1 VocalisStudio.xcodeproj/xcshareddata/xcschemes/VocalisStudio-UIOnly.xcscheme
```

## トラブルシューティング

### スキーマが表示されない場合

1. Xcodeを再起動
2. プロジェクトを閉じて再度開く
3. スキーマファイルの存在確認:
   ```bash
   ls -la VocalisStudio.xcodeproj/xcshareddata/xcschemes/
   ```

### テストが実行されない場合

1. 選択されているスキーマを確認
2. Destination(シミュレータ)が正しいか確認
3. Derived Dataをクリーンアップ:
   ```bash
   rm -rf ~/Library/Developer/Xcode/DerivedData/VocalisStudio-*
   ```

## スキーマファイルの構造

各`.xcscheme`ファイルは以下の情報を含みます:

```xml
<TestableReference skipped="NO|YES" parallelizable="YES">
  <BuildableReference ... BlueprintName="VocalisStudioTests" />
</TestableReference>
```

- `skipped="NO"`: テスト実行対象
- `skipped="YES"`: テスト実行対象外(ビルドもスキップ)

## 関連ファイル

- **スキーマファイル**: `VocalisStudio.xcodeproj/xcshareddata/xcschemes/*.xcscheme`
- **テスト実行スクリプト**: `scripts/test-runner.sh`
- **このドキュメント**: `claudedocs/test-scheme-management.md`

## 更新履歴

- **2025-11-06**: 初版作成 - 3つのスキーマとtest-runner.shスクリプトを追加
