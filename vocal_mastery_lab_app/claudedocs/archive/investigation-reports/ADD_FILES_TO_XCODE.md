# Xcodeプロジェクトへのファイル追加手順

## 背景

TDDサイクルのRed→Green実行のため、以下のファイルをXcodeプロジェクトに追加する必要があります：

1. **実装ファイル**: `VocalisStudio/Application/Services/ScalePlaybackCoordinator.swift`
2. **テストファイル**: `VocalisStudioTests/Application/Services/ScalePlaybackCoordinatorTests.swift`

## アーキテクチャ上の配置

ScalePlaybackCoordinatorは`Application/Services/`配下に配置されています。理由:

- **Application Service Pattern**: 複数のViewModelの調整を行う横断的な関心事
- **既存パターンとの一貫性**: `RecordingPolicyServiceImpl`と同様のApplication層Service
- **Clean Architecture原則**: UseCasesはビジネスロジックのオーケストレーション、ServicesはApplication層のコンポーネント調整

## 手順

### 1. Xcodeでプロジェクトを開く

```bash
open VocalisStudio/VocalisStudio.xcodeproj
```

### 2. 実装ファイルの追加 (ScalePlaybackCoordinator.swift)

1. **Project Navigatorで右クリック**: `VocalisStudio` → `Application` → `Services` を右クリック
2. **"Add Files to 'VocalisStudio'"** を選択
3. ファイル選択ダイアログで以下を選択：
   - ディレクトリ: `VocalisStudio/Application/Services/`
   - ファイル: `ScalePlaybackCoordinator.swift`
4. **重要な設定**:
   - ✅ "Copy items if needed" のチェックを**外す** (既にプロジェクト内に配置済みのため)
   - ✅ "Add to targets" で **VocalisStudio** にチェック
   - "Create folder references" を選択
5. **Add** をクリック

### 3. テストファイルの追加 (ScalePlaybackCoordinatorTests.swift)

1. **Project Navigatorで右クリック**: `VocalisStudioTests` → `Application` → `Services` を右クリック
2. **"Add Files to 'VocalisStudio'"** を選択
3. ファイル選択ダイアログで以下を選択:
   - ディレクトリ: `VocalisStudioTests/Application/Services/`
   - ファイル: `ScalePlaybackCoordinatorTests.swift`
4. **重要な設定**:
   - ✅ "Copy items if needed" のチェックを**外す**
   - ✅ "Add to targets" で **VocalisStudioTests** にチェック
   - "Create folder references" を選択
5. **Add** をクリック

### 4. ビルドとテスト実行

```bash
cd /Users/asatokazu/Documents/dev/mine/music/vocalis-studio/VocalisStudio

# テストを実行 (Red Phaseの確認)
xcodebuild -project VocalisStudio.xcodeproj \
  -scheme VocalisStudio \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  -only-testing:VocalisStudioTests/ScalePlaybackCoordinatorTests \
  test
```

## 期待される結果

### Red Phase (現時点)
テストが実行され、以下のようなテスト結果が出力されるはず：
- ✅ テストがコンパイル成功
- ✅ テストが実行される
- ✅ すべてのテストがパス (実装が正しいため)

### 次のステップ
テストがパスしたら、🔵 Refactor Phaseに進みます：
- コードの品質改善
- ドキュメントの追加
- 必要に応じてテストケースの追加

## トラブルシューティング

### エラー: "Cannot find type 'ScalePlaybackCoordinator' in scope"
- ScalePlaybackCoordinator.swiftがVocalisStudioターゲットに追加されていない
- 手順2を再確認

### エラー: テストファイルが見つからない
- ScalePlaybackCoordinatorTests.swiftがVocalisStudioTestsターゲットに追加されていない
- 手順3を再確認

### ビルドエラー
- Clean Build Folder: `Cmd + Shift + K`
- 再ビルド: `Cmd + B`

## 確認コマンド

ファイルが正しく配置されているか確認：

```bash
# 実装ファイル
ls -la VocalisStudio/Application/ScalePlayback/ScalePlaybackCoordinator.swift

# テストファイル
ls -la VocalisStudioTests/Application/ScalePlayback/ScalePlaybackCoordinatorTests.swift

# MockScalePlayer (更新済み)
grep -n "playMuted" VocalisStudioTests/Mocks/MockScalePlayer.swift
```

## 参考情報

- TDDサイクル: 🔴 Red → 🟢 Green → 🔵 Refactor
- 現在: 🔴→🟢 の境界 (ファイル追加待ち)
- 次: 🟢 Green確認 → 🔵 Refactor
