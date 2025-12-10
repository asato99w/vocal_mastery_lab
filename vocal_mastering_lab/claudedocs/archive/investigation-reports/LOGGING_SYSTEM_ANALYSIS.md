# ロギングシステムの現状分析と改善提案

**作成日**: 2025-10-26
**最終更新**: 2025-10-30
**調査理由**: ピッチ検出バグのデバッグ時に、ログが取得できない問題を調査

## ⚠️ 重要: UIテスト後の調査にはFileLoggerを第一選択とする

**結論**: これまでの取得失敗の主因（時刻窓ずれ／UDID誤り／起動前提コマンド）はそのまま有効。**UIテスト後の確実な追跡にはFileLoggerを第一選択にする**。

**新事実 (2025-10-30)**:
- `Logger.viewModel.info/debug/error()` は**OSLogのみ**（FileLoggerには書かれない）
- FileLoggerに確実に残すには `Logger.viewModel.logToFile()` または `FileLogger.shared.log()` を使う
- コンテナIDは再インストールで変わるため、最新ログを動的に探す処理が必須

**推奨順序の更新**:
1. **FileLogger (`logToFile`) を既定** - UIテスト後の解析で確実
2. **OSLogは補助** - リアルタイムデバッグやシステム連携向け
3. **重要イベントは二重書き込み** - `Logger.info()` + `Logger.logToFile()` 併用

## 現状のロギングアーキテクチャ

### 1. 利用可能なロギング機構

VocalisStudioには3つのロギング機構が存在します：

#### A. FileLogger (Infrastructure層)
- **ファイルパス**: `VocalisStudio/Infrastructure/Logging/FileLogger.swift`
- **目的**: DEBUGビルド時にファイルベースのログを出力
- **出力先**: `Documents/logs/vocalis_YYYY-MM-DDTHH:mm:ss.log`
- **特徴**:
  - シングルトン (`FileLogger.shared`)
  - `#if DEBUG` ブロック内でのみ動作
  - 非同期書き込み (`DispatchQueue`)
  - 自動ローテーション (5MB制限、最大5ファイル保持)

**使用方法**:
```swift
FileLogger.shared.log(level: "INFO", category: "viewmodel", message: "Recording started")
```

**⚠️ コンテナID変動対策**: 再インストール後は新しいContainer UUIDが作成されるため、最新ログを動的に検索:
```bash
# 最新ログファイルを検索（推奨）
UDID="508462B0-4692-4B9B-88F9-73A63F9B91F5"
find ~/Library/Developer/CoreSimulator/Devices/$UDID/data/Containers/Data/Application \
  -name "vocalis_*.log" -type f -exec stat -f "%m %N" {} + 2>/dev/null \
  | sort -rn | head -1 | cut -d' ' -f2-
```

#### B. Logger+Extensions (Presentation層)
- **ファイルパス**: `VocalisStudio/Infrastructure/Logging/Logger+Extensions.swift`
- **目的**: OSLogのカテゴリ別ラッパー
- **特徴**:
  - Apple標準のOSLogを使用
  - カテゴリ別のLoggerインスタンスを提供 (`Logger.viewModel`, `Logger.recording`など)
  - **⚠️重要**: `Logger.info()`, `Logger.debug()`などはFileLoggerに記録**されない**

**使用方法**:
```swift
Logger.viewModel.info("Recording started")  // OSLogのみ、ファイルには記録されない
```

**FileLoggerに記録されるメソッド** (明示的にFileLogger.shared.log()を呼ぶもののみ):
```swift
Logger.viewModel.logError(error)     // FileLoggerに記録される
Logger.viewModel.logCritical(message) // FileLoggerに記録される
```

#### C. OSLogAdapter (Infrastructure層)
- **ファイルパス**: `VocalisStudio/Infrastructure/Logging/OSLogAdapter.swift`
- **目的**: Domain層のLoggerProtocolを実装
- **特徴**:
  - Clean Architectureに準拠した依存性逆転
  - **すべてのログメソッドがFileLoggerに記録される**
  - Application層（UseCaseなど）で使用

**使用方法**:
```swift
let logger = OSLogAdapter(category: "useCase")
logger.info("Recording started", category: "useCase")  // OSLog + FileLoggerの両方
```

### 2. 実際のログ記録状況

#### テスト実行時のログファイル内容の例

**スケール設定あり (settings != nil)**:
```
2025-10-26 13:24:28.559 [INFO] [viewmodel] RecordingViewModel.startRecording() called, settings = present
2025-10-26 13:24:28.563 [DEBUG] [viewmodel] Recording started through state VM
2025-10-26 13:24:28.569 [INFO] [viewmodel] Settings present, starting pitch detection...
2025-10-26 13:24:28.569 [DEBUG] [viewmodel] ✅ Target pitch monitoring started
2025-10-26 13:24:31.071 [INFO] [viewmodel] ✅ Realtime pitch detection started
2025-10-26 13:24:31.072 [INFO] [viewmodel] RecordingViewModel.startRecording() completed
```

**スケール設定なし (settings = nil)**:
```
2025-10-26 13:25:44.732 [INFO] [viewmodel] RecordingViewModel.startRecording() called, settings = nil
2025-10-26 13:25:44.733 [DEBUG] [viewmodel] Recording started through state VM
2025-10-26 13:25:44.733 [WARNING] [viewmodel] ⚠️ No settings provided, pitch detection NOT started
2025-10-26 13:25:44.733 [INFO] [viewmodel] RecordingViewModel.startRecording() completed
```

**実アプリ実行時のログファイル** (2025-10-25のログ):
- `[useCase]`, `[audio]`, `[recording]`, `[scalePlayer]` カテゴリのみ記録
- `[viewmodel]`, `[pitch]` カテゴリは**一切記録されていない**

### 3. 各層でのロギング使用状況

| 層 | 使用しているLogger | FileLoggerに記録 |
|----|-------------------|------------------|
| Domain | なし (純粋なビジネスロジック) | - |
| Application | OSLogAdapter | ✅ Yes |
| Infrastructure | OSLogAdapter | ✅ Yes |
| Presentation | Logger+Extensions | ❌ No (logError/logCriticalのみYes) |

## 問題点

### 1. 一貫性のないロギングAPI

- **Presentation層**: `Logger.viewModel.info()` → ファイルに記録されない
- **Application層**: `OSLogAdapter.info()` → ファイルに記録される

同じ「info」メソッドでも、使う場所によって動作が異なるため混乱を招く。

### 2. Logger+Extensionsのコメントが不正確

**Logger+Extensions.swift:88-90**のコメント:
```swift
// Note: OSLog methods (info, debug, warning, error) automatically log to both
// system log and file in debug builds through OSLog observation.
```

このコメントは**誤り**です。実際には`info()`, `debug()`, `warning()`, `error()`はFileLoggerに記録されません。

### 3. Presentation層のログがファイルに残らない

ViewModelやViewのログはOSLogにしか記録されないため：
- テスト実行時にログファイルから確認できない
- OSLogは揮発性のため、後から確認しづらい
- デバッグが困難

### 4. FileLoggerの直接呼び出しが必要

Presentation層でファイルログを残すには、`FileLogger.shared.log()`を直接呼ぶ必要がある：
```swift
FileLogger.shared.log(level: "INFO", category: "viewmodel", message: "...")
```

これはクリーンアーキテクチャの観点から望ましくない（Presentation層がInfrastructure層の具象に依存）。

## 改善提案

### 提案1: Logger+ExtensionsをOSLogAdapterと統一

**目的**: すべての層で一貫したロギングAPI

**実装方法**:
```swift
// Logger+Extensions.swift を修正
extension Logger {
    static let viewModel = Logger(subsystem: subsystem, category: "viewmodel")

    // すべてのログメソッドでFileLoggerも呼び出す
    func info(_ message: String) {
        self.log(level: .info, "\(message)")
        FileLogger.shared.log(level: "INFO", category: self.category, message: message)
    }

    func debug(_ message: String) {
        self.log(level: .debug, "\(message)")
        FileLogger.shared.log(level: "DEBUG", category: self.category, message: message)
    }

    // warning, errorも同様
}
```

**メリット**:
- すべての層でファイルログが残る
- 既存コードの変更が最小限
- 一貫したAPI

**デメリット**:
- OSLogのString interpolation最適化が使えなくなる可能性

### 提案2: Presentation層専用のLoggerProtocolを導入

**目的**: Clean Architectureを維持しつつ、Presentation層でもファイルログを残す

**実装方法**:
1. Domain層に`LoggerProtocol`を定義（既存）
2. Presentation層用の`PresentationLogger`をInfrastructure層に実装
3. DependencyContainerで`PresentationLogger`を注入

```swift
// Infrastructure/Logging/PresentationLogger.swift
public final class PresentationLogger: LoggerProtocol {
    private let osLogger: Logger
    private let fileLogger: FileLogger

    public init(category: String) {
        self.osLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "", category: category)
        self.fileLogger = FileLogger.shared
    }

    public func info(_ message: String, category: String) {
        osLogger.info("\(message)")
        fileLogger.log(level: "INFO", category: category, message: message)
    }

    // debug, warning, errorも同様
}

// DependencyContainer.swift
let viewModelLogger = PresentationLogger(category: "viewmodel")

// RecordingViewModel.swift
public class RecordingViewModel: ObservableObject {
    private let logger: LoggerProtocol

    public init(..., logger: LoggerProtocol) {
        self.logger = logger
    }

    public func startRecording(...) async {
        logger.info("Recording started", category: "viewmodel")
    }
}
```

**メリット**:
- Clean Architecture準拠
- テスト時にモックLoggerを注入可能
- すべての層で統一されたAPI

**デメリット**:
- ViewModelの初期化にlogger引数追加が必要
- やや複雑

### 提案3: 現状維持 + ドキュメント整備

**目的**: 最小限の変更で運用改善

**実装方法**:
1. Logger+Extensions.swiftの不正確なコメントを修正
2. 使い分けガイドラインをドキュメント化
3. デバッグ時は`FileLogger.shared.log()`を直接使用することを明記

**メリット**:
- コード変更が最小限
- すぐに実施可能

**デメリット**:
- 一貫性の問題は解決されない
- Presentation層のログがファイルに残らない問題は継続

## 推奨アプローチ

**短期的 (即座に実施)**:
- 提案3を実施してドキュメント整備
- Logger+Extensions.swiftのコメント修正

**中長期的 (次のリファクタリング時)**:
- 提案1を実施してLogger+Extensionsを修正
- パフォーマンステストを実施してOSLogの最適化への影響を確認

## テスト時のログ確認方法

### 方法1: ユニットテストからのログファイル確認

```swift
// テストコード例
func testSomething() async throws {
    // 1. ログを直接書き込み
    FileLogger.shared.log(level: "INFO", category: "test", message: "Test started")

    // 2. テスト対象コードを実行
    await sut.someMethod()

    // 3. ログファイルパスを取得
    let logPath = FileLogger.shared.currentLogPath

    // 4. 待機してログが書き込まれるのを確保
    try? await Task.sleep(nanoseconds: 500_000_000) // 0.5秒

    // 5. ログファイルを読み込み
    if let logContent = try? String(contentsOfFile: logPath, encoding: .utf8) {
        print(logContent)
        // または /tmp/test_result.txt に出力
        try? logContent.write(toFile: "/tmp/test_result.txt", atomically: true, encoding: .utf8)
    }
}
```

### 方法2: UIテスト実行時のログ確認（✅ 2025-10-28成功）

**これまでの失敗理由**:
1. **時刻指定の問題** (`--start`/`--end`):
   - `date` コマンドで取得した時刻とログシステムの時刻がずれていた
   - 時刻フォーマットが厳密で、少しでも間違うとログが取得できない
   - テスト実行前後の時刻を正確に記録するのが�煩雑

2. **プロセス名の問題**:
   - UIテスト実行時はアプリプロセス名が通常実行と異なる場合がある
   - `VocalisStudio` だけでなく `VocalisStudio-Runner` なども候補になる
   - プロセスIDが毎回変わるため特定が困難

3. **ログレベルの問題**:
   - デフォルトでは `info` レベル以下のログが表示されない
   - `--debug` オプションを明示的に指定しないとデバッグログが取得できない
   - OSLogの仕様で、`debug` レベルのログはメモリに保持される期間が短い

4. **subsystemの問題**:
   - `subsystem == "com.kazuasato.VocalisStudio"` だけでは不十分
   - カテゴリ指定（`category == "viewmodel"`）を追加すると複雑になりすぎてマッチしない
   - OR条件の書き方が不適切だった（括弧の位置など）

5. **FileLoggerへの誤解**:
   - `Logger.viewModel.info()` がFileLoggerに記録されると誤解していた
   - UIテスト実行時にFileLoggerがどこに書き込むか不明瞭だった
   - シミュレータ内のログファイルパスを特定するのが困難

**成功した方法**:
```bash
# シンプルな方法: 最近5分間のログから対象プロセスのみをフィルタ
xcrun simctl spawn <SIMULATOR_UDID> log show \
  --style syslog \
  --predicate 'process == "VocalisStudio" OR subsystem == "com.kazuasato.VocalisStudio"' \
  --last 5m \
  --debug --info \
  | grep -E "\[DIAG\]|RecordingStateViewModel|startRecording|executeRecording" \
  | tail -100
```

**具体例** (実際に成功したコマンド):
```bash
xcrun simctl spawn 508462B0-4692-4B9B-88F9-73A63F9B91F5 log show \
  --style syslog \
  --predicate 'process == "VocalisStudio" OR subsystem == "com.kazuasato.VocalisStudio"' \
  --last 5m \
  --debug --info \
  | grep -E "\[DIAG\]|RecordingStateViewModel|startRecording|executeRecording" \
  | tail -100
```

**成功の要因**:
1. ✅ **`--last 5m` の使用**: 時刻指定の複雑さを完全に回避
   - `--start`/`--end` を使わず、相対時刻で指定
   - テスト実行後すぐにコマンドを実行すれば確実にログが取得できる

2. ✅ **OR条件の正しい書き方**: `process == "VocalisStudio" OR subsystem == "com.kazuasato.VocalisStudio"`
   - プロセス名とsubsystemの両方を条件にすることで取りこぼしを防ぐ
   - カテゴリ指定を含めない（複雑になりすぎてマッチしない）

3. ✅ **`--debug --info` の明示的指定**: デフォルトでは取得できないデバッグログを確実に取得
   - `Logger.viewModel.debug()` や `print("[DIAG]...")` が取得できる

4. ✅ **`grep` による後処理**: predicateで絞り込むのではなく、取得後にフィルタ
   - predicateを複雑にすると失敗しやすい
   - シンプルなpredicateで全体を取得し、grepで必要な部分を抽出

5. ✅ **`tail -100` で可読性向上**: 膨大なログから最新の関連部分のみ表示

**取得できたログの例**:
```
2025-10-28 16:22:45.065987+0900  localhost VocalisStudio[68666]: (VocalisStudio.debug.dylib) [com.kazuasato.VocalisStudio:viewmodel] RecordingStateViewModel initialized
2025-10-28 16:22:53.786781+0900  localhost VocalisStudio[68666]: (VocalisStudio.debug.dylib) [com.kazuasato.VocalisStudio:viewmodel] [RecordingStateViewModel.swift:325] executeRecording(settings:) - Error: この機能を使用するにはプレミアムプランが必要です
```

**重要な注意点**:
- UIテスト実行直後にコマンドを実行すること（ログが残っている間に）
- `--last 5m` の時間は必要に応じて調整可能（`1m`, `10m`, `1h`など）
- `grep` のパターンは調査対象に応じて変更
- シミュレータUDIDは `xcrun simctl list devices` で確認

**デバッグマーカーの活用**:
コード内に `[DIAG]` などのマーカーをprintに追加しておくと、grepで簡単にフィルタリングできます：
```swift
print("[DIAG] startRecording START: state=\(recordingState)")
```

### 方法3: 実アプリのログ確認

シミュレーター実行時:
```bash
# 最新のログファイルを検索
find ~/Library/Developer/CoreSimulator/Devices -name "vocalis_*.log" -type f -exec ls -lt {} + | head -1

# ログファイルを表示
cat <ログファイルパス>
```

## 関連ファイル

- `VocalisStudio/Infrastructure/Logging/FileLogger.swift` - ファイルベースロガー
- `VocalisStudio/Infrastructure/Logging/Logger+Extensions.swift` - OSLogラッパー
- `VocalisStudio/Infrastructure/Logging/OSLogAdapter.swift` - LoggerProtocol実装
- `VocalisDomain/RepositoryProtocols/LoggerProtocol.swift` - Domain層のLogger抽象

## 参考: 今回のバグ調査で判明したこと

**バグ**: ピッチ検出が全く動作しない

**原因**: RecordingViewModel.startRecording()で`settings = nil`の場合、ピッチ検出が開始されない

**ログからの証拠**:
```
[WARNING] [viewmodel] ⚠️ No settings provided, pitch detection NOT started
```

**調査が困難だった理由**:
- Presentation層のログがFileLoggerに記録されていなかった
- Logger.viewModel.info()がファイルに残らないことを知らなかった
- ログシステムの仕様が明確にドキュメント化されていなかった

## 診断ログのベストプラクティス (2025-10-28 追加)

### ❌ 避けるべきパターン

**1. `print()` 文による診断ログ**
```swift
print("[DIAG] playLastRecording called")
print("[DIAG] lastRecordingURL: \(lastRecordingURL)")
```

**問題点**:
- OSLogに検索可能な形で出力されない
- `log show` や `grep` でフィルタリングできない
- FileLoggerにも記録されない

**2. `Logger.viewModel.debug()` でのselfキャプチャ忘れ**
```swift
// ❌ コンパイルエラー: selfの明示的なキャプチャが必要
Logger.viewModel.debug("URL: \(lastRecordingURL)")
```

**エラーメッセージ**:
```
error: reference to property 'lastRecordingURL' in closure requires explicit use of 'self' to make capture semantics explicit
```

### ✅ 推奨パターン

**`Logger.viewModel.debug()` を使用し、selfを明示的にキャプチャ**
```swift
Logger.viewModel.debug("🔵 playLastRecording() called")
Logger.viewModel.debug("🔵 lastRecordingURL: \(String(describing: self.lastRecordingURL))")
Logger.viewModel.debug("🔵 lastRecordingSettings: \(String(describing: self.lastRecordingSettings))")
```

**メリット**:
- OSLogとFileLoggerの両方に出力される
- `log show --predicate 'category == "viewmodel"'` でフィルタリング可能
- 絵文字マーカー (🔵) で視認性向上
- `self` を明示することでSwiftの所有権セマンティクスが明確になる

**確認されたログ出力例** (2025-10-28 17:50のテスト実行):
```
# OSLog
2025-10-28 17:50:34.842387+0900 VocalisStudio[93733]: [com.kazuasato.VocalisStudio:viewmodel] Starting recording with settings: 5-tone scale

# FileLogger
2025-10-28 17:50:34.852 [INFO] [viewmodel] RecordingViewModel.startRecording() called, settings = present
```
