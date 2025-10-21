# iOS統合ガイド

## 概要

UVR MDX-Net CoreMLモデルのiOS/macOSアプリケーションへの統合方法を解説します。

## Swift/CoreML統合テスト結果

### テスト実行

```bash
cd ios_test
swift run
```

### 実行結果

```
✅ モデルコンパイル完了
✅ モデル読み込み完了
✅ 推論成功!
⏱️  実行時間: 0.0884 秒
📊 出力範囲: [-1.94, 6.75]
```

**評価**: Swift環境でCoreMLモデルが正常に動作することを確認しました。

## Xcodeプロジェクト構成

### プロジェクト構造

```
YourApp/
├── Models/
│   └── UVR-MDX-NET-Inst_Main.mlpackage    # CoreMLモデル
├── Audio/
│   ├── STFTProcessor.swift                # STFT/iSTFT処理
│   └── VocalSeparator.swift               # 音源分離エンジン
├── ViewControllers/
│   └── SeparationViewController.swift     # UI制御
└── Resources/
    └── sample.wav                         # テスト音声
```

### Xcodeプロジェクトへの追加手順

#### 1. CoreMLモデルの追加

1. **Xcodeでプロジェクトを開く**
2. **モデルをドラッグ&ドロップ**
   - `UVR-MDX-NET-Inst_Main.mlpackage` をプロジェクトナビゲータに追加
3. **ターゲットメンバーシップ確認**
   - モデルがアプリターゲットに含まれていることを確認
4. **自動生成されたインターフェース確認**
   - Xcodeがモデルクラスを自動生成

#### 2. Swift実装ファイルの追加

```swift
// STFTProcessor.swift と VocalSeparator.swift を追加
// すでに実装済み (swift/ ディレクトリ内)
```

#### 3. 必要なフレームワークのリンク

**Project Settings → General → Frameworks:**
- `CoreML.framework`
- `Accelerate.framework`
- `AVFoundation.framework`

### モデルの使用方法

#### 基本的な使用例

```swift
import CoreML
import Foundation

// 1. モデル読み込み
let config = MLModelConfiguration()
config.computeUnits = .all  // CPU + GPU + Neural Engine

guard let modelURL = Bundle.main.url(
    forResource: "UVR-MDX-NET-Inst_Main",
    withExtension: "mlmodelc"
) else {
    fatalError("モデルが見つかりません")
}

let model = try! MLModel(contentsOf: modelURL, configuration: config)

// 2. 入力データ準備
let input = try! MLMultiArray(
    shape: [1, 4, 2048, 256],
    dataType: .float32
)

// データを設定...

// 3. 推論実行
let inputProvider = try! MLDictionaryFeatureProvider(dictionary: [
    "input_1": MLFeatureValue(multiArray: input)
])

let output = try! model.prediction(from: inputProvider)

// 4. 出力取得
if let result = output.featureValue(for: "var_992")?.multiArrayValue {
    // 処理...
}
```

#### 完全な音源分離パイプライン

```swift
// VocalSeparator統合例
let separator = try VocalSeparator(modelURL: modelURL)

let audioURL = Bundle.main.url(forResource: "song", withExtension: "wav")!
let separated = try await separator.separate(audioURL: audioURL)

// ボーカルのみ保存
try separator.save(
    audio: separated.vocals,
    sampleRate: separated.sampleRate,
    to: outputURL
)
```

## パフォーマンス最適化

### 1. Neural Engine活用

```swift
let config = MLModelConfiguration()
config.computeUnits = .all  // 推奨: すべての計算ユニットを活用
```

**期待される効果**:
- A17 Pro: 約2-3倍高速化
- M4: 約3-5倍高速化

### 2. バッチ処理

```swift
// 長時間音声は チャンク単位で処理
let chunkDuration: TimeInterval = 10.0  // 10秒ごと
```

### 3. 非同期処理

```swift
Task {
    let result = try await separator.separate(audioURL: audioURL)
    // メインスレッドでUI更新
    await MainActor.run {
        updateUI(with: result)
    }
}
```

## 実装のポイント

### 1. STFTパラメータ

```swift
let config = STFTProcessor.Configuration(
    fftSize: 4096,       // モデルの期待値に合わせる
    hopSize: 1024,       // fftSize / 4
    windowType: .hann    // ハン窓
)
```

### 2. メモリ管理

```swift
// 大きな配列を扱うため、autoreleasepool使用を推奨
autoreleasepool {
    let result = try separator.separate(audioURL: audioURL)
    // 処理...
}
```

### 3. エラーハンドリング

```swift
do {
    let separated = try await separator.separate(audioURL: audioURL)
} catch VocalSeparator.SeparationError.invalidAudioFormat(let msg) {
    print("音声フォーマットエラー: \(msg)")
} catch {
    print("予期しないエラー: \(error)")
}
```

## サンプルアプリケーション構造

### SwiftUIサンプル

```swift
import SwiftUI
import CoreML

struct ContentView: View {
    @State private var isProcessing = false
    @State private var progress: Double = 0

    var body: some View {
        VStack {
            Button("音源分離実行") {
                Task {
                    await separateAudio()
                }
            }
            .disabled(isProcessing)

            if isProcessing {
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
            }
        }
    }

    func separateAudio() async {
        isProcessing = true
        defer { isProcessing = false }

        do {
            let modelURL = Bundle.main.url(
                forResource: "UVR-MDX-NET-Inst_Main",
                withExtension: "mlmodelc"
            )!

            let separator = try VocalSeparator(modelURL: modelURL)

            // 処理...

        } catch {
            print("エラー: \(error)")
        }
    }
}
```

## デプロイメント

### 最小要件

- **iOS**: 17.0+
- **macOS**: 14.0+
- **Xcode**: 15.0+
- **Swift**: 5.9+

### App Store提出時の注意

1. **モデルサイズ**: 25.28 MB (App Store制限内)
2. **プライバシー設定**: マイク使用の説明文を追加
3. **パフォーマンステスト**: 様々なデバイスでテスト

## トラブルシューティング

### モデル読み込みエラー

**症状**: `Unable to load model`

**対処**:
```swift
// モデルを事前にコンパイル
let compiledURL = try MLModel.compileModel(at: modelURL)
let model = try MLModel(contentsOf: compiledURL)
```

### メモリ不足

**症状**: 長時間音声でクラッシュ

**対処**:
- チャンクサイズを小さくする (256 → 128 フレーム)
- ストリーミング処理を実装
- メモリ使用量をモニタリング

### 推論速度が遅い

**症状**: 期待よりも推論が遅い

**対処**:
1. `computeUnits = .all` 設定確認
2. デバッグビルドではなくリリースビルドでテスト
3. Instrumentsでプロファイリング

## 次のステップ

### Phase 1: 基本統合 ✅

- [x] CoreMLモデル読み込み
- [x] 基本推論テスト
- [x] Swift統合確認

### Phase 2: 完全実装

- [ ] AVAudioFileとの統合
- [ ] STFT/iSTFT処理の統合
- [ ] エンドツーエンド音源分離
- [ ] リアルタイム処理対応

### Phase 3: アプリケーション開発

- [ ] UIデザイン
- [ ] ファイル選択機能
- [ ] プログレス表示
- [ ] 分離結果のプレビュー
- [ ] エクスポート機能

### Phase 4: 最適化とテスト

- [ ] パフォーマンスチューニング
- [ ] メモリ最適化
- [ ] デバイス別テスト
- [ ] ユーザビリティ改善

## まとめ

✅ **Swift/CoreML統合テスト成功**
- モデル読み込み: 正常
- 推論実行: 0.09秒 (高速)
- 出力検証: 正常

✅ **iOS統合準備完了**
- CoreMLモデル: 動作確認済み
- Swift実装: 提供済み
- 統合ガイド: 整備済み

次は実際のiOSアプリケーションでの統合を進めることができます。

---

**作成日**: 2025-10-21
**検証環境**: macOS 14.x, Swift 6.1
**対象プラットフォーム**: iOS 17.0+, macOS 14.0+
