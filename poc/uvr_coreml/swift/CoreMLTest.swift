#!/usr/bin/swift
import Foundation
import CoreML
import Accelerate

// CoreMLモデルテストスクリプト
// Swift単体でCoreMLモデルの動作を確認

print("=" + String(repeating: "=", count: 79))
print("🍎 CoreML Swift統合テスト")
print("=" + String(repeating: "=", count: 79))

// モデルパス
let currentDir = FileManager.default.currentDirectoryPath
let modelPath = URL(fileURLWithPath: currentDir)
    .deletingLastPathComponent()
    .appendingPathComponent("models/coreml/UVR-MDX-NET-Inst_Main.mlpackage")

print("\n📂 モデルパス: \(modelPath.path)")

// モデル存在確認
guard FileManager.default.fileExists(atPath: modelPath.path) else {
    print("❌ エラー: モデルが見つかりません")
    exit(1)
}

print("✅ モデルファイル確認完了")

// モデル読み込み
print("\n🔄 モデル読み込み中...")

let config = MLModelConfiguration()
config.computeUnits = .all  // CPU + GPU + Neural Engine

guard let model = try? MLModel(contentsOf: modelPath, configuration: config) else {
    print("❌ エラー: モデル読み込み失敗")
    exit(1)
}

print("✅ モデル読み込み完了")

// モデル情報表示
let modelDescription = model.modelDescription
print("\n📋 モデル情報:")

print("  入力:")
for input in modelDescription.inputDescriptionsByName {
    print("    \(input.key): \(input.value)")
}

print("  出力:")
for output in modelDescription.outputDescriptionsByName {
    print("    \(output.key): \(output.value)")
}

// テストデータ生成
print("\n🎯 テストデータ生成中...")

let batchSize = 1
let channels = 4
let freqBins = 2048
let timeFrames = 256

// MLMultiArray作成
guard let inputArray = try? MLMultiArray(
    shape: [batchSize, channels, freqBins, timeFrames] as [NSNumber],
    dataType: .float32
) else {
    print("❌ エラー: 入力配列作成失敗")
    exit(1)
}

// ランダムデータで初期化
let count = inputArray.count
for i in 0..<count {
    inputArray[i] = NSNumber(value: Float.random(in: -1...1))
}

print("  入力形状: [\(batchSize), \(channels), \(freqBins), \(timeFrames)]")
print("  要素数: \(count)")

// 推論実行
print("\n🔄 推論実行中...")

let startTime = Date()

let input = try! MLDictionaryFeatureProvider(dictionary: [
    "input_1": MLFeatureValue(multiArray: inputArray)
])

guard let output = try? model.prediction(from: input) else {
    print("❌ エラー: 推論失敗")
    exit(1)
}

let elapsed = Date().timeIntervalSince(startTime)

print("✅ 推論成功!")
print("⏱️  実行時間: \(String(format: "%.4f", elapsed)) 秒")

// 出力確認
if let outputArray = output.featureValue(for: "var_992")?.multiArrayValue {
    print("\n📊 出力情報:")
    print("  形状: \(outputArray.shape)")
    print("  型: \(outputArray.dataType)")

    // 統計計算
    var sum: Float = 0
    var minVal: Float = Float.greatestFiniteMagnitude
    var maxVal: Float = -Float.greatestFiniteMagnitude

    for i in 0..<outputArray.count {
        let val = outputArray[i].floatValue
        sum += val
        minVal = min(minVal, val)
        maxVal = max(maxVal, val)
    }

    let mean = sum / Float(outputArray.count)

    print("  範囲: [\(String(format: "%.6f", minVal)), \(String(format: "%.6f", maxVal))]")
    print("  平均: \(String(format: "%.6f", mean))")
}

print("\n" + String(repeating: "=", count: 80))
print("✅ Swift/CoreML統合テスト完了")
print(String(repeating: "=", count: 80))

print("\n次のステップ:")
print("  1. STFT処理の統合")
print("  2. AVAudioFileとの連携")
print("  3. 完全な音源分離パイプライン実装")
