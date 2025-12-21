import Foundation
import CoreML
import AVFoundation
import Accelerate

// VocalSeparatorCompleteを使用したボーカル抽出テスト
@available(iOS 17.0, macOS 14.0, *)
func runProperSeparationTest() throws {
    print(String(repeating: "=", count: 80))
    print("🎵 ボーカル抽出テスト (VocalSeparatorComplete)")
    print(String(repeating: "=", count: 80))

    // パス設定（カレントディレクトリから相対パス）
    let currentDir = FileManager.default.currentDirectoryPath
    let baseURL = URL(fileURLWithPath: currentDir)
    let modelPath = baseURL.appendingPathComponent("models/coreml/UVR-MDX-NET-Inst_Main.mlpackage")
    let inputPath = baseURL.appendingPathComponent("tests/output/hollow_crown_from_flac.wav")
    let outputDir = baseURL.appendingPathComponent("tests/swift_output")

    try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

    let vocalsPath = outputDir.appendingPathComponent("hollow_crown_vocals.wav")

    print("\n📂 入力:")
    print("  音声: \(inputPath.lastPathComponent)")
    print("  モデル: \(modelPath.lastPathComponent)")

    // モデル設定
    let config = VocalSeparatorComplete.ModelConfiguration(
        fftSize: 4096,
        hopSize: 1024,
        sampleRate: 44100,
        chunkSize: 256
    )

    // VocalSeparator初期化
    print("\n🔧 VocalSeparatorComplete初期化中...")
    let separator = try VocalSeparatorComplete(
        modelURL: modelPath,
        configuration: config
    )

    // ボーカル抽出実行
    print("\n🎵 ボーカル抽出実行中...")
    print("  注: 処理には数分かかる場合があります")

    let startTime = Date()

    let result = try separator.separate(audioURL: inputPath)

    let elapsed = Date().timeIntervalSince(startTime)

    print("✅ 抽出完了")
    print("  処理時間: \(String(format: "%.2f", elapsed))秒")

    // 結果保存
    print("\n💾 結果保存中...")
    try separator.save(
        separatedAudio: result,
        vocalsURL: vocalsPath
    )

    print("\n" + String(repeating: "=", count: 80))
    print("✅ テスト完了")
    print(String(repeating: "=", count: 80))

    print("\n📂 出力ファイル:")
    print("  ボーカル: \(vocalsPath.path)")
}
