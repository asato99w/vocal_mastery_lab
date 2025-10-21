import Foundation
import CoreML
import AVFoundation
import Accelerate

// VocalSeparatorCompleteを使用した本格的なテスト
@available(iOS 17.0, macOS 14.0, *)
func runProperSeparationTest() throws {
    print(String(repeating: "=", count: 80))
    print("🎵 本格的音源分離テスト (VocalSeparatorComplete)")
    print(String(repeating: "=", count: 80))

    // パス設定
    let modelPath = URL(fileURLWithPath: "/Users/asatokazu/Documents/dev/mine/music/vocal_mastery_lab/poc/uvr_coreml/models/coreml/UVR-MDX-NET-Inst_Main.mlpackage")
    let inputPath = URL(fileURLWithPath: "/Users/asatokazu/Documents/dev/mine/music/vocal_mastery_lab/poc/uvr_coreml/tests/output/hollow_crown_from_flac.wav")
    let outputDir = URL(fileURLWithPath: "/Users/asatokazu/Documents/dev/mine/music/vocal_mastery_lab/poc/uvr_coreml/tests/swift_output")

    try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

    let vocalsPath = outputDir.appendingPathComponent("hollow_crown_vocals_proper.wav")
    let instrumentalPath = outputDir.appendingPathComponent("hollow_crown_instrumental_proper.wav")

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

    // 音源分離実行
    print("\n🎵 音源分離実行中...")
    print("  注: 処理には数分かかる場合があります")

    let startTime = Date()

    // 音源分離実行（同期版に修正済み）
    let result = try separator.separate(audioURL: inputPath)

    let elapsed = Date().timeIntervalSince(startTime)

    print("✅ 分離完了")
    print("  処理時間: \(String(format: "%.2f", elapsed))秒")

    // 結果保存
    print("\n💾 結果保存中...")
    try separator.save(
        separatedAudio: result,
        vocalsURL: vocalsPath,
        instrumentalURL: instrumentalPath
    )

    print("\n" + String(repeating: "=", count: 80))
    print("✅ テスト完了")
    print(String(repeating: "=", count: 80))

    print("\n📂 出力ファイル:")
    print("  ボーカル: \(vocalsPath.path)")
    print("  伴奏: \(instrumentalPath.path)")
}
