import Foundation
import AVFoundation

/// MDXNetSeparatorVocFTテスト
@available(iOS 17.0, macOS 14.0, *)
func runMDXNetVocFTTest() throws {
    print(String(repeating: "=", count: 60))
    print("🎵 MDXNet Voc_FT ボーカル分離テスト")
    print(String(repeating: "=", count: 60))

    let currentDir = FileManager.default.currentDirectoryPath
    let baseURL = URL(fileURLWithPath: currentDir)
    let modelPath = baseURL.appendingPathComponent("models/coreml/UVR-MDX-NET-Voc_FT.mlpackage")
    let inputPath = baseURL.appendingPathComponent("tests/output/hollow_crown.wav")
    let outputDir = baseURL.appendingPathComponent("tests/swift_output")

    try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

    // モデル初期化
    let separator = try MDXNetSeparatorVocFT(modelURL: modelPath)

    // 分離実行
    print("\n🎵 ボーカル分離実行中...")
    let startTime = Date()
    let (vocals, instrumental) = try separator.separate(audioURL: inputPath, denoise: true)
    let elapsed = Date().timeIntervalSince(startTime)
    print("✅ 完了: \(String(format: "%.2f", elapsed))秒")

    // 保存
    let vocalsPath = outputDir.appendingPathComponent("hollow_crown_vocals_vocft.wav")
    let instPath = outputDir.appendingPathComponent("hollow_crown_instrumental_vocft.wav")

    try saveAudioVocFT(vocals, to: vocalsPath, sampleRate: 44100)
    try saveAudioVocFT(instrumental, to: instPath, sampleRate: 44100)

    print("\n出力:")
    print("  ボーカル: \(vocalsPath.path)")
    print("  伴奏: \(instPath.path)")

    // 統計
    let vocalsRMS = sqrt(vocals.reduce(0) { $0 + $1 * $1 } / Float(vocals.count))
    let instRMS = sqrt(instrumental.reduce(0) { $0 + $1 * $1 } / Float(instrumental.count))
    print("\n統計:")
    print("  ボーカル RMS: \(String(format: "%.6f", vocalsRMS))")
    print("  伴奏 RMS: \(String(format: "%.6f", instRMS))")

    print("\n" + String(repeating: "=", count: 60))
    print("✅ テスト完了")
    print(String(repeating: "=", count: 60))
}

private func saveAudioVocFT(_ samples: [Float], to url: URL, sampleRate: Int) throws {
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: UInt32(samples.count)) else {
        throw NSError(domain: "Audio", code: 1, userInfo: [NSLocalizedDescriptionKey: "Buffer creation failed"])
    }

    buffer.frameLength = UInt32(samples.count)
    memcpy(buffer.floatChannelData![0], samples, samples.count * MemoryLayout<Float>.size)

    let file = try AVAudioFile(forWriting: url, settings: format.settings)
    try file.write(from: buffer)
}
