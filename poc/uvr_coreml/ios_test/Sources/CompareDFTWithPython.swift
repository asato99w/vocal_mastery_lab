import Foundation
import Accelerate

/// vDSP_DFTとPython librosaの1フレーム比較テスト
///
/// 440Hzの単一周波数信号でFFT結果を比較
func testDFTComparison() {
    print("\n" + String(repeating: "=", count: 80))
    print("🔍 vDSP_DFT vs Python librosa - Single Frame Comparison")
    print(String(repeating: "=", count: 80))

    // パラメータ (librosaと同じ)
    let sampleRate: Float = 44100.0
    let frequency: Float = 440.0  // A4音
    let fftSize = 4096
    let duration: Float = Float(fftSize) / sampleRate

    print("\n📊 テストパラメータ:")
    print("  サンプルレート: \(sampleRate) Hz")
    print("  周波数: \(frequency) Hz")
    print("  FFTサイズ: \(fftSize)")
    print("  信号長: \(fftSize) samples (\(String(format: "%.4f", duration))秒)")

    // 440Hzの正弦波を生成
    var signal = [Float](repeating: 0, count: fftSize)
    for i in 0..<fftSize {
        let t = Float(i) / sampleRate
        signal[i] = sin(2.0 * Float.pi * frequency * t)
    }

    print("\n🎵 テスト信号 (440Hz正弦波):")
    print("  最初の10サンプル: \(signal[0..<10].map { String(format: "%.6f", $0) })")

    // DFT setupを作成
    guard let dftSetup = vDSP_DFT_zop_CreateSetup(
        nil,
        vDSP_Length(fftSize),
        vDSP_DFT_Direction.FORWARD
    ) else {
        print("❌ DFT setup failed")
        return
    }
    defer { vDSP_DFT_DestroySetup(dftSetup) }

    // Hann窓を適用 (librosaのデフォルト)
    var window = [Float](repeating: 0, count: fftSize)
    vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))

    var windowedSignal = [Float](repeating: 0, count: fftSize)
    vDSP_vmul(signal, 1, window, 1, &windowedSignal, 1, vDSP_Length(fftSize))

    // 複素数入力 (実部=窓適用信号、虚部=0)
    var realInput = windowedSignal
    var imagInput = [Float](repeating: 0, count: fftSize)
    var realOutput = [Float](repeating: 0, count: fftSize)
    var imagOutput = [Float](repeating: 0, count: fftSize)

    // DFT実行
    vDSP_DFT_Execute(dftSetup, &realInput, &imagInput, &realOutput, &imagOutput)

    // 振幅と位相を計算
    let frequencyBins = fftSize / 2 + 1
    var magnitude = [Float](repeating: 0, count: frequencyBins)
    var phase = [Float](repeating: 0, count: frequencyBins)

    for i in 0..<frequencyBins {
        let re = realOutput[i]
        let im = imagOutput[i]
        magnitude[i] = sqrtf(re * re + im * im)
        phase[i] = atan2f(im, re)
    }

    // 440Hz binを見つける
    let expectedBin = Int(round(frequency * Float(fftSize) / sampleRate))

    print("\n🎯 440Hz bin [位置 \(expectedBin)]:")
    print("  Magnitude: \(String(format: "%.6f", magnitude[expectedBin]))")
    print("  Phase: \(String(format: "%.6f", phase[expectedBin]))")

    // DC成分
    print("\n📊 DC bin [0]:")
    print("  Magnitude: \(String(format: "%.6f", magnitude[0]))")
    print("  Phase: \(String(format: "%.6f", phase[0]))")

    // Nyquist成分
    print("\n📊 Nyquist bin [\(fftSize/2)]:")
    print("  Magnitude: \(String(format: "%.6f", magnitude[fftSize/2]))")
    print("  Phase: \(String(format: "%.6f", phase[fftSize/2]))")

    // 最初の10ビンの詳細
    print("\n🔍 最初の10ビンの詳細:")
    for i in 0..<min(10, frequencyBins) {
        let freq = Float(i) * sampleRate / Float(fftSize)
        print("  Bin \(i) (\(String(format: "%.1f", freq))Hz): mag=\(String(format: "%.6f", magnitude[i])), phase=\(String(format: "%.6f", phase[i]))")
    }

    // 440Hz周辺の詳細 (±5 bins)
    print("\n🎵 440Hz周辺のビン詳細:")
    let startBin = max(0, expectedBin - 5)
    let endBin = min(frequencyBins - 1, expectedBin + 5)
    for i in startBin...endBin {
        let freq = Float(i) * sampleRate / Float(fftSize)
        let marker = i == expectedBin ? " ← 440Hz" : ""
        print("  Bin \(i) (\(String(format: "%.1f", freq))Hz): mag=\(String(format: "%.6f", magnitude[i])), phase=\(String(format: "%.6f", phase[i]))\(marker)")
    }

    // Python比較用にデータを保存
    let outputDir = "../tests/swift_output"
    try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

    // 実部・虚部を保存
    let realPath = "\(outputDir)/dft_real_swift.txt"
    let imagPath = "\(outputDir)/dft_imag_swift.txt"
    let magPath = "\(outputDir)/dft_magnitude_swift.txt"
    let phasePath = "\(outputDir)/dft_phase_swift.txt"

    let realStr = (0..<frequencyBins).map { String(format: "%.10f", realOutput[$0]) }.joined(separator: "\n")
    let imagStr = (0..<frequencyBins).map { String(format: "%.10f", imagOutput[$0]) }.joined(separator: "\n")
    let magStr = magnitude.map { String(format: "%.10f", $0) }.joined(separator: "\n")
    let phaseStr = phase.map { String(format: "%.10f", $0) }.joined(separator: "\n")

    try? realStr.write(toFile: realPath, atomically: true, encoding: .utf8)
    try? imagStr.write(toFile: imagPath, atomically: true, encoding: .utf8)
    try? magStr.write(toFile: magPath, atomically: true, encoding: .utf8)
    try? phaseStr.write(toFile: phasePath, atomically: true, encoding: .utf8)

    print("\n💾 比較用データ保存:")
    print("  \(realPath)")
    print("  \(imagPath)")
    print("  \(magPath)")
    print("  \(phasePath)")

    print("\n" + String(repeating: "=", count: 80))
}
