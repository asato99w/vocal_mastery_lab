import Foundation
import Accelerate

@available(iOS 17.0, macOS 14.0, *)
func debugSTFTComparison() {
    print(String(repeating: "=", count: 80))
    print("🔍 Swift STFT/iSTFT 中間データダンプ (Python比較用)")
    print(String(repeating: "=", count: 80))
    
    // テスト信号生成 (440Hz, 1秒)
    let sr: Float = 44100.0
    let duration: Float = 1.0
    let freq: Float = 440.0
    let sampleCount = Int(sr * duration)
    
    let testSignal: [Float] = (0..<sampleCount).map { i in
        let t = Float(i) / sr
        return sin(2.0 * Float.pi * freq * t)
    }
    
    print("\n📊 テスト信号 (440Hz, 1秒):")
    let maxVal = testSignal.map { abs($0) }.max()!
    let rms = sqrt(testSignal.map { $0 * $0 }.reduce(0, +) / Float(sampleCount))
    print("  Max: \(maxVal)")
    print("  RMS: \(rms)")
    print("  サンプル数: \(sampleCount)")
    
    // STFTプロセッサ初期化
    let stftProcessor = STFTProcessor(fftSize: 4096, hopSize: 1024, windowType: .hann)
    
    print("\n🔄 STFT実行 (n_fft=4096, hop=1024):")
    let stftResult = stftProcessor.computeSTFT(audio: testSignal)
    
    print("  周波数ビン数: \(stftResult.frequencyBins)")
    print("  時間フレーム数: \(stftResult.timeFrames)")
    
    // STFT統計
    let magnitudeFlat = stftResult.magnitude.flatMap { $0 }
    let phaseFlat = stftResult.phase.flatMap { $0 }
    
    print("\n📊 STFT統計:")
    print("  Magnitude Max: \(magnitudeFlat.max()!)")
    print("  Magnitude Mean: \(magnitudeFlat.reduce(0, +) / Float(magnitudeFlat.count))")
    print("  Phase range: \(phaseFlat.min()!) ~ \(phaseFlat.max()!)")
    
    // 440Hz binの確認
    let freq440Bin = Int(freq * 4096.0 / sr)
    let mag440 = stftResult.magnitude[freq440Bin].reduce(0, +) / Float(stftResult.timeFrames)
    print("\n🎵 440Hz bin [\(freq440Bin)]:")
    print("  Magnitude平均: \(mag440)")
    print("  Phase (frame 0): \(stftResult.phase[freq440Bin][0])")
    
    // 最初の3フレームの詳細
    print("\n🔬 最初の3フレームの詳細 (440Hz bin):")
    for i in 0..<min(3, stftResult.timeFrames) {
        let mag = stftResult.magnitude[freq440Bin][i]
        let phase = stftResult.phase[freq440Bin][i]
        print("  Frame \(i): mag=\(mag), phase=\(phase)")
    }
    
    // iSTFT
    print("\n🔄 iSTFT実行:")
    let reconstructed = stftProcessor.computeISTFT(
        magnitude: stftResult.magnitude,
        phase: stftResult.phase
    )
    
    let reconstructedMax = reconstructed.map { abs($0) }.max()!
    let reconstructedRMS = sqrt(reconstructed.map { $0 * $0 }.reduce(0, +) / Float(reconstructed.count))
    
    print("  Max: \(reconstructedMax)")
    print("  RMS: \(reconstructedRMS)")
    print("  長さ: \(reconstructed.count)")
    
    // 再構成誤差
    let minLen = min(testSignal.count, reconstructed.count)
    let error: [Float] = (0..<minLen).map { i in
        testSignal[i] - reconstructed[i]
    }
    
    let errorMax = error.map { abs($0) }.max()!
    let errorRMS = sqrt(error.map { $0 * $0 }.reduce(0, +) / Float(minLen))
    
    print("\n📏 再構成誤差:")
    print("  Max error: \(errorMax)")
    print("  RMS error: \(errorRMS)")
    
    // テキストファイルに保存
    let outputDir = "/Users/asatokazu/Documents/dev/mine/music/vocal_mastery_lab/poc/uvr_coreml/tests/swift_output"
    
    // 最初の100サンプル
    let testSignal100 = testSignal.prefix(100).map { "\($0)" }.joined(separator: "\n")
    try? testSignal100.write(toFile: "\(outputDir)/test_signal_swift.txt", atomically: true, encoding: .utf8)
    
    // フレーム0のmagnitude
    let mag0 = stftResult.magnitude.map { $0[0] }.map { "\($0)" }.joined(separator: "\n")
    try? mag0.write(toFile: "\(outputDir)/magnitude_frame0_swift.txt", atomically: true, encoding: .utf8)
    
    // フレーム0のphase
    let phase0 = stftResult.phase.map { $0[0] }.map { "\($0)" }.joined(separator: "\n")
    try? phase0.write(toFile: "\(outputDir)/phase_frame0_swift.txt", atomically: true, encoding: .utf8)
    
    // 再構成結果の最初の100サンプル
    let reconstructed100 = reconstructed.prefix(100).map { "\($0)" }.joined(separator: "\n")
    try? reconstructed100.write(toFile: "\(outputDir)/reconstructed_swift.txt", atomically: true, encoding: .utf8)
    
    print("\n💾 デバッグデータ保存完了:")
    print("  \(outputDir)/test_signal_swift.txt (最初の100サンプル)")
    print("  \(outputDir)/magnitude_frame0_swift.txt (フレーム0)")
    print("  \(outputDir)/phase_frame0_swift.txt (フレーム0)")
    print("  \(outputDir)/reconstructed_swift.txt (最初の100サンプル)")
    
    print("\n" + String(repeating: "=", count: 80))
}
