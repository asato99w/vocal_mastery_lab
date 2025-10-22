import Foundation
import CoreML
import Accelerate

@available(iOS 17.0, macOS 14.0, *)
func debugModelOutput() {
    print(String(repeating: "=", count: 80))
    print("🔍 CoreML Model Output Debug")
    print(String(repeating: "=", count: 80))
    
    // テスト信号生成 (440Hz, 1秒) - Pythonと同じ
    let sr: Float = 44100.0
    let duration: Float = 1.0
    let freq: Float = 440.0
    let sampleCount = Int(sr * duration)
    
    let testSignal: [Float] = (0..<sampleCount).map { i in
        let t = Float(i) / sr
        return sin(2.0 * Float.pi * freq * t)
    }
    
    print("\n📊 テスト信号 (440Hz, 1秒):")
    print("  Max: \(testSignal.map { abs($0) }.max()!)")
    
    // STFT
    let stftProcessor = STFTProcessor(fftSize: 4096, hopSize: 1024, windowType: .hann)
    let stftResult = stftProcessor.computeSTFT(audio: testSignal)
    
    print("\n🔄 STFT実行:")
    print("  形状: [\(stftResult.frequencyBins), \(stftResult.timeFrames)]")
    
    // 実部・虚部の範囲確認 (magnitude/phaseから復元)
    var realVals: [Float] = []
    var imagVals: [Float] = []
    
    for f in 0..<min(2048, stftResult.frequencyBins) {
        for t in 0..<stftResult.timeFrames {
            let mag = stftResult.magnitude[f][t]
            let ph = stftResult.phase[f][t]
            realVals.append(mag * cos(ph))
            imagVals.append(mag * sin(ph))
        }
    }
    
    print("  Real range: \(realVals.min()!) ~ \(realVals.max()!)")
    print("  Imag range: \(imagVals.min()!) ~ \(imagVals.max()!)")
    
    // モデル入力作成 (最初のチャンク)
    let chunkSize = 256
    let freqBins = 2048
    
    let inputArray = try! MLMultiArray(
        shape: [1, 4, freqBins, chunkSize] as [NSNumber],
        dataType: .float32
    )
    
    // データコピー
    for t in 0..<min(chunkSize, stftResult.timeFrames) {
        for f in 0..<freqBins {
            let mag = stftResult.magnitude[f][t]
            let ph = stftResult.phase[f][t]
            let real = mag * cos(ph)
            let imag = mag * sin(ph)
            
            inputArray[[0, 0, f, t] as [NSNumber]] = NSNumber(value: real)  // Left Real
            inputArray[[0, 1, f, t] as [NSNumber]] = NSNumber(value: imag)  // Left Imag
            inputArray[[0, 2, f, t] as [NSNumber]] = NSNumber(value: real)  // Right Real
            inputArray[[0, 3, f, t] as [NSNumber]] = NSNumber(value: imag)  // Right Imag
        }
    }
    
    // パディング
    if stftResult.timeFrames < chunkSize {
        for t in stftResult.timeFrames..<chunkSize {
            for f in 0..<freqBins {
                for ch in 0..<4 {
                    inputArray[[0, ch, f, t] as [NSNumber]] = 0
                }
            }
        }
    }
    
    print("\n📥 CoreML入力:")
    print("  形状: [1, 4, \(freqBins), \(chunkSize)]")
    
    // 入力の値範囲を確認
    var inputMin: Float = Float.infinity
    var inputMax: Float = -Float.infinity
    
    for idx in 0..<inputArray.count {
        let val = inputArray[idx].floatValue
        inputMin = min(inputMin, val)
        inputMax = max(inputMax, val)
    }
    
    print("  値の範囲: \(inputMin) ~ \(inputMax)")
    
    // CoreML推論
    let modelPath = "/Users/asatokazu/Documents/dev/mine/music/vocal_mastery_lab/poc/uvr_coreml/models/coreml/UVR-MDX-NET-Inst_Main.mlpackage"
    
    print("\n🤖 CoreML推論実行:")
    
    guard let compiledURL = try? MLModel.compileModel(at: URL(fileURLWithPath: modelPath)) else {
        print("❌ Model compile failed")
        return
    }
    
    let config = MLModelConfiguration()
    config.computeUnits = .all
    
    guard let model = try? MLModel(contentsOf: compiledURL, configuration: config) else {
        print("❌ Model load failed")
        return
    }
    
    let inputProvider = try! MLDictionaryFeatureProvider(dictionary: [
        "input_1": MLFeatureValue(multiArray: inputArray)
    ])
    
    guard let output = try? model.prediction(from: inputProvider) else {
        print("❌ Prediction failed")
        return
    }
    
    guard let outputArray = output.featureValue(for: "var_992")?.multiArrayValue else {
        print("❌ Output extraction failed")
        return
    }
    
    print("\n📤 CoreML出力:")
    print("  形状: \(outputArray.shape)")
    
    // 出力の値範囲
    var outputMin: Float = Float.infinity
    var outputMax: Float = -Float.infinity
    
    for idx in 0..<outputArray.count {
        let val = outputArray[idx].floatValue
        outputMin = min(outputMin, val)
        outputMax = max(outputMax, val)
    }
    
    print("  値の範囲: \(outputMin) ~ \(outputMax)")
    
    // チャンネル別統計
    print("\n📊 チャンネル別出力統計:")
    for ch in 0..<4 {
        var chVals: [Float] = []
        for f in 0..<2048 {
            for t in 0..<256 {
                let val = outputArray[[0, ch, f, t] as [NSNumber]].floatValue
                chVals.append(val)
            }
        }
        let chMin = chVals.min()!
        let chMax = chVals.max()!
        let chMean = chVals.reduce(0, +) / Float(chVals.count)
        print("  Ch\(ch): min=\(chMin), max=\(chMax), mean=\(chMean)")
    }
    
    // 440Hz binの出力
    let freq440Bin = 40
    print("\n🎵 440Hz bin [\(freq440Bin)] の出力 (最初の3フレーム):")
    for frameIdx in 0..<3 {
        print("  Frame \(frameIdx):")
        for ch in 0..<4 {
            let val = outputArray[[0, ch, freq440Bin, frameIdx] as [NSNumber]].floatValue
            print("    Ch\(ch): \(val)")
        }
    }
    
    // デバッグデータ保存
    let outputDir = "/Users/asatokazu/Documents/dev/mine/music/vocal_mastery_lab/poc/uvr_coreml/tests/swift_output"
    
    // Ch0, Frame0を保存
    var ch0Frame0: [Float] = []
    for f in 0..<2048 {
        let val = outputArray[[0, 0, f, 0] as [NSNumber]].floatValue
        ch0Frame0.append(val)
    }
    
    let ch0Frame0Text = ch0Frame0.map { "\($0)" }.joined(separator: "\n")
    try? ch0Frame0Text.write(toFile: "\(outputDir)/model_output_ch0_frame0_swift.txt", atomically: true, encoding: .utf8)
    
    print("\n💾 デバッグデータ保存完了:")
    print("  \(outputDir)/model_output_ch0_frame0_swift.txt")
    
    print("\n" + String(repeating: "=", count: 80))
}
