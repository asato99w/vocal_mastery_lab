import Foundation
import CoreML
import AVFoundation
import Accelerate

// AudioFileProcessor.swift の内容をインライン化
@available(iOS 17.0, macOS 14.0, *)
struct AudioData {
    let samples: [[Float]]
    let sampleRate: Double
    let frameCount: Int

    var channelCount: Int { samples.count }
}

@available(iOS 17.0, macOS 14.0, *)
class SimpleAudioProcessor {

    static func loadAudio(from url: URL) throws -> AudioData {
        print("📂 音声読み込み: \(url.lastPathComponent)")

        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(audioFile.length)
        ) else {
            throw NSError(domain: "AudioProcessor", code: 1)
        }

        try audioFile.read(into: buffer)

        let channelCount = Int(format.channelCount)
        let frameLength = Int(buffer.frameLength)

        var samples: [[Float]] = []
        for channel in 0..<channelCount {
            guard let channelData = buffer.floatChannelData?[channel] else { continue }
            let channelSamples = Array(UnsafeBufferPointer(start: channelData, count: frameLength))
            samples.append(channelSamples)
        }

        if samples.count == 1 {
            samples.append(samples[0])
        }

        print("  サンプルレート: \(format.sampleRate) Hz")
        print("  チャンネル数: \(samples.count)")
        print("  サンプル数: \(samples[0].count)")
        print("  長さ: \(String(format: "%.2f", Double(samples[0].count) / format.sampleRate)) 秒")

        return AudioData(
            samples: samples,
            sampleRate: format.sampleRate,
            frameCount: frameLength
        )
    }

    static func saveAudio(_ audioData: AudioData, to url: URL) throws {
        print("💾 音声保存: \(url.lastPathComponent)")

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: audioData.sampleRate,
            channels: AVAudioChannelCount(audioData.channelCount),
            interleaved: false
        ) else {
            throw NSError(domain: "AudioProcessor", code: 2)
        }

        let audioFile = try AVAudioFile(forWriting: url, settings: outputFormat.settings)

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(audioData.frameCount)
        ) else {
            throw NSError(domain: "AudioProcessor", code: 3)
        }

        buffer.frameLength = AVAudioFrameCount(audioData.frameCount)

        for channel in 0..<audioData.channelCount {
            guard let channelData = buffer.floatChannelData?[channel] else { continue }
            audioData.samples[channel].withUnsafeBufferPointer { srcBuffer in
                channelData.update(from: srcBuffer.baseAddress!, count: audioData.frameCount)
            }
        }

        try audioFile.write(from: buffer)

        let fileSize = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
        let fileSizeMB = Double(fileSize) / (1024 * 1024)
        print("  ファイルサイズ: \(String(format: "%.2f", fileSizeMB)) MB")
    }
}

// エンドツーエンドテスト実行
@available(iOS 17.0, macOS 14.0, *)
func runEndToEndTest() throws {
    print(String(repeating: "=", count: 80))
    print("🎵 エンドツーエンド音源分離テスト")
    print(String(repeating: "=", count: 80))

    // パス設定
    let modelPath = URL(fileURLWithPath: "/Users/asatokazu/Documents/dev/mine/music/vocal_mastery_lab/poc/uvr_coreml/models/coreml/UVR-MDX-NET-Inst_Main.mlpackage")
    let inputPath = URL(fileURLWithPath: "/Users/asatokazu/Documents/dev/mine/music/vocal_mastery_lab/poc/uvr_coreml/tests/output/hollow_crown.wav")
    let outputDir = URL(fileURLWithPath: "/Users/asatokazu/Documents/dev/mine/music/vocal_mastery_lab/poc/uvr_coreml/tests/swift_output")

    // 出力ディレクトリ作成
    try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

    let vocalsPath = outputDir.appendingPathComponent("hollow_crown_vocals.wav")
    let instrumentalPath = outputDir.appendingPathComponent("hollow_crown_instrumental.wav")

    // 1. 音声読み込み
    print("\n📂 ステップ1: 音声ファイル読み込み")
    let audioData = try SimpleAudioProcessor.loadAudio(from: inputPath)

    // 2. モデル読み込み
    print("\n🤖 ステップ2: CoreMLモデル読み込み")
    let compiledURL = try MLModel.compileModel(at: modelPath)

    let config = MLModelConfiguration()
    config.computeUnits = .all
    let model = try MLModel(contentsOf: compiledURL, configuration: config)
    print("✅ モデル読み込み完了")

    // 3. 実際のCoreML推論を使用した音源分離（簡易版）
    print("\n🔄 ステップ3: 音源分離実行（CoreML推論）")
    print("   注: この例では最初の5秒のみ処理")
    print("   完全な実装はVocalSeparatorComplete.swiftを参照")

    // 処理サンプル数を制限（デモ用: 最初の5秒）
    let fftSize = 4096
    let hopSize = 1024
    let maxSamples = min(audioData.frameCount, Int(audioData.sampleRate * 5.0))

    // 簡易STFT（デモ用の単純な実装）
    func simpleSTFT(_ audio: [Float]) -> [[Float]] {
        let numFrames = (maxSamples - fftSize) / hopSize + 1
        var magnitudes: [[Float]] = Array(
            repeating: Array(repeating: 0, count: numFrames),
            count: 2048
        )

        // 実際のSTFT処理の代わりに、簡易的な周波数分解をシミュレート
        for frameIdx in 0..<numFrames {
            let startIdx = frameIdx * hopSize
            for freqIdx in 0..<2048 {
                let windowSize = min(10, maxSamples - startIdx)
                var sum: Float = 0
                for i in 0..<windowSize {
                    if startIdx + i < audio.count {
                        sum += abs(audio[startIdx + i])
                    }
                }
                magnitudes[freqIdx][frameIdx] = sum / Float(windowSize)
            }
        }
        return magnitudes
    }

    // 左チャンネルのSTFT
    let leftMag = simpleSTFT(Array(audioData.samples[0].prefix(maxSamples)))

    // モデルは固定サイズ256フレームを期待
    let modelTimeFrames = 256
    print("   STFT出力: \(leftMag[0].count)フレーム → \(modelTimeFrames)フレームにパディング")

    // CoreML推論用の入力準備（固定256フレーム）
    let inputShape = [1, 4, 2048, modelTimeFrames] as [NSNumber]
    let inputArray = try MLMultiArray(shape: inputShape, dataType: .float32)

    // ゼロで初期化
    for i in 0..<inputArray.count {
        inputArray[i] = 0
    }

    // 入力データ設定（実データ + パディング）
    let actualFrames = min(leftMag[0].count, modelTimeFrames)
    for t in 0..<actualFrames {
        for f in 0..<2048 {
            inputArray[[0, 0, f, t] as [NSNumber]] = NSNumber(value: leftMag[f][t])
            inputArray[[0, 1, f, t] as [NSNumber]] = 0  // 虚部
            inputArray[[0, 2, f, t] as [NSNumber]] = NSNumber(value: leftMag[f][t])  // 右チャンネル
            inputArray[[0, 3, f, t] as [NSNumber]] = 0  // 虚部
        }
    }

    print("   入力形状: \(inputShape)")
    print("   推論実行中...")

    // CoreML推論
    let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
        "input_1": MLFeatureValue(multiArray: inputArray)
    ])

    let startTime = Date()
    let output = try model.prediction(from: inputProvider)
    let inferenceTime = Date().timeIntervalSince(startTime)

    print("   推論時間: \(String(format: "%.4f", inferenceTime))秒")

    guard let outputArray = output.featureValue(for: "var_992")?.multiArrayValue else {
        throw NSError(domain: "EndToEndTest", code: 4, userInfo: [
            NSLocalizedDescriptionKey: "出力取得失敗"
        ])
    }

    print("   出力形状: \(outputArray.shape)")

    // 簡易的なiSTFT（デモ用）
    func simpleISTFT(_ magnitudes: [[Float]]) -> [Float] {
        let frameCount = magnitudes[0].count
        let outputLength = (frameCount - 1) * hopSize + fftSize
        var output = [Float](repeating: 0, count: min(outputLength, maxSamples))

        for frameIdx in 0..<frameCount {
            let startIdx = frameIdx * hopSize
            for i in 0..<min(hopSize, output.count - startIdx) {
                var sum: Float = 0
                for freqIdx in 0..<min(100, magnitudes.count) {
                    sum += magnitudes[freqIdx][frameIdx]
                }
                if startIdx + i < output.count {
                    output[startIdx + i] += sum / 100.0 * 0.1
                }
            }
        }
        return output
    }

    // マスク抽出とiSTFT
    var vocalMag: [[Float]] = Array(repeating: [], count: 2048)
    var instrMag: [[Float]] = Array(repeating: [], count: 2048)

    for f in 0..<2048 {
        for t in 0..<leftMag[0].count {
            let vocalValue = outputArray[[0, 0, f, t] as [NSNumber]].floatValue
            let instrValue = outputArray[[0, 1, f, t] as [NSNumber]].floatValue
            vocalMag[f].append(vocalValue * leftMag[f][t])
            instrMag[f].append(instrValue * leftMag[f][t])
        }
    }

    let vocalAudio = simpleISTFT(vocalMag)
    let instrAudio = simpleISTFT(instrMag)

    // 結果をステレオに拡張
    var vocalSamples = [vocalAudio, vocalAudio]
    var instrumentalSamples = [instrAudio, instrAudio]

    // 元の長さに合わせてパディング
    for i in 0..<2 {
        while vocalSamples[i].count < audioData.frameCount {
            vocalSamples[i].append(0)
        }
        while instrumentalSamples[i].count < audioData.frameCount {
            instrumentalSamples[i].append(0)
        }
    }

    print("✅ 分離完了（処理時間: \(String(format: "%.2f", Double(maxSamples) / audioData.sampleRate))秒分）")

    // 4. 保存
    print("\n💾 ステップ4: 分離音声保存")

    let vocalsData = AudioData(
        samples: vocalSamples,
        sampleRate: audioData.sampleRate,
        frameCount: audioData.frameCount
    )

    let instrumentalData = AudioData(
        samples: instrumentalSamples,
        sampleRate: audioData.sampleRate,
        frameCount: audioData.frameCount
    )

    try SimpleAudioProcessor.saveAudio(vocalsData, to: vocalsPath)
    try SimpleAudioProcessor.saveAudio(instrumentalData, to: instrumentalPath)

    print("\n" + String(repeating: "=", count: 80))
    print("✅ エンドツーエンドテスト完了")
    print(String(repeating: "=", count: 80))

    print("\n📂 出力ファイル:")
    print("  ボーカル: \(vocalsPath.path)")
    print("  伴奏: \(instrumentalPath.path)")

    print("\n📝 注意:")
    print("  このデモでは簡易的なフィルタを使用しています。")
    print("  完全な実装には以下が必要です:")
    print("  - STFT/iSTFT処理")
    print("  - CoreML推論統合")
    print("  - チャンク処理")
    print("  詳細は VocalSeparatorComplete.swift を参照してください。")
}

