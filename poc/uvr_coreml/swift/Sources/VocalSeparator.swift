import Foundation
import Accelerate
import CoreML
import AVFoundation

/// UVR-MDX-NET ボーカル分離器 (Swift/CoreML)
public class VocalSeparator {
    // MARK: - Parameters

    private let nFFT: Int = 6144
    private let dimF: Int = 3072
    private let dimT: Int = 256  // 2^8
    private let hop: Int = 1024
    private let sampleRate: Int = 44100
    private let dimC: Int = 4

    private var nBins: Int { nFFT / 2 + 1 }
    private var chunkSize: Int { hop * (dimT - 1) }

    private let model: MLModel
    private let dftSetupForward: OpaquePointer
    private let dftSetupInverse: OpaquePointer
    private var window: [Float]

    // MARK: - Initialization

    public init(modelURL: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all

        // .mlpackage の場合はコンパイルが必要
        let compiledURL: URL
        if modelURL.pathExtension == "mlpackage" {
            compiledURL = try MLModel.compileModel(at: modelURL)
        } else {
            compiledURL = modelURL
        }
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)

        // DFTセットアップ（vDSP_DFT_Execute用）
        guard let fwdSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .FORWARD) else {
            throw SeparatorError.fftInitFailed
        }
        self.dftSetupForward = fwdSetup

        guard let invSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .INVERSE) else {
            throw SeparatorError.fftInitFailed
        }
        self.dftSetupInverse = invSetup

        // Hann window (periodic, PyTorch互換)
        self.window = [Float](repeating: 0, count: nFFT)
        for i in 0..<nFFT {
            window[i] = 0.5 * (1.0 - cos(2.0 * .pi * Float(i) / Float(nFFT)))
        }
    }

    deinit {
        vDSP_DFT_DestroySetup(dftSetupForward)
        vDSP_DFT_DestroySetup(dftSetupInverse)
    }

    // MARK: - Separation

    public func separate(audioURL: URL) throws -> (vocals: [Float], instrumental: [Float]) {
        // Load audio
        let (mix, _) = try loadAudio(url: audioURL)
        print("入力サンプル数: \(mix.count / 2)")

        // Prepare chunks
        let trim = nFFT / 2
        let genSize = chunkSize - 2 * trim
        let nSample = mix.count / 2
        let pad = genSize - nSample % genSize

        // 各チャンネルを分離してパディング（Pythonと同じ方法）
        let lChannel = Array(mix[0..<nSample])
        let rChannel = Array(mix[nSample..<(2 * nSample)])

        let lPadded = [Float](repeating: 0, count: trim) + lChannel +
                      [Float](repeating: 0, count: pad) +
                      [Float](repeating: 0, count: trim)
        let rPadded = [Float](repeating: 0, count: trim) + rChannel +
                      [Float](repeating: 0, count: pad) +
                      [Float](repeating: 0, count: trim)

        var chunks: [[Float]] = []
        var i = 0
        while i < nSample + pad {
            let chunkL = Array(lPadded[i..<(i + chunkSize)])
            let chunkR = Array(rPadded[i..<(i + chunkSize)])
            chunks.append(chunkL + chunkR)
            i += genSize
        }

        print("チャンク数: \(chunks.count)")

        // Process each chunk
        var outputChunks: [[Float]] = []
        for (idx, chunk) in chunks.enumerated() {
            // STFT
            let stftResult = performSTFT(chunk)

            // Model inference
            let modelOutput = try runModel(input: stftResult)

            // iSTFT
            let audioOutput = performISTFT(modelOutput)

            // Trim
            let trimmedL = Array(audioOutput[trim..<(chunkSize - trim)])
            let trimmedR = Array(audioOutput[(chunkSize + trim)..<(2 * chunkSize - trim)])
            outputChunks.append(trimmedL + trimmedR)

            print("  チャンク \(idx + 1)/\(chunks.count) 処理完了")
        }

        // Concatenate and remove padding
        // Voc_FT モデルはボーカルを直接出力する
        var vocals = [Float](repeating: 0, count: 2 * nSample)
        var offset = 0
        for chunk in outputChunks {
            let copyLen = min(chunk.count / 2, nSample - offset)
            if copyLen <= 0 { break }

            for j in 0..<copyLen {
                vocals[j + offset] = chunk[j]
                vocals[nSample + j + offset] = chunk[chunk.count / 2 + j]
            }
            offset += copyLen
        }

        // Instrumental = Mix - Vocals
        var instrumental = [Float](repeating: 0, count: 2 * nSample)
        for i in 0..<(2 * nSample) {
            instrumental[i] = mix[i] - vocals[i]
        }

        return (vocals, instrumental)
    }

    // MARK: - STFT

    private func performSTFT(_ audio: [Float]) -> [Float] {
        let numFrames = dimT
        var result = [Float](repeating: 0, count: dimC * dimF * dimT)
        let pad = nFFT / 2  // center=True に相当するパディング

        for ch in 0..<2 {
            let channelOffset = ch * chunkSize
            let channelAudio = Array(audio[channelOffset..<(channelOffset + chunkSize)])

            // center=True: 前後に nFFT/2 のパディングを追加
            let paddedAudio = [Float](repeating: 0, count: pad) + channelAudio + [Float](repeating: 0, count: pad)

            for frame in 0..<numFrames {
                let start = frame * hop

                // Windowedセグメントを作成
                var windowedReal = [Float](repeating: 0, count: nFFT)
                var windowedImag = [Float](repeating: 0, count: nFFT)

                for i in 0..<nFFT {
                    windowedReal[i] = paddedAudio[start + i] * window[i]
                }

                // DFT実行
                var outputReal = [Float](repeating: 0, count: nFFT)
                var outputImag = [Float](repeating: 0, count: nFFT)

                vDSP_DFT_Execute(dftSetupForward, windowedReal, windowedImag, &outputReal, &outputImag)

                // Store in [4, dimF, dimT] format: [L_real, L_imag, R_real, R_imag]
                for f in 0..<dimF {
                    let realIdx = (ch * 2) * dimF * dimT + f * dimT + frame
                    let imagIdx = (ch * 2 + 1) * dimF * dimT + f * dimT + frame
                    result[realIdx] = outputReal[f]
                    result[imagIdx] = outputImag[f]
                }
            }
        }

        return result
    }

    // MARK: - Model Inference

    private func runModel(input: [Float]) throws -> [Float] {
        let inputShape = [1, dimC, dimF, dimT] as [NSNumber]
        guard let inputArray = try? MLMultiArray(shape: inputShape, dataType: .float32) else {
            throw SeparatorError.modelInputFailed
        }

        for i in 0..<input.count {
            inputArray[i] = NSNumber(value: input[i])
        }

        let inputFeature = try MLDictionaryFeatureProvider(
            dictionary: ["input": MLFeatureValue(multiArray: inputArray)]
        )

        let output = try model.prediction(from: inputFeature)

        guard let outputArray = output.featureValue(for: "var_992")?.multiArrayValue else {
            throw SeparatorError.modelOutputFailed
        }

        var result = [Float](repeating: 0, count: dimC * dimF * dimT)
        for i in 0..<result.count {
            result[i] = outputArray[i].floatValue
        }

        return result
    }

    // MARK: - iSTFT

    private func performISTFT(_ spectrum: [Float]) -> [Float] {
        var audio = [Float](repeating: 0, count: 2 * chunkSize)
        let pad = nFFT / 2  // center=True に相当するパディング
        let paddedLen = chunkSize + nFFT  // パディング後の長さ

        for ch in 0..<2 {
            var paddedAudio = [Float](repeating: 0, count: paddedLen)
            var windowSum = [Float](repeating: 0, count: paddedLen)

            for frame in 0..<dimT {
                // 完全なスペクトルを構築（対称性を利用）
                var inputReal = [Float](repeating: 0, count: nFFT)
                var inputImag = [Float](repeating: 0, count: nFFT)

                // モデル出力からスペクトルを取得 (0 to dimF-1)
                for f in 0..<dimF {
                    let realIdx = (ch * 2) * dimF * dimT + f * dimT + frame
                    let imagIdx = (ch * 2 + 1) * dimF * dimT + f * dimT + frame
                    inputReal[f] = spectrum[realIdx]
                    inputImag[f] = spectrum[imagIdx]
                }

                // 共役対称性を利用して後半を埋める
                // X[k] = conj(X[N-k]) for k > N/2
                for f in (nFFT / 2 + 1)..<nFFT {
                    inputReal[f] = inputReal[nFFT - f]
                    inputImag[f] = -inputImag[nFFT - f]
                }

                // iDFT実行
                var outputReal = [Float](repeating: 0, count: nFFT)
                var outputImag = [Float](repeating: 0, count: nFFT)

                vDSP_DFT_Execute(dftSetupInverse, inputReal, inputImag, &outputReal, &outputImag)

                // スケーリング（iDFTはnFFTで割る）と窓関数適用
                let scale = 1.0 / Float(nFFT)
                for i in 0..<nFFT {
                    outputReal[i] = outputReal[i] * scale * window[i]
                }

                // Overlap-add (パディング込みの配列に)
                let start = frame * hop
                for i in 0..<nFFT {
                    paddedAudio[start + i] += outputReal[i]
                    windowSum[start + i] += window[i] * window[i]
                }
            }

            // Normalize
            for i in 0..<paddedLen {
                if windowSum[i] > 1e-8 {
                    paddedAudio[i] /= windowSum[i]
                }
            }

            // center=True のパディングを除去して出力
            for i in 0..<chunkSize {
                audio[ch * chunkSize + i] = paddedAudio[pad + i]
            }
        }

        return audio
    }

    // MARK: - Audio I/O

    private func loadAudio(url: URL) throws -> ([Float], Int) {
        let file = try AVAudioFile(forReading: url)
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 2)!

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(file.length)) else {
            throw SeparatorError.audioLoadFailed
        }

        try file.read(into: buffer)

        let frameCount = Int(buffer.frameLength)
        var samples = [Float](repeating: 0, count: frameCount * 2)

        if let channelData = buffer.floatChannelData {
            for i in 0..<frameCount {
                samples[i] = channelData[0][i]
                samples[frameCount + i] = channelData[1][i]
            }
        }

        return (samples, sampleRate)
    }

    public func saveAudio(_ samples: [Float], to url: URL) throws {
        let frameCount = samples.count / 2
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 2)!

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(frameCount)) else {
            throw SeparatorError.audioSaveFailed
        }

        buffer.frameLength = AVAudioFrameCount(frameCount)

        if let channelData = buffer.floatChannelData {
            for i in 0..<frameCount {
                channelData[0][i] = samples[i]
                channelData[1][i] = samples[frameCount + i]
            }
        }

        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    }
}

// MARK: - Errors

enum SeparatorError: Error {
    case fftInitFailed
    case modelInputFailed
    case modelOutputFailed
    case audioLoadFailed
    case audioSaveFailed
}
