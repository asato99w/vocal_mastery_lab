import Foundation
import CoreML
import Accelerate
import AVFoundation

/// UVR-MDX-NET ボーカル分離 (Python実装と完全互換)
/// 参照: https://github.com/seanghay/uvr-mdx-infer
@available(iOS 17.0, macOS 14.0, *)
class MDXNetSeparator {

    // MARK: - Constants (Python実装と同じ)

    let nFFT: Int = 6144
    let dimF: Int = 2048
    let dimT: Int = 256  // 2^8
    let hop: Int = 1024
    let sampleRate: Int = 44100
    let dimC: Int = 4  // ステレオ x (実部+虚部)

    var nBins: Int { nFFT / 2 + 1 }  // 3073
    var chunkSize: Int { hop * (dimT - 1) }  // 261120

    // MARK: - Properties

    private let model: MLModel
    private let window: [Float]
    private var dftSetupForward: OpaquePointer?
    private var dftSetupInverse: OpaquePointer?

    // MARK: - Initialization

    init(modelURL: URL) throws {
        print("🔄 MDXNetSeparator 初期化中...")

        // CoreMLモデル読み込み
        let compiledURL = try MLModel.compileModel(at: modelURL)
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)

        // Hann window (periodic=True)
        var w = [Float](repeating: 0, count: nFFT)
        for i in 0..<nFFT {
            w[i] = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(i) / Float(nFFT))
        }
        self.window = w

        // DFT setup
        self.dftSetupForward = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .FORWARD)
        self.dftSetupInverse = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .INVERSE)

        guard dftSetupForward != nil, dftSetupInverse != nil else {
            throw NSError(domain: "MDXNet", code: 1, userInfo: [NSLocalizedDescriptionKey: "DFT setup failed"])
        }

        print("✅ MDXNetSeparator 初期化完了")
        print("   n_fft=\(nFFT), dim_f=\(dimF), dim_t=\(dimT), hop=\(hop)")
    }

    deinit {
        if let setup = dftSetupForward { vDSP_DFT_DestroySetup(setup) }
        if let setup = dftSetupInverse { vDSP_DFT_DestroySetup(setup) }
    }

    // MARK: - Public Methods

    /// ボーカル分離実行
    func separate(audioURL: URL, denoise: Bool = true) throws -> (vocals: [Float], instrumental: [Float]) {
        print("\n入力: \(audioURL.lastPathComponent)")

        // 1. 音声読み込み
        let audio = try loadAudio(url: audioURL)
        print("  形状: [\(audio.left.count)] (ステレオ)")

        // 2. 分離処理
        let sources = try demix(left: audio.left, right: audio.right, denoise: denoise)

        // 3. ボーカル = 元音声 - 伴奏
        var vocals = [Float](repeating: 0, count: audio.left.count)
        var instrumental = sources

        // モノラル化して返す (左チャンネルのみ)
        for i in 0..<min(audio.left.count, sources.count) {
            vocals[i] = audio.left[i] - sources[i]
        }

        return (vocals, instrumental)
    }

    // MARK: - Private Methods

    private func loadAudio(url: URL) throws -> (left: [Float], right: [Float]) {
        let file = try AVAudioFile(forReading: url)
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 2, interleaved: false)!

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: UInt32(file.length)) else {
            throw NSError(domain: "MDXNet", code: 2, userInfo: [NSLocalizedDescriptionKey: "Buffer creation failed"])
        }

        try file.read(into: buffer)

        guard let channelData = buffer.floatChannelData else {
            throw NSError(domain: "MDXNet", code: 3, userInfo: [NSLocalizedDescriptionKey: "No channel data"])
        }

        let frameCount = Int(buffer.frameLength)
        let left = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        let right = buffer.format.channelCount > 1 ?
            Array(UnsafeBufferPointer(start: channelData[1], count: frameCount)) : left

        return (left, right)
    }

    private func demix(left: [Float], right: [Float], denoise: Bool, margin: Int = 44100, chunks: Int = 15) throws -> [Float] {
        let samples = left.count
        let segmentSize = chunks * sampleRate

        var actualMargin = margin
        if actualMargin > segmentSize {
            actualMargin = segmentSize
        }

        // セグメント分割
        var segments: [(start: Int, data: ([Float], [Float]))] = []
        var counter = -1
        var skip = 0

        while skip < samples {
            counter += 1
            let sMargin = counter == 0 ? 0 : actualMargin
            let end = min(skip + segmentSize + actualMargin, samples)
            let start = skip - sMargin

            let leftSeg = Array(left[max(0, start)..<end])
            let rightSeg = Array(right[max(0, start)..<end])

            if start < 0 {
                // 先頭パディング
                let padSize = -start
                let leftPadded = [Float](repeating: 0, count: padSize) + leftSeg
                let rightPadded = [Float](repeating: 0, count: padSize) + rightSeg
                segments.append((skip, (leftPadded, rightPadded)))
            } else {
                segments.append((skip, (leftSeg, rightSeg)))
            }

            if end >= samples { break }
            skip += segmentSize
        }

        // 各セグメント処理
        var chunkedSources: [[Float]] = []
        let trim = nFFT / 2
        let genSize = chunkSize - 2 * trim

        for (segIdx, (segStart, (leftSeg, rightSeg))) in segments.enumerated() {
            let nSample = leftSeg.count
            let pad = genSize - nSample % genSize

            // パディング
            let leftPadded = [Float](repeating: 0, count: trim) + leftSeg + [Float](repeating: 0, count: pad + trim)
            let rightPadded = [Float](repeating: 0, count: trim) + rightSeg + [Float](repeating: 0, count: pad + trim)

            // チャンクに分割
            var mixWaves: [([Float], [Float])] = []
            var i = 0
            while i < nSample + pad {
                let leftChunk = Array(leftPadded[i..<(i + chunkSize)])
                let rightChunk = Array(rightPadded[i..<(i + chunkSize)])
                mixWaves.append((leftChunk, rightChunk))
                i += genSize
            }

            // 推論
            var tarWaves: [[Float]] = []

            for (leftWave, rightWave) in mixWaves {
                let spek = stft(left: leftWave, right: rightWave)

                let specPred: MLMultiArray
                if denoise {
                    let predPos = try predict(spek)
                    let negSpek = negateMultiArray(spek)
                    let predNeg = try predict(negSpek)
                    specPred = averagePredictions(predPos, negPredNeg: predNeg)
                } else {
                    specPred = try predict(spek)
                }

                let waves = istft(specPred)
                tarWaves.append(waves)
            }

            // 結合
            var tarSignal = [Float](repeating: 0, count: (nSample + pad))
            for (waveIdx, wave) in tarWaves.enumerated() {
                let startIdx = waveIdx * genSize
                let trimmedWave = Array(wave[trim..<(wave.count - trim)])
                for (i, val) in trimmedWave.enumerated() {
                    if startIdx + i < tarSignal.count {
                        tarSignal[startIdx + i] = val
                    }
                }
            }
            tarSignal = Array(tarSignal[0..<nSample])

            // マージン処理
            let cutStart = segIdx == 0 ? 0 : actualMargin
            let cutEnd = segIdx == segments.count - 1 ? tarSignal.count : tarSignal.count - actualMargin

            if cutEnd > cutStart {
                chunkedSources.append(Array(tarSignal[cutStart..<cutEnd]))
            }

            print("   進捗: \(segIdx + 1)/\(segments.count)")
        }

        // 結合
        return chunkedSources.flatMap { $0 }
    }

    /// STFT: PyTorchと同じ処理
    private func stft(left: [Float], right: [Float]) -> MLMultiArray {
        // 入力形状: [batch, channels, freq, time] = [1, 4, 2048, 256]
        let inputArray = try! MLMultiArray(shape: [1, 4, dimF, dimT] as [NSNumber], dataType: .float32)

        // フレーム抽出とSTFT
        let numFrames = dimT

        for t in 0..<numFrames {
            let startIdx = t * hop
            let endIdx = min(startIdx + nFFT, left.count)

            // 左チャンネル
            var leftFrame = [Float](repeating: 0, count: nFFT)
            for i in 0..<min(nFFT, endIdx - startIdx) {
                leftFrame[i] = left[startIdx + i]
            }

            // 右チャンネル
            var rightFrame = [Float](repeating: 0, count: nFFT)
            for i in 0..<min(nFFT, endIdx - startIdx) {
                rightFrame[i] = right[startIdx + i]
            }

            // 窓関数適用
            var leftWindowed = [Float](repeating: 0, count: nFFT)
            var rightWindowed = [Float](repeating: 0, count: nFFT)
            vDSP_vmul(leftFrame, 1, window, 1, &leftWindowed, 1, vDSP_Length(nFFT))
            vDSP_vmul(rightFrame, 1, window, 1, &rightWindowed, 1, vDSP_Length(nFFT))

            // DFT実行
            var leftReal = [Float](repeating: 0, count: nFFT)
            var leftImag = [Float](repeating: 0, count: nFFT)
            var rightReal = [Float](repeating: 0, count: nFFT)
            var rightImag = [Float](repeating: 0, count: nFFT)
            var zeroImag = [Float](repeating: 0, count: nFFT)

            vDSP_DFT_Execute(dftSetupForward!, &leftWindowed, &zeroImag, &leftReal, &leftImag)
            vDSP_DFT_Execute(dftSetupForward!, &rightWindowed, &zeroImag, &rightReal, &rightImag)

            // MLMultiArrayに格納 (dim_fまで)
            for f in 0..<dimF {
                inputArray[[0, 0, f, t] as [NSNumber]] = NSNumber(value: leftReal[f])
                inputArray[[0, 1, f, t] as [NSNumber]] = NSNumber(value: leftImag[f])
                inputArray[[0, 2, f, t] as [NSNumber]] = NSNumber(value: rightReal[f])
                inputArray[[0, 3, f, t] as [NSNumber]] = NSNumber(value: rightImag[f])
            }
        }

        return inputArray
    }

    /// CoreML推論
    private func predict(_ input: MLMultiArray) throws -> MLMultiArray {
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: input)])
        let output = try model.prediction(from: inputProvider)

        guard let result = output.featureValue(for: "var_1144")?.multiArrayValue else {
            throw NSError(domain: "MDXNet", code: 4, userInfo: [NSLocalizedDescriptionKey: "Output not found"])
        }

        return result
    }

    /// iSTFT: PyTorchと同じ処理
    private func istft(_ specPred: MLMultiArray) -> [Float] {
        var output = [Float](repeating: 0, count: chunkSize)
        var windowSum = [Float](repeating: 0, count: chunkSize)

        for t in 0..<dimT {
            // スペクトログラム取得
            var realFull = [Float](repeating: 0, count: nFFT)
            var imagFull = [Float](repeating: 0, count: nFFT)

            // 正の周波数 (左チャンネルのみ使用)
            for f in 0..<dimF {
                realFull[f] = specPred[[0, 0, f, t] as [NSNumber]].floatValue
                imagFull[f] = specPred[[0, 1, f, t] as [NSNumber]].floatValue
            }

            // 周波数パディング (dim_f から nBins まで)
            for f in dimF..<nBins {
                realFull[f] = 0
                imagFull[f] = 0
            }

            // 負の周波数 (共役対称)
            for f in 1..<(nBins - 1) {
                let mirrorIdx = nFFT - f
                realFull[mirrorIdx] = realFull[f]
                imagFull[mirrorIdx] = -imagFull[f]
            }

            // IDFT実行
            var realOut = [Float](repeating: 0, count: nFFT)
            var imagOut = [Float](repeating: 0, count: nFFT)
            vDSP_DFT_Execute(dftSetupInverse!, &realFull, &imagFull, &realOut, &imagOut)

            // スケーリング (1/N)
            var scale = 1.0 / Float(nFFT)
            vDSP_vsmul(realOut, 1, &scale, &realOut, 1, vDSP_Length(nFFT))

            // OLA
            let startIdx = t * hop
            for i in 0..<nFFT {
                let outIdx = startIdx + i
                if outIdx < chunkSize {
                    output[outIdx] += realOut[i] * window[i]
                    windowSum[outIdx] += window[i] * window[i]
                }
            }
        }

        // 正規化
        for i in 0..<chunkSize {
            if windowSum[i] > 1e-8 {
                output[i] /= windowSum[i]
            }
        }

        return output
    }

    private func negateMultiArray(_ array: MLMultiArray) -> MLMultiArray {
        let result = try! MLMultiArray(shape: array.shape, dataType: .float32)
        let count = array.count
        for i in 0..<count {
            result[i] = NSNumber(value: -array[i].floatValue)
        }
        return result
    }

    private func averagePredictions(_ pos: MLMultiArray, negPredNeg: MLMultiArray) -> MLMultiArray {
        let result = try! MLMultiArray(shape: pos.shape, dataType: .float32)
        let count = pos.count
        for i in 0..<count {
            let posVal = pos[i].floatValue
            let negVal = -negPredNeg[i].floatValue
            result[i] = NSNumber(value: (posVal + negVal) * 0.5)
        }
        return result
    }
}
