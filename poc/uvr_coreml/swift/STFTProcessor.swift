import Foundation
import Accelerate

/// STFT (Short-Time Fourier Transform) プロセッサ
///
/// vDSP/Accelerate Frameworkを使用した高速STFT/iSTFT実装
/// UVR MDX-Net音源分離用に最適化
class STFTProcessor {

    // MARK: - Properties

    /// FFTサイズ（周波数分解能）
    private let fftSize: Int

    /// ホップサイズ（時間分解能）
    private let hopSize: Int

    /// 窓関数タイプ
    private let windowType: WindowType

    /// FFTセットアップ
    private var fftSetup: vDSP_DFT_Setup?

    /// 窓関数バッファ
    private var windowBuffer: [Float]

    /// 周波数ビン数
    var frequencyBins: Int {
        return fftSize / 2 + 1
    }

    // MARK: - Types

    enum WindowType {
        case hann
        case hamming
        case blackman

        var vdspType: vDSP_Window_Type {
            switch self {
            case .hann: return .hanningDenormalized
            case .hamming: return .hammingDenormalized
            case .blackman: return .blackmanDenormalized
            }
        }
    }

    struct SpectrogramData {
        /// 振幅スペクトログラム [frequencyBins, timeFrames]
        let magnitude: [[Float]]

        /// 位相スペクトログラム [frequencyBins, timeFrames]
        let phase: [[Float]]

        /// 時間フレーム数
        var timeFrames: Int { magnitude[0].count }

        /// 周波数ビン数
        var frequencyBins: Int { magnitude.count }
    }

    // MARK: - Initialization

    /// イニシャライザ
    ///
    /// - Parameters:
    ///   - fftSize: FFTサイズ（デフォルト: 4096）
    ///   - hopSize: ホップサイズ（デフォルト: 1024 = fftSize/4）
    ///   - windowType: 窓関数タイプ（デフォルト: hann）
    init(fftSize: Int = 4096, hopSize: Int = 1024, windowType: WindowType = .hann) {
        self.fftSize = fftSize
        self.hopSize = hopSize
        self.windowType = windowType

        // 窓関数生成
        self.windowBuffer = [Float](repeating: 0, count: fftSize)
        vDSP_blkman_window(&self.windowBuffer, vDSP_Length(fftSize), 0)

        // FFTセットアップ作成
        self.fftSetup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(fftSize),
            .FORWARD
        )

        guard fftSetup != nil else {
            fatalError("Failed to create FFT setup")
        }
    }

    deinit {
        if let setup = fftSetup {
            vDSP_DFT_DestroySetup(setup)
        }
    }

    // MARK: - STFT

    /// STFT実行（モノラル）
    ///
    /// - Parameter audio: 音声信号（モノラル）
    /// - Returns: スペクトログラムデータ（振幅・位相）
    func computeSTFT(audio: [Float]) -> SpectrogramData {
        let numFrames = (audio.count - fftSize) / hopSize + 1

        var magnitudes: [[Float]] = Array(
            repeating: Array(repeating: 0, count: numFrames),
            count: frequencyBins
        )
        var phases: [[Float]] = Array(
            repeating: Array(repeating: 0, count: numFrames),
            count: frequencyBins
        )

        // 各フレームでSTFT
        for frameIndex in 0..<numFrames {
            let startIndex = frameIndex * hopSize
            let endIndex = startIndex + fftSize

            guard endIndex <= audio.count else { break }

            // フレーム抽出
            let frame = Array(audio[startIndex..<endIndex])

            // 窓関数適用
            var windowedFrame = [Float](repeating: 0, count: fftSize)
            vDSP_vmul(frame, 1, windowBuffer, 1, &windowedFrame, 1, vDSP_Length(fftSize))

            // FFT実行
            let (magnitude, phase) = performFFT(windowedFrame)

            // スペクトログラムに格納
            for binIndex in 0..<frequencyBins {
                magnitudes[binIndex][frameIndex] = magnitude[binIndex]
                phases[binIndex][frameIndex] = phase[binIndex]
            }
        }

        return SpectrogramData(magnitude: magnitudes, phase: phases)
    }

    /// STFT実行（ステレオ）
    ///
    /// - Parameter audio: 音声信号（ステレオ、インターリーブ）
    /// - Returns: 左右チャンネルのスペクトログラムデータ
    func computeSTFTStereo(audio: [Float]) -> (left: SpectrogramData, right: SpectrogramData) {
        // ステレオをLRに分離
        var leftChannel = [Float]()
        var rightChannel = [Float]()

        for i in stride(from: 0, to: audio.count, by: 2) {
            leftChannel.append(audio[i])
            if i + 1 < audio.count {
                rightChannel.append(audio[i + 1])
            }
        }

        // 各チャンネルでSTFT
        let leftSTFT = computeSTFT(audio: leftChannel)
        let rightSTFT = computeSTFT(audio: rightChannel)

        return (leftSTFT, rightSTFT)
    }

    // MARK: - iSTFT

    /// iSTFT実行（逆STFT）
    ///
    /// - Parameters:
    ///   - magnitude: 振幅スペクトログラム
    ///   - phase: 位相スペクトログラム
    /// - Returns: 再構成された音声信号
    func computeISTFT(magnitude: [[Float]], phase: [[Float]]) -> [Float] {
        guard magnitude.count == frequencyBins,
              phase.count == frequencyBins else {
            fatalError("Invalid spectrogram dimensions")
        }

        let numFrames = magnitude[0].count
        let audioLength = (numFrames - 1) * hopSize + fftSize

        var output = [Float](repeating: 0, count: audioLength)
        var windowSum = [Float](repeating: 0, count: audioLength)

        // 各フレームでiFFT
        for frameIndex in 0..<numFrames {
            // フレームのスペクトル抽出
            var frameMagnitude = [Float](repeating: 0, count: frequencyBins)
            var framePhase = [Float](repeating: 0, count: frequencyBins)

            for binIndex in 0..<frequencyBins {
                frameMagnitude[binIndex] = magnitude[binIndex][frameIndex]
                framePhase[binIndex] = phase[binIndex][frameIndex]
            }

            // iFFT実行
            let timeSignal = performIFFT(magnitude: frameMagnitude, phase: framePhase)

            // オーバーラップ加算
            let startIndex = frameIndex * hopSize
            for i in 0..<fftSize {
                let outputIndex = startIndex + i
                if outputIndex < audioLength {
                    output[outputIndex] += timeSignal[i] * windowBuffer[i]
                    windowSum[outputIndex] += windowBuffer[i] * windowBuffer[i]
                }
            }
        }

        // 正規化
        for i in 0..<audioLength {
            if windowSum[i] > 1e-8 {
                output[i] /= windowSum[i]
            }
        }

        return output
    }

    // MARK: - Private Helpers

    /// FFT実行（実数 → 複素数）
    private func performFFT(_ input: [Float]) -> (magnitude: [Float], phase: [Float]) {
        guard let setup = fftSetup else {
            fatalError("FFT setup not initialized")
        }

        // 実数・虚数部用バッファ
        var real = [Float](repeating: 0, count: fftSize)
        var imaginary = [Float](repeating: 0, count: fftSize)

        // 入力を実数部にコピー
        real = input

        // FFT実行
        vDSP_DFT_Execute(setup, real, imaginary, &real, &imaginary)

        // 振幅・位相計算
        var magnitude = [Float](repeating: 0, count: frequencyBins)
        var phase = [Float](repeating: 0, count: frequencyBins)

        for i in 0..<frequencyBins {
            let re = real[i]
            let im = imaginary[i]

            // 振幅: sqrt(re^2 + im^2)
            magnitude[i] = sqrtf(re * re + im * im)

            // 位相: atan2(im, re)
            phase[i] = atan2f(im, re)
        }

        return (magnitude, phase)
    }

    /// iFFT実行（複素数 → 実数）
    private func performIFFT(magnitude: [Float], phase: [Float]) -> [Float] {
        guard let setup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(fftSize),
            .INVERSE
        ) else {
            fatalError("Failed to create iFFT setup")
        }

        defer {
            vDSP_DFT_DestroySetup(setup)
        }

        // 複素数を実部・虚部に変換
        var real = [Float](repeating: 0, count: fftSize)
        var imaginary = [Float](repeating: 0, count: fftSize)

        for i in 0..<frequencyBins {
            let mag = magnitude[i]
            let ph = phase[i]

            real[i] = mag * cosf(ph)
            imaginary[i] = mag * sinf(ph)
        }

        // 対称性を利用して残りを埋める
        for i in frequencyBins..<fftSize {
            let mirrorIndex = fftSize - i
            real[i] = real[mirrorIndex]
            imaginary[i] = -imaginary[mirrorIndex]
        }

        // iFFT実行
        var output = [Float](repeating: 0, count: fftSize)
        var dummy = [Float](repeating: 0, count: fftSize)

        vDSP_DFT_Execute(setup, real, imaginary, &output, &dummy)

        // スケーリング
        var scale = Float(1.0 / Float(fftSize))
        vDSP_vsmul(output, 1, &scale, &output, 1, vDSP_Length(fftSize))

        return output
    }
}

// MARK: - Convenience Extensions

extension STFTProcessor {
    /// スペクトログラムをCoreML入力形式に変換
    ///
    /// CoreMLは [batch, channel, frequency, time] の4D配列を期待
    /// - Parameter spectrogram: スペクトログラムデータ
    /// - Returns: CoreML入力形式のデータ
    static func spectrogramToCoreMLInput(_ spectrogram: SpectrogramData) -> [Float] {
        let freqBins = spectrogram.frequencyBins
        let timeFrames = spectrogram.timeFrames

        // [frequency, time] → flatten
        var flatArray = [Float]()
        for t in 0..<timeFrames {
            for f in 0..<freqBins {
                flatArray.append(spectrogram.magnitude[f][t])
            }
        }

        return flatArray
    }

    // MARK: - AudioData Integration

    /// AudioDataからSTFT実行
    @available(iOS 17.0, macOS 14.0, *)
    func computeSTFT(audioData: AudioFileProcessor.AudioData) -> (left: SpectrogramData, right: SpectrogramData) {
        let leftSTFT = computeSTFT(audio: audioData.samples[0])
        let rightSTFT = audioData.channelCount > 1 ?
            computeSTFT(audio: audioData.samples[1]) :
            leftSTFT

        return (leftSTFT, rightSTFT)
    }

    /// iSTFT結果をAudioDataに変換
    @available(iOS 17.0, macOS 14.0, *)
    func createAudioData(
        leftMagnitude: [[Float]],
        leftPhase: [[Float]],
        rightMagnitude: [[Float]],
        rightPhase: [[Float]],
        sampleRate: Double
    ) -> AudioFileProcessor.AudioData {
        let leftAudio = computeISTFT(magnitude: leftMagnitude, phase: leftPhase)
        let rightAudio = computeISTFT(magnitude: rightMagnitude, phase: rightPhase)

        return AudioFileProcessor.AudioData(
            samples: [leftAudio, rightAudio],
            sampleRate: sampleRate,
            frameCount: leftAudio.count
        )
    }
}
