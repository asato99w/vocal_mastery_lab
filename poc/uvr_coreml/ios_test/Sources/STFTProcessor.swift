import Foundation
import Accelerate

/// STFT (Short-Time Fourier Transform) プロセッサ
///
/// vDSP/Accelerate Frameworkを使用した高速STFT/iSTFT実装
/// UVR MDX-Net音源分離用に最適化
///
/// 修正版: 正しいvDSP_fft_zrip（実数FFT）を使用
class STFTProcessor {

    // MARK: - Properties

    /// FFTサイズ（周波数分解能）
    private let fftSize: Int

    /// ホップサイズ（時間分解能）
    private let hopSize: Int

    /// 窓関数タイプ
    private let windowType: WindowType

    /// FFTセットアップ (実数FFT用)
    private var fftSetup: FFTSetup?

    /// 窓関数バッファ
    private var windowBuffer: [Float]

    /// log2(fftSize)
    private let log2n: vDSP_Length

    /// 周波数ビン数
    var frequencyBins: Int {
        return fftSize / 2 + 1
    }

    // MARK: - Types

    enum WindowType {
        case hann
        case hamming
        case blackman
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
    ///   - fftSize: FFTサイズ（デフォルト: 4096、2の累乗である必要がある）
    ///   - hopSize: ホップサイズ（デフォルト: 1024 = fftSize/4）
    ///   - windowType: 窓関数タイプ（デフォルト: hann）
    init(fftSize: Int = 4096, hopSize: Int = 1024, windowType: WindowType = .hann) {
        self.fftSize = fftSize
        self.hopSize = hopSize
        self.windowType = windowType
        self.log2n = vDSP_Length(log2(Float(fftSize)))

        // FFTサイズが2の累乗であることを確認
        assert(fftSize == (1 << Int(log2n)), "fftSize must be a power of 2")

        // 窓関数生成
        self.windowBuffer = [Float](repeating: 0, count: fftSize)

        switch windowType {
        case .hann:
            vDSP_hann_window(&self.windowBuffer, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        case .hamming:
            vDSP_hamm_window(&self.windowBuffer, vDSP_Length(fftSize), 0)
        case .blackman:
            vDSP_blkman_window(&self.windowBuffer, vDSP_Length(fftSize), 0)
        }

        // 実数FFT用セットアップ作成
        self.fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))

        guard fftSetup != nil else {
            fatalError("Failed to create FFT setup")
        }
    }

    deinit {
        if let setup = fftSetup {
            vDSP_destroy_fftsetup(setup)
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
        guard magnitude.count == phase.count else {
            fatalError("Magnitude and phase dimensions must match: \(magnitude.count) vs \(phase.count)")
        }

        let actualFreqBins = magnitude.count
        guard actualFreqBins > 0 && magnitude[0].count > 0 else {
            fatalError("Empty spectrogram")
        }

        let numFrames = magnitude[0].count
        let audioLength = (numFrames - 1) * hopSize + fftSize

        var output = [Float](repeating: 0, count: audioLength)
        var windowSum = [Float](repeating: 0, count: audioLength)

        // 各フレームでiFFT
        for frameIndex in 0..<numFrames {
            // フレームのスペクトル抽出
            var frameMagnitude = [Float](repeating: 0, count: actualFreqBins)
            var framePhase = [Float](repeating: 0, count: actualFreqBins)

            for binIndex in 0..<actualFreqBins {
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
    ///
    /// vDSP_fft_zrip を使用した正しい実数FFT実装
    ///
    /// 重要:
    /// - vDSP_fft_zripは実数FFTで、N個の実数入力をN/2個の複素数として扱う
    /// - データはeven-oddパッキング: {A[0],A[2],...,A[n-1],A[1],A[3],...A[n]}
    /// - FFT結果は2×スケールされるため、1/2の補正が必要
    /// 詳細: https://developer.apple.com/documentation/accelerate/1449930-vdsp_fft_zrip
    private func performFFT(_ input: [Float]) -> (magnitude: [Float], phase: [Float]) {
        guard let setup = fftSetup else {
            fatalError("FFT setup not initialized")
        }

        let halfSize = fftSize / 2

        // Split complex format用バッファ
        var realPart = [Float](repeating: 0, count: halfSize)
        var imagPart = [Float](repeating: 0, count: halfSize)

        // Even-oddパッキング: 偶数インデックスをrealp、奇数インデックスをimagpに配置
        for i in 0..<halfSize {
            realPart[i] = input[2*i]      // 偶数インデックス
            imagPart[i] = input[2*i + 1]  // 奇数インデックス
        }

        // FFT実行
        realPart.withUnsafeMutableBufferPointer { realBuffer in
            imagPart.withUnsafeMutableBufferPointer { imagBuffer in
                var splitComplex = DSPSplitComplex(realp: realBuffer.baseAddress!, imagp: imagBuffer.baseAddress!)
                vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
            }
        }

        // vDSP_fft_zripは2×スケールするため、1/2で補正
        var scale = Float(0.5)
        vDSP_vsmul(realPart, 1, &scale, &realPart, 1, vDSP_Length(halfSize))
        vDSP_vsmul(imagPart, 1, &scale, &imagPart, 1, vDSP_Length(halfSize))

        // 結果をmagnitude/phaseに変換
        var magnitude = [Float](repeating: 0, count: frequencyBins)
        var phase = [Float](repeating: 0, count: frequencyBins)

        // DC成分 (bin 0) - realp[0]に格納されている
        magnitude[0] = abs(realPart[0])
        phase[0] = realPart[0] >= 0 ? 0 : Float.pi

        // Nyquist成分 (bin N/2) - imagp[0]に格納されている
        if frequencyBins > halfSize {
            magnitude[halfSize] = abs(imagPart[0])
            phase[halfSize] = imagPart[0] >= 0 ? 0 : Float.pi
        }

        // 中間周波数成分 (bin 1 ~ N/2-1)
        for i in 1..<halfSize {
            let re = realPart[i]
            let im = imagPart[i]
            magnitude[i] = sqrtf(re * re + im * im)
            phase[i] = atan2f(im, re)
        }

        return (magnitude, phase)
    }

    /// iFFT実行（複素数 → 実数）
    ///
    /// vDSP_fft_zrip を使用した正しい実数iFFT実装
    ///
    /// 重要:
    /// - FFTの逆なので、magnitude/phaseを複素数に変換してiFFT実行
    /// - iFFT結果は2×スケールされるため、1/2の補正が必要
    /// - さらにFFT長Nの補正が必要なため、合計で1/(2N)のスケーリング
    /// - Even-oddアンパッキング: {realp[0],imagp[0],realp[1],imagp[1],...}
    private func performIFFT(magnitude: [Float], phase: [Float]) -> [Float] {
        guard let setup = fftSetup else {
            fatalError("FFT setup not initialized")
        }

        let inputBins = magnitude.count
        let halfSize = fftSize / 2

        // Split complex format用バッファ
        var realPart = [Float](repeating: 0, count: halfSize)
        var imagPart = [Float](repeating: 0, count: halfSize)

        // Magnitude/PhaseをSplit Complexに変換
        // DC成分（bin 0）→ realp[0]に配置
        realPart[0] = magnitude[0] * cos(phase[0])

        // Nyquist成分（bin N/2）→ imagp[0]に配置
        imagPart[0] = inputBins > halfSize ? magnitude[halfSize] * cos(phase[halfSize]) : 0

        // 中間周波数成分（bin 1 ~ N/2-1）
        let usedBins = min(inputBins, halfSize)
        for i in 1..<usedBins {
            realPart[i] = magnitude[i] * cos(phase[i])
            imagPart[i] = magnitude[i] * sin(phase[i])
        }

        // iFFT実行
        realPart.withUnsafeMutableBufferPointer { realBuffer in
            imagPart.withUnsafeMutableBufferPointer { imagBuffer in
                var splitComplex = DSPSplitComplex(realp: realBuffer.baseAddress!, imagp: imagBuffer.baseAddress!)
                vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFTDirection(FFT_INVERSE))
            }
        }

        // スケーリング (1/(2N))
        // - vDSP_fft_zripのiFFTは2×スケール → 1/2で補正
        // - FFT長Nの補正 → 1/Nで補正
        // - 合計: 1/(2N) = 1/2 * 1/N
        var scale = Float(1.0) / Float(fftSize * 2)
        vDSP_vsmul(realPart, 1, &scale, &realPart, 1, vDSP_Length(halfSize))
        vDSP_vsmul(imagPart, 1, &scale, &imagPart, 1, vDSP_Length(halfSize))

        // Even-oddアンパッキング: split complex → real signal
        var output = [Float](repeating: 0, count: fftSize)
        for i in 0..<halfSize {
            output[2*i] = realPart[i]      // 偶数インデックス
            output[2*i + 1] = imagPart[i]  // 奇数インデックス
        }

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
