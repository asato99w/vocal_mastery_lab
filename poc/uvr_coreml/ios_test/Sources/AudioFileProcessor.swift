import Foundation
import AVFoundation
import Accelerate

/// AVAudioFileを使用した音声ファイル処理
@available(iOS 17.0, macOS 14.0, *)
class AudioFileProcessor {

    // MARK: - Types

    struct AudioData {
        let samples: [[Float]]  // [channels][samples]
        let sampleRate: Double
        let frameCount: Int

        var channelCount: Int { samples.count }
    }

    enum ProcessingError: Error {
        case fileNotFound(String)
        case unsupportedFormat(String)
        case readError(String)
        case writeError(String)
        case conversionError(String)
    }

    // MARK: - Audio Loading

    /// 音声ファイルを読み込む
    static func loadAudio(from url: URL, targetSampleRate: Double? = nil) throws -> AudioData {
        print("📂 音声読み込み: \(url.lastPathComponent)")

        // ファイル存在確認
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ProcessingError.fileNotFound("ファイルが見つかりません: \(url.path)")
        }

        // AVAudioFile読み込み
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat

        print("  元の形式:")
        print("    サンプルレート: \(format.sampleRate) Hz")
        print("    チャンネル数: \(format.channelCount)")
        print("    フレーム数: \(audioFile.length)")

        // PCMバッファ作成
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(audioFile.length)
        ) else {
            throw ProcessingError.readError("バッファ作成失敗")
        }

        // データ読み込み
        try audioFile.read(into: buffer)

        // Float配列に変換
        var samples = convertBufferToFloatArray(buffer)

        // サンプルレート変換が必要な場合
        if let targetRate = targetSampleRate, targetRate != format.sampleRate {
            print("  サンプルレート変換: \(format.sampleRate) Hz → \(targetRate) Hz")
            samples = try resample(
                samples,
                from: format.sampleRate,
                to: targetRate
            )
        }

        let finalSampleRate = targetSampleRate ?? format.sampleRate

        print("  読み込み完了:")
        print("    サンプルレート: \(finalSampleRate) Hz")
        print("    チャンネル数: \(samples.count)")
        print("    サンプル数: \(samples[0].count)")
        print("    長さ: \(String(format: "%.2f", Double(samples[0].count) / finalSampleRate)) 秒")

        return AudioData(
            samples: samples,
            sampleRate: finalSampleRate,
            frameCount: samples[0].count
        )
    }

    // MARK: - Audio Saving

    /// 音声ファイルを保存
    static func saveAudio(
        _ audioData: AudioData,
        to url: URL,
        format: AVAudioCommonFormat = .pcmFormatFloat32
    ) throws {
        print("💾 音声保存: \(url.lastPathComponent)")

        // 出力フォーマット作成
        guard let outputFormat = AVAudioFormat(
            commonFormat: format,
            sampleRate: audioData.sampleRate,
            channels: AVAudioChannelCount(audioData.channelCount),
            interleaved: false
        ) else {
            throw ProcessingError.writeError("フォーマット作成失敗")
        }

        // AVAudioFile作成
        let audioFile = try AVAudioFile(
            forWriting: url,
            settings: outputFormat.settings
        )

        // PCMバッファ作成
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(audioData.frameCount)
        ) else {
            throw ProcessingError.writeError("バッファ作成失敗")
        }

        buffer.frameLength = AVAudioFrameCount(audioData.frameCount)

        // データをバッファにコピー
        for channel in 0..<audioData.channelCount {
            guard let channelData = buffer.floatChannelData?[channel] else {
                throw ProcessingError.writeError("チャンネルデータアクセス失敗")
            }

            audioData.samples[channel].withUnsafeBufferPointer { srcBuffer in
                channelData.update(from: srcBuffer.baseAddress!, count: audioData.frameCount)
            }
        }

        // ファイルに書き込み
        try audioFile.write(from: buffer)

        let fileSize = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
        let fileSizeMB = Double(fileSize) / (1024 * 1024)

        print("  保存完了:")
        print("    ファイルサイズ: \(String(format: "%.2f", fileSizeMB)) MB")
        print("    パス: \(url.path)")
    }

    // MARK: - Format Conversion

    /// AVAudioPCMBufferをFloat配列に変換
    private static func convertBufferToFloatArray(_ buffer: AVAudioPCMBuffer) -> [[Float]] {
        let channelCount = Int(buffer.format.channelCount)
        let frameLength = Int(buffer.frameLength)

        var samples: [[Float]] = []

        for channel in 0..<channelCount {
            guard let channelData = buffer.floatChannelData?[channel] else {
                continue
            }

            let channelSamples = Array(UnsafeBufferPointer(
                start: channelData,
                count: frameLength
            ))

            samples.append(channelSamples)
        }

        // モノラルの場合はステレオに変換
        if samples.count == 1 {
            samples.append(samples[0])
        }

        return samples
    }

    /// サンプルレート変換
    private static func resample(
        _ samples: [[Float]],
        from sourceSampleRate: Double,
        to targetSampleRate: Double
    ) throws -> [[Float]] {

        guard sourceSampleRate != targetSampleRate else {
            return samples
        }

        let ratio = targetSampleRate / sourceSampleRate
        let sourceLength = samples[0].count
        let targetLength = Int(Double(sourceLength) * ratio)

        var resampledSamples: [[Float]] = []

        for channel in samples {
            var resampled = [Float](repeating: 0, count: targetLength)

            // 線形補間によるリサンプリング
            for i in 0..<targetLength {
                let srcIndex = Double(i) / ratio
                let index0 = Int(srcIndex)
                let index1 = min(index0 + 1, sourceLength - 1)
                let fraction = Float(srcIndex - Double(index0))

                resampled[i] = channel[index0] * (1.0 - fraction) + channel[index1] * fraction
            }

            resampledSamples.append(resampled)
        }

        return resampledSamples
    }

    // MARK: - Utility

    /// ステレオからモノラルに変換
    static func convertToMono(_ audioData: AudioData) -> AudioData {
        guard audioData.channelCount > 1 else {
            return audioData
        }

        let monoSamples = zip(audioData.samples[0], audioData.samples[1]).map { (l, r) in
            (l + r) / 2.0
        }

        return AudioData(
            samples: [monoSamples],
            sampleRate: audioData.sampleRate,
            frameCount: monoSamples.count
        )
    }

    /// モノラルからステレオに変換
    static func convertToStereo(_ audioData: AudioData) -> AudioData {
        guard audioData.channelCount == 1 else {
            return audioData
        }

        return AudioData(
            samples: [audioData.samples[0], audioData.samples[0]],
            sampleRate: audioData.sampleRate,
            frameCount: audioData.frameCount
        )
    }

    /// 音量正規化
    static func normalize(_ audioData: AudioData) -> AudioData {
        var normalizedSamples: [[Float]] = []

        // 全チャンネルの最大値を取得
        var maxValue: Float = 0
        for channel in audioData.samples {
            for sample in channel {
                maxValue = max(maxValue, abs(sample))
            }
        }

        guard maxValue > 0 else {
            return audioData
        }

        // 正規化
        let scale = 1.0 / maxValue
        for channel in audioData.samples {
            let normalized = channel.map { $0 * scale }
            normalizedSamples.append(normalized)
        }

        return AudioData(
            samples: normalizedSamples,
            sampleRate: audioData.sampleRate,
            frameCount: audioData.frameCount
        )
    }
}
