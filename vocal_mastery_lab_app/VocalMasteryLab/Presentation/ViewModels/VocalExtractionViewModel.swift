import Foundation
import VocalisDomain
import Combine

/// State of vocal extraction process
public enum VocalExtractionState: Equatable {
    case idle
    case processing(progress: Double, stage: String)
    case completed(result: ExtractionResultData)
    case error(message: String)

    public static func == (lhs: VocalExtractionState, rhs: VocalExtractionState) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle):
            return true
        case (.processing(let lProgress, let lStage), .processing(let rProgress, let rStage)):
            return lProgress == rProgress && lStage == rStage
        case (.completed(let lResult), .completed(let rResult)):
            return lResult.vocalURL == rResult.vocalURL
        case (.error(let lMessage), .error(let rMessage)):
            return lMessage == rMessage
        default:
            return false
        }
    }
}

/// Data representing extraction result for UI display
public struct ExtractionResultData: Equatable {
    public let vocalURL: URL
    public let instrumentalURL: URL?
    public let duration: Duration

    public init(vocalURL: URL, instrumentalURL: URL? = nil, duration: Duration) {
        self.vocalURL = vocalURL
        self.instrumentalURL = instrumentalURL
        self.duration = duration
    }
}

/// ViewModel for vocal extraction
@MainActor
public class VocalExtractionViewModel: ObservableObject {
    @Published public var state: VocalExtractionState = .idle
    @Published public var isSaving: Bool = false

    private let recording: Recording
    private let extractor: VocalExtractorProtocol
    private let extractedAudioRepository: ExtractedAudioRepositoryProtocol
    private let audioPlayer: AudioPlayerProtocol

    public init(
        recording: Recording,
        extractor: VocalExtractorProtocol,
        extractedAudioRepository: ExtractedAudioRepositoryProtocol,
        audioPlayer: AudioPlayerProtocol
    ) {
        self.recording = recording
        self.extractor = extractor
        self.extractedAudioRepository = extractedAudioRepository
        self.audioPlayer = audioPlayer
    }

    /// Start the extraction process
    public func startExtraction() async {
        state = .processing(progress: 0.0, stage: "準備中...")

        do {
            let result = try await extractor.extract(from: recording.fileURL) { [weak self] progress, stage in
                Task { @MainActor in
                    self?.state = .processing(progress: progress, stage: stage)
                }
            }

            state = .completed(result: ExtractionResultData(
                vocalURL: result.vocalFileURL,
                instrumentalURL: result.instrumentalFileURL,
                duration: result.duration
            ))
        } catch {
            state = .error(message: error.localizedDescription)
        }
    }

    /// Save the extraction result
    public func saveExtraction() async -> Bool {
        guard case .completed(let result) = state else {
            return false
        }

        isSaving = true
        defer { isSaving = false }

        do {
            // Save vocal track
            let vocalAudio = ExtractedAudio(
                sourceRecordingId: recording.id,
                type: .vocal,
                fileURL: result.vocalURL,
                duration: result.duration
            )
            try await extractedAudioRepository.save(vocalAudio)

            // Save instrumental track if available
            if let instrumentalURL = result.instrumentalURL {
                let instrumentalAudio = ExtractedAudio(
                    sourceRecordingId: recording.id,
                    type: .instrumental,
                    fileURL: instrumentalURL,
                    duration: result.duration
                )
                try await extractedAudioRepository.save(instrumentalAudio)
            }

            return true
        } catch {
            state = .error(message: "保存に失敗しました: \(error.localizedDescription)")
            return false
        }
    }

    /// Reset to idle state
    public func reset() {
        // Clean up temporary files if not saved
        if case .completed(let result) = state {
            try? FileManager.default.removeItem(at: result.vocalURL)
            if let instrumentalURL = result.instrumentalURL {
                try? FileManager.default.removeItem(at: instrumentalURL)
            }
        }
        state = .idle
    }

    /// Play original audio
    public func playOriginal() async {
        try? await audioPlayer.play(url: recording.fileURL)
    }

    /// Play vocal track
    public func playVocal() async {
        guard case .completed(let result) = state else { return }
        try? await audioPlayer.play(url: result.vocalURL)
    }

    /// Play instrumental track
    public func playInstrumental() async {
        guard case .completed(let result) = state,
              let instrumentalURL = result.instrumentalURL else { return }
        try? await audioPlayer.play(url: instrumentalURL)
    }

    /// Stop playback
    public func stopPlayback() async {
        await audioPlayer.stop()
    }

    /// Get recording info
    public var recordingTitle: String {
        recording.title ?? "録音"
    }

    public var recordingDuration: String {
        formatTime(recording.duration.seconds)
    }

    public var recordingDate: String {
        recording.formattedDate
    }

    private func formatTime(_ seconds: Double) -> String {
        let minutes = Int(seconds) / 60
        let remainingSeconds = Int(seconds) % 60
        return String(format: "%d:%02d", minutes, remainingSeconds)
    }
}
