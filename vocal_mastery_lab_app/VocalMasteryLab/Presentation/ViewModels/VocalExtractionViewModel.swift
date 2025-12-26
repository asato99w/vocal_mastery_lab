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

/// Audio source type for playback
public enum PlaybackSource {
    case none
    case original
    case vocal
    case instrumental
}

/// ViewModel for vocal extraction
@MainActor
public class VocalExtractionViewModel: ObservableObject {
    @Published public var state: VocalExtractionState = .idle
    @Published public var isSaving: Bool = false
    @Published public private(set) var playingSource: PlaybackSource = .none
    @Published public private(set) var currentTime: TimeInterval = 0.0
    @Published public private(set) var isPlaying: Bool = false
    @Published public private(set) var extractionCount: Int = 0

    private let recording: Recording
    private let extractor: VocalExtractorProtocol
    private let extractedAudioRepository: ExtractedAudioRepositoryProtocol
    private let audioPlayer: AudioPlayerProtocol
    private var positionTrackingTask: Task<Void, Never>?

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

    deinit {
        positionTrackingTask?.cancel()
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

            extractionCount = 1
            state = .completed(result: ExtractionResultData(
                vocalURL: result.vocalFileURL,
                instrumentalURL: result.instrumentalFileURL,
                duration: result.duration
            ))
        } catch {
            state = .error(message: error.localizedDescription)
        }
    }

    /// Start secondary extraction (extract from already extracted vocal for cleaner result)
    public func startSecondaryExtraction() async {
        guard case .completed(let currentResult) = state else { return }

        // Store the original instrumental URL and its data for verification
        let originalInstrumentalURL = currentResult.instrumentalURL
        let originalInstrumentalData: Data?
        if let url = originalInstrumentalURL {
            originalInstrumentalData = try? Data(contentsOf: url)
        } else {
            originalInstrumentalData = nil
        }

        // Keep reference to old vocal URL for cleanup
        let oldVocalURL = currentResult.vocalURL

        state = .processing(progress: 0.0, stage: "2次抽出 準備中...")

        do {
            // Extract from the current vocal (not the original recording)
            let result = try await extractor.extract(from: oldVocalURL) { [weak self] progress, stage in
                Task { @MainActor in
                    self?.state = .processing(progress: progress, stage: "2次抽出: \(stage)")
                }
            }

            // Verify instrumental hasn't changed (if it existed)
            if let originalURL = originalInstrumentalURL, let originalData = originalInstrumentalData {
                // The instrumental from secondary extraction should be ignored
                // We keep the original instrumental
                assert(FileManager.default.fileExists(atPath: originalURL.path),
                       "Original instrumental file should still exist")
                let currentData = try? Data(contentsOf: originalURL)
                assert(currentData == originalData,
                       "Instrumental audio data should not have changed during secondary extraction")
            }

            // Clean up the old vocal file
            try? FileManager.default.removeItem(at: oldVocalURL)

            // Clean up the new instrumental from secondary extraction (we don't need it)
            if let newInstrumentalURL = result.instrumentalFileURL {
                try? FileManager.default.removeItem(at: newInstrumentalURL)
            }

            extractionCount += 1
            // Update state with new vocal but keep original instrumental
            state = .completed(result: ExtractionResultData(
                vocalURL: result.vocalFileURL,
                instrumentalURL: originalInstrumentalURL,  // Keep original instrumental
                duration: result.duration
            ))
        } catch {
            // Restore previous state on error
            state = .completed(result: currentResult)
            state = .error(message: "2次抽出に失敗しました: \(error.localizedDescription)")
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
        await stopPlayback()
        playingSource = .original
        isPlaying = true
        currentTime = 0
        startPositionTracking()
        try? await audioPlayer.play(url: recording.fileURL)
        await handlePlaybackFinished()
    }

    /// Play vocal track
    public func playVocal() async {
        guard case .completed(let result) = state else { return }
        await stopPlayback()
        playingSource = .vocal
        isPlaying = true
        currentTime = 0
        startPositionTracking()
        try? await audioPlayer.play(url: result.vocalURL)
        await handlePlaybackFinished()
    }

    /// Play instrumental track
    public func playInstrumental() async {
        guard case .completed(let result) = state,
              let instrumentalURL = result.instrumentalURL else { return }
        await stopPlayback()
        playingSource = .instrumental
        isPlaying = true
        currentTime = 0
        startPositionTracking()
        try? await audioPlayer.play(url: instrumentalURL)
        await handlePlaybackFinished()
    }

    /// Toggle play/pause
    public func togglePlayPause() {
        if audioPlayer.isPlaying {
            audioPlayer.pause()
            isPlaying = false
        } else {
            audioPlayer.resume()
            isPlaying = true
        }
    }

    /// Seek to position
    public func seek(to time: TimeInterval) {
        audioPlayer.seek(to: time)
        currentTime = time
    }

    /// Stop playback
    public func stopPlayback() async {
        stopPositionTracking()
        await audioPlayer.stop()
        playingSource = .none
        isPlaying = false
        currentTime = 0
    }

    /// Current duration based on playing source
    public var currentDuration: TimeInterval {
        switch playingSource {
        case .none:
            return recording.duration.seconds
        case .original:
            return recording.duration.seconds
        case .vocal, .instrumental:
            if case .completed(let result) = state {
                return result.duration.seconds
            }
            return 0
        }
    }

    // MARK: - Private Helpers

    private func startPositionTracking() {
        stopPositionTracking()
        positionTrackingTask = Task { @MainActor in
            while !Task.isCancelled {
                currentTime = audioPlayer.currentTime
                try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }
        }
    }

    private func stopPositionTracking() {
        positionTrackingTask?.cancel()
        positionTrackingTask = nil
    }

    private func handlePlaybackFinished() async {
        stopPositionTracking()
        playingSource = .none
        isPlaying = false
        currentTime = 0
    }

    /// Get recording info
    public var recordingTitle: String {
        recording.title ?? "録音"
    }

    public var recordingDuration: String {
        formatTime(recording.duration.seconds)
    }

    /// Original recording duration in seconds
    public var originalDurationSeconds: TimeInterval {
        recording.duration.seconds
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
