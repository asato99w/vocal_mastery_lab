import Foundation
import VocalisDomain
import OSLog

/// Use case for stopping a recording session
public protocol StopRecordingUseCaseProtocol {
    func setRecordingContext(url: URL, settings: ScaleSettings?)
    func execute() async throws -> StopRecordingResult
}

public class StopRecordingUseCase: StopRecordingUseCaseProtocol {
    private let audioRecorder: AudioRecorderProtocol
    private let scalePlayer: ScalePlayerProtocol
    private let recordingRepository: RecordingRepositoryProtocol
    private var currentRecordingURL: URL?
    private var currentSettings: ScaleSettings?

    public init(
        audioRecorder: AudioRecorderProtocol,
        scalePlayer: ScalePlayerProtocol,
        recordingRepository: RecordingRepositoryProtocol
    ) {
        self.audioRecorder = audioRecorder
        self.scalePlayer = scalePlayer
        self.recordingRepository = recordingRepository
    }

    /// Set the current recording context (called by StartRecordingUseCase)
    public func setRecordingContext(url: URL, settings: ScaleSettings?) {
        self.currentRecordingURL = url
        self.currentSettings = settings
    }

    public func execute() async throws -> StopRecordingResult {
        // Stop the scale player first (only if it was playing)
        if currentSettings != nil {
            await scalePlayer.stop()
        }

        // Stop the audio recorder
        let duration = try await audioRecorder.stopRecording()

        // Get playback timeline from scale player (if scale recording was active)
        var playbackTimeline: ScalePlaybackTimeline?
        if currentSettings != nil {
            playbackTimeline = scalePlayer.getPlaybackTimeline()
            scalePlayer.stopTimestampRecording()

            // DEBUG: Log first few note segments for timing analysis
            if let timeline = playbackTimeline {
                let segments = timeline.noteSegments
                FileLogger.shared.log(
                    level: "DEBUG",
                    category: "timing_analysis",
                    message: "PlaybackTimeline: \(segments.count) segments"
                )
                for (index, segment) in segments.enumerated() {
                    FileLogger.shared.log(
                        level: "DEBUG",
                        category: "timing_analysis",
                        message: "Segment[\(index)]: MIDI=\(segment.note.value), start=\(String(format: "%.3f", segment.startTime))s, end=\(String(format: "%.3f", segment.endTime))s, duration=\(String(format: "%.3f", segment.duration))s"
                    )
                }
            }
        }

        // Save recording to repository if we have URL context
        // Note: settings can be nil (when recording without scale)
        var savedRecordingId: RecordingId?
        if let url = currentRecordingURL {
            let recording = Recording(
                fileURL: url,
                createdAt: Date(),
                duration: Duration(seconds: duration),
                scaleSettings: currentSettings,  // Can be nil for scale-off recordings
                playbackTimeline: playbackTimeline  // Timeline for pitch bar UI
            )
            try await recordingRepository.save(recording)
            savedRecordingId = recording.id
        }

        // Clear context
        currentRecordingURL = nil
        currentSettings = nil

        // Reset audio session mode cache to allow fresh mode selection for next recording
        AudioSessionManager.shared.resetSessionMode()
        Logger.useCase.info("Audio session mode cache reset after recording stop")

        // Return result with recording ID
        return StopRecordingResult(duration: duration, recordingId: savedRecordingId)
    }
}
