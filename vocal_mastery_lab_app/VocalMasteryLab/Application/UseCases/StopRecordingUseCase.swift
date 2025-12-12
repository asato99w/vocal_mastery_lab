import Foundation
import VocalisDomain
import OSLog

/// Use case for stopping a recording session
public protocol StopRecordingUseCaseProtocol {
    func setRecordingContext(url: URL)
    func execute() async throws -> StopRecordingResult
}

public class StopRecordingUseCase: StopRecordingUseCaseProtocol {
    private let audioRecorder: AudioRecorderProtocol
    private let recordingRepository: RecordingRepositoryProtocol
    private var currentRecordingURL: URL?

    public init(
        audioRecorder: AudioRecorderProtocol,
        recordingRepository: RecordingRepositoryProtocol
    ) {
        self.audioRecorder = audioRecorder
        self.recordingRepository = recordingRepository
    }

    /// Set the current recording context (called by StartRecordingUseCase)
    public func setRecordingContext(url: URL) {
        self.currentRecordingURL = url
    }

    public func execute() async throws -> StopRecordingResult {
        // Stop the audio recorder
        let duration = try await audioRecorder.stopRecording()

        // Save recording to repository if we have URL context
        var savedRecordingId: RecordingId?
        if let url = currentRecordingURL {
            let recording = Recording(
                fileURL: url,
                createdAt: Date(),
                duration: Duration(seconds: duration)
            )
            try await recordingRepository.save(recording)
            savedRecordingId = recording.id
        }

        // Clear context
        currentRecordingURL = nil

        // Reset audio session mode cache to allow fresh mode selection for next recording
        AudioSessionManager.shared.resetSessionMode()
        Logger.useCase.info("Audio session mode cache reset after recording stop")

        // Return result with recording ID
        return StopRecordingResult(duration: duration, recordingId: savedRecordingId)
    }
}
