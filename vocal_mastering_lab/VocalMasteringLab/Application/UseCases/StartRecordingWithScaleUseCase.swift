import Foundation
import VocalisDomain

/// Use case for starting a recording session with scale playback
public protocol StartRecordingWithScaleUseCaseProtocol {
    func execute(user: User, settings: ScaleSettings) async throws -> RecordingSession
}

public class StartRecordingWithScaleUseCase: StartRecordingWithScaleUseCaseProtocol {
    private let scalePlayer: ScalePlayerProtocol
    private let audioRecorder: AudioRecorderProtocol
    private let recordingPolicyService: RecordingPolicyService
    private let logger: LoggerProtocol

    public init(
        scalePlayer: ScalePlayerProtocol,
        audioRecorder: AudioRecorderProtocol,
        recordingPolicyService: RecordingPolicyService,
        logger: LoggerProtocol
    ) {
        self.scalePlayer = scalePlayer
        self.audioRecorder = audioRecorder
        self.recordingPolicyService = recordingPolicyService
        self.logger = logger
    }

    public func execute(user: User, settings: ScaleSettings) async throws -> RecordingSession {
        // Check recording permission using domain service
        let permission = try await recordingPolicyService.canStartRecording(user: user, settings: settings)

        guard case .allowed = permission else {
            if case .denied(let reason) = permission {
                logger.warning("Recording denied: \(reason)", category: "useCase")
                throw RecordingPermissionError.from(reason)
            }
            logger.error("Unexpected permission state", category: "useCase")
            throw RecordingPermissionError.unexpectedState
        }

        logger.info("Recording permission granted", category: "useCase")
        let tempo = settings.tempo.secondsPerNote
        logger.info("Starting recording with scale settings, tempo: \(tempo)s", category: "useCase")

        // 1. Generate scale elements with key change chords
        let scaleElements = settings.generateScaleWithKeyChange()
        logger.debug("Generated \(scaleElements.count) scale elements", category: "useCase")

        // 2. Prepare recording - get the URL where audio will be saved
        let recordingURL = try await audioRecorder.prepareRecording()
        logger.info("Recording prepared: \(recordingURL.lastPathComponent)", category: "recording")

        // 3. Load scale elements into player
        try await scalePlayer.loadScaleElements(scaleElements, tempo: settings.tempo)
        logger.debug("Scale elements loaded", category: "scalePlayer")

        // 4. Start recording
        try await audioRecorder.startRecording()
        logger.info("Recording started", category: "recording")

        // 5. Start timestamp recording for pitch bar UI
        let recordingStartTime = Date()
        scalePlayer.startTimestampRecording(recordingStartTime: recordingStartTime)
        logger.debug("Timestamp recording started", category: "scalePlayer")

        // 6. Start audible scale playback immediately (non-blocking)
        // This ensures audio output reaches microphone for pitch detection
        let playbackTask = Task {
            try await scalePlayer.play(muted: false)
        }
        logger.info("Scale playback started", category: "scalePlayer")

        // Give playback a moment to start
        try await Task.sleep(nanoseconds: 100_000_000) // 0.1 second

        // Check if playback failed to start
        if playbackTask.isCancelled {
            logger.error("Playback task was cancelled", category: "scalePlayer")
            _ = try await audioRecorder.stopRecording()
            throw ScalePlayerError.playbackFailed("Playback task was cancelled")
        }

        // 7. Return session info
        logger.info("Recording session created successfully", category: "useCase")
        return RecordingSession(
            recordingURL: recordingURL,
            settings: settings,
            startedAt: Date()
        )
    }
}
