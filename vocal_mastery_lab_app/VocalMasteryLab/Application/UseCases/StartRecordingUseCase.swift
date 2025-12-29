import Foundation
import VocalisDomain

/// Use case for starting a recording session
/// Separates preparation (audio session setup) from actual recording start
public protocol StartRecordingUseCaseProtocol {
    /// Prepare for recording: check permissions, configure audio session, prepare recorder
    /// Call this during the "preparing" phase before countdown
    /// - Returns: URL where the recording will be saved
    func prepare(user: User) async throws -> URL

    /// Start the actual recording
    /// Call this after countdown completes
    /// - Returns: RecordingSession with recording info
    func start() async throws -> RecordingSession

    /// Legacy method for backward compatibility (combines prepare + start)
    func execute(user: User) async throws -> RecordingSession
}

public class StartRecordingUseCase: StartRecordingUseCaseProtocol {
    private let audioRecorder: AudioRecorderProtocol
    private let recordingPolicyService: RecordingPolicyService

    /// Stores the prepared recording URL between prepare() and start() calls
    private var preparedRecordingURL: URL?

    public init(
        audioRecorder: AudioRecorderProtocol,
        recordingPolicyService: RecordingPolicyService
    ) {
        self.audioRecorder = audioRecorder
        self.recordingPolicyService = recordingPolicyService
    }

    public func prepare(user: User) async throws -> URL {
        // Check recording permission using domain service
        let permission = try await recordingPolicyService.canStartRecording(user: user)

        guard case .allowed = permission else {
            if case .denied(let reason) = permission {
                throw RecordingPermissionError.from(reason)
            }
            throw RecordingPermissionError.unexpectedState
        }

        // Prepare recording - configure audio session and get the URL
        let recordingURL = try await audioRecorder.prepareRecording()
        preparedRecordingURL = recordingURL

        return recordingURL
    }

    public func start() async throws -> RecordingSession {
        guard let recordingURL = preparedRecordingURL else {
            throw AudioRecorderError.notPrepared
        }

        // Start recording (audio session already configured in prepare())
        try await audioRecorder.startRecording()

        // Clear prepared URL
        preparedRecordingURL = nil

        // Return session info
        return RecordingSession(
            recordingURL: recordingURL,
            startedAt: Date()
        )
    }

    /// Legacy method for backward compatibility
    public func execute(user: User) async throws -> RecordingSession {
        let _ = try await prepare(user: user)
        return try await start()
    }
}
