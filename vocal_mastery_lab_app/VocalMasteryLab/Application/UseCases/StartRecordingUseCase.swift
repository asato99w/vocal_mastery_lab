import Foundation
import VocalisDomain

/// Use case for starting a recording session
public protocol StartRecordingUseCaseProtocol {
    func execute(user: User) async throws -> RecordingSession
}

public class StartRecordingUseCase: StartRecordingUseCaseProtocol {
    private let audioRecorder: AudioRecorderProtocol
    private let recordingPolicyService: RecordingPolicyService

    public init(
        audioRecorder: AudioRecorderProtocol,
        recordingPolicyService: RecordingPolicyService
    ) {
        self.audioRecorder = audioRecorder
        self.recordingPolicyService = recordingPolicyService
    }

    public func execute(user: User) async throws -> RecordingSession {
        // Check recording permission using domain service
        let permission = try await recordingPolicyService.canStartRecording(user: user)

        guard case .allowed = permission else {
            if case .denied(let reason) = permission {
                throw RecordingPermissionError.from(reason)
            }
            throw RecordingPermissionError.unexpectedState
        }
        // Prepare recording - get the URL where audio will be saved
        let recordingURL = try await audioRecorder.prepareRecording()

        // Start recording
        try await audioRecorder.startRecording()

        // Return session info
        return RecordingSession(
            recordingURL: recordingURL,
            startedAt: Date()
        )
    }
}
