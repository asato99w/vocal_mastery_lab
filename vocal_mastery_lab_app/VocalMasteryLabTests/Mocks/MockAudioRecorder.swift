import Foundation
import VocalisDomain
@testable import VocalMasteryLab

final class MockAudioRecorder: AudioRecorderProtocol {
    var prepareRecordingCalled = false
    var startRecordingCalled = false
    var stopRecordingCalled = false

    var prepareRecordingResult: URL?
    var stopRecordingResult: TimeInterval = 0.0

    var prepareRecordingShouldFail = false
    var startRecordingShouldFail = false
    var stopRecordingShouldFail = false

    var prepareRecordingCallTime: Date?
    var startRecordingCallTime: Date?
    var stopRecordingCallTime: Date?

    var _isRecording = false

    var isRecording: Bool {
        _isRecording
    }

    func prepareRecording() async throws -> URL {
        prepareRecordingCalled = true
        prepareRecordingCallTime = Date()

        if prepareRecordingShouldFail {
            throw AudioRecorderError.notPrepared
        }

        guard let url = prepareRecordingResult else {
            throw AudioRecorderError.notPrepared
        }

        return url
    }

    func startRecording() async throws {
        startRecordingCalled = true
        startRecordingCallTime = Date()

        if startRecordingShouldFail {
            throw AudioRecorderError.recordingFailed("Mock recording error")
        }

        _isRecording = true
    }

    func stopRecording() async throws -> TimeInterval {
        stopRecordingCalled = true
        stopRecordingCallTime = Date()

        if stopRecordingShouldFail {
            throw AudioRecorderError.notRecording
        }

        _isRecording = false
        return stopRecordingResult
    }

    func reset() {
        prepareRecordingCalled = false
        startRecordingCalled = false
        stopRecordingCalled = false
        prepareRecordingResult = nil
        stopRecordingResult = 0.0
        prepareRecordingShouldFail = false
        startRecordingShouldFail = false
        stopRecordingShouldFail = false
        prepareRecordingCallTime = nil
        startRecordingCallTime = nil
        stopRecordingCallTime = nil
        _isRecording = false
    }
}
