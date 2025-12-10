import Foundation
import VocalisDomain
import AVFoundation

/// Notification posted when audio playback finishes
public extension Notification.Name {
    static let audioPlaybackDidFinish = Notification.Name("audioPlaybackDidFinish")
}

/// Wrapper for AVAudioPlayer
public class AVAudioPlayerWrapper: NSObject, AudioPlayerProtocol {

    private let settingsRepository: AudioSettingsRepositoryProtocol
    private var audioPlayer: AVAudioPlayer?
    private var playbackContinuation: CheckedContinuation<Void, Error>?

    public init(settingsRepository: AudioSettingsRepositoryProtocol) {
        self.settingsRepository = settingsRepository
        super.init()
    }

    public var isPlaying: Bool {
        return audioPlayer?.isPlaying ?? false
    }

    public var currentTime: TimeInterval {
        return audioPlayer?.currentTime ?? 0
    }

    public var duration: TimeInterval {
        return audioPlayer?.duration ?? 0
    }

    public func pause() {
        audioPlayer?.pause()

        // DO NOT resume continuation on manual pause
        // The continuation should only complete when playback naturally finishes
        // Resuming here would trigger AnalysisViewModel's completion handler prematurely
        // Continuation will be resumed when:
        // 1. Playback naturally completes (audioPlayerDidFinishPlaying)
        // 2. User calls stop()
        // 3. User resumes and playback completes
    }

    public func resume() {
        audioPlayer?.play()
    }

    public func seek(to time: TimeInterval) {
        audioPlayer?.currentTime = time
    }

    public func play(url: URL, withPitchDetection: Bool) async throws {
        // Check if file exists
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioPlayerError.fileNotFound
        }

        do {
            // Create audio player
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.delegate = self

            // Apply recording playback volume from settings
            let settings = settingsRepository.get()
            audioPlayer?.volume = settings.recordingPlaybackVolume

            // Configure audio session based on whether pitch detection is needed
            if withPitchDetection {
                // Use playAndRecord for pitch detection support
                try AudioSessionManager.shared.configureForRecordingAndPlayback()
            } else {
                // Use playback for maximum volume (no recording needed)
                try AudioSessionManager.shared.configureForPlayback()
            }
            try AudioSessionManager.shared.activate()

            // Register as active audio component BEFORE starting playback
            // This prevents AudioSession from being deactivated while we're using it
            AudioSessionManager.shared.registerActiveComponent("AVAudioPlayerWrapper")

            // Play
            let success = audioPlayer?.play() ?? false
            if !success {
                throw AudioPlayerError.playbackFailed("Failed to start playback")
            }

            // Wait for playback to complete
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                self.playbackContinuation = continuation
            }

        } catch let error as AudioPlayerError {
            throw error
        } catch {
            throw AudioPlayerError.playbackFailed(error.localizedDescription)
        }
    }

    public func stop() async {
        audioPlayer?.stop()
        audioPlayer = nil

        // Resume continuation if waiting
        playbackContinuation?.resume()
        playbackContinuation = nil

        // Unregister as active audio component
        // AudioSession will only deactivate if this was the last component
        AudioSessionManager.shared.unregisterActiveComponent("AVAudioPlayerWrapper")

        // Deactivate is now safe - it will only actually deactivate if no other components are active
        // This prevents invalidating AVAudioEngine's node formats when pitch detection is still running
        try? AudioSessionManager.shared.deactivate()
    }
}

// MARK: - AVAudioPlayerDelegate

extension AVAudioPlayerWrapper: AVAudioPlayerDelegate {

    public func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        // Post notification for UI update
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: .audioPlaybackDidFinish, object: nil)
        }

        if flag {
            playbackContinuation?.resume()
        } else {
            playbackContinuation?.resume(throwing: AudioPlayerError.playbackFailed("Playback did not complete successfully"))
        }
        playbackContinuation = nil
    }

    public func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        if let error = error {
            playbackContinuation?.resume(throwing: AudioPlayerError.playbackFailed(error.localizedDescription))
            playbackContinuation = nil
        }
    }
}
