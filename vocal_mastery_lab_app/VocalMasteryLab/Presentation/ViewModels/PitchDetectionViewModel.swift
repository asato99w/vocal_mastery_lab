import Foundation
import VocalisDomain
import Combine

/// ViewModel for pitch detection functionality
/// Manages real-time pitch detection during recording/playback
@MainActor
public class PitchDetectionViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published public private(set) var detectedPitch: DetectedPitch?
    @Published public private(set) var pitchAccuracy: PitchAccuracy = .none

    // MARK: - Dependencies

    private let pitchDetector: PitchDetectorProtocol
    private let audioPlayer: AudioPlayerProtocol

    // MARK: - Private Properties

    private var cancellables = Set<AnyCancellable>()
    private var pitchDetectionTask: Task<Void, Never>?

    // MARK: - Configuration

    private let playbackPitchPollingIntervalNanoseconds: UInt64

    /// 検出ピッチを保持する時間（秒）
    private let pitchRetentionDuration: TimeInterval = 4.0
    private var lastValidPitchTime: Date?

    // MARK: - Initialization

    public init(
        pitchDetector: PitchDetectorProtocol,
        audioPlayer: AudioPlayerProtocol,
        playbackPitchPollingIntervalNanoseconds: UInt64 = 50_000_000
    ) {
        self.pitchDetector = pitchDetector
        self.audioPlayer = audioPlayer
        self.playbackPitchPollingIntervalNanoseconds = playbackPitchPollingIntervalNanoseconds

        setupPitchDetectorSubscription()
    }

    // MARK: - Setup

    private func setupPitchDetectorSubscription() {
        // Subscribe to pitch detector's publisher to get immediate updates
        pitchDetector.detectedPitchPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] pitch in
                self?.updateDetectedPitch(pitch)
            }
            .store(in: &cancellables)
    }

    // MARK: - Public Methods

    /// Start pitch detection during playback for analysis view
    public func startPlaybackPitchDetection(url: URL) async throws {
        // Start pitch detector
        try await pitchDetector.startRealtimeDetection()

        // Monitor audio player to stop detection when playback ends
        pitchDetectionTask = Task { [weak self] in
            guard let self = self else { return }
            let pollingInterval = await self.playbackPitchPollingIntervalNanoseconds
            while !Task.isCancelled {
                let isPlaying = await MainActor.run { self.audioPlayer.isPlaying }
                guard isPlaying else { break }

                try? await Task.sleep(nanoseconds: pollingInterval)
            }
        }
    }

    /// Stop playback pitch detection
    public func stopPlaybackPitchDetection() {
        pitchDetectionTask?.cancel()
        pitchDetectionTask = nil
        pitchDetector.stopRealtimeDetection()
        lastValidPitchTime = nil
    }

    /// Reset all pitch detection state
    public func reset() {
        detectedPitch = nil
        pitchAccuracy = .none
        lastValidPitchTime = nil

        pitchDetectionTask?.cancel()
        pitchDetectionTask = nil
    }

    // MARK: - Private Methods

    private func updateDetectedPitch(_ pitch: DetectedPitch?) {
        guard let pitch = pitch else {
            // Debounce: retain last valid pitch for a duration
            if let lastValid = lastValidPitchTime {
                let timeSinceLastValid = Date().timeIntervalSince(lastValid)
                if timeSinceLastValid < pitchRetentionDuration {
                    return  // Keep pitch, don't clear
                }
            }
            detectedPitch = nil
            pitchAccuracy = .none
            return
        }

        // Record time of valid pitch detection
        lastValidPitchTime = Date()

        // Validate frequency is reasonable
        guard pitch.frequency > 0 && pitch.frequency < 10000 else {
            detectedPitch = nil
            pitchAccuracy = .none
            return
        }

        detectedPitch = pitch
        pitchAccuracy = PitchAccuracy.from(cents: pitch.cents)
    }

    deinit {
        pitchDetectionTask?.cancel()
    }
}
