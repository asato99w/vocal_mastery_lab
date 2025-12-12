import Foundation
import VocalisDomain
import Combine
import OSLog
import AVFoundation

/// Analysis state for the analysis screen
public enum AnalysisState: Equatable {
    case loading(progress: Double)  // progress: 0.0 to 1.0
    case ready(result: AnalysisResult)
    case error(message: String)
}

/// ViewModel for the analysis screen
@MainActor
public class AnalysisViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published public private(set) var state: AnalysisState = .loading(progress: 0.0)
    @Published public private(set) var isPlaying: Bool = false
    @Published public private(set) var currentTime: Double = 0.0

    // MARK: - Dependencies

    private let recording: Recording
    private let audioPlayer: AudioPlayerProtocol
    private let analyzeRecordingUseCase: AnalyzeRecordingUseCase
    private let logger = Logger(subsystem: "com.kazuasato.VocalMasteryLab", category: "AnalysisViewModel")

    // MARK: - Private Properties

    private var playbackTimer: Timer?
    private var cancellables = Set<AnyCancellable>()

    // Output latency compensation for playback cursor
    // The cursor should show the time at which the user hears the sound,
    // which is (audioPlayer.currentTime - outputLatency) since there's a delay
    // between the audio data position and when it actually reaches the output device.
    private var outputLatency: TimeInterval {
        AVAudioSession.sharedInstance().outputLatency
    }

    // Backward-jump prevention: track last displayed time to avoid cursor jumping
    // backwards when outputLatency suddenly increases (e.g., switching to Bluetooth)
    private var lastDisplayedTime: Double = 0.0

    /// Playback state machine for managing play/pause/completion flow
    /// This explicit state enum makes state transitions clear and prevents bugs
    /// related to forgetting to set/clear state variables
    private enum PlaybackState {
        case idle                           // Stopped state
        case playing(startedAt: Double)     // Playing from specific position
        case paused(at: Double)            // Paused at specific position
    }

    private var playbackState: PlaybackState = .idle

    // MARK: - Computed Properties

    public var duration: Double {
        recording.duration.seconds
    }

    public var analysisResult: AnalysisResult? {
        if case .ready(let result) = state {
            return result
        }
        return nil
    }

    /// Returns true if analysis is in progress
    public var isAnalyzing: Bool {
        if case .loading = state {
            return true
        }
        return false
    }

    // MARK: - Initialization

    public init(
        recording: Recording,
        audioPlayer: AudioPlayerProtocol,
        analyzeRecordingUseCase: AnalyzeRecordingUseCase
    ) {
        self.recording = recording
        self.audioPlayer = audioPlayer
        self.analyzeRecordingUseCase = analyzeRecordingUseCase
    }

    // MARK: - Public Methods

    /// Start analysis when view appears
    public func startAnalysis() async {
        logger.info("Starting analysis for recording: \(self.recording.id.value.uuidString)")

        state = .loading(progress: 0.0)

        do {
            // Execute analysis use case with progress reporting
            let result = try await analyzeRecordingUseCase.execute(recording: recording) { [weak self] progress in
                guard let self = self else { return }
                self.state = .loading(progress: progress)
            }

            state = .ready(result: result)
            logger.info("Analysis completed successfully")

        } catch {
            logger.error("Analysis failed: \(error.localizedDescription)")
            state = .error(message: error.localizedDescription)
        }
    }

    /// Toggle playback
    public func togglePlayback() {
        if isPlaying {
            pause()
        } else {
            play()
        }
    }

    /// Seek to specific time
    public func seek(to time: Double) {
        currentTime = min(max(0, time), duration)
        // Reset backward-jump prevention tracker for manual seeks
        lastDisplayedTime = currentTime
        audioPlayer.seek(to: currentTime)
    }

    /// Skip backward 5 seconds
    public func skipBackward() {
        seek(to: currentTime - 5.0)
    }

    /// Skip forward 5 seconds
    public func skipForward() {
        seek(to: currentTime + 5.0)
    }

    /// Stop playback completely (used when navigating away from the screen)
    public func stopPlayback() async {
        // Stop timer first
        playbackTimer?.invalidate()
        playbackTimer = nil

        // Stop audio player
        await audioPlayer.stop()

        // Reset state
        isPlaying = false
        currentTime = 0.0
        playbackState = .idle
    }

    // MARK: - Private Methods

    private func play() {
        guard case .ready = state else {
            return
        }

        isPlaying = true

        // Check playback state to determine if resuming or starting fresh
        switch playbackState {
        case .paused(let pausedPosition):
            // Resume from paused position
            audioPlayer.resume()
            playbackState = .playing(startedAt: pausedPosition)
            // Reset backward-jump prevention tracker to allow normal updates
            lastDisplayedTime = pausedPosition
            logger.debug("Playback resumed from time: \(pausedPosition)")

        case .idle, .playing:
            // Start fresh playback from beginning
            let startPosition = currentTime
            playbackState = .playing(startedAt: startPosition)
            // Reset backward-jump prevention tracker to allow normal updates
            lastDisplayedTime = startPosition

            Task { [weak self] in
                guard let self = self else { return }
                do {
                    // Pitch data is pre-analyzed, no real-time detection needed
                    try await self.audioPlayer.play(url: self.recording.fileURL, withPitchDetection: false)

                    // Playback finished - check state to determine next action
                    await MainActor.run {
                        switch self.playbackState {
                        case .paused(let time):
                            // Manual pause occurred - restore position
                            self.logger.debug("🎵 COMPLETION: Manual pause detected, restoring time: \(time)")
                            self.currentTime = time
                            self.playbackState = .idle

                        case .playing:
                            // Natural completion - reset to beginning
                            self.logger.debug("🎵 COMPLETION: Natural completion, resetting")
                            self.isPlaying = false

                            // Stop timer if still running
                            self.playbackTimer?.invalidate()
                            self.playbackTimer = nil

                            // Reset position to beginning
                            self.currentTime = 0.0
                            self.playbackState = .idle
                            self.logger.debug("🎵 COMPLETION: Reset complete. isPlaying=\(self.isPlaying), currentTime=\(self.currentTime)")

                        case .idle:
                            // Already handled or stopped
                            break
                        }
                    }
                } catch {
                    self.logger.error("Audio playback failed: \(error.localizedDescription)")
                    await MainActor.run {
                        self.pause()
                    }
                }
            }
            logger.debug("Playback started from time: \(startPosition)")
        }

        // Start playback timer to update currentTime
        playbackTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            guard let self = self else { return }

            Task { @MainActor in
                // Only update if still playing (avoid race with completion handler)
                guard self.isPlaying else { return }

                // Only update if audio player has actually started
                // This prevents the seekbar from jerking before playback begins
                guard self.audioPlayer.isPlaying else { return }

                // Apply output latency compensation:
                // audioPlayer.currentTime is the position in the audio data being processed
                // The user hears that audio outputLatency seconds later
                // So for visual sync, we show (rawTime - outputLatency)
                let rawTime = self.audioPlayer.currentTime
                let compensatedTime = max(0, rawTime - self.outputLatency)

                // Backward-jump prevention:
                // When outputLatency suddenly increases (e.g., switching to Bluetooth),
                // the compensated time could jump backwards. Instead of showing
                // a backwards-moving cursor, we hold at the current position and wait.
                if compensatedTime >= self.lastDisplayedTime {
                    self.currentTime = compensatedTime
                    self.lastDisplayedTime = compensatedTime
                }
                // If compensatedTime < lastDisplayedTime, we skip the update
                // and the cursor holds steady until rawTime catches up
            }
        }
    }

    private func pause() {
        // CRITICAL: Stop timer FIRST before reading position to prevent race condition
        playbackTimer?.invalidate()
        playbackTimer = nil

        // Get actual playback position and transition to paused state
        // CRITICAL: Use audioPlayer.currentTime (actual position), not self.currentTime
        // to avoid race condition with timer updates
        let pausedPosition = audioPlayer.currentTime
        playbackState = .paused(at: pausedPosition)

        isPlaying = false

        audioPlayer.pause()

        // Update currentTime to match the actual paused position
        currentTime = pausedPosition
    }

    deinit {
        playbackTimer?.invalidate()
    }
}
