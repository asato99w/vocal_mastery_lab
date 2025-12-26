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
    private let logger = Logger(subsystem: "com.kazuasato.VocalisStudio", category: "AnalysisViewModel")

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

    // Latency measurement: track if first timer tick was logged
    private var firstTimerTickLogged: Bool = false
    // Latency measurement: track if first currentTime change was logged
    private var firstCurrentTimeChangeLogged: Bool = false
    // Playback verification: track toggle timestamp for latency calculation
    private var playbackToggleTimestamp: Double = 0.0
    // Playback verification: track previous currentTime for delta calculation
    private var previousTickTime: Double = 0.0
    // Playback verification: tick counter for logging
    private var tickCounter: Int = 0

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
        let toggleTime = CFAbsoluteTimeGetCurrent()
        playbackToggleTimestamp = toggleTime
        FileLogger.shared.log(level: "INFO", category: "latency", message: "🔴 LATENCY_TOGGLE: togglePlayback called at \(toggleTime), isPlaying=\(self.isPlaying)")
        logger.debug("🎵 TOGGLE: togglePlayback() called, isPlaying=\(self.isPlaying)")
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
        let playEnterTime = CFAbsoluteTimeGetCurrent()
        FileLogger.shared.log(level: "INFO", category: "latency", message: "🔴 LATENCY_PLAY_ENTER: play() entered at \(playEnterTime)")
        logger.debug("🎵 PLAY: Entered play(), state=\(String(describing: self.state))")
        guard case .ready = state else {
            logger.debug("🎵 PLAY: Early return - state is NOT .ready")
            return
        }

        logger.debug("🎵 PLAY: State is .ready, setting isPlaying=true")
        isPlaying = true
        firstTimerTickLogged = false  // Reset for latency measurement
        firstCurrentTimeChangeLogged = false  // Reset for latency measurement
        previousTickTime = currentTime  // Reset for delta calculation
        tickCounter = 0  // Reset tick counter
        FileLogger.shared.log(level: "INFO", category: "latency", message: "🔴 LATENCY_ISPLAYING: isPlaying set to true at \(CFAbsoluteTimeGetCurrent())")

        // Check playback state to determine if resuming or starting fresh
        switch playbackState {
        case .paused(let pausedPosition):
            // Resume from current position (which may differ from pausedPosition if user seeked via UI)
            let resumePosition = currentTime

            // Seek BEFORE resume if position changed since pause
            if abs(resumePosition - pausedPosition) > 0.01 {
                audioPlayer.seek(to: resumePosition)
                logger.debug("Seeking to \(resumePosition) before resume (was paused at \(pausedPosition))")
            }

            audioPlayer.resume()
            playbackState = .playing(startedAt: resumePosition)
            // Reset backward-jump prevention tracker to current position
            // (not pausedPosition, which may be stale if user seeked via UI after pause)
            lastDisplayedTime = resumePosition
            logger.debug("Playback resumed from time: \(resumePosition)")

        case .idle, .playing:
            // Start fresh playback from the current position
            let startPosition = currentTime
            playbackState = .playing(startedAt: startPosition)
            // CRITICAL: Reset backward-jump prevention tracker to 0.0 when starting fresh playback
            // The new audioPlayer always starts from 0.0, then seeks if needed.
            // If we set lastDisplayedTime = startPosition (e.g., 0.698), but audioPlayer starts at 0.0,
            // the backward-jump prevention would block all updates until audioPlayer catches up.
            lastDisplayedTime = 0.0

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
        FileLogger.shared.log(level: "INFO", category: "latency", message: "🔴 LATENCY_TIMER_START: Timer starting at \(CFAbsoluteTimeGetCurrent())")
        playbackTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            guard let self = self else { return }

            Task { @MainActor in
                // Only update if still playing (avoid race with completion handler)
                guard self.isPlaying else { return }

                // Only update if audio player has actually started
                // This prevents the seekbar from jerking before playback begins
                if !self.audioPlayer.isPlaying {
                    let timestamp = CFAbsoluteTimeGetCurrent()
                    FileLogger.shared.log(level: "DEBUG", category: "latency", message: "🔵 LATENCY_WAIT_AUDIOPLAYER: timestamp=\(timestamp), audioPlayer.isPlaying=false, waiting...")
                    return
                }

                // Log first timer tick for latency measurement
                if self.firstTimerTickLogged == false {
                    self.firstTimerTickLogged = true
                    let outputLatencyMs = self.outputLatency * 1000
                    FileLogger.shared.log(level: "INFO", category: "latency", message: "🔴 LATENCY_FIRST_TICK: First valid timer tick at \(CFAbsoluteTimeGetCurrent()), currentTime=\(self.audioPlayer.currentTime), outputLatency=\(outputLatencyMs)ms")
                }

                // Apply output latency compensation:
                // audioPlayer.currentTime is the position in the audio data being processed
                // The user hears that audio outputLatency seconds later
                // So for visual sync, we show (rawTime - outputLatency)
                let rawTime = self.audioPlayer.currentTime
                let compensatedTime = max(0, rawTime - self.outputLatency)

                // Log compensation calculation for debugging
                if self.firstTimerTickLogged && self.tickCounter < 5 {
                    FileLogger.shared.log(level: "DEBUG", category: "latency", message: "🔵 LATENCY_COMPENSATION: rawTime=\(rawTime), outputLatency=\(self.outputLatency), compensatedTime=\(compensatedTime), lastDisplayedTime=\(self.lastDisplayedTime)")
                }

                // Backward-jump prevention:
                // When outputLatency suddenly increases (e.g., switching to Bluetooth),
                // the compensated time could jump backwards. Instead of showing
                // a backwards-moving cursor, we hold at the current position and wait.
                if compensatedTime >= self.lastDisplayedTime {
                    let previousTime = self.currentTime
                    self.currentTime = compensatedTime
                    self.lastDisplayedTime = compensatedTime

                    // Log first actual currentTime change for latency measurement
                    if self.firstCurrentTimeChangeLogged == false && abs(self.currentTime - previousTime) > 0.001 {
                        self.firstCurrentTimeChangeLogged = true
                        FileLogger.shared.log(level: "INFO", category: "latency", message: "🔴 LATENCY_CURRENTTIME_CHANGED: currentTime updated at \(CFAbsoluteTimeGetCurrent()), from \(previousTime) to \(self.currentTime)")
                    }

                    // Playback verification: log each tick for regression testing
                    self.tickCounter += 1
                    let delta = self.currentTime - self.previousTickTime
                    let timestamp = CFAbsoluteTimeGetCurrent()
                    FileLogger.shared.log(level: "INFO", category: "playback", message: "🟢 PLAYBACK_TICK: tick=\(self.tickCounter), timestamp=\(timestamp), currentTime=\(self.currentTime), delta=\(delta)")
                    self.previousTickTime = self.currentTime
                } else {
                    // Log skipped update due to backward-jump prevention
                    let timestamp = CFAbsoluteTimeGetCurrent()
                    FileLogger.shared.log(level: "WARN", category: "playback", message: "🟡 PLAYBACK_SKIP: timestamp=\(timestamp), compensatedTime=\(compensatedTime), lastDisplayedTime=\(self.lastDisplayedTime)")
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
