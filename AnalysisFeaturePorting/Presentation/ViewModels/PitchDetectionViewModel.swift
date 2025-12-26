import Foundation
import VocalisDomain
import Combine

/// ViewModel for pitch detection functionality
/// Manages target pitch monitoring and real-time pitch detection during recording/playback
@MainActor
public class PitchDetectionViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published public private(set) var targetPitch: DetectedPitch?
    @Published public private(set) var detectedPitch: DetectedPitch?
    @Published public private(set) var pitchAccuracy: PitchAccuracy = .none

    // MARK: - Dependencies

    private let pitchDetector: PitchDetectorProtocol
    private let scalePlaybackCoordinator: ScalePlaybackCoordinator
    private let audioPlayer: AudioPlayerProtocol

    // MARK: - Private Properties

    private var cancellables = Set<AnyCancellable>()
    private var progressMonitorTask: Task<Void, Never>?
    private var pitchDetectionTask: Task<Void, Never>?

    // MARK: - Configuration

    private let targetPitchPollingIntervalNanoseconds: UInt64
    private let playbackPitchPollingIntervalNanoseconds: UInt64

    /// 検出ピッチを保持する時間（秒）
    /// 再生中はAVAudioSessionの制約でピッチ検出成功率が低いため、
    /// 録音中に検出したピッチを一定時間保持してUI表示を維持する
    private let pitchRetentionDuration: TimeInterval = 4.0
    private var lastValidPitchTime: Date?

    // MARK: - Initialization

    public init(
        pitchDetector: PitchDetectorProtocol,
        scalePlaybackCoordinator: ScalePlaybackCoordinator,
        audioPlayer: AudioPlayerProtocol,
        targetPitchPollingIntervalNanoseconds: UInt64 = 100_000_000,
        playbackPitchPollingIntervalNanoseconds: UInt64 = 50_000_000
    ) {
        self.pitchDetector = pitchDetector
        self.scalePlaybackCoordinator = scalePlaybackCoordinator
        self.audioPlayer = audioPlayer
        self.targetPitchPollingIntervalNanoseconds = targetPitchPollingIntervalNanoseconds
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

    /// Start monitoring target pitch during recording with scale playback
    public func startTargetPitchMonitoring(settings: ScaleSettings) async throws {
        // NOTE: Scale playback should already be started by caller (RecordingViewModel)
        // This method only starts the monitoring task to poll current scale element

        FileLogger.shared.log(
            level: "INFO",
            category: "pitch_monitoring",
            message: "🔵 Target pitch monitoring started (polling interval: \(targetPitchPollingIntervalNanoseconds / 1_000_000)ms)"
        )

        // Start monitoring task
        progressMonitorTask = Task { [weak self] in
            guard let self = self else { return }
            let pollingInterval = await self.targetPitchPollingIntervalNanoseconds
            var loopCount = 0
            var lastDebugLogTime = Date()

            while !Task.isCancelled {
                loopCount += 1
                let now = Date()

                // 🔍 Log every loop iteration with detailed state
                let interval = now.timeIntervalSince(lastDebugLogTime) * 1000
                FileLogger.shared.log(level: "INFO", category: "pitch_monitoring", message: "[DIAG] Loop #\(loopCount) START: isCancelled=\(Task.isCancelled), interval=\(String(format: "%.0f", interval))ms")
                lastDebugLogTime = now

                // Check scale player current element via coordinator
                if let currentElement = self.scalePlaybackCoordinator.currentScaleElement {
                    FileLogger.shared.log(level: "INFO", category: "pitch_monitoring", message: "[DIAG] Loop #\(loopCount) Before updateTargetPitch: targetPitch=\(String(describing: targetPitch))")
                    await self.updateTargetPitchFromScaleElement(currentElement)
                    FileLogger.shared.log(level: "INFO", category: "pitch_monitoring", message: "[DIAG] Loop #\(loopCount) After updateTargetPitch: targetPitch=\(String(describing: targetPitch))")
                } else {
                    await MainActor.run { self.targetPitch = nil }
                    FileLogger.shared.log(level: "INFO", category: "pitch_monitoring", message: "[DIAG] Loop #\(loopCount) currentScaleElement is nil")
                }

                // Note: Detected pitch is now automatically updated via Combine subscription
                // No manual polling needed here

                try? await Task.sleep(nanoseconds: pollingInterval)
            }

            FileLogger.shared.log(level: "INFO", category: "pitch_monitoring", message: "[DIAG] Loop EXITED: final targetPitch=\(String(describing: targetPitch)), iterations=\(loopCount)")
        }
    }

    /// Stop target pitch monitoring
    public func stopTargetPitchMonitoring() async {
        FileLogger.shared.log(
            level: "INFO",
            category: "pitch_monitoring",
            message: "🔴 stopTargetPitchMonitoring START: targetPitch=\(String(describing: targetPitch)), taskExists=\(progressMonitorTask != nil)"
        )

        progressMonitorTask?.cancel()
        FileLogger.shared.log(level: "DEBUG", category: "pitch_monitoring", message: "🔴 Task.cancel() called")

        _ = await progressMonitorTask?.value
        FileLogger.shared.log(
            level: "DEBUG",
            category: "pitch_monitoring",
            message: "🔴 Task.value returned: targetPitch=\(String(describing: targetPitch))"
        )

        progressMonitorTask = nil
        targetPitch = nil
        FileLogger.shared.log(level: "INFO", category: "pitch_monitoring", message: "🔴 stopTargetPitchMonitoring END: targetPitch set to nil")
    }

    /// Start pitch detection during playback for analysis view
    public func startPlaybackPitchDetection(url: URL) async throws {
        // Start pitch detector
        // Pitch updates are automatically handled by Combine subscription
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
        targetPitch = nil
        detectedPitch = nil
        pitchAccuracy = .none
        lastValidPitchTime = nil

        progressMonitorTask?.cancel()
        progressMonitorTask = nil
        pitchDetectionTask?.cancel()
        pitchDetectionTask = nil
    }

    // MARK: - Private Methods

    private func updateTargetPitchFromScaleElement(_ element: ScaleElement) {
        switch element {
        case .scaleNote(let note):
            let pitch = DetectedPitch.fromFrequency(
                note.frequency,
                confidence: 1.0
            )
            targetPitch = pitch

        case .longNote(let note, _):
            // Long note uses the same pitch as scale note
            let pitch = DetectedPitch.fromFrequency(
                note.frequency,
                confidence: 1.0
            )
            targetPitch = pitch

        case .chordLong(let notes), .chordShort(let notes):
            // Use root note of chord as target
            if let rootNote = notes.first {
                let pitch = DetectedPitch.fromFrequency(
                    rootNote.frequency,
                    confidence: 1.0
                )
                targetPitch = pitch
            } else {
                targetPitch = nil
            }

        case .silence:
            targetPitch = nil
        }
    }

    // MARK: - Logging Support

    private var pitchUpdateCount = 0
    private var lastLogTime = Date()

    private func updateDetectedPitch(_ pitch: DetectedPitch?) {
        pitchUpdateCount += 1

        // 🔍 Log every call to this method for first 50 calls to understand frequency
        if pitchUpdateCount <= 50 {
            let now = Date()
            let interval = now.timeIntervalSince(lastLogTime) * 1000
            FileLogger.shared.log(
                level: "DEBUG",
                category: "pitch_update",
                message: "📞 updateDetectedPitch called #\(pitchUpdateCount) (interval: \(String(format: "%.1f", interval))ms, hasValue: \(pitch != nil))"
            )
        }

        guard let pitch = pitch else {
            // Debounce: 最後の有効なピッチ検出から一定時間は保持
            // 再生中はAVAudioSessionの制約でピッチ検出成功率が低いため、
            // 録音中に検出したピッチを一定時間保持してUI表示を維持する
            if let lastValid = lastValidPitchTime {
                let timeSinceLastValid = Date().timeIntervalSince(lastValid)
                if timeSinceLastValid < pitchRetentionDuration {
                    return  // ピッチを保持、クリアしない
                }
            }
            detectedPitch = nil
            pitchAccuracy = .none
            return
        }

        // 有効なピッチ検出時に時間を記録
        lastValidPitchTime = Date()

        // Validate frequency is reasonable (avoid NaN/infinite in log calculation)
        guard pitch.frequency > 0 && pitch.frequency < 10000 else {
            detectedPitch = nil
            pitchAccuracy = .none
            return
        }

        // Log at startup and every 500ms during monitoring
        let now = Date()
        let timeSinceLastLog = now.timeIntervalSince(lastLogTime)
        let shouldLog = pitchUpdateCount == 1 || timeSinceLastLog >= 0.5

        // 🔍 Debug why 500ms condition is not met frequently
        if pitchUpdateCount <= 20 || (pitchUpdateCount % 50 == 0) {
            FileLogger.shared.log(
                level: "DEBUG",
                category: "pitch_update",
                message: "⏱️ Time check: lastLog=\(String(format: "%.3f", timeSinceLastLog))s, shouldLog=\(shouldLog), count=\(pitchUpdateCount)"
            )
        }

        // If we have a target pitch, calculate cents difference
        if let target = targetPitch {
            // Calculate MIDI note numbers from frequency
            // A4 (440 Hz) = MIDI note 69
            let targetMIDI = Int(round(12 * log2(target.frequency / 440.0) + 69))
            let detectedMIDI = Int(round(12 * log2(pitch.frequency / 440.0) + 69))

            // Calculate cents difference for detailed logging
            let centsError = 1200.0 * log2(pitch.frequency / target.frequency)

            if shouldLog {
                FileLogger.shared.log(
                    level: "INFO",
                    category: "pitch_tracking",
                    message: String(format: "🎯 UPDATE #%d | Target: %.1f Hz (MIDI %d) → Detected: %.1f Hz (MIDI %d) | Error: %+.1f cents | Confidence: %.2f | Interval: %.0fms",
                        pitchUpdateCount,
                        target.frequency,
                        targetMIDI,
                        pitch.frequency,
                        detectedMIDI,
                        centsError,
                        pitch.confidence,
                        now.timeIntervalSince(lastLogTime) * 1000
                    )
                )
                lastLogTime = now
            }
            let targetFreq = target.frequency
            let detectedFreq = pitch.frequency

            guard targetFreq > 0 else {
                detectedPitch = pitch
                pitchAccuracy = .none
                return
            }

            let cents = Int(round(1200 * log2(detectedFreq / targetFreq)))

            detectedPitch = DetectedPitch(
                noteName: pitch.noteName,
                frequency: pitch.frequency,
                confidence: pitch.confidence,
                cents: cents
            )

            pitchAccuracy = PitchAccuracy.from(cents: cents)
        } else {
            detectedPitch = pitch
            pitchAccuracy = PitchAccuracy.from(cents: pitch.cents)
        }
    }

    deinit {
        progressMonitorTask?.cancel()
        pitchDetectionTask?.cancel()
    }
}
