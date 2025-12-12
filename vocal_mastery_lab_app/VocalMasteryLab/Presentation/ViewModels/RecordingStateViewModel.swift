import Foundation
import Combine
import OSLog
import VocalisDomain
import SubscriptionDomain

/// Type of alert message to display
public enum AlertMessageType {
    case error
    case limitReached
}

/// ViewModel for recording state management
/// Manages core recording functionality including countdown, start, stop, and duration monitoring
@MainActor
public class RecordingStateViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published public private(set) var recordingState: RecordingState = .idle
    @Published public private(set) var currentSession: RecordingSession?
    @Published public private(set) var errorMessage: String?
    @Published public private(set) var alertMessageType: AlertMessageType = .error
    @Published public private(set) var progress: Double = 0.0
    @Published public private(set) var countdownValue: Int = 3
    @Published public private(set) var lastRecordingURL: URL?
    @Published public private(set) var lastRecordingId: RecordingId?
    @Published internal var isPlayingRecording: Bool = false
    @Published public private(set) var isCountdownComplete: Bool = false

    // MARK: - Subscription Properties

    @Published public private(set) var currentTier: SubscriptionTier = .free
    @Published public private(set) var dailyRecordingCount: Int = 0
    @Published public private(set) var recordingLimit: RecordingLimit = RecordingLimit(dailyCount: 2, maxDuration: 30)

    // MARK: - Callbacks

    /// Called when automatic stop is needed (e.g., duration limit reached)
    /// This allows parent coordinator to perform proper cleanup (stop pitch detection, etc.) before stopping recording
    public var onAutoStopNeeded: (() async -> Void)?

    // MARK: - Dependencies

    private let startRecordingUseCase: StartRecordingUseCaseProtocol
    private let stopRecordingUseCase: StopRecordingUseCaseProtocol
    internal let audioPlayer: AudioPlayerProtocol
    private let subscriptionViewModel: SubscriptionViewModel
    private let usageTracker: RecordingUsageTracker

    // MARK: - Private Properties

    private var countdownTask: Task<Void, Never>?
    private var durationMonitorTask: Task<Void, Never>?
    private var recordingStartTime: Date?
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Constants

    private static let durationMonitoringIntervalNanoseconds: UInt64 = 500_000_000 // 500ms

    // MARK: - Configuration

    private let countdownDuration: Int
    private let recordingLimitConfig: RecordingLimit.Configuration

    // MARK: - Initialization

    public init(
        startRecordingUseCase: StartRecordingUseCaseProtocol,
        stopRecordingUseCase: StopRecordingUseCaseProtocol,
        audioPlayer: AudioPlayerProtocol,
        subscriptionViewModel: SubscriptionViewModel,
        usageTracker: RecordingUsageTracker = RecordingUsageTracker(),
        countdownDuration: Int = 3,
        recordingLimitConfig: RecordingLimit.Configuration = .production
    ) {
        self.startRecordingUseCase = startRecordingUseCase
        self.stopRecordingUseCase = stopRecordingUseCase
        self.audioPlayer = audioPlayer
        self.subscriptionViewModel = subscriptionViewModel
        self.usageTracker = usageTracker
        self.countdownDuration = countdownDuration
        self.recordingLimitConfig = recordingLimitConfig
        self.countdownValue = countdownDuration

        // Subscribe to subscription status updates
        subscriptionViewModel.$currentStatus
            .sink { [weak self] status in
                guard let self = self else { return }
                Task { @MainActor in
                    Logger.viewModel.error("🔴 RECORDING_LIMIT_MARK: currentStatus updated, tier=\(status?.tier.rawValue ?? "nil")")
                    FileLogger.shared.log(level: "ERROR", category: "recording_limit", message: "🔴 currentStatus updated, tier=\(status?.tier.rawValue ?? "nil")")
                    if let status = status {
                        self.currentTier = status.tier
                        self.recordingLimit = RecordingLimit.forTier(status.tier, configuration: self.recordingLimitConfig)
                        Logger.viewModel.error("🔴 RECORDING_LIMIT_MARK: recordingLimit updated, dailyCount=\(self.recordingLimit.dailyCount?.description ?? "nil"), maxDuration=\(self.recordingLimit.maxDuration?.description ?? "nil")")
                        FileLogger.shared.log(level: "ERROR", category: "recording_limit", message: "🔴 recordingLimit updated, dailyCount=\(self.recordingLimit.dailyCount?.description ?? "nil"), maxDuration=\(self.recordingLimit.maxDuration?.description ?? "nil")")
                    }
                }
            }
            .store(in: &cancellables)

        // Initialize usage count
        dailyRecordingCount = usageTracker.getTodayCount()

        Logger.viewModel.error("🔴 RECORDING_LIMIT_MARK: RecordingStateViewModel initialized, defaultLimit dailyCount=\(self.recordingLimit.dailyCount?.description ?? "nil"), maxDuration=\(self.recordingLimit.maxDuration?.description ?? "nil")")
        FileLogger.shared.log(level: "ERROR", category: "recording_limit", message: "🔴 RecordingStateViewModel initialized, defaultLimit dailyCount=\(self.recordingLimit.dailyCount?.description ?? "nil"), maxDuration=\(self.recordingLimit.maxDuration?.description ?? "nil")")
        Logger.viewModel.info("RecordingStateViewModel initialized")
    }

    // MARK: - Public Methods

    /// Clear error message
    public func clearError() {
        errorMessage = nil
        alertMessageType = .error
    }

    /// Cleanup when leaving recording screen
    /// Deactivates audio session if not actively recording
    public func cleanup() {
        // Only cleanup if not actively recording
        guard recordingState == .idle else {
            Logger.viewModel.info("Cleanup skipped: recording in progress")
            return
        }

        // Reset audio session mode cache
        AudioSessionManager.shared.resetSessionMode()

        // Force deactivate audio session to release microphone
        try? AudioSessionManager.shared.forceDeactivate()
        Logger.viewModel.info("Audio session cleaned up on screen exit")
    }

    /// Start the recording process with countdown
    public func startRecording() async {
        Logger.viewModel.error("🔴 RECORDING_LIMIT_MARK: startRecording START, state=\(String(describing: self.recordingState))")

        // Don't start if already recording or in countdown
        guard recordingState == .idle else {
            Logger.viewModel.error("🔴 RECORDING_LIMIT_MARK: startRecording REJECTED - already in state \(String(describing: self.recordingState))")
            Logger.viewModel.warning("Start recording ignored: already in state \(String(describing: self.recordingState))")
            return
        }

        // IMMEDIATE visual feedback - set preparing state before any async work
        recordingState = .preparing
        Logger.viewModel.info("✅ State changed to .preparing - immediate visual feedback")

        // Pre-configure audio session to reduce latency during recording start
        try? AudioSessionManager.shared.configureForRecordingAndPlayback()
        try? AudioSessionManager.shared.activate()

        // Check recording count limit
        self.dailyRecordingCount = usageTracker.getTodayCount()
        Logger.viewModel.error("🔴 RECORDING_LIMIT_MARK: Recording count check: current=\(self.dailyRecordingCount), limit=\(self.recordingLimit.dailyCount?.description ?? "nil")")
        FileLogger.shared.log(level: "ERROR", category: "recording_limit", message: "🔴 Recording count check: current=\(self.dailyRecordingCount), limit=\(self.recordingLimit.dailyCount?.description ?? "nil")")
        if !recordingLimit.isCountWithinLimit(self.dailyRecordingCount) {
            Logger.viewModel.error("🔴 RECORDING_LIMIT_MARK: Recording REJECTED - count limit reached")
            FileLogger.shared.log(level: "ERROR", category: "recording_limit", message: "🔴 Recording REJECTED - count limit reached")
            Logger.viewModel.warning("Recording limit reached: \(self.dailyRecordingCount)")
            let errorMsg = String(format: "recording.limit_reached".localized, currentTier.displayName)
            alertMessageType = .limitReached
            errorMessage = errorMsg
            FileLogger.shared.log(level: "ERROR", category: "recording", message: "❌ Recording rejected - User error message: \(errorMsg)")
            recordingState = .idle  // Reset to idle on limit failure
            // Deactivate audio session since recording was rejected
            try? AudioSessionManager.shared.forceDeactivate()
            return
        }

        Logger.viewModel.info("Starting recording")
        Logger.viewModel.error("🔴 RECORDING_LIMIT_MARK: startRecording PASSED checks")

        // Clear any previous error
        errorMessage = nil

        // If countdown is 0, skip countdown and start recording immediately
        if countdownDuration == 0 {
            isCountdownComplete = true
            await executeRecording()
            return
        }

        // Start countdown
        recordingState = .countdown
        countdownValue = countdownDuration

        // Create countdown task
        countdownTask = Task { [weak self] in
            guard let self = self else { return }

            // Countdown: countdownDuration, ..., 2, 1
            for value in (1...self.countdownDuration).reversed() {
                if Task.isCancelled { return }

                await MainActor.run {
                    self.countdownValue = value
                }

                try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
            }

            if Task.isCancelled { return }

            // Countdown complete, set flag before executing recording
            await MainActor.run { self.isCountdownComplete = true }
            await self.executeRecording()
        }
    }

    /// Cancel the countdown before recording starts
    public func cancelCountdown() async {
        guard recordingState == .countdown else { return }

        countdownTask?.cancel()
        countdownTask = nil
        recordingState = .idle
        countdownValue = countdownDuration
        isCountdownComplete = false
    }

    /// Stop the current recording
    public func stopRecording() async {
        guard recordingState == .recording else { return }

        Logger.viewModel.info("Stopping recording")

        // Stop monitoring tasks
        durationMonitorTask?.cancel()
        durationMonitorTask = nil
        recordingStartTime = nil

        do {
            // Save the recording URL before clearing currentSession
            let recordingURL = currentSession?.recordingURL

            // Stop recording via use case
            let result = try await stopRecordingUseCase.execute()

            let filename = recordingURL?.lastPathComponent ?? "unknown"
            Logger.viewModel.info("Recording stopped successfully: \(filename)")

            // Increment recording count
            usageTracker.incrementCount()
            self.dailyRecordingCount = usageTracker.getTodayCount()
            Logger.viewModel.info("Daily recording count: \(self.dailyRecordingCount)")

            // Update state
            recordingState = .idle
            currentSession = nil
            progress = 0.0
            isCountdownComplete = false

            // Save the recording URL and ID for playback
            lastRecordingURL = recordingURL
            lastRecordingId = result.recordingId

        } catch {
            // Handle error
            Logger.viewModel.logError(error)
            let errorMsg = error.localizedDescription
            errorMessage = errorMsg
            FileLogger.shared.log(level: "ERROR", category: "recording", message: "❌ Recording failed - User error message: \(errorMsg)")
            recordingState = .idle
            currentSession = nil
            progress = 0.0
            isCountdownComplete = false
        }
    }

    /// Play the last recording
    public func playLastRecording() async {
        Logger.viewModel.debug("🔵 playLastRecording() called in RecordingStateViewModel")

        guard let url = lastRecordingURL else {
            Logger.viewModel.warning("Play recording failed: no recording available")
            let errorMsg = "No recording available"
            errorMessage = errorMsg
            FileLogger.shared.log(level: "ERROR", category: "playback", message: "❌ Playback failed - User error message: \(errorMsg)")
            return
        }

        guard !isPlayingRecording else {
            Logger.viewModel.warning("⚠️ playLastRecording() blocked: isPlayingRecording = true")
            return
        }

        Logger.viewModel.info("Starting playback: \(url.lastPathComponent)")

        do {
            isPlayingRecording = true

            // Play the actual recording (blocks until playback completes)
            try await audioPlayer.play(url: url)

            isPlayingRecording = false
            Logger.viewModel.info("Playback completed")

        } catch {
            Logger.viewModel.logError(error)
            errorMessage = error.localizedDescription
            isPlayingRecording = false
        }
    }

    /// Stop playing the recording
    public func stopPlayback() async {
        await audioPlayer.stop()
        isPlayingRecording = false
        Logger.viewModel.info("Playback stopped")
    }

    // MARK: - Private Methods

    /// Create User object from current state
    private func createCurrentUser() -> User {
        let stats = RecordingStats(
            todayCount: dailyRecordingCount,
            lastResetDate: Date(),
            totalCount: 0
        )

        // Use current subscription status, or default to free tier
        let status = subscriptionViewModel.currentStatus ?? .defaultFree(cohort: .v2_0)

        return User(
            id: UserId(),
            subscriptionStatus: status,
            recordingStats: stats
        )
    }

    /// Execute the actual recording after countdown
    private func executeRecording() async {
        do {
            // Create user object from current state
            let user = createCurrentUser()

            // Start recording
            let session = try await startRecordingUseCase.execute(user: user)
            Logger.viewModel.info("Recording started")

            // Set recording context for StopRecordingUseCase
            stopRecordingUseCase.setRecordingContext(url: session.recordingURL)

            // Update state
            recordingState = .recording
            currentSession = session
            progress = 0.0
            recordingStartTime = Date()

            // Start duration monitoring
            startDurationMonitoring()

            Logger.viewModel.info("Recording in progress")

        } catch {
            Logger.viewModel.logError(error)
            errorMessage = error.localizedDescription
            recordingState = .idle
            currentSession = nil
            progress = 0.0
            isCountdownComplete = false  // Reset flag to prevent pitch detection from starting
        }
    }

    /// Monitor recording duration and enforce time limit
    private func startDurationMonitoring() {
        durationMonitorTask = Task { [weak self] in
            guard let self = self else { return }

            let startTime = await MainActor.run { self.recordingStartTime }
            guard let startTime = startTime else { return }

            let maxDuration = await MainActor.run { self.recordingLimit.maxDuration }
            guard let maxDuration = maxDuration else {
                // No duration limit, no monitoring needed
                return
            }

            while !Task.isCancelled {
                let elapsed = Date().timeIntervalSince(startTime)
                let progress = min(elapsed / maxDuration, 1.0)

                await MainActor.run {
                    self.progress = progress
                }

                // Check if time limit reached
                if elapsed >= maxDuration {
                    // Only show error message for free tier (premium stops silently)
                    let tier = await MainActor.run { self.currentTier }
                    if tier == .free {
                        await MainActor.run {
                            let tierName = self.currentTier.displayName
                            self.alertMessageType = .limitReached
                            self.errorMessage = "録音時間の上限に達しました (\(tierName)プラン: \(Int(maxDuration))秒)"
                        }
                    }

                    // Call the auto-stop handler if provided (allows parent coordinator to cleanup pitch detection, etc.)
                    let callback = await MainActor.run { self.onAutoStopNeeded }
                    if let callback = callback {
                        await callback()
                    } else {
                        // Fallback: just stop recording state (but this won't cleanup pitch detection)
                        await self.stopRecording()
                    }
                    break
                }

                try? await Task.sleep(nanoseconds: Self.durationMonitoringIntervalNanoseconds)
            }
        }
    }

    deinit {
        countdownTask?.cancel()
        durationMonitorTask?.cancel()
    }
}
