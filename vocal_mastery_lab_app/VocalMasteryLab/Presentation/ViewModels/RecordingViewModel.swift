import Foundation
import SubscriptionDomain
import VocalisDomain
import Combine
import OSLog

/// Recording state for the main recording screen
public enum RecordingState: Equatable {
    case idle
    case preparing    // Immediate feedback after start button tap
    case countdown
    case recording
}

/// Coordinator ViewModel for the main recording screen
/// Delegates responsibilities to RecordingStateViewModel and PitchDetectionViewModel
@MainActor
public class RecordingViewModel: ObservableObject {
    // MARK: - Child ViewModels

    public let recordingStateVM: RecordingStateViewModel
    public let pitchDetectionVM: PitchDetectionViewModel
    public let subscriptionViewModel: SubscriptionViewModel

    // MARK: - Dependencies

    private let pitchDetector: any PitchDetectorProtocol & ObservableObject
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Forwarded Properties (for backward compatibility)

    @Published public var recordingState: RecordingState = .idle
    @Published public var currentSession: RecordingSession?
    @Published public var errorMessage: String?
    @Published public var progress: Double = 0.0
    @Published public var countdownValue: Int = 0 // Forwarded from RecordingStateViewModel
    @Published public var lastRecordingURL: URL?
    @Published public var lastRecordingId: RecordingId?
    @Published public var lastRecordingDate: Date?
    @Published public var lastRecordingDuration: TimeInterval?
    @Published public var isPlayingRecording: Bool = false

    @Published public var currentTier: SubscriptionTier = .free
    @Published public var dailyRecordingCount: Int = 0
    @Published public var recordingLimit: RecordingLimit = RecordingLimit(dailyCount: 5, maxDuration: 30)

    @Published public var detectedPitch: DetectedPitch?
    @Published public var pitchAccuracy: PitchAccuracy = .none
    @Published public var spectrum: [Float]?
    @Published public var audioLevel: Float = -160.0  // dB value (-160 to 0)

    // MARK: - Initialization

    public init(
        startRecordingUseCase: StartRecordingUseCaseProtocol,
        stopRecordingUseCase: StopRecordingUseCaseProtocol,
        audioPlayer: AudioPlayerProtocol,
        pitchDetector: any PitchDetectorProtocol & ObservableObject,
        subscriptionViewModel: SubscriptionViewModel,
        usageTracker: RecordingUsageTracker = RecordingUsageTracker(),
        countdownDuration: Int = 3,
        recordingLimitConfig: RecordingLimit.Configuration = .production
    ) {
        self.pitchDetector = pitchDetector
        self.subscriptionViewModel = subscriptionViewModel

        // Initialize child ViewModels
        self.recordingStateVM = RecordingStateViewModel(
            startRecordingUseCase: startRecordingUseCase,
            stopRecordingUseCase: stopRecordingUseCase,
            audioPlayer: audioPlayer,
            subscriptionViewModel: subscriptionViewModel,
            usageTracker: usageTracker,
            countdownDuration: countdownDuration,
            recordingLimitConfig: recordingLimitConfig
        )

        self.pitchDetectionVM = PitchDetectionViewModel(
            pitchDetector: pitchDetector,
            audioPlayer: audioPlayer
        )

        setupBindings()
        setupCallbacks()

        Logger.viewModel.info("RecordingViewModel initialized with child ViewModels")
    }

    // MARK: - Setup

    private func setupBindings() {
        // Forward RecordingStateVM properties
        recordingStateVM.$recordingState
            .assign(to: &$recordingState)

        recordingStateVM.$currentSession
            .assign(to: &$currentSession)

        recordingStateVM.$errorMessage
            .assign(to: &$errorMessage)

        recordingStateVM.$progress
            .assign(to: &$progress)

        recordingStateVM.$countdownValue
            .assign(to: &$countdownValue)

        recordingStateVM.$lastRecordingURL
            .assign(to: &$lastRecordingURL)

        recordingStateVM.$lastRecordingId
            .assign(to: &$lastRecordingId)

        recordingStateVM.$lastRecordingDate
            .assign(to: &$lastRecordingDate)

        recordingStateVM.$lastRecordingDuration
            .assign(to: &$lastRecordingDuration)

        recordingStateVM.$currentTier
            .assign(to: &$currentTier)

        recordingStateVM.$dailyRecordingCount
            .assign(to: &$dailyRecordingCount)

        recordingStateVM.$recordingLimit
            .assign(to: &$recordingLimit)

        // Forward PitchDetectionVM properties
        pitchDetectionVM.$detectedPitch
            .assign(to: &$detectedPitch)

        pitchDetectionVM.$pitchAccuracy
            .assign(to: &$pitchAccuracy)

        // Subscribe to spectrum updates from pitch detector
        if let realtimePitchDetector = pitchDetector as? RealtimePitchDetector {
            realtimePitchDetector.$spectrum
                .sink { [weak self] spectrum in
                    guard let self = self else { return }
                    Task { @MainActor in
                        self.spectrum = spectrum
                    }
                }
                .store(in: &cancellables)

            // Subscribe to audio level updates from pitch detector
            realtimePitchDetector.$audioLevel
                .sink { [weak self] level in
                    guard let self = self else { return }
                    Task { @MainActor in
                        self.audioLevel = level
                    }
                }
                .store(in: &cancellables)
        }
    }

    private func setupCallbacks() {
        // Set up auto-stop callback for duration limit
        recordingStateVM.onAutoStopNeeded = { [weak self] in
            guard let self = self else { return }
            Logger.viewModel.info("Duration limit reached - calling RecordingViewModel.stopRecording()")
            await self.stopRecording()
        }
    }

    // MARK: - Public Methods (Coordinator)

    /// Set preparing state immediately for instant visual feedback
    public func setPreparingState() {
        guard self.recordingState == .idle else { return }
        self.recordingState = .preparing
        Logger.viewModel.info("RecordingViewModel: State changed to .preparing")
    }

    /// Start the recording process with countdown
    public func startRecording() async {
        Logger.viewModel.info("RecordingViewModel.startRecording() called")

        guard self.recordingState == .idle || self.recordingState == .preparing else {
            Logger.viewModel.warning("Start recording ignored: already in state \(String(describing: self.recordingState))")
            return
        }

        if self.recordingState == .idle {
            self.recordingState = .preparing
        }

        // Start recording through state VM
        await recordingStateVM.startRecording()

        // Wait for recording to actually start
        Logger.viewModel.info("Waiting for recording to start...")

        while recordingStateVM.recordingState != .recording {
            if recordingStateVM.recordingState == .idle {
                Logger.viewModel.warning("Recording failed to start - skipping pitch detection")
                return
            }
            try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 second
        }
        Logger.viewModel.info("Recording started - starting pitch detection")

        do {
            try await pitchDetector.startRealtimeDetection()
            Logger.viewModel.info("Realtime pitch detection started")
        } catch {
            Logger.viewModel.error("Error starting pitch detection: \(error.localizedDescription)")
            errorMessage = error.localizedDescription
        }

        Logger.viewModel.info("RecordingViewModel.startRecording() completed")
    }

    /// Cancel the countdown before recording starts
    public func cancelCountdown() async {
        await recordingStateVM.cancelCountdown()
    }

    /// Stop the current recording
    public func stopRecording() async {
        // Stop pitch detection
        pitchDetector.stopRealtimeDetection()

        // Stop recording
        await recordingStateVM.stopRecording()

        // Reset pitch detection state
        pitchDetectionVM.reset()
    }

    /// Play the last recording
    public func playLastRecording() async {
        Logger.viewModel.debug("playLastRecording() called")

        guard let url = lastRecordingURL else {
            Logger.viewModel.debug("No recording URL - cannot play")
            await recordingStateVM.playLastRecording()
            return
        }

        Logger.viewModel.debug("Starting playback")

        do {
            isPlayingRecording = true
            Logger.viewModel.info("Starting audio playback")

            try await recordingStateVM.audioPlayer.play(url: url)
            Logger.viewModel.info("Audio playback completed")

            isPlayingRecording = false
            recordingStateVM.isPlayingRecording = false
            Logger.viewModel.info("isPlayingRecording = false (normal completion)")

        } catch {
            Logger.viewModel.error("Error during playback: \(error.localizedDescription)")
            Logger.viewModel.logError(error)
            errorMessage = error.localizedDescription

            isPlayingRecording = false
            recordingStateVM.isPlayingRecording = false
        }
    }

    /// Stop playing the recording
    public func stopPlayback() async {
        Logger.viewModel.debug("stopPlayback() called")

        await recordingStateVM.audioPlayer.stop()

        // Reset pitch detection state
        pitchDetectionVM.reset()

        isPlayingRecording = false
        recordingStateVM.isPlayingRecording = false
        Logger.viewModel.info("isPlayingRecording = false (manual stop)")

        Logger.viewModel.debug("stopPlayback() completed")
    }

    /// Reload audio detection settings from repository and update pitch detector
    public func reloadAudioSettings(from repository: AudioSettingsRepositoryProtocol) {
        let settings = repository.get()
        if let pitchDetector = pitchDetector as? RealtimePitchDetector {
            pitchDetector.updateSettings(settings)
            Logger.viewModel.info("Audio settings reloaded: RMS=\(settings.rmsSilenceThreshold), Confidence=\(settings.confidenceThreshold)")
        }
    }
}
