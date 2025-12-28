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

/// Backing track source type
public enum BackingTrackSource: String, CaseIterable {
    case original = "元音源"
    case vocal = "ボーカル"
    case instrumental = "伴奏"
}

/// Backing track info for selection
public struct BackingTrackInfo: Identifiable, Equatable {
    public let id: RecordingId
    public let recording: Recording
    public let availableSources: [BackingTrackSource]
    public let extractedAudios: [ExtractedAudio]

    public var displayTitle: String {
        recording.title ?? recording.formattedDate
    }

    public func fileURL(for source: BackingTrackSource) -> URL? {
        switch source {
        case .original:
            return recording.fileURL
        case .vocal:
            return extractedAudios.first { $0.type == .vocal }?.fileURL
        case .instrumental:
            return extractedAudios.first { $0.type == .instrumental }?.fileURL
        }
    }
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

    // MARK: - Backing Track Properties

    @Published public private(set) var availableBackingTracks: [BackingTrackInfo] = []
    @Published public var selectedBackingTrack: BackingTrackInfo?
    @Published public var selectedBackingSource: BackingTrackSource?
    @Published public private(set) var isBackingPlaying: Bool = false
    @Published public private(set) var backingCurrentTime: TimeInterval = 0
    @Published public private(set) var backingDuration: TimeInterval = 0
    private var backingTrackPlayer: AudioPlayerProtocol?
    private var recordingRepository: RecordingRepositoryProtocol?
    private var extractedAudioRepository: ExtractedAudioRepositoryProtocol?
    private var backingHasStarted: Bool = false

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

        // Reload backing tracks to include the new recording
        await loadBackingTracks()
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

    // MARK: - Backing Track Methods

    /// Set repositories for backing track functionality
    public func setBackingTrackRepositories(
        recordingRepository: RecordingRepositoryProtocol,
        extractedAudioRepository: ExtractedAudioRepositoryProtocol,
        backingTrackPlayer: AudioPlayerProtocol
    ) {
        self.recordingRepository = recordingRepository
        self.extractedAudioRepository = extractedAudioRepository
        self.backingTrackPlayer = backingTrackPlayer
    }

    /// Load available backing tracks (all recordings)
    public func loadBackingTracks() async {
        guard let recordingRepo = recordingRepository,
              let extractedRepo = extractedAudioRepository else {
            Logger.viewModel.warning("Backing track repositories not set")
            return
        }

        do {
            let recordings = try await recordingRepo.findAll()
            let allExtracted = try await extractedRepo.findAll()

            Logger.viewModel.info("[BackingTrack] Total recordings from repository: \(recordings.count)")
            Logger.viewModel.info("[BackingTrack] Total extracted audios: \(allExtracted.count)")

            // Group extracted audio by recording ID
            var extractedByRecording: [RecordingId: [ExtractedAudio]] = [:]
            for extracted in allExtracted {
                extractedByRecording[extracted.sourceRecordingId, default: []].append(extracted)
            }

            // Build backing track info list (all recordings)
            var tracks: [BackingTrackInfo] = []
            for recording in recordings {
                let extractedAudios = extractedByRecording[recording.id] ?? []

                // Always include original source
                var sources: [BackingTrackSource] = [.original]

                // Add vocal/instrumental if extracted
                if extractedAudios.contains(where: { $0.type == .vocal }) {
                    sources.append(.vocal)
                }
                if extractedAudios.contains(where: { $0.type == .instrumental }) {
                    sources.append(.instrumental)
                }

                Logger.viewModel.debug("[BackingTrack] Adding: \(recording.title ?? recording.formattedDate), sources: \(sources.map { $0.rawValue })")

                tracks.append(BackingTrackInfo(
                    id: recording.id,
                    recording: recording,
                    availableSources: sources,
                    extractedAudios: extractedAudios
                ))
            }

            availableBackingTracks = tracks
            Logger.viewModel.info("[BackingTrack] Final tracks count: \(tracks.count)")

        } catch {
            Logger.viewModel.error("Failed to load backing tracks: \(error.localizedDescription)")
        }
    }

    /// Clear backing track selection
    public func clearBackingTrack() {
        selectedBackingTrack = nil
        selectedBackingSource = nil
    }

    /// Select a backing track and reset playback state
    public func selectBackingTrack(_ track: BackingTrackInfo?) async {
        // Stop current playback if playing
        if isBackingPlaying {
            await stopBacking()
        }

        // Reset playback state for new track
        backingHasStarted = false
        backingCurrentTime = 0
        backingDuration = 0

        // Set new track
        selectedBackingTrack = track
        selectedBackingSource = track?.availableSources.first

        // Pre-load audio file for instant playback
        if let track = track,
           let source = track.availableSources.first,
           let url = track.fileURL(for: source),
           let player = backingTrackPlayer {
            do {
                try player.prepare(url: url)
            } catch {
                Logger.viewModel.error("Failed to prepare backing track: \(error.localizedDescription)")
            }
        }

        Logger.viewModel.info("Selected backing track: \(track?.displayTitle ?? "none")")
    }

    /// Select a backing source and reset playback state
    public func selectBackingSource(_ source: BackingTrackSource?) async {
        // Stop current playback if playing
        if isBackingPlaying {
            await stopBacking()
        }

        // Reset playback state for new source
        backingHasStarted = false
        backingCurrentTime = 0

        // Set new source
        selectedBackingSource = source

        // Pre-load audio file for instant playback
        if let track = selectedBackingTrack,
           let source = source,
           let url = track.fileURL(for: source),
           let player = backingTrackPlayer {
            do {
                try player.prepare(url: url)
            } catch {
                Logger.viewModel.error("Failed to prepare backing track: \(error.localizedDescription)")
            }
        }

        Logger.viewModel.info("Selected backing source: \(source?.rawValue ?? "none")")
    }

    /// Start playing the selected backing track
    public func startBackingTrackPlayback() async {
        // Guard against multiple simultaneous playback attempts
        guard !isBackingPlaying else {
            Logger.viewModel.info("Backing track already playing, skipping startBackingTrackPlayback")
            return
        }

        guard let track = selectedBackingTrack,
              let source = selectedBackingSource,
              let url = track.fileURL(for: source),
              let player = backingTrackPlayer else {
            return
        }

        // Set playing state immediately to prevent race conditions
        isBackingPlaying = true
        backingHasStarted = true
        Logger.viewModel.info("Starting backing track playback: \(url.lastPathComponent)")

        // Fire and forget - don't await completion
        Task {
            do {
                try await player.play(url: url)
                isBackingPlaying = false
                backingHasStarted = false
                backingCurrentTime = 0
                Logger.viewModel.info("Backing track playback finished")
            } catch {
                isBackingPlaying = false
                backingHasStarted = false
                Logger.viewModel.error("Backing track playback error: \(error.localizedDescription)")
            }
        }
    }

    /// Stop playing the backing track
    public func stopBackingTrackPlayback() async {
        guard let player = backingTrackPlayer else { return }
        await player.stop()
        Logger.viewModel.info("Backing track playback stopped")
    }

    // MARK: - Backing Track Player Control Methods

    /// Toggle backing track playback (play/pause)
    public func toggleBackingPlayback() async {
        guard let track = selectedBackingTrack,
              let source = selectedBackingSource,
              let url = track.fileURL(for: source),
              let player = backingTrackPlayer else {
            return
        }

        if isBackingPlaying {
            // Pause
            player.pause()
            isBackingPlaying = false
            Logger.viewModel.info("Backing track paused")
        } else if backingHasStarted && backingCurrentTime > 0 {
            // Resume from paused position
            player.resume()
            isBackingPlaying = true
            Logger.viewModel.info("Backing track resumed")
        } else {
            // Start fresh playback
            backingHasStarted = true
            isBackingPlaying = true
            Logger.viewModel.info("Starting backing track playback: \(url.lastPathComponent)")

            Task {
                do {
                    try await player.play(url: url)
                    // Playback finished
                    isBackingPlaying = false
                    backingHasStarted = false
                    backingCurrentTime = 0
                    Logger.viewModel.info("Backing track playback finished")
                } catch {
                    isBackingPlaying = false
                    backingHasStarted = false
                    Logger.viewModel.error("Backing track playback error: \(error.localizedDescription)")
                }
            }
        }
    }

    /// Seek to a specific time in the backing track
    public func seekBacking(to time: TimeInterval) {
        backingTrackPlayer?.seek(to: time)
        backingCurrentTime = time
    }

    /// Stop backing track and reset
    public func stopBacking() async {
        guard let player = backingTrackPlayer else { return }
        await player.stop()
        isBackingPlaying = false
        backingHasStarted = false
        backingCurrentTime = 0
        Logger.viewModel.info("Backing track stopped and reset")
    }

    /// Update backing player state from the player
    public func updateBackingPlayerState() {
        guard let player = backingTrackPlayer else { return }
        backingCurrentTime = player.currentTime
        backingDuration = player.duration
        isBackingPlaying = player.isPlaying
    }

    /// Clear backing track selection and stop playback
    public func clearBackingTrackWithStop() async {
        await stopBacking()
        selectedBackingTrack = nil
        selectedBackingSource = nil
    }
}
