import SwiftUI
import SubscriptionDomain
import VocalisDomain

/// Simple recording screen view based on UI_DESIGN.md specification
public struct RecordingView: View {
    @StateObject private var viewModel: RecordingViewModel
    @StateObject private var localization = LocalizationManager.shared
    @State private var showingAlert: Bool = false
    @State private var elapsedTime: TimeInterval = 0
    @State private var timer: Timer?
    @State private var extractingRecording: Recording?

    private let vocalExtractor: VocalExtractorProtocol
    private let extractedAudioRepository: ExtractedAudioRepositoryProtocol
    private let audioPlayer: AudioPlayerProtocol

    public init(
        viewModel: RecordingViewModel,
        vocalExtractor: VocalExtractorProtocol,
        extractedAudioRepository: ExtractedAudioRepositoryProtocol,
        audioPlayer: AudioPlayerProtocol
    ) {
        _viewModel = StateObject(wrappedValue: viewModel)
        self.vocalExtractor = vocalExtractor
        self.extractedAudioRepository = extractedAudioRepository
        self.audioPlayer = audioPlayer
    }

    public var body: some View {
        VStack(spacing: 24) {
            Spacer()

            // Timer display
            timerSection

            // Record controls (uses existing RecordingControls component)
            RecordingControls(
                recordingState: viewModel.recordingState,
                hasLastRecording: viewModel.lastRecordingURL != nil,
                isPlayingRecording: viewModel.isPlayingRecording,
                canStartRecording: true,
                onStart: startRecording,
                onStop: stopRecording,
                onCancel: cancelCountdown,
                onPlayLast: togglePlayback,
                onAnalyze: nil  // No analyze button in simple mode
            )

            // Background hint
            backgroundHintSection

            Spacer()

            // Last recording info section (only shown when recording exists)
            if viewModel.lastRecordingURL != nil {
                lastRecordingInfoSection
            }

            Spacer()
        }
        .padding()
        .navigationTitle("recording.title".localized)
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(viewModel.recordingState == .recording || viewModel.recordingState == .countdown)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                NavigationLink(destination: RecordingListView(
                    viewModel: RecordingListViewModel(
                        recordingRepository: DependencyContainer.shared.recordingRepository,
                        extractedAudioRepository: DependencyContainer.shared.extractedAudioRepository,
                        audioPlayer: DependencyContainer.shared.audioPlayer
                    ),
                    audioPlayer: DependencyContainer.shared.audioPlayer,
                    analyzeRecordingUseCase: DependencyContainer.shared.analyzeRecordingUseCase,
                    extractedAudioRepository: DependencyContainer.shared.extractedAudioRepository,
                    vocalExtractor: DependencyContainer.shared.vocalExtractor
                )) {
                    HStack(spacing: 4) {
                        Image(systemName: "list.bullet")
                        Text("recording.list_button".localized)
                    }
                }
                .accessibilityIdentifier("RecordingListButton")
                .disabled(viewModel.recordingState == .recording || viewModel.recordingState == .countdown)
            }
        }
        .alert(
            viewModel.recordingStateVM.alertMessageType == .limitReached ? "alert.notice".localized : "error".localized,
            isPresented: $showingAlert
        ) {
            Button("ok".localized) {
                viewModel.recordingStateVM.clearError()
            }
            .accessibilityIdentifier(
                viewModel.recordingStateVM.alertMessageType == .limitReached
                    ? "RecordingLimitAlertOKButton"
                    : "ErrorAlertOKButton"
            )
        } message: {
            Text(viewModel.errorMessage ?? "")
        }
        .onChange(of: viewModel.errorMessage) { errorMessage in
            showingAlert = errorMessage != nil
        }
        .onChange(of: viewModel.recordingState) { newState in
            if newState == .recording {
                startTimer()
            } else {
                stopTimer()
                if newState == .idle {
                    elapsedTime = 0
                }
            }
        }
        .onAppear {
            // Reload audio settings when returning to recording screen
            viewModel.reloadAudioSettings(from: DependencyContainer.shared.audioSettingsRepository)
        }
        .onDisappear {
            stopTimer()
            if viewModel.isPlayingRecording {
                Task {
                    await viewModel.stopPlayback()
                }
            }
            viewModel.recordingStateVM.cleanup()
        }
        .navigationDestination(item: $extractingRecording) { recording in
            VocalExtractionView(
                viewModel: VocalExtractionViewModel(
                    recording: recording,
                    extractor: vocalExtractor,
                    extractedAudioRepository: extractedAudioRepository,
                    audioPlayer: audioPlayer
                )
            )
        }
    }

    // MARK: - Timer Section

    private var timerSection: some View {
        VStack(spacing: 8) {
            HStack {
                if viewModel.recordingState == .recording {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 12, height: 12)
                    Text("recording.recording_label".localized)
                        .font(.subheadline)
                        .foregroundColor(.red)
                }
            }
            .frame(height: 20)

            Text(formatTime(elapsedTime))
                .font(.system(size: 48, weight: .light, design: .monospaced))
                .accessibilityIdentifier("RecordingTimerLabel")
        }
    }

    // MARK: - Background Hint Section

    private var backgroundHintSection: some View {
        HStack {
            Image(systemName: "lightbulb.fill")
                .foregroundColor(.yellow)
            Text("recording.background_hint".localized)
                .font(.caption)
                .foregroundColor(ColorPalette.text.opacity(0.6))
        }
        .accessibilityIdentifier("BackgroundRecordingHint")
    }

    // MARK: - Last Recording Info Section

    private var lastRecordingInfoSection: some View {
        VStack(spacing: 12) {
            Divider()

            // Recording info
            VStack(spacing: 4) {
                if let date = viewModel.lastRecordingDate {
                    Text(formatDate(date))
                        .font(.subheadline)
                        .foregroundColor(ColorPalette.text.opacity(0.8))
                        .accessibilityIdentifier("LastRecordingDateLabel")
                }

                if let duration = viewModel.lastRecordingDuration {
                    Text(formatDuration(duration))
                        .font(.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.6))
                        .accessibilityIdentifier("LastRecordingDurationLabel")
                }
            }

            // Vocal extraction button
            Button(action: vocalExtraction) {
                HStack {
                    Image(systemName: "waveform")
                    Text("recording.vocal_extraction_button".localized)
                }
            }
            .buttonStyle(SecondaryButtonStyle())
            .accessibilityIdentifier("VocalExtractionButton")
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("LastRecordingSection")
    }

    // MARK: - Timer Management

    private func startTimer() {
        elapsedTime = 0
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            elapsedTime += 0.1
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    // MARK: - Formatting

    private func formatTime(_ time: TimeInterval) -> String {
        let hours = Int(time) / 3600
        let minutes = (Int(time) % 3600) / 60
        let seconds = Int(time) % 60
        return String(format: "%02d:%02d:%02d", hours, minutes, seconds)
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy/MM/dd HH:mm"
        return formatter.string(from: date)
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        if minutes > 0 {
            return "\(minutes)分\(seconds)秒"
        } else {
            return "\(seconds)秒"
        }
    }

    // MARK: - Actions

    private func startRecording() {
        viewModel.setPreparingState()
        Task { @MainActor in
            await viewModel.startRecording()
        }
    }

    private func stopRecording() {
        Task { @MainActor in
            await viewModel.stopRecording()
        }
    }

    private func cancelCountdown() {
        Task { @MainActor in
            await viewModel.cancelCountdown()
        }
    }

    private func togglePlayback() {
        if viewModel.isPlayingRecording {
            Task {
                await viewModel.stopPlayback()
            }
        } else {
            viewModel.isPlayingRecording = true
            Task {
                await viewModel.playLastRecording()
            }
        }
    }

    private func vocalExtraction() {
        guard let url = viewModel.lastRecordingURL,
              let id = viewModel.lastRecordingId,
              let duration = viewModel.lastRecordingDuration else {
            return
        }

        let recording = Recording(
            id: id,
            fileURL: url,
            createdAt: viewModel.lastRecordingDate ?? Date(),
            duration: Duration(seconds: duration)
        )

        extractingRecording = recording
    }
}

// MARK: - Preview

#if DEBUG
struct RecordingView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationStack {
            RecordingView(
                viewModel: RecordingViewModel(
                    startRecordingUseCase: PreviewMockStartRecordingUseCase(),
                    stopRecordingUseCase: PreviewMockStopRecordingUseCase(),
                    audioPlayer: PreviewMockAudioPlayer(),
                    pitchDetector: RealtimePitchDetector(),
                    subscriptionViewModel: SubscriptionViewModel(
                        getStatusUseCase: PreviewMockGetStatusUseCase(),
                        purchaseUseCase: PreviewMockPurchaseUseCase(),
                        restoreUseCase: PreviewMockRestoreUseCase()
                    )
                ),
                vocalExtractor: PreviewMockVocalExtractor(),
                extractedAudioRepository: PreviewMockExtractedAudioRepository(),
                audioPlayer: PreviewMockAudioPlayer()
            )
        }
    }
}

// MARK: - Preview Mocks

private class PreviewMockStartRecordingUseCase: StartRecordingUseCaseProtocol {
    func execute(user: User) async throws -> RecordingSession {
        try await Task.sleep(nanoseconds: 1_000_000_000)
        return RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/preview.m4a"),
            startedAt: Date()
        )
    }
}

private class PreviewMockStopRecordingUseCase: StopRecordingUseCaseProtocol {
    func setRecordingContext(url: URL) {}

    func execute() async throws -> StopRecordingResult {
        try await Task.sleep(nanoseconds: 500_000_000)
        return StopRecordingResult(duration: 5.0)
    }
}

private class PreviewMockAudioPlayer: AudioPlayerProtocol {
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 10.0

    func play(url: URL, withPitchDetection: Bool) async throws {
        isPlaying = true
    }

    func stop() async {
        isPlaying = false
    }

    func pause() {
        isPlaying = false
    }

    func resume() {
        isPlaying = true
    }

    func seek(to time: TimeInterval) {
        currentTime = time
    }
}

private class PreviewMockGetStatusUseCase: GetSubscriptionStatusUseCaseProtocol {
    func execute() async throws -> SubscriptionStatus {
        return SubscriptionStatus(tier: .free, cohort: .v2_0)
    }
}

private class PreviewMockPurchaseUseCase: PurchaseSubscriptionUseCaseProtocol {
    func execute(tier: SubscriptionTier) async throws {}
}

private class PreviewMockRestoreUseCase: RestorePurchasesUseCaseProtocol {
    func execute() async throws {}
}

private class PreviewMockVocalExtractor: VocalExtractorProtocol {
    func extract(from sourceURL: URL, progressHandler: @escaping (Double, String) -> Void) async throws -> VocalExtractionResult {
        VocalExtractionResult(vocalFileURL: sourceURL, duration: Duration(seconds: 10))
    }
}

private class PreviewMockExtractedAudioRepository: ExtractedAudioRepositoryProtocol {
    func save(_ audio: ExtractedAudio) async throws {}
    func findById(_ id: ExtractedAudioId) async throws -> ExtractedAudio? { nil }
    func findByRecording(_ recordingId: RecordingId) async throws -> [ExtractedAudio] { [] }
    func findAll() async throws -> [ExtractedAudio] { [] }
    func delete(_ id: ExtractedAudioId) async throws {}
    func deleteByRecording(_ recordingId: RecordingId) async throws {}
}
#endif
