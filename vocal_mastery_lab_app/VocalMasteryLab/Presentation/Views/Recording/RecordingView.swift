import SwiftUI
import SubscriptionDomain
import VocalisDomain

/// Main recording screen view with settings panel and real-time visualization
public struct RecordingView: View {
    @StateObject private var viewModel: RecordingViewModel
    @StateObject private var settingsViewModel = RecordingSettingsViewModel()
    @StateObject private var presetViewModel = DependencyContainer.shared.makeScalePresetViewModel()
    @StateObject private var localization = LocalizationManager.shared
    @Environment(\.horizontalSizeClass) var horizontalSizeClass
    @Environment(\.uiTestAnimationsDisabled) var uiTestAnimationsDisabled
    @State private var isSettingsPanelVisible: Bool = true
    @State private var showingAlert: Bool = false
    @State private var recordingForAnalysis: Recording?

    public init(viewModel: RecordingViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    public var body: some View {
        GeometryReader { geometry in
            Group {
                if geometry.size.width > geometry.size.height {
                    // Landscape layout
                    landscapeLayout
                } else {
                    // Portrait layout
                    portraitLayout
                }
            }
        }
        .navigationTitle("recording.title".localized)
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(viewModel.recordingState == .recording || viewModel.recordingState == .countdown)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                NavigationLink(destination: RecordingListView(
                    viewModel: RecordingListViewModel(
                        recordingRepository: DependencyContainer.shared.recordingRepository,
                        audioPlayer: DependencyContainer.shared.audioPlayer
                    ),
                    audioPlayer: DependencyContainer.shared.audioPlayer,
                    analyzeRecordingUseCase: DependencyContainer.shared.analyzeRecordingUseCase
                )) {
                    HStack(spacing: 4) {
                        Image(systemName: "list.bullet")
                        Text("recording.list_button".localized)
                    }
                }
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
        .onChange(of: viewModel.recordingState) { newState in
            // Auto-hide settings panel when recording starts
            if newState == .recording {
                if uiTestAnimationsDisabled {
                    isSettingsPanelVisible = false
                } else {
                    withAnimation {
                        isSettingsPanelVisible = false
                    }
                }
            }
        }
        .onChange(of: viewModel.errorMessage) { errorMessage in
            // Show alert when error message is set
            showingAlert = errorMessage != nil
        }
        .onAppear {
            // Reload audio settings when returning to recording screen
            // (in case user modified settings in SettingsView)
            viewModel.reloadAudioSettings(from: DependencyContainer.shared.audioSettingsRepository)
        }
        .onDisappear {
            // Stop playback when navigating away from this screen
            if viewModel.isPlayingRecording {
                Task {
                    await viewModel.stopPlayback()
                }
            }
            // Cleanup audio session to release microphone
            viewModel.recordingStateVM.cleanup()
        }
    }

    // MARK: - Landscape Layout

    private var landscapeLayout: some View {
        HStack(spacing: 0) {
            // Left side: Settings panel (collapsible)
            if isSettingsPanelVisible {
                RecordingSettingsPanel(viewModel: settingsViewModel, presetViewModel: presetViewModel)
                    .frame(width: 240)
                    .transition(.move(edge: .leading))

                Divider()
            }

            // Right side: Real-time display and controls
            VStack(spacing: 8) {
                // Toggle button for settings panel
                settingsToggleButton

                RealtimeDisplayArea(
                    recordingState: viewModel.recordingState,
                    isPlayingRecording: viewModel.isPlayingRecording,
                    targetPitch: viewModel.targetPitch,
                    detectedPitch: viewModel.detectedPitch,
                    pitchAccuracy: viewModel.pitchAccuracy,
                    spectrum: viewModel.spectrum,
                    audioLevel: viewModel.audioLevel,
                    isSettingsPanelVisible: isSettingsPanelVisible
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                RecordingControls(
                    recordingState: viewModel.recordingState,
                    hasLastRecording: viewModel.lastRecordingURL != nil,
                    isPlayingRecording: viewModel.isPlayingRecording,
                    canStartRecording: settingsViewModel.canStartRecording,
                    onStart: startRecording,
                    onStop: stopRecording,
                    onCancel: cancelCountdown,
                    onPlayLast: togglePlayback,
                    onAnalyze: navigateToAnalysisScreen,
                    isCompactLayout: true  // Horizontal button layout for landscape
                )
                .padding(.horizontal, 12)
                .padding(.bottom, 12)
            }
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: - Portrait Layout

    private var portraitLayout: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Toggle button for settings panel
                // Hide settings toggle during recording
                if viewModel.recordingState != .recording {
                    HStack {
                        Button(action: {
                            if uiTestAnimationsDisabled {
                                isSettingsPanelVisible.toggle()
                            } else {
                                withAnimation {
                                    isSettingsPanelVisible.toggle()
                                }
                            }
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: isSettingsPanelVisible ? "chevron.up" : "gearshape.fill")
                                Text(isSettingsPanelVisible ? "recording.hide_settings".localized : "recording.show_settings".localized)
                            }
                        }
                        .buttonStyle(CompactButtonStyle())

                        Spacer()
                    }
                }

                // Settings panel (collapsible)
                if isSettingsPanelVisible {
                    RecordingSettingsCompact(viewModel: settingsViewModel, presetViewModel: presetViewModel)
                        .transition(.opacity.combined(with: .move(edge: .top)))
                }

                RealtimeDisplayArea(
                    recordingState: viewModel.recordingState,
                    isPlayingRecording: viewModel.isPlayingRecording,
                    targetPitch: viewModel.targetPitch,
                    detectedPitch: viewModel.detectedPitch,
                    pitchAccuracy: viewModel.pitchAccuracy,
                    spectrum: viewModel.spectrum,
                    audioLevel: viewModel.audioLevel,
                    isSettingsPanelVisible: isSettingsPanelVisible
                )
                .frame(height: isSettingsPanelVisible ? 0 : 350)

                RecordingControls(
                    recordingState: viewModel.recordingState,
                    hasLastRecording: viewModel.lastRecordingURL != nil,
                    isPlayingRecording: viewModel.isPlayingRecording,
                    canStartRecording: settingsViewModel.canStartRecording,
                    onStart: startRecording,
                    onStop: stopRecording,
                    onCancel: cancelCountdown,
                    onPlayLast: togglePlayback,
                    onAnalyze: navigateToAnalysisScreen
                )
            }
            .padding()
        }
        .navigationDestination(item: $recordingForAnalysis) { recording in
            AnalysisView(
                recording: recording,
                audioPlayer: DependencyContainer.shared.audioPlayer,
                analyzeRecordingUseCase: DependencyContainer.shared.analyzeRecordingUseCase
            )
        }
    }

    // MARK: - Settings Toggle Button

    @ViewBuilder
    private var settingsToggleButton: some View {
        // Hide settings toggle during recording
        if viewModel.recordingState != .recording {
            HStack {
                Button(action: {
                    if uiTestAnimationsDisabled {
                        isSettingsPanelVisible.toggle()
                    } else {
                        withAnimation {
                            isSettingsPanelVisible.toggle()
                        }
                    }
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: isSettingsPanelVisible ? "sidebar.left" : "gearshape.fill")
                        Text(isSettingsPanelVisible ? "recording.hide_settings".localized : "recording.show_settings".localized)
                    }
                }
                .buttonStyle(CompactButtonStyle())

                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
        }
    }

    // MARK: - Action Handlers

    private func startRecording() {
        // Immediate visual feedback - set preparing state synchronously before async work
        viewModel.setPreparingState()

        Task { @MainActor in
            let settings = settingsViewModel.generateScaleSettings()
            await viewModel.startRecording(settings: settings)
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
        // Synchronous state update for immediate UI response
        if viewModel.isPlayingRecording {
            Task {
                await viewModel.stopPlayback()
            }
        } else {
            // Pre-set playing state BEFORE async operation for immediate UI update
            viewModel.isPlayingRecording = true
            Task {
                await viewModel.playLastRecording()
            }
        }
    }

    private func navigateToAnalysisScreen() {
        // Stop playback before navigating to analysis
        if viewModel.isPlayingRecording {
            Task {
                await viewModel.stopPlayback()
            }
        }

        // Fetch saved Recording from Repository (Single Source of Truth)
        guard let recordingId = viewModel.lastRecordingId else { return }
        Task {
            if let recording = try? await DependencyContainer.shared.recordingRepository.findById(recordingId) {
                recordingForAnalysis = recording
            }
        }
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
                    startRecordingWithScaleUseCase: PreviewMockStartRecordingWithScaleUseCase(),
                    stopRecordingUseCase: PreviewMockStopRecordingUseCase(),
                    audioPlayer: PreviewMockAudioPlayer(),
                    pitchDetector: RealtimePitchDetector(),
                    scalePlaybackCoordinator: ScalePlaybackCoordinator(scalePlayer: PreviewMockScalePlayer()),
                    subscriptionViewModel: SubscriptionViewModel(
                        getStatusUseCase: PreviewMockGetStatusUseCase(),
                        purchaseUseCase: PreviewMockPurchaseUseCase(),
                        restoreUseCase: PreviewMockRestoreUseCase()
                    )
                )
            )
        }
        .previewInterfaceOrientation(.landscapeLeft)
    }
}

// MARK: - Preview Mocks

private class PreviewMockScalePlayer: ScalePlayerProtocol {
    var isPlaying: Bool = false
    var currentNoteIndex: Int = 0
    var progress: Double = 0.0
    var currentScaleElement: ScaleElement? = nil

    func loadScale(_ notes: [MIDINote], tempo: Tempo) async throws {}
    func loadScaleElements(_ elements: [ScaleElement], tempo: Tempo) async throws {}
    func play(muted: Bool) async throws {}
    func stop() async {}
    func startTimestampRecording(recordingStartTime: Date) {}
    func stopTimestampRecording() {}
    func getPlaybackTimeline() -> ScalePlaybackTimeline? { nil }
}

private class PreviewMockStartRecordingUseCase: StartRecordingUseCaseProtocol {
    func execute(user: User) async throws -> RecordingSession {
        try await Task.sleep(nanoseconds: 1_000_000_000)
        return RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/preview.m4a"),
            settings: nil,
            startedAt: Date()
        )
    }
}

private class PreviewMockStartRecordingWithScaleUseCase: StartRecordingWithScaleUseCaseProtocol {
    func execute(user: User, settings: ScaleSettings) async throws -> RecordingSession {
        try await Task.sleep(nanoseconds: 1_000_000_000)
        return RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/preview.m4a"),
            settings: settings,
            startedAt: Date()
        )
    }
}

private class PreviewMockStopRecordingUseCase: StopRecordingUseCaseProtocol {
    func setRecordingContext(url: URL, settings: ScaleSettings?) {
        // Preview mock doesn't need to track context
    }

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
    func execute(tier: SubscriptionTier) async throws {
        // Mock implementation for preview
    }
}

private class PreviewMockRestoreUseCase: RestorePurchasesUseCaseProtocol {
    func execute() async throws {
        // Mock implementation for preview
    }
}
#endif
