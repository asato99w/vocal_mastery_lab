import SwiftUI
import VocalisDomain
import os.log

/// Analysis screen - displays spectrogram and pitch analysis for a recording
public struct AnalysisView: View {
    let recording: Recording
    @StateObject private var viewModel: AnalysisViewModel
    @StateObject private var localization = LocalizationManager.shared
    @Environment(\.horizontalSizeClass) var horizontalSizeClass

    // MARK: - Expanded Graph State
    @State private var expandedGraph: ExpandedGraphType? = nil

    // MARK: - Statistics Sheet State
    @State private var showStatisticsSheet: Bool = false

    enum ExpandedGraphType: Identifiable {
        case spectrogram
        case pitchAnalysis

        var id: Self { self }
    }

    public init(
        recording: Recording,
        audioPlayer: AudioPlayerProtocol,
        analyzeRecordingUseCase: AnalyzeRecordingUseCase
    ) {
        self.recording = recording
        _viewModel = StateObject(wrappedValue: AnalysisViewModel(
            recording: recording,
            audioPlayer: audioPlayer,
            analyzeRecordingUseCase: analyzeRecordingUseCase
        ))
    }

    public var body: some View {
        ZStack {
            GeometryReader { geometry in
                if geometry.size.width > geometry.size.height {
                    // Landscape layout
                    landscapeLayout
                } else {
                    // Portrait layout
                    portraitLayout
                }
            }

            // Loading overlay
            if case .loading(let progress) = viewModel.state {
                ColorPalette.background.opacity(0.4)
                    .ignoresSafeArea()

                VStack(spacing: 16) {
                    Text("analysis.analyzing".localized)
                        .font(.headline)
                        .foregroundColor(ColorPalette.text)

                    VStack(spacing: 8) {
                        ProgressView(value: progress, total: 1.0)
                            .progressViewStyle(LinearProgressViewStyle(tint: ColorPalette.primary))
                            .frame(width: 200)

                        Text("\(Int(progress * 100))%")
                            .font(.subheadline)
                            .foregroundColor(ColorPalette.text)
                            .monospacedDigit()
                    }
                }
                .padding(32)
                .background(ColorPalette.secondary)
                .cornerRadius(16)
            }

            // Error overlay
            if case .error(let message) = viewModel.state {
                ColorPalette.background.opacity(0.4)
                    .ignoresSafeArea()

                VStack(spacing: 16) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 48))
                        .foregroundColor(ColorPalette.alertActive)

                    Text("analysis.error".localized)
                        .font(.headline)
                        .foregroundColor(ColorPalette.text)

                    Text(message)
                        .font(.subheadline)
                        .foregroundColor(ColorPalette.text.opacity(0.6))
                        .multilineTextAlignment(.center)
                }
                .padding(32)
                .background(ColorPalette.background)
                .cornerRadius(16)
                .shadow(radius: 10)
            }
        }
        .navigationTitle("analysis.title".localized)
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(viewModel.isAnalyzing)
        .fullScreenCover(item: $expandedGraph) { graphType in
            expandedGraphFullScreen(for: graphType)
        }
        .sheet(isPresented: $showStatisticsSheet) {
            StatisticsSheetView(
                recording: recording,
                statistics: calculateStatistics()
            )
        }
        .task {
            await viewModel.startAnalysis()
        }
        .onDisappear {
            // Stop playback when navigating away from this screen
            // Always stop to reset AudioPlayer state (even if paused)
            Task {
                await viewModel.stopPlayback()
            }
        }
    }

    // MARK: - Statistics Calculation

    private func calculateStatistics() -> RecordingStatistics? {
        guard let analysisResult = viewModel.analysisResult else { return nil }

        // Use algorithm-aware calculator for vibrato detection optimization
        let algorithm = recording.analysisAlgorithm ?? .default
        let calculator = RecordingStatisticsCalculator(algorithm: algorithm)
        return calculator.calculate(
            pitchData: analysisResult.pitchData,
            playbackTimeline: recording.playbackTimeline,
            scaleSettings: recording.scaleSettings,
            spectrogramData: analysisResult.spectrogramData
        )
    }

    // MARK: - Landscape Layout

    private var landscapeLayout: some View {
        HStack(spacing: 0) {
            // Left side: Recording info and playback controls
            VStack(spacing: 12) {
                RecordingInfoPanel(recording: recording, showStatisticsSheet: $showStatisticsSheet)

                PlaybackControl(
                    isPlaying: viewModel.isPlaying,
                    currentTime: viewModel.currentTime,
                    duration: recording.duration.seconds,
                    onPlayPause: { viewModel.togglePlayback() },
                    onSeek: { time in viewModel.seek(to: time) }
                )

                Spacer()
            }
            .frame(width: 240)
            .padding(12)

            Divider()

            // Right side: Visualization area
            VStack(spacing: 12) {
                // Spectrogram (top half)
                SpectrogramView(
                    currentTime: viewModel.currentTime,
                    spectrogramData: viewModel.analysisResult?.spectrogramData,
                    onExpand: {
                        expandedGraph = .spectrogram
                    },
                    onPlayPause: { viewModel.togglePlayback() },
                    onSeek: { time in viewModel.seek(to: time) }
                )
                .frame(maxHeight: .infinity)

                Divider()

                // Pitch analysis graph (bottom half)
                PitchAnalysisView(
                    currentTime: viewModel.currentTime,
                    pitchData: viewModel.analysisResult?.pitchData,
                    scaleSettings: viewModel.analysisResult?.scaleSettings,
                    playbackTimeline: recording.playbackTimeline,
                    onExpand: {
                        expandedGraph = .pitchAnalysis
                    },
                    onPlayPause: { viewModel.togglePlayback() },
                    onSeek: { time in viewModel.seek(to: time) }
                )
                .frame(maxHeight: .infinity)
            }
            .padding(12)
        }
    }

    // MARK: - Portrait Layout

    private var portraitLayout: some View {
        ScrollView {
            VStack(spacing: 16) {
                RecordingInfoCompact(recording: recording, showStatisticsSheet: $showStatisticsSheet)

                PlaybackControl(
                    isPlaying: viewModel.isPlaying,
                    currentTime: viewModel.currentTime,
                    duration: recording.duration.seconds,
                    onPlayPause: { viewModel.togglePlayback() },
                    onSeek: { time in viewModel.seek(to: time) }
                )

                SpectrogramView(
                    currentTime: viewModel.currentTime,
                    spectrogramData: viewModel.analysisResult?.spectrogramData,
                    onExpand: {
                        expandedGraph = .spectrogram
                    },
                    onPlayPause: { viewModel.togglePlayback() },
                    onSeek: { time in viewModel.seek(to: time) }
                )
                .frame(height: 200)

                PitchAnalysisView(
                    currentTime: viewModel.currentTime,
                    pitchData: viewModel.analysisResult?.pitchData,
                    scaleSettings: viewModel.analysisResult?.scaleSettings,
                    playbackTimeline: recording.playbackTimeline,
                    onExpand: {
                        expandedGraph = .pitchAnalysis
                    },
                    onPlayPause: { viewModel.togglePlayback() },
                    onSeek: { time in viewModel.seek(to: time) }
                )
                .frame(height: 200)
            }
            .padding()
        }
    }

    // MARK: - Expanded Graph Full Screen

    @ViewBuilder
    private func expandedGraphFullScreen(for type: ExpandedGraphType) -> some View {
        ZStack {
            // Background
            ColorPalette.background
                .ignoresSafeArea()

            // Graph content
            VStack(spacing: 0) {
                // Graph area (maximized)
                switch type {
                case .spectrogram:
                    SpectrogramView(
                        currentTime: viewModel.currentTime,
                        spectrogramData: viewModel.analysisResult?.spectrogramData,
                        isExpanded: true,
                        onCollapse: {
                            withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                                expandedGraph = nil
                            }
                        },
                        onPlayPause: { viewModel.togglePlayback() },
                        onSeek: { time in viewModel.seek(to: time) }
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                case .pitchAnalysis:
                    PitchAnalysisView(
                        currentTime: viewModel.currentTime,
                        pitchData: viewModel.analysisResult?.pitchData,
                        scaleSettings: viewModel.analysisResult?.scaleSettings,
                        playbackTimeline: recording.playbackTimeline,
                        isExpanded: true,
                        onCollapse: {
                            withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                                expandedGraph = nil
                            }
                        },
                        onPlayPause: { viewModel.togglePlayback() },
                        onSeek: { time in viewModel.seek(to: time) }
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier(type == .spectrogram ? "ExpandedSpectrogramView" : "ExpandedPitchGraphView")
    }
}

// MARK: - Preview

#if DEBUG
private class PreviewAudioPlayer: AudioPlayerProtocol {
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0.0
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

private class PreviewAudioFileAnalyzer: AudioFileAnalyzerProtocol {
    func analyze(fileURL: URL, progress: @escaping @MainActor (Double) async -> Void) async throws -> (pitchData: PitchAnalysisData, spectrogramData: SpectrogramData) {
        // Simulate progress updates
        await progress(0.0)
        try await Task.sleep(nanoseconds: 500_000_000)  // 0.5s
        await progress(0.5)
        try await Task.sleep(nanoseconds: 500_000_000)  // 0.5s
        await progress(1.0)

        let pitchData = PitchAnalysisData(
            timeStamps: [0.0, 0.05, 0.10],
            frequencies: [261.6, 262.3, 261.9],
            confidences: [0.85, 0.92, 0.88],
            targetNotes: [nil, nil, nil]
        )

        let spectrogramData = SpectrogramData(
            timeStamps: [0.0, 0.1, 0.2],
            frequencyBins: [80, 180, 280],
            magnitudes: [[0.1, 0.3, 0.8], [0.2, 0.4, 0.7], [0.3, 0.5, 0.6]]
        )

        return (pitchData, spectrogramData)
    }

    func analyzeSpectrogramOnly(fileURL: URL, progress: @escaping @MainActor (Double) async -> Void) async throws -> SpectrogramData {
        // Simulate progress updates
        await progress(0.0)
        try await Task.sleep(nanoseconds: 300_000_000)  // 0.3s
        await progress(0.5)
        try await Task.sleep(nanoseconds: 300_000_000)  // 0.3s
        await progress(1.0)

        return SpectrogramData(
            timeStamps: [0.0, 0.1, 0.2],
            frequencyBins: [80, 180, 280],
            magnitudes: [[0.1, 0.3, 0.8], [0.2, 0.4, 0.7], [0.3, 0.5, 0.6]]
        )
    }
}

private class PreviewLogger: LoggerProtocol {
    func debug(_ message: String, category: String) {}
    func info(_ message: String, category: String) {}
    func warning(_ message: String, category: String) {}
    func error(_ message: String, category: String) {}
}

struct AnalysisView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationStack {
            AnalysisView(
                recording: Recording(
                    id: RecordingId(),
                    fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
                    createdAt: Date(),
                    duration: Duration(seconds: 10.0),
                    scaleSettings: ScaleSettings(
                        startNote: try! MIDINote(60), // C3
                        endNote: try! MIDINote(72),   // C4
                        notePattern: .fiveToneScale,
                        tempo: try! Tempo(secondsPerNote: 0.5)
                    )
                ),
                audioPlayer: PreviewAudioPlayer(),
                analyzeRecordingUseCase: AnalyzeRecordingUseCase(
                    audioFileAnalyzer: PreviewAudioFileAnalyzer(),
                    analysisCache: AnalysisCache(),
                    logger: PreviewLogger()
                )
            )
        }
        .previewInterfaceOrientation(.landscapeLeft)
    }
}
#endif
