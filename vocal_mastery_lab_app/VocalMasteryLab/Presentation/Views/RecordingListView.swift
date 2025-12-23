import SwiftUI
import VocalisDomain

/// Recording list screen
public struct RecordingListView: View {
    @StateObject private var viewModel: RecordingListViewModel
    @StateObject private var localization = LocalizationManager.shared
    @State private var selectedRecording: Recording?
    @State private var extractingRecording: Recording?
    @State private var editingRecording: Recording?
    @State private var editingTitle: String = ""
    @State private var showingRenameAlert: Bool = false
    @State private var deletingRecording: Recording?
    @State private var showingDeleteAlert: Bool = false

    private let audioPlayer: AudioPlayerProtocol
    private let analyzeRecordingUseCase: AnalyzeRecordingUseCase
    private let extractedAudioRepository: ExtractedAudioRepositoryProtocol
    private let vocalExtractor: VocalExtractorProtocol

    public init(
        viewModel: RecordingListViewModel,
        audioPlayer: AudioPlayerProtocol,
        analyzeRecordingUseCase: AnalyzeRecordingUseCase,
        extractedAudioRepository: ExtractedAudioRepositoryProtocol,
        vocalExtractor: VocalExtractorProtocol
    ) {
        _viewModel = StateObject(wrappedValue: viewModel)
        self.audioPlayer = audioPlayer
        self.analyzeRecordingUseCase = analyzeRecordingUseCase
        self.extractedAudioRepository = extractedAudioRepository
        self.vocalExtractor = vocalExtractor
    }

    public var body: some View {
        VStack(spacing: 0) {
            if viewModel.isLoading {
                Spacer()
                ProgressView()
                Spacer()
            } else if viewModel.recordings.isEmpty {
                emptyState
            } else {
                recordingList
            }

            // Fixed bottom playback control panel
            if !viewModel.recordings.isEmpty {
                PlaybackControlPanel(viewModel: viewModel)
            }
        }
        .navigationTitle("list.title".localized)
        .navigationBarTitleDisplayMode(.large)
        .navigationDestination(item: $selectedRecording) { recording in
            AnalysisView(
                recording: recording,
                audioPlayer: audioPlayer,
                analyzeRecordingUseCase: analyzeRecordingUseCase
            )
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
        .task {
            await viewModel.loadRecordings()
        }
        .onAppear {
            // Refresh recordings list when screen appears (to update cache state indicators)
            Task {
                await viewModel.loadRecordings()
            }
        }
        .onDisappear {
            // Stop playback when navigating away from this screen
            // Always stop to reset AudioPlayer state (even if paused)
            Task {
                await viewModel.stopPlayback()
            }
        }
        .alert(isPresented: .constant(viewModel.errorMessage != nil)) {
            Alert(
                title: Text("error".localized),
                message: Text(viewModel.errorMessage ?? ""),
                dismissButton: .default(Text("ok".localized))
            )
        }
        .alert("recording.rename.title".localized, isPresented: $showingRenameAlert) {
            TextField("recording.rename.placeholder".localized, text: $editingTitle)
            Button("cancel".localized, role: .cancel) {
                editingRecording = nil
                editingTitle = ""
            }
            Button("save".localized) {
                if let recording = editingRecording {
                    let newTitle = editingTitle  // Capture value before reset
                    Task {
                        await viewModel.renameRecording(recording, newTitle: newTitle)
                    }
                }
                editingRecording = nil
                editingTitle = ""
            }
        } message: {
            Text("recording.rename.message".localized)
        }
        .alert("recording.delete.title".localized, isPresented: $showingDeleteAlert) {
            Button("cancel".localized, role: .cancel) {
                deletingRecording = nil
            }
            Button("delete".localized, role: .destructive) {
                if let recording = deletingRecording {
                    Task {
                        await viewModel.deleteRecording(recording)
                    }
                }
                deletingRecording = nil
            }
            .accessibilityIdentifier("DeleteConfirmButton")
        } message: {
            Text("recording.delete.message".localized)
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 20) {
            Image(systemName: "mic.slash")
                .font(.system(size: 60))
                .foregroundColor(ColorPalette.text.opacity(0.5))

            Text("list.empty_title".localized)
                .font(.title2)
                .foregroundColor(ColorPalette.text)

            Text("list.empty_message".localized)
                .font(.body)
                .foregroundColor(ColorPalette.text.opacity(0.6))
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
    }

    // MARK: - Recording List

    private var recordingList: some View {
        List {
            ForEach(viewModel.recordings) { recording in
                RecordingRow(
                    recording: recording,
                    isSelected: viewModel.selectedRecording?.id == recording.id,
                    isPlaying: viewModel.playingRecordingId == recording.id,
                    isExtracted: viewModel.hasExtractedAudio(recording),
                    availableSources: viewModel.availableSources(for: recording),
                    onTap: {
                        Task {
                            await viewModel.selectAndPlay(recording)
                        }
                    },
                    onAnalyze: {
                        selectedRecording = recording
                    },
                    onExtract: {
                        extractingRecording = recording
                    },
                    onRename: {
                        editingRecording = recording
                        editingTitle = recording.title ?? ""
                        showingRenameAlert = true
                    },
                    onDelete: {
                        deletingRecording = recording
                        showingDeleteAlert = true
                    }
                )
                .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                    Button(role: .destructive) {
                        deletingRecording = recording
                        showingDeleteAlert = true
                    } label: {
                        Label("delete".localized, systemImage: "trash")
                    }
                    .accessibilityIdentifier("DeleteRecordingButton_\(recording.id.value.uuidString)")
                }
            }
        }
    }
}

// MARK: - Scale Button Style

/// Button style with scale and opacity animation for immediate visual feedback
private struct ScaleButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.92 : 1.0)
            .opacity(configuration.isPressed ? 0.8 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

// MARK: - Recording Row

private struct RecordingRow: View {
    let recording: Recording
    let isSelected: Bool
    let isPlaying: Bool
    let isExtracted: Bool
    let availableSources: [AudioSourceType]
    let onTap: () -> Void
    let onAnalyze: () -> Void
    let onExtract: () -> Void
    let onRename: () -> Void
    let onDelete: () -> Void

    private var hasVocal: Bool {
        availableSources.contains(.vocal)
    }

    private var hasInstrumental: Bool {
        availableSources.contains(.instrumental)
    }

    var body: some View {
        HStack(spacing: 0) {
            // Selection indicator bar
            Rectangle()
                .fill(isSelected ? ColorPalette.primary : Color.clear)
                .frame(width: 4)

            // Main content - tappable area for playback
            VStack(alignment: .leading, spacing: 4) {
                Text(recording.title ?? "recording.title".localized)
                    .font(.headline)
                    .foregroundColor(ColorPalette.text)

                HStack(spacing: 8) {
                    // Date and duration
                    HStack(spacing: 4) {
                        Text(recording.formattedDate)
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.6))

                        Text("•")
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.4))

                        Text(formatTime(recording.duration.seconds))
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                    }

                    // Extraction status indicators
                    if hasVocal || hasInstrumental {
                        HStack(spacing: 4) {
                            if hasVocal {
                                Image(systemName: "person.wave.2")
                                    .font(.system(size: 10))
                                    .foregroundColor(ColorPalette.primary)
                            }
                            if hasInstrumental {
                                Image(systemName: "music.note.list")
                                    .font(.system(size: 10))
                                    .foregroundColor(ColorPalette.primary)
                            }
                        }
                        .accessibilityIdentifier("ExtractionIndicators")
                    }
                }
            }
            .padding(.vertical, 12)
            .padding(.horizontal, 12)
            .contentShape(Rectangle())
            .onTapGesture {
                onTap()
            }

            Spacer()

            // Menu button
            Menu {
                // Play
                Button {
                    onTap()
                } label: {
                    Label("再生", systemImage: "play.fill")
                }

                Divider()

                // Analysis - disabled if not extracted
                Button {
                    onAnalyze()
                } label: {
                    Label("分析", systemImage: "chart.xyaxis.line")
                }
                .disabled(!isExtracted)

                // Extract/Re-extract - always enabled
                Button {
                    onExtract()
                } label: {
                    Label(isExtracted ? "再抽出" : "ボーカル抽出", systemImage: "waveform.path.ecg")
                }

                Divider()

                Button {
                    onRename()
                } label: {
                    Label("recording.rename".localized, systemImage: "pencil")
                }

                Button(role: .destructive) {
                    onDelete()
                } label: {
                    Label("delete".localized, systemImage: "trash")
                }
            } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(ColorPalette.text.opacity(0.6))
                    .frame(width: 44, height: 44)
                    .contentShape(Rectangle())
            }
            .accessibilityIdentifier("MenuButton_\(recording.id.value.uuidString)")
        }
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isSelected ? ColorPalette.primary.opacity(0.05) : ColorPalette.background)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isSelected ? ColorPalette.primary.opacity(0.3) : Color.clear, lineWidth: 1)
        )
    }

    /// Format time in seconds to MM:SS format
    private func formatTime(_ seconds: Double) -> String {
        let minutes = Int(seconds) / 60
        let remainingSeconds = Int(seconds) % 60
        return String(format: "%d:%02d", minutes, remainingSeconds)
    }
}
