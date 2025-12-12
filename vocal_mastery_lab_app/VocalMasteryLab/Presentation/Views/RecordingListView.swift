import SwiftUI
import VocalisDomain

/// Recording list screen
public struct RecordingListView: View {
    @StateObject private var viewModel: RecordingListViewModel
    @StateObject private var localization = LocalizationManager.shared
    @State private var selectedRecording: Recording?
    @State private var editingRecording: Recording?
    @State private var editingTitle: String = ""
    @State private var showingRenameAlert: Bool = false
    @State private var deletingRecording: Recording?
    @State private var showingDeleteAlert: Bool = false

    private let audioPlayer: AudioPlayerProtocol
    private let analyzeRecordingUseCase: AnalyzeRecordingUseCase

    public init(
        viewModel: RecordingListViewModel,
        audioPlayer: AudioPlayerProtocol,
        analyzeRecordingUseCase: AnalyzeRecordingUseCase
    ) {
        _viewModel = StateObject(wrappedValue: viewModel)
        self.audioPlayer = audioPlayer
        self.analyzeRecordingUseCase = analyzeRecordingUseCase
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
                    hasCachedData: analyzeRecordingUseCase.hasCachedData(for: recording),
                    selectedRecording: $selectedRecording,
                    onTap: {
                        Task {
                            await viewModel.selectAndPlay(recording)
                        }
                    },
                    onAnalyze: {
                        selectedRecording = recording
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
                .swipeActions(edge: .leading) {
                    Button {
                        editingRecording = recording
                        // Show current display name in text field
                        editingTitle = recording.title ?? localizedScaleDisplayName(for: recording) ?? ""
                        showingRenameAlert = true
                    } label: {
                        Label("recording.rename".localized, systemImage: "pencil")
                    }
                    .tint(.blue)
                    .accessibilityIdentifier("RenameRecordingButton_\(recording.id.value.uuidString)")
                }
            }
        }
    }

    /// Get localized scale display name for a recording
    private func localizedScaleDisplayName(for recording: Recording) -> String? {
        guard let scaleDisplayNameKey = recording.scaleDisplayNameKey,
              let startNoteName = recording.scaleStartNoteName else {
            return nil
        }
        return "\(startNoteName) \(scaleDisplayNameKey.localized)"
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
    let hasCachedData: Bool
    @Binding var selectedRecording: Recording?
    let onTap: () -> Void
    let onAnalyze: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            // Selection indicator bar
            Rectangle()
                .fill(isSelected ? ColorPalette.primary : Color.clear)
                .frame(width: 4)

            // Main content - tappable area for playback
            VStack(alignment: .leading, spacing: 8) {
                // Recording name (title > scaleDisplayName > default)
                Text(recording.title ?? localizedScaleDisplayName(for: recording) ?? "recording.title".localized)
                    .font(.headline)
                    .foregroundColor(ColorPalette.text)

                // Date and duration on same line
                HStack {
                    Text(recording.formattedDate)
                        .font(.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.6))

                    Text("•")
                        .font(.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.4))

                    Text(formatTime(recording.duration.seconds))
                        .font(.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.6))

                    Spacer()
                }
            }
            .padding(.vertical, 12)
            .padding(.horizontal, 12)
            .contentShape(Rectangle())
            .onTapGesture {
                onTap()
            }

            // Analysis button - separated from onTapGesture to prevent gesture conflict
            Button(action: onAnalyze) {
                HStack(spacing: 4) {
                    Image(systemName: "waveform.path.ecg")
                        .font(.system(size: 14, weight: .medium))
                }
                .foregroundColor(hasCachedData ? .white : ColorPalette.text)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    Capsule()
                        .fill(hasCachedData ? ColorPalette.primary : ColorPalette.alertActive)
                )
            }
            .buttonStyle(ScaleButtonStyle())
            .accessibilityIdentifier("AnalysisNavigationLink_\(hasCachedData ? "cached" : "uncached")_\(recording.id.value.uuidString)")
            .padding(.trailing, 12)
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

    /// Get localized scale display name for a recording
    private func localizedScaleDisplayName(for recording: Recording) -> String? {
        guard let scaleDisplayNameKey = recording.scaleDisplayNameKey,
              let startNoteName = recording.scaleStartNoteName else {
            return nil
        }
        return "\(startNoteName) \(scaleDisplayNameKey.localized)"
    }
}
