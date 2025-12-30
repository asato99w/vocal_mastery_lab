import SwiftUI
import VocalisDomain

/// Vocal extraction screen
public struct VocalExtractionView: View {
    @StateObject private var viewModel: VocalExtractionViewModel
    @Environment(\.dismiss) private var dismiss

    /// Optional callback when save completes successfully
    /// Used by RecordingView to navigate to RecordingListView after save
    var onSaveComplete: (() -> Void)?

    public init(viewModel: VocalExtractionViewModel, onSaveComplete: (() -> Void)? = nil) {
        _viewModel = StateObject(wrappedValue: viewModel)
        self.onSaveComplete = onSaveComplete
    }

    public var body: some View {
        VStack(spacing: 24) {
            // Recording info section
            recordingInfoSection

            Spacer()

            // Main content based on state
            stateContent

            Spacer()

            // Action buttons
            actionButtons
        }
        .padding()
        .navigationTitle("ボーカル抽出")
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear {
            Task {
                await viewModel.stopPlayback()
            }
        }
    }

    // MARK: - Recording Info Section

    private var recordingInfoSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(viewModel.recordingTitle)
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(ColorPalette.text)

            HStack {
                Label(viewModel.recordingDate, systemImage: "calendar")
                    .font(.caption)
                    .foregroundColor(ColorPalette.text.opacity(0.6))

                Text("•")
                    .foregroundColor(ColorPalette.text.opacity(0.4))

                Label(viewModel.recordingDuration, systemImage: "clock")
                    .font(.caption)
                    .foregroundColor(ColorPalette.text.opacity(0.6))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(ColorPalette.secondary)
        )
    }

    // MARK: - State Content

    @ViewBuilder
    private var stateContent: some View {
        switch viewModel.state {
        case .idle:
            idleContent
        case .processing(let progress, let stage):
            processingContent(progress: progress, stage: stage)
        case .completed(let result):
            completedContent(result: result)
        case .error(let message):
            errorContent(message: message)
        }
    }

    private var idleContent: some View {
        VStack(spacing: 20) {
            Image(systemName: "waveform.path.ecg")
                .font(.system(size: 60))
                .foregroundColor(ColorPalette.primary)

            Text("録音からボーカルを抽出します")
                .font(.body)
                .foregroundColor(ColorPalette.text.opacity(0.8))
                .multilineTextAlignment(.center)
        }
    }

    private func processingContent(progress: Double, stage: String) -> some View {
        VStack(spacing: 20) {
            ProgressView(value: progress)
                .progressViewStyle(LinearProgressViewStyle(tint: ColorPalette.primary))
                .scaleEffect(y: 2)

            Text("\(Int(progress * 100))%")
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(ColorPalette.primary)

            Text(stage)
                .font(.body)
                .foregroundColor(ColorPalette.text.opacity(0.8))
        }
        .padding(.horizontal, 40)
    }

    private func completedContent(result: ExtractionResultData) -> some View {
        VStack(spacing: 20) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 60))
                .foregroundColor(.green)

            Text(viewModel.extractionCount > 1 ? "抽出完了（\(viewModel.extractionCount)次）" : "抽出完了")
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(ColorPalette.text)

            // Preview section - Mini player style
            VStack(spacing: 0) {
                MiniPlayerRow(
                    title: "元の音声",
                    icon: "waveform",
                    isActive: viewModel.playingSource == .original,
                    isPlaying: viewModel.playingSource == .original && viewModel.isPlaying,
                    currentTime: viewModel.playingSource == .original ? viewModel.currentTime : 0,
                    duration: viewModel.originalDurationSeconds,
                    onPlayPause: {
                        if viewModel.playingSource == .original {
                            viewModel.togglePlayPause()
                        } else {
                            Task { await viewModel.playOriginal() }
                        }
                    },
                    onSeek: { time in
                        if viewModel.playingSource == .original {
                            viewModel.seek(to: time)
                        }
                    }
                )

                Divider()

                MiniPlayerRow(
                    title: "ボーカル",
                    icon: "person.wave.2",
                    isActive: viewModel.playingSource == .vocal,
                    isPlaying: viewModel.playingSource == .vocal && viewModel.isPlaying,
                    currentTime: viewModel.playingSource == .vocal ? viewModel.currentTime : 0,
                    duration: result.duration.seconds,
                    onPlayPause: {
                        if viewModel.playingSource == .vocal {
                            viewModel.togglePlayPause()
                        } else {
                            Task { await viewModel.playVocal() }
                        }
                    },
                    onSeek: { time in
                        if viewModel.playingSource == .vocal {
                            viewModel.seek(to: time)
                        }
                    }
                )

                if result.instrumentalURL != nil {
                    Divider()

                    MiniPlayerRow(
                        title: "伴奏",
                        icon: "music.note.list",
                        isActive: viewModel.playingSource == .instrumental,
                        isPlaying: viewModel.playingSource == .instrumental && viewModel.isPlaying,
                        currentTime: viewModel.playingSource == .instrumental ? viewModel.currentTime : 0,
                        duration: result.duration.seconds,
                        onPlayPause: {
                            if viewModel.playingSource == .instrumental {
                                viewModel.togglePlayPause()
                            } else {
                                Task { await viewModel.playInstrumental() }
                            }
                        },
                        onSeek: { time in
                            if viewModel.playingSource == .instrumental {
                                viewModel.seek(to: time)
                            }
                        }
                    )
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(ColorPalette.secondary)
            )
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }

    private func errorContent(message: String) -> some View {
        VStack(spacing: 20) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 60))
                .foregroundColor(.red)

            Text("エラー")
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(ColorPalette.text)

            Text(message)
                .font(.body)
                .foregroundColor(ColorPalette.text.opacity(0.8))
                .multilineTextAlignment(.center)
        }
    }

    // MARK: - Action Buttons

    @ViewBuilder
    private var actionButtons: some View {
        switch viewModel.state {
        case .idle:
            Button(action: {
                Task { await viewModel.startExtraction() }
            }) {
                Text("抽出開始")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(ColorPalette.primary)
                    .cornerRadius(12)
            }

        case .processing:
            EmptyView()

        case .completed:
            VStack(spacing: 12) {
                // 2次抽出ボタン
                Button(action: {
                    Task { await viewModel.startSecondaryExtraction() }
                }) {
                    HStack {
                        Image(systemName: "waveform.badge.plus")
                        Text("2次抽出")
                    }
                    .font(.headline)
                    .foregroundColor(ColorPalette.primary)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(ColorPalette.primary.opacity(0.1))
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(ColorPalette.primary, lineWidth: 1)
                    )
                }

                HStack(spacing: 16) {
                    Button(action: {
                        viewModel.reset()
                    }) {
                        Text("やり直し")
                            .font(.headline)
                            .foregroundColor(ColorPalette.text)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(ColorPalette.secondary)
                            .cornerRadius(12)
                    }

                    Button(action: {
                        Task {
                            let success = await viewModel.saveExtraction()
                            if success {
                                if let onSaveComplete = onSaveComplete {
                                    // Custom navigation (e.g., to RecordingListView)
                                    onSaveComplete()
                                } else {
                                    // Default: pop back to previous screen
                                    dismiss()
                                }
                            }
                        }
                    }) {
                        if viewModel.isSaving {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(ColorPalette.primary)
                                .cornerRadius(12)
                        } else {
                            Text("保存")
                                .font(.headline)
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(ColorPalette.primary)
                                .cornerRadius(12)
                        }
                    }
                    .disabled(viewModel.isSaving)
                }
            }

        case .error:
            Button(action: {
                viewModel.reset()
            }) {
                Text("戻る")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(ColorPalette.primary)
                    .cornerRadius(12)
            }
        }
    }
}

// MARK: - Mini Player Row

private struct MiniPlayerRow: View {
    let title: String
    let icon: String
    let isActive: Bool
    let isPlaying: Bool
    let currentTime: TimeInterval
    let duration: TimeInterval
    let onPlayPause: () -> Void
    let onSeek: (TimeInterval) -> Void

    var body: some View {
        HStack(spacing: 12) {
            // Icon and title
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 16))
                    .frame(width: 20)
                Text(title)
                    .font(.subheadline)
                    .fontWeight(isActive ? .semibold : .regular)
            }
            .foregroundColor(isActive ? ColorPalette.primary : ColorPalette.text)
            .frame(width: 90, alignment: .leading)

            // Play/Pause button
            Button(action: onPlayPause) {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 16))
                    .foregroundColor(isActive ? ColorPalette.primary : ColorPalette.text.opacity(0.6))
                    .frame(width: 24, height: 24)
            }

            // Custom seek bar with hidden thumb
            SeekBar(
                value: currentTime,
                range: 0...max(duration, 0.1),
                isEnabled: isActive,
                accentColor: isActive ? ColorPalette.primary : ColorPalette.text.opacity(0.3),
                onSeek: onSeek
            )

            // Time
            Text(formatTime(isActive ? currentTime : 0))
                .font(.caption)
                .foregroundColor(ColorPalette.text.opacity(0.6))
                .frame(width: 36, alignment: .trailing)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(isActive ? ColorPalette.primary.opacity(0.08) : Color.clear)
    }

    private func formatTime(_ seconds: TimeInterval) -> String {
        let minutes = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", minutes, secs)
    }
}

// MARK: - Custom Seek Bar

private struct SeekBar: View {
    let value: TimeInterval
    let range: ClosedRange<TimeInterval>
    let isEnabled: Bool
    let accentColor: Color
    let onSeek: (TimeInterval) -> Void

    @State private var isDragging: Bool = false
    @State private var dragValue: TimeInterval?

    private let trackHeight: CGFloat = 4
    private let thumbSize: CGFloat = 14

    private var displayValue: TimeInterval {
        dragValue ?? value
    }

    private var progress: Double {
        let rangeSpan = range.upperBound - range.lowerBound
        guard rangeSpan > 0 else { return 0 }
        return (displayValue - range.lowerBound) / rangeSpan
    }

    var body: some View {
        GeometryReader { geometry in
            let trackWidth = geometry.size.width

            ZStack(alignment: .leading) {
                // Background track
                Capsule()
                    .fill(ColorPalette.text.opacity(0.15))
                    .frame(height: trackHeight)

                // Progress track
                Capsule()
                    .fill(accentColor)
                    .frame(width: max(0, trackWidth * progress), height: trackHeight)

                // Thumb (visible only when dragging or recently interacted)
                if isDragging && isEnabled {
                    Circle()
                        .fill(accentColor)
                        .frame(width: thumbSize, height: thumbSize)
                        .offset(x: max(0, min(trackWidth - thumbSize, trackWidth * progress - thumbSize / 2)))
                        .shadow(color: .black.opacity(0.2), radius: 2, x: 0, y: 1)
                }
            }
            .frame(height: max(trackHeight, thumbSize))
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { gesture in
                        guard isEnabled else { return }
                        isDragging = true
                        let newProgress = gesture.location.x / trackWidth
                        let clampedProgress = max(0, min(1, newProgress))
                        let rangeSpan = range.upperBound - range.lowerBound
                        dragValue = range.lowerBound + clampedProgress * rangeSpan
                    }
                    .onEnded { gesture in
                        guard isEnabled else { return }
                        let newProgress = gesture.location.x / trackWidth
                        let clampedProgress = max(0, min(1, newProgress))
                        let rangeSpan = range.upperBound - range.lowerBound
                        let newValue = range.lowerBound + clampedProgress * rangeSpan
                        onSeek(newValue)
                        dragValue = nil
                        isDragging = false
                    }
            )
        }
        .frame(height: max(trackHeight, thumbSize))
    }
}
