import SwiftUI
import VocalisDomain

/// Fixed bottom playback control panel
struct PlaybackControlPanel: View {
    @ObservedObject var viewModel: RecordingListViewModel

    var body: some View {
        VStack(spacing: 12) {
            // Recording info or placeholder message
            VStack(spacing: 4) {
                if let recording = viewModel.selectedRecording {
                    HStack(spacing: 4) {
                        if let title = recording.title {
                            Text(title)
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundColor(ColorPalette.text)
                        }

                        // Show current playing source if not original
                        if viewModel.currentPlayingSource != .original {
                            Text("- \(viewModel.currentPlayingSource.displayName)")
                                .font(.subheadline)
                                .foregroundColor(ColorPalette.primary)
                        }
                    }

                    Text(recording.formattedDate)
                        .font(.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.6))
                } else {
                    Text("list.select_recording".localized)
                        .font(.subheadline)
                        .foregroundColor(ColorPalette.text.opacity(0.5))

                    Text(" ")
                        .font(.caption)
                }
            }
            .accessibilityIdentifier("PlaybackControlPanel_RecordingInfo")

            // Audio source segment control (only shown when recording is selected)
            if viewModel.selectedRecording != nil {
                audioSourcePicker
            }

            // Playback controls (always visible)
            HStack(spacing: 32) {
                // Previous button
                Button(action: {
                    Task {
                        await viewModel.playPrevious()
                    }
                }) {
                    Image(systemName: "backward.fill")
                        .font(.title2)
                        .foregroundColor(viewModel.canPlayPrevious ? ColorPalette.text : ColorPalette.text.opacity(0.3))
                }
                .disabled(!viewModel.canPlayPrevious)
                .accessibilityIdentifier("PlaybackControlPanel_PreviousButton")
                .accessibilityLabel("previous.recording".localized)

                // Play/Pause button
                Button(action: {
                    Task {
                        await viewModel.togglePlayback()
                    }
                }) {
                    Image(systemName: viewModel.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 44))
                        .foregroundColor(viewModel.selectedRecording != nil ? ColorPalette.primary : ColorPalette.primary.opacity(0.3))
                }
                .disabled(viewModel.selectedRecording == nil)
                .accessibilityIdentifier("PlaybackControlPanel_PlayPauseButton")
                .accessibilityLabel(viewModel.isPlaying ? "pause".localized : "play".localized)

                // Next button
                Button(action: {
                    Task {
                        await viewModel.playNext()
                    }
                }) {
                    Image(systemName: "forward.fill")
                        .font(.title2)
                        .foregroundColor(viewModel.canPlayNext ? ColorPalette.text : ColorPalette.text.opacity(0.3))
                }
                .disabled(!viewModel.canPlayNext)
                .accessibilityIdentifier("PlaybackControlPanel_NextButton")
                .accessibilityLabel("next.recording".localized)
            }

            // Slider and time (always visible)
            VStack(spacing: 4) {
                Slider(
                    value: Binding(
                        get: {
                            if let recording = viewModel.selectedRecording {
                                return viewModel.currentPlaybackPosition[recording.id] ?? 0.0
                            }
                            return 0.0
                        },
                        set: { newValue in
                            if let recording = viewModel.selectedRecording {
                                // Synchronous UI update for responsive slider
                                viewModel.updatePositionImmediate(newValue, for: recording.id)
                                // Async audio seek
                                viewModel.seekAudio(to: newValue)
                            }
                        }
                    ),
                    in: 0...max(currentDuration, 0.1)
                )
                .accentColor(ColorPalette.primary)
                .disabled(viewModel.selectedRecording == nil)
                .accessibilityIdentifier("PlaybackControlPanel_Slider")

                HStack {
                    Text(formatTime(viewModel.selectedRecording.flatMap { viewModel.currentPlaybackPosition[$0.id] } ?? 0.0))
                        .font(.caption2)
                        .foregroundColor(ColorPalette.text.opacity(0.5))
                        .accessibilityIdentifier("PlaybackControlPanel_CurrentTime")

                    Spacer()

                    Text(formatTime(currentDuration))
                        .font(.caption2)
                        .foregroundColor(ColorPalette.text.opacity(0.5))
                        .accessibilityIdentifier("PlaybackControlPanel_TotalTime")
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(ColorPalette.background)
                .shadow(color: Color.black.opacity(0.1), radius: 8, x: 0, y: -4)
        )
        // Note: accessibilityIdentifier on VStack propagates to children in SwiftUI,
        // which overrides individual child identifiers. Removed to allow child identifiers to work.
        .accessibilityElement(children: .contain)
    }

    // MARK: - Audio Source Picker

    private var audioSourcePicker: some View {
        HStack(spacing: 0) {
            ForEach(AudioSourceType.allCases, id: \.self) { source in
                audioSourceButton(for: source)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(ColorPalette.secondary)
        )
        // Use .contain to allow child button identifiers to work
        .accessibilityElement(children: .contain)
    }

    private func audioSourceButton(for source: AudioSourceType) -> some View {
        let isSelected = viewModel.selectedAudioSource == source
        let isAvailable = viewModel.isSourceAvailable(source)

        return Button(action: {
            guard isAvailable else { return }
            Task {
                await viewModel.switchAudioSource(to: source)
            }
        }) {
            VStack(spacing: 2) {
                Image(systemName: source.iconName)
                    .font(.system(size: 14))
                Text(source.displayName)
                    .font(.system(size: 10))
            }
            .foregroundColor(
                isAvailable
                    ? (isSelected ? .white : ColorPalette.text)
                    : ColorPalette.text.opacity(0.3)
            )
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isSelected ? ColorPalette.primary : Color.clear)
            )
            .padding(2)
        }
        .disabled(!isAvailable)
        .accessibilityIdentifier("AudioSourceButton_\(source.rawValue)")
    }

    // MARK: - Helpers

    /// Get current duration based on selected source
    private var currentDuration: Double {
        guard let recording = viewModel.selectedRecording else { return 1.0 }
        return viewModel.getDuration(for: recording, source: viewModel.currentPlayingSource)
    }

    /// Format time in seconds to MM:SS format
    private func formatTime(_ seconds: Double) -> String {
        let minutes = Int(seconds) / 60
        let remainingSeconds = Int(seconds) % 60
        return String(format: "%d:%02d", minutes, remainingSeconds)
    }
}
