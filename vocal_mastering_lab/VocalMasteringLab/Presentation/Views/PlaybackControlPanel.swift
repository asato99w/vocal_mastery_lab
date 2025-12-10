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
                    if let scaleDisplayNameKey = recording.scaleDisplayNameKey,
                       let startNoteName = recording.scaleStartNoteName {
                        Text("\(startNoteName) \(scaleDisplayNameKey.localized)")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .foregroundColor(ColorPalette.text)
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
                    let isPlaying = viewModel.selectedRecording.map { viewModel.playingRecordingId == $0.id } ?? false
                    Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 44))
                        .foregroundColor(viewModel.selectedRecording != nil ? ColorPalette.primary : ColorPalette.primary.opacity(0.3))
                }
                .disabled(viewModel.selectedRecording == nil)
                .accessibilityIdentifier("PlaybackControlPanel_PlayPauseButton")
                .accessibilityLabel(viewModel.selectedRecording.map { viewModel.playingRecordingId == $0.id } ?? false ? "pause".localized : "play".localized)

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
                                Task {
                                    await viewModel.seek(to: newValue, for: recording.id)
                                }
                            }
                        }
                    ),
                    in: 0...max(viewModel.selectedRecording?.duration.seconds ?? 1.0, 0.1)
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

                    Text(formatTime(viewModel.selectedRecording?.duration.seconds ?? 0.0))
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
        .accessibilityIdentifier("PlaybackControlPanel")
    }

    /// Format time in seconds to MM:SS format
    private func formatTime(_ seconds: Double) -> String {
        let minutes = Int(seconds) / 60
        let remainingSeconds = Int(seconds) % 60
        return String(format: "%d:%02d", minutes, remainingSeconds)
    }
}
