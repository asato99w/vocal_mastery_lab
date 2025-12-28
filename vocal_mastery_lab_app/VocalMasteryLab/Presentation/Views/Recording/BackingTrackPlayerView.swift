import SwiftUI

/// Backing track player control component
/// Displays track info, playback position, and control buttons
struct BackingTrackPlayerView: View {
    let trackName: String
    let sourceName: String
    let isPlaying: Bool
    let currentTime: TimeInterval
    let duration: TimeInterval
    let onTogglePlayback: () -> Void
    let onSeek: (TimeInterval) -> Void
    let onStop: () -> Void

    @State private var isDragging = false
    @State private var dragProgress: Double = 0

    var body: some View {
        VStack(spacing: 8) {
            // Track info row
            HStack {
                Image(systemName: "music.note")
                    .foregroundColor(ColorPalette.primary)
                Text("\(trackName) (\(sourceName))")
                    .font(.subheadline)
                    .foregroundColor(ColorPalette.text)
                    .lineLimit(1)
                Spacer()
            }
            .accessibilityIdentifier("BackingTrackInfoLabel")

            // Progress bar with time display
            VStack(spacing: 4) {
                // Seek slider
                GeometryReader { geometry in
                    let progress = isDragging ? dragProgress : (duration > 0 ? currentTime / duration : 0)

                    ZStack(alignment: .leading) {
                        // Background track
                        RoundedRectangle(cornerRadius: 2)
                            .fill(ColorPalette.secondary)
                            .frame(height: 4)

                        // Progress fill
                        RoundedRectangle(cornerRadius: 2)
                            .fill(ColorPalette.primary)
                            .frame(width: geometry.size.width * CGFloat(progress), height: 4)

                        // Thumb
                        Circle()
                            .fill(ColorPalette.primary)
                            .frame(width: 12, height: 12)
                            .offset(x: geometry.size.width * CGFloat(progress) - 6)
                    }
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                isDragging = true
                                let newProgress = max(0, min(1, value.location.x / geometry.size.width))
                                dragProgress = newProgress
                            }
                            .onEnded { value in
                                let newProgress = max(0, min(1, value.location.x / geometry.size.width))
                                let newTime = newProgress * duration
                                onSeek(newTime)
                                isDragging = false
                            }
                    )
                }
                .frame(height: 20)
                .accessibilityIdentifier("BackingTrackSeekSlider")

                // Time labels
                HStack {
                    Text(formatTime(currentTime))
                        .font(.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.6))
                        .accessibilityIdentifier("BackingTrackCurrentTimeLabel")

                    Spacer()

                    Text(formatTime(duration))
                        .font(.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.6))
                        .accessibilityIdentifier("BackingTrackDurationLabel")
                }
            }

            // Control buttons
            HStack(spacing: 16) {
                // Stop button
                Button(action: onStop) {
                    Image(systemName: "stop.fill")
                        .font(.title3)
                        .foregroundColor(ColorPalette.text.opacity(0.8))
                }
                .accessibilityIdentifier("BackingTrackStopButton")

                // Play/Pause button
                Button(action: onTogglePlayback) {
                    Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                        .font(.title2)
                        .foregroundColor(ColorPalette.primary)
                }
                .accessibilityIdentifier("BackingTrackPlayPauseButton")

                Spacer()

                // Playing indicator
                if isPlaying {
                    HStack(spacing: 4) {
                        Image(systemName: "speaker.wave.2.fill")
                            .font(.caption)
                        Text("再生中")
                            .font(.caption)
                    }
                    .foregroundColor(ColorPalette.primary)
                    .accessibilityIdentifier("BackingTrackPlayingIndicator")
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(ColorPalette.background)
                .shadow(color: Color.black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(ColorPalette.primary.opacity(0.3), lineWidth: 1)
        )
        .accessibilityIdentifier("BackingTrackPlayerView")
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

// MARK: - Preview

#if DEBUG
struct BackingTrackPlayerView_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 20) {
            // Playing state
            BackingTrackPlayerView(
                trackName: "練習曲1",
                sourceName: "伴奏",
                isPlaying: true,
                currentTime: 45.0,
                duration: 180.0,
                onTogglePlayback: {},
                onSeek: { _ in },
                onStop: {}
            )

            // Paused state
            BackingTrackPlayerView(
                trackName: "My Song",
                sourceName: "元音源",
                isPlaying: false,
                currentTime: 0,
                duration: 120.0,
                onTogglePlayback: {},
                onSeek: { _ in },
                onStop: {}
            )
        }
        .padding()
        .background(Color.gray.opacity(0.2))
    }
}
#endif
