//
//  PlaybackComponents.swift
//  VocalisStudio
//
//  Playback control UI components for AnalysisView
//  Extracted from AnalysisView.swift for better code organization
//

import SwiftUI

// MARK: - Compact Playback Control

struct CompactPlaybackControl: View {
    let isPlaying: Bool
    let onPlayPause: () -> Void

    var body: some View {
        HStack {
            Button(action: onPlayPause) {
                Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                    .font(.title2)
                    .foregroundColor(ColorPalette.primary)
            }
            .accessibilityIdentifier("ExpandedAnalysisPlayPauseButton")

            Text(isPlaying ? "analysis.playing".localized : "analysis.paused".localized)
                .font(.caption)
                .foregroundColor(ColorPalette.text.opacity(0.6))
        }
    }
}

// MARK: - Playback Control

struct PlaybackControl: View {
    let isPlaying: Bool
    let currentTime: Double
    let duration: Double
    let onPlayPause: () -> Void
    let onSeek: (Double) -> Void

    var body: some View {
        VStack(spacing: 8) {
            Text("analysis.playback_title".localized)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(ColorPalette.text)

            // Playback buttons
            HStack(spacing: 20) {
                Button(action: { onSeek(max(0, currentTime - 5)) }) {
                    Image(systemName: "backward.fill")
                        .font(.callout)
                        .foregroundColor(ColorPalette.primary)
                }
                .accessibilityIdentifier("AnalysisSeekBackButton")

                Button(action: onPlayPause) {
                    Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 40))
                        .foregroundColor(ColorPalette.primary)
                        .frame(width: 60, height: 60)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("AnalysisPlayPauseButton")
                .accessibilityValue(isPlaying ? "playing" : "paused")

                Button(action: { onSeek(min(duration, currentTime + 5)) }) {
                    Image(systemName: "forward.fill")
                        .font(.callout)
                        .foregroundColor(ColorPalette.primary)
                }
                .accessibilityIdentifier("AnalysisSeekForwardButton")
            }

            // Progress bar
            VStack(spacing: 3) {
                Slider(value: Binding(
                    get: { currentTime },
                    set: { onSeek($0) }
                ), in: 0...duration)
                .tint(ColorPalette.primary)
                .accessibilityIdentifier("AnalysisProgressSlider")

                HStack {
                    Text(formatTime(currentTime))
                    Spacer()
                    Text(formatTime(duration))
                }
                .font(.caption2)
                .foregroundColor(ColorPalette.text.opacity(0.6))
            }
        }
        .padding(10)
        .background(ColorPalette.secondary)
        .cornerRadius(8)
    }

    private func formatTime(_ seconds: Double) -> String {
        let minutes = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%02d:%02d", minutes, secs)
    }
}
