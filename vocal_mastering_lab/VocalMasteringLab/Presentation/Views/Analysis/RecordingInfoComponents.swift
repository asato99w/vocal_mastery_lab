//
//  RecordingInfoComponents.swift
//  VocalMasteringLab
//
//  Recording information UI components for AnalysisView
//  Extracted from AnalysisView.swift for better code organization
//

import SwiftUI
import VocalisDomain

// MARK: - Recording Info Panel (Landscape)

struct RecordingInfoPanel: View {
    let recording: Recording
    @Binding var showStatisticsSheet: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Header with title and statistics button
            HStack {
                Text("analysis.info_title".localized)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(ColorPalette.text)

                Spacer()

                Button(action: { showStatisticsSheet = true }) {
                    Image(systemName: "chart.bar.xaxis")
                        .font(.system(size: 16))
                        .foregroundColor(ColorPalette.primary)
                }
                .accessibilityLabel("analysis.statistics_button".localized)
                .accessibilityIdentifier("StatisticsButton")
            }

            // Recording title or date
            if let title = recording.title, !title.isEmpty {
                Text(title)
                    .font(.headline)
                    .foregroundColor(ColorPalette.text)
                    .lineLimit(1)
            }

            // Date and duration row
            HStack(spacing: 12) {
                Label(formatDate(recording.createdAt), systemImage: "calendar")
                Label(recording.duration.formatted, systemImage: "clock")
            }
            .font(.caption)
            .foregroundColor(ColorPalette.text.opacity(0.7))

            Divider()
                .background(ColorPalette.text.opacity(0.2))

            // Scale settings (if available)
            if let settings = recording.scaleSettings {
                VStack(alignment: .leading, spacing: 6) {
                    // Scale pattern and start note
                    HStack {
                        Image(systemName: "music.note.list")
                            .foregroundColor(ColorPalette.primary)
                        Text(settings.notePattern.displayNameKey.localized)
                        Spacer()
                        Text(settings.startNote.noteName)
                            .fontWeight(.medium)
                    }
                    .font(.caption)

                    // Tempo
                    HStack {
                        Image(systemName: "metronome")
                            .foregroundColor(ColorPalette.primary)
                        Text("\(Int(60.0 / settings.tempo.secondsPerNote)) " + "recording.tempo_unit".localized)
                        Spacer()
                    }
                    .font(.caption)

                    // Key progression
                    HStack {
                        Image(systemName: "arrow.up.arrow.down")
                            .foregroundColor(ColorPalette.primary)
                        Text("↑\(settings.ascendingKeyCount) ↓\(settings.descendingKeyCount)")
                        Spacer()
                    }
                    .font(.caption)
                }
                .foregroundColor(ColorPalette.text.opacity(0.8))
            } else {
                Text("analysis.no_scale".localized)
                    .font(.caption)
                    .foregroundColor(ColorPalette.text.opacity(0.5))
                    .italic()
            }
        }
        .padding(12)
        .background(ColorPalette.secondary)
        .cornerRadius(10)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("RecordingInfoPanel")
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MM/dd HH:mm"
        return formatter.string(from: date)
    }
}

// MARK: - Recording Info Compact (Portrait)

struct RecordingInfoCompact: View {
    let recording: Recording
    @Binding var showStatisticsSheet: Bool

    var body: some View {
        VStack(spacing: 10) {
            // Header row
            HStack {
                // Title or date
                if let title = recording.title, !title.isEmpty {
                    Text(title)
                        .font(.headline)
                        .foregroundColor(ColorPalette.text)
                        .lineLimit(1)
                } else {
                    Text(formatDateFull(recording.createdAt))
                        .font(.headline)
                        .foregroundColor(ColorPalette.text)
                }

                Spacer()

                // Statistics button
                Button(action: { showStatisticsSheet = true }) {
                    HStack(spacing: 4) {
                        Image(systemName: "chart.bar.xaxis")
                        Text("analysis.statistics_short".localized)
                    }
                    .font(.subheadline)
                    .foregroundColor(ColorPalette.primary)
                }
                .accessibilityLabel("analysis.statistics_button".localized)
                .accessibilityIdentifier("StatisticsButtonCompact")
            }

            // Info pills row
            HStack(spacing: 8) {
                // Duration pill
                InfoPill(icon: "clock", text: recording.duration.formatted)

                // Scale info pills (if available)
                if let settings = recording.scaleSettings {
                    InfoPill(
                        icon: "music.note.list",
                        text: settings.notePattern.displayNameKey.localized
                    )
                    InfoPill(
                        icon: "music.note",
                        text: "\(settings.startNote.noteName) \(Int(60.0 / settings.tempo.secondsPerNote))BPM"
                    )
                }

                Spacer()
            }
        }
        .padding(12)
        .background(ColorPalette.secondary)
        .cornerRadius(10)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("RecordingInfoCompact")
    }

    private func formatDateFull(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

// MARK: - Info Pill Component

struct InfoPill: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption2)
            Text(text)
                .font(.caption)
        }
        .foregroundColor(ColorPalette.text.opacity(0.7))
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(ColorPalette.background.opacity(0.5))
        .cornerRadius(12)
    }
}
