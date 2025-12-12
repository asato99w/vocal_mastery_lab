//
//  RecordingInfoComponents.swift
//  VocalMasteryLab
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
