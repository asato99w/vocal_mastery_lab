//
//  StatisticsComponents.swift
//  VocalisStudio
//
//  Statistics-related UI components for AnalysisView
//  Extracted from AnalysisView.swift for better code organization
//

import SwiftUI
import VocalisDomain

// MARK: - Statistics Sheet View

struct StatisticsSheetView: View {
    let recording: Recording
    let statistics: RecordingStatistics?
    @Environment(\.dismiss) private var dismiss

    // Pitch Analysis section states
    @State private var isPitchAnalysisSectionExpanded: Bool = true
    @State private var isPositionSectionExpanded: Bool = false
    @State private var isPitchSectionExpanded: Bool = false
    @State private var isVibratoSectionExpanded: Bool = false

    // Spectrum Analysis section states
    @State private var isSpectrumAnalysisSectionExpanded: Bool = true

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    if let stats = statistics {
                        // Pitch Analysis Section (major section)
                        pitchAnalysisSection(stats)

                        // Spectrum Analysis Section (major section)
                        spectrumAnalysisSection(stats)
                    } else {
                        noDataView
                    }
                }
                .padding()
            }
            .background(ColorPalette.background)
            .navigationTitle("statistics.title".localized)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(action: { dismiss() }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                    }
                    .accessibilityIdentifier("StatisticsSheetCloseButton")
                    .accessibilityLabel("analysis.close".localized)
                }
            }
        }
        .accessibilityIdentifier("StatisticsSheetView")
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }

    // MARK: - Pitch Analysis Section (Major)

    private func pitchAnalysisSection(_ stats: RecordingStatistics) -> some View {
        VStack(spacing: 0) {
            // Major section header
            Button(action: { withAnimation { isPitchAnalysisSectionExpanded.toggle() } }) {
                HStack {
                    Image(systemName: "waveform.path.ecg")
                        .foregroundColor(ColorPalette.primary)
                        .font(.title3)
                    Text("statistics.pitch_analysis".localized)
                        .font(.title3.bold())
                        .foregroundColor(ColorPalette.text)

                    Spacer()

                    Image(systemName: isPitchAnalysisSectionExpanded ? "chevron.up" : "chevron.down")
                        .foregroundColor(ColorPalette.text.opacity(0.5))
                }
                .padding()
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("PitchAnalysisSectionToggleButton")

            if isPitchAnalysisSectionExpanded {
                VStack(spacing: 12) {
                    // Overall subsection (always expanded within pitch analysis)
                    overallSubsection(stats.overall)

                    // Position subsection (collapsible)
                    if !stats.positionStatistics.isEmpty {
                        positionSection(stats.positionStatistics)
                    }

                    // Pitch subsection (collapsible)
                    if !stats.pitchStatistics.isEmpty {
                        pitchSection(stats.pitchStatistics)
                    }

                    // Vibrato subsection (collapsible)
                    vibratoSection(stats.vibratoStatistics)
                }
                .padding(.horizontal)
                .padding(.bottom)
            }
        }
        .background(ColorPalette.secondary)
        .cornerRadius(12)
        .accessibilityIdentifier("PitchAnalysisSection")
    }

    // MARK: - Spectrum Analysis Section (Major)

    private func spectrumAnalysisSection(_ stats: RecordingStatistics) -> some View {
        VStack(spacing: 0) {
            // Major section header
            Button(action: { withAnimation { isSpectrumAnalysisSectionExpanded.toggle() } }) {
                HStack {
                    Image(systemName: "chart.bar.fill")
                        .foregroundColor(ColorPalette.primary)
                        .font(.title3)
                    Text("statistics.spectrum_analysis".localized)
                        .font(.title3.bold())
                        .foregroundColor(ColorPalette.text)

                    Spacer()

                    Image(systemName: isSpectrumAnalysisSectionExpanded ? "chevron.up" : "chevron.down")
                        .foregroundColor(ColorPalette.text.opacity(0.5))
                }
                .padding()
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("SpectrumAnalysisSectionToggleButton")

            if isSpectrumAnalysisSectionExpanded {
                VStack(spacing: 12) {
                    // High Frequency Resonance subsection
                    highFrequencyResonanceSubsection(
                        sf: stats.singersFormantStatistics,
                        hf: stats.highFrequencyStatistics
                    )
                }
                .padding(.horizontal)
                .padding(.bottom)
            }
        }
        .background(ColorPalette.secondary)
        .cornerRadius(12)
        .accessibilityIdentifier("SpectrumAnalysisSection")
    }

    // MARK: - Overall Subsection (within Pitch Analysis)

    private func overallSubsection(_ overall: RecordingStatistics.OverallStatistics) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("statistics.overall".localized)
                .font(.headline)
                .foregroundColor(ColorPalette.text)

            VStack(spacing: 8) {
                StatisticsRow(
                    label: "statistics.avg_deviation".localized,
                    value: formatCents(overall.averageDeviationCents)
                )
                StatisticsRow(
                    label: "statistics.deviation_stddev".localized,
                    value: "±" + formatCents(overall.deviationStdDev)
                )
                StatisticsRow(
                    label: "statistics.median_deviation".localized,
                    value: formatCents(overall.medianDeviationCents)
                )
                StatisticsRow(
                    label: "statistics.detection_rate".localized,
                    value: formatPercent(overall.detectionRate)
                )

                // Vocal range
                if let lowest = overall.lowestNoteName, let highest = overall.highestNoteName {
                    StatisticsRow(
                        label: "statistics.vocal_range".localized,
                        value: "\(lowest) 〜 \(highest)"
                    )
                }
            }
        }
        .padding()
        .background(ColorPalette.background.opacity(0.5))
        .cornerRadius(8)
        .accessibilityIdentifier("OverallSubsection")
    }

    // MARK: - High Frequency Resonance Subsection

    private func highFrequencyResonanceSubsection(
        sf: RecordingStatistics.SingersFormantStatistics?,
        hf: RecordingStatistics.HighFrequencyStatistics?
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("statistics.high_freq_resonance".localized)
                .font(.headline)
                .foregroundColor(ColorPalette.text)

            // Show data if either SF or HF statistics are available
            if sf != nil || hf != nil {
                VStack(spacing: 10) {
                    // Singer's Formant with bar
                    SpectrumBarRow(
                        label: "SF",
                        bandInfo: "(2.5-3.5kHz)",
                        ratio: sf?.ratioPercentage,
                        maxRatio: 50.0,
                        intensity: sf?.averageIntensity
                    )

                    // Brightness (4-6 kHz)
                    SpectrumBarRow(
                        label: "statistics.brightness_short".localized,
                        bandInfo: "(4-6kHz)",
                        ratio: hf?.brightnessPercentage,
                        maxRatio: 50.0,
                        intensity: nil
                    )

                    // Airiness (6-9 kHz)
                    SpectrumBarRow(
                        label: "statistics.air_short".localized,
                        bandInfo: "(6-9kHz)",
                        ratio: hf?.airinessPercentage,
                        maxRatio: 50.0,
                        intensity: nil
                    )
                }
            } else {
                Text("statistics.sf_no_data".localized)
                    .font(.subheadline)
                    .foregroundColor(ColorPalette.text.opacity(0.5))
            }
        }
        .padding()
        .background(ColorPalette.background.opacity(0.5))
        .cornerRadius(8)
        .accessibilityIdentifier("HighFreqResonanceSubsection")
    }

    // MARK: - Position Subsection

    private func positionSection(_ positions: [RecordingStatistics.PositionStatistics]) -> some View {
        VStack(spacing: 0) {
            // Header with expand/collapse
            Button(action: { withAnimation { isPositionSectionExpanded.toggle() } }) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "list.number")
                            .foregroundColor(ColorPalette.primary)
                        Text("statistics.by_position".localized)
                            .font(.headline)
                            .foregroundColor(ColorPalette.text)

                        Spacer()

                        Image(systemName: isPositionSectionExpanded ? "chevron.up" : "chevron.down")
                            .foregroundColor(ColorPalette.text.opacity(0.5))
                    }

                    if !isPositionSectionExpanded {
                        Text(positionSummaryText(positions))
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                            .padding(.leading, 28) // Align with title (icon width + spacing)
                    }
                }
                .padding()
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("PositionSectionToggleButton")
            .accessibilityAddTraits(.isButton)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("statistics.by_position".localized)

            // Expandable content
            if isPositionSectionExpanded {
                VStack(spacing: 0) {
                    // Column headers
                    HStack(spacing: 8) {
                        Text("#")
                            .font(.caption.bold())
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                            .frame(width: 24, alignment: .leading)

                        Text("statistics.detection".localized)
                            .font(.caption.bold())
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                            .frame(width: 50, alignment: .center)

                        Text("statistics.timing".localized)
                            .font(.caption.bold())
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                            .frame(maxWidth: .infinity, alignment: .center)

                        Text("statistics.pitch_short".localized)
                            .font(.caption.bold())
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                            .frame(maxWidth: .infinity, alignment: .center)
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 6)

                    Divider()
                        .background(ColorPalette.text.opacity(0.2))
                        .padding(.horizontal)

                    ForEach(positions) { position in
                        VStack(spacing: 6) {
                            HStack(spacing: 8) {
                                // Position number
                                Text("\(position.position)")
                                    .font(.subheadline.monospacedDigit())
                                    .foregroundColor(ColorPalette.text)
                                    .frame(width: 24, alignment: .leading)
                                    .accessibilityIdentifier("PositionLabel_\(position.position)")

                                // Detection rate (n/m format) - no color, just data
                                Text("\(position.notesDetected)/\(position.noteOccurrences)")
                                    .font(.subheadline.monospacedDigit())
                                    .foregroundColor(ColorPalette.text.opacity(0.8))
                                    .frame(width: 50, alignment: .center)

                                // Timing meter
                                TimingMeterView(
                                    errorMs: position.averageOnsetErrorMs,
                                    maxErrorMs: 100.0
                                )
                                .frame(maxWidth: .infinity)

                                // Pitch deviation bar (existing style)
                                DeviationBarView(
                                    deviation: position.averageDeviationCents,
                                    maxDeviation: 100.0
                                )
                                .frame(maxWidth: .infinity)
                            }

                            // Second row: numeric values
                            HStack(spacing: 8) {
                                Spacer()
                                    .frame(width: 24)

                                Spacer()
                                    .frame(width: 50)

                                // Timing value
                                Text(formatTimingMs(position.averageOnsetErrorMs))
                                    .font(.caption.monospacedDigit())
                                    .foregroundColor(ColorPalette.text.opacity(0.6))
                                    .frame(maxWidth: .infinity, alignment: .center)

                                // Pitch value
                                Text(formatSignedCents(position.averageDeviationCents))
                                    .font(.caption.monospacedDigit())
                                    .foregroundColor(deviationColor(abs(position.averageDeviationCents)))
                                    .frame(maxWidth: .infinity, alignment: .center)
                            }
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 6)
                        .accessibilityIdentifier("PositionRow_\(position.position)")

                        if position.id != positions.last?.id {
                            Divider()
                                .background(ColorPalette.text.opacity(0.1))
                                .padding(.horizontal)
                        }
                    }
                }
                .padding(.bottom)
                .accessibilityIdentifier("PositionSectionContent")
            }
        }
        .background(ColorPalette.background.opacity(0.5))
        .cornerRadius(8)
    }

    // MARK: - Pitch Subsection

    private func pitchSection(_ pitches: [RecordingStatistics.PitchStatistics]) -> some View {
        // Sort by absolute deviation descending (worst accuracy first)
        let sortedPitches = pitches.sorted { abs($0.averageDeviationCents) > abs($1.averageDeviationCents) }

        return VStack(spacing: 0) {
            // Header with expand/collapse
            Button(action: { withAnimation { isPitchSectionExpanded.toggle() } }) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "music.note")
                            .foregroundColor(ColorPalette.primary)
                        Text("statistics.by_pitch".localized)
                            .font(.headline)
                            .foregroundColor(ColorPalette.text)

                        Spacer()

                        Image(systemName: isPitchSectionExpanded ? "chevron.up" : "chevron.down")
                            .foregroundColor(ColorPalette.text.opacity(0.5))
                    }

                    if !isPitchSectionExpanded {
                        Text(pitchSummaryText(sortedPitches))
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                            .padding(.leading, 28) // Align with title (icon width + spacing)
                    }
                }
                .padding()
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("PitchSectionToggleButton")
            .accessibilityAddTraits(.isButton)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("statistics.by_pitch".localized)

            // Expandable content
            if isPitchSectionExpanded {
                VStack(spacing: 6) {
                    ForEach(Array(sortedPitches.enumerated()), id: \.element.id) { index, pitch in
                        VStack(spacing: 4) {
                            HStack {
                                Text(pitch.noteName)
                                    .font(.subheadline.monospaced())
                                    .foregroundColor(ColorPalette.text)
                                    .frame(width: 45, alignment: .leading)
                                    .accessibilityIdentifier("PitchNoteLabel_\(pitch.noteName)")

                                Spacer()

                                Text(formatSignedCents(pitch.averageDeviationCents))
                                    .font(.subheadline.monospacedDigit())
                                    .foregroundColor(deviationColor(abs(pitch.averageDeviationCents)))

                                Text("±" + formatCents(pitch.deviationStdDev))
                                    .font(.caption.monospacedDigit())
                                    .foregroundColor(ColorPalette.text.opacity(0.6))
                                    .frame(width: 70, alignment: .trailing)

                                Text("\(pitch.occurrenceCount)" + "statistics.times".localized)
                                    .font(.caption)
                                    .foregroundColor(ColorPalette.text.opacity(0.5))
                                    .frame(width: 35, alignment: .trailing)
                            }

                            // Deviation bar
                            DeviationBarView(
                                deviation: pitch.averageDeviationCents,
                                maxDeviation: 100.0  // 100 cents = 1 semitone for full range visibility
                            )
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 4)
                        .accessibilityIdentifier("PitchRow_\(pitch.noteName)")

                        if index != sortedPitches.count - 1 {
                            Divider()
                                .background(ColorPalette.text.opacity(0.1))
                                .padding(.horizontal)
                        }
                    }
                }
                .padding(.bottom)
                .accessibilityIdentifier("PitchSectionContent")
            }
        }
        .background(ColorPalette.background.opacity(0.5))
        .cornerRadius(8)
    }

    // MARK: - Vibrato Subsection

    private func vibratoSection(_ vibrato: RecordingStatistics.VibratoStatistics?) -> some View {
        VStack(spacing: 0) {
            // Header with expand/collapse
            Button(action: { withAnimation { isVibratoSectionExpanded.toggle() } }) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "waveform.path")
                            .foregroundColor(ColorPalette.primary)
                        Text("statistics.vibrato".localized)
                            .font(.headline)
                            .foregroundColor(ColorPalette.text)

                        Spacer()

                        Image(systemName: isVibratoSectionExpanded ? "chevron.up" : "chevron.down")
                            .foregroundColor(ColorPalette.text.opacity(0.5))
                    }

                    if !isVibratoSectionExpanded {
                        Text(vibratoSummaryText(vibrato))
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                            .padding(.leading, 28) // Align with title (icon width + spacing)
                    }
                }
                .padding()
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("VibratoSectionToggleButton")
            .accessibilityAddTraits(.isButton)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("statistics.vibrato".localized)

            // Expandable content
            if isVibratoSectionExpanded {
                if let vibrato = vibrato {
                    VStack(spacing: 10) {
                        StatisticsRow(
                            label: "statistics.vibrato_rate".localized,
                            value: formatHz(vibrato.averageRate)
                        )
                        StatisticsRow(
                            label: "statistics.vibrato_extent".localized,
                            value: formatVibratoExtent(vibrato.averageExtent)
                        )
                        StatisticsRow(
                            label: "statistics.vibrato_regularity".localized,
                            value: formatPercent(Double(vibrato.averageRegularity))
                        )
                        StatisticsRow(
                            label: "statistics.vibrato_presence".localized,
                            value: formatPercent(Double(vibrato.presenceRate))
                        )
                    }
                    .padding(.horizontal)
                    .padding(.bottom)
                    .accessibilityIdentifier("VibratoSectionContent")
                } else {
                    HStack {
                        Text("statistics.vibrato_no_data".localized)
                            .font(.subheadline)
                            .foregroundColor(ColorPalette.text.opacity(0.5))
                        Spacer()
                    }
                    .padding(.horizontal)
                    .padding(.bottom)
                    .accessibilityIdentifier("VibratoSectionNoData")
                }
            }
        }
        .background(ColorPalette.background.opacity(0.5))
        .cornerRadius(8)
    }

    private func vibratoSummaryText(_ vibrato: RecordingStatistics.VibratoStatistics?) -> String {
        guard let vibrato = vibrato else {
            return "statistics.vibrato_no_data".localized
        }
        // Format: "検出率: 75%（6.0 Hz）"
        let presencePercent = String(format: "%.0f%%", vibrato.presenceRate * 100)
        let rateHz = String(format: "%.1f Hz", vibrato.averageRate)
        return "statistics.vibrato_presence".localized + ": " + presencePercent + "（" + rateHz + "）"
    }

    private func formatHz(_ value: Float) -> String {
        return String(format: "%.1f Hz", value)
    }

    private func formatVibratoExtent(_ value: Float) -> String {
        return "±" + String(format: "%.0f", value) + " " + "statistics.cents".localized
    }

    // MARK: - No Data View

    private var noDataView: some View {
        VStack(spacing: 16) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 48))
                .foregroundColor(ColorPalette.text.opacity(0.3))

            Text("statistics.no_data".localized)
                .font(.headline)
                .foregroundColor(ColorPalette.text.opacity(0.6))
        }
        .frame(maxWidth: .infinity, minHeight: 200)
        .background(ColorPalette.secondary)
        .cornerRadius(12)
    }

    // MARK: - Formatting Helpers

    private func formatCents(_ value: Double) -> String {
        if value < 1 {
            return String(format: "%.1f", value) + " " + "statistics.cents".localized
        } else {
            return String(format: "%.1f", value) + " " + "statistics.cents".localized
        }
    }

    private func formatSignedCents(_ value: Double) -> String {
        let sign = value >= 0 ? "+" : ""
        return sign + String(format: "%.1f", value) + " " + "statistics.cents".localized
    }

    private func formatPercent(_ value: Double) -> String {
        return String(format: "%.0f%%", value * 100)
    }

    private func positionSummaryText(_ positions: [RecordingStatistics.PositionStatistics]) -> String {
        guard !positions.isEmpty else { return "" }
        // Sort by absolute deviation to find the worst position
        let sortedByDeviation = positions.sorted { abs($0.averageDeviationCents) > abs($1.averageDeviationCents) }
        let worstDeviation = abs(sortedByDeviation.first!.averageDeviationCents)
        // Format: "16ポジション（46 cents）"
        return "\(positions.count)" + "statistics.positions".localized + "（" + String(format: "%.0f", worstDeviation) + " " + "statistics.cents".localized + "）"
    }

    private func pitchSummaryText(_ pitches: [RecordingStatistics.PitchStatistics]) -> String {
        guard !pitches.isEmpty else { return "" }
        // pitches are already sorted by deviation (worst first)
        let worstDeviation = abs(pitches.first!.averageDeviationCents)
        // Format: "8 notes（46 cents）"
        return "\(pitches.count)" + "statistics.notes_detected".localized + "（" + String(format: "%.0f", worstDeviation) + " " + "statistics.cents".localized + "）"
    }

    /// Returns color based on deviation magnitude
    /// Relaxed thresholds for better user motivation while keeping numeric details visible
    /// - Green: < 30 cents (good - within typical perception threshold)
    /// - Yellow: 30-50 cents (acceptable - noticeable but reasonable)
    /// - Orange: 50-75 cents (needs work - clearly off pitch)
    /// - Red: > 75 cents (poor - approaching semitone error)
    private func deviationColor(_ absDeviation: Double) -> Color {
        if absDeviation < 30 {
            return .green
        } else if absDeviation < 50 {
            return .yellow
        } else if absDeviation < 75 {
            return .orange
        } else {
            return .red
        }
    }

    /// Format timing error in milliseconds with sign
    private func formatTimingMs(_ value: Double) -> String {
        let sign = value >= 0 ? "+" : ""
        return sign + String(format: "%.0fms", value)
    }
}

// MARK: - Timing Meter View

/// Visual meter showing timing error with graduated scale (like a ruler)
/// Center = perfect timing, left = early, right = late
struct TimingMeterView: View {
    let errorMs: Double  // Signed milliseconds (+late, -early)
    let maxErrorMs: Double  // Maximum error for full scale (e.g., 100ms)

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let centerX = width / 2
            let normalizedError = min(max(errorMs / maxErrorMs, -1), 1)
            let indicatorOffset = normalizedError * centerX

            ZStack {
                // Background with graduated marks (メモリ)
                HStack(spacing: 0) {
                    ForEach(0..<11, id: \.self) { index in
                        let isCenter = index == 5
                        let isMajor = index == 0 || index == 5 || index == 10

                        Rectangle()
                            .fill(isCenter ? ColorPalette.text.opacity(0.5) : ColorPalette.text.opacity(0.2))
                            .frame(width: isMajor ? 2 : 1, height: isMajor ? 12 : 8)

                        if index < 10 {
                            Spacer()
                        }
                    }
                }

                // Indicator triangle pointing to the error position
                Triangle()
                    .fill(indicatorColor)
                    .frame(width: 8, height: 6)
                    .offset(x: indicatorOffset, y: 6)
            }
        }
        .frame(height: 18)
    }

    private var indicatorColor: Color {
        // Subtle color - timing is secondary info, pitch is primary
        ColorPalette.text.opacity(0.7)
    }
}

/// Triangle shape for timing indicator
struct Triangle: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.midX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}

// MARK: - Deviation Bar View

/// Visual bar showing pitch deviation direction and magnitude
struct DeviationBarView: View {
    let deviation: Double  // Signed cents (+sharp, -flat)
    let maxDeviation: Double

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let centerX = width / 2
            let normalizedDeviation = min(max(deviation / maxDeviation, -1), 1)
            let barWidth = abs(normalizedDeviation) * centerX

            ZStack(alignment: .center) {
                // Background track
                Rectangle()
                    .fill(ColorPalette.text.opacity(0.1))
                    .frame(height: 6)
                    .cornerRadius(3)

                // Center line indicator
                Rectangle()
                    .fill(ColorPalette.text.opacity(0.3))
                    .frame(width: 2, height: 10)

                // Deviation bar
                if deviation != 0 {
                    Rectangle()
                        .fill(barColor)
                        .frame(width: barWidth, height: 6)
                        .cornerRadius(3)
                        .offset(x: deviation > 0 ? barWidth / 2 : -barWidth / 2)
                }
            }
        }
        .frame(height: 10)
    }

    private var barColor: Color {
        let absDeviation = abs(deviation)
        // Relaxed thresholds matching deviationColor() for consistency
        if absDeviation < 30 {
            return .green
        } else if absDeviation < 50 {
            return .yellow
        } else if absDeviation < 75 {
            return .orange
        } else {
            return .red
        }
    }
}

// MARK: - Statistics Section Component

struct StatisticsSectionView<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(ColorPalette.primary)
                Text(title)
                    .font(.headline)
                    .foregroundColor(ColorPalette.text)
            }

            content
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(ColorPalette.secondary)
        .cornerRadius(12)
    }
}

// MARK: - Statistics Row Component

struct StatisticsRow: View {
    let label: String
    let value: String
    var color: Color? = nil

    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundColor(ColorPalette.text.opacity(0.7))

            Spacer()

            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(color ?? ColorPalette.text)
        }
    }
}

// MARK: - Spectrum Bar Row Component

/// A row displaying spectrum analysis data with a horizontal bar graph
/// Two-line layout: Label on first line, bar graph on second line
struct SpectrumBarRow: View {
    let label: String           // e.g., "SF", "輝き", "空気感"
    let bandInfo: String        // e.g., "(2.5-3.5kHz)"
    let ratio: Float?           // Percentage value (0-100), nil if no data
    let maxRatio: Float         // Maximum value for bar scaling (e.g., 20%)
    let intensity: Float?       // dB value, only shown for SF

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Line 1: Label with band info and values
            HStack {
                // Label + band info (allow shrinking for long translations)
                HStack(spacing: 4) {
                    Text(label)
                        .font(.subheadline.bold())
                        .foregroundColor(ColorPalette.text)
                    Text(bandInfo)
                        .font(.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.5))
                }
                .lineLimit(1)
                .minimumScaleFactor(0.75)

                Spacer()

                // Value display (fixed width to prevent layout shift)
                if let ratio = ratio {
                    // Intensity (dB) - only for SF, shown first (left)
                    if let intensity = intensity {
                        Text(formatIntensity(intensity))
                            .font(.caption.monospacedDigit())
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                    }

                    // Percentage shown last (right)
                    Text(String(format: "%.1f%%", ratio))
                        .font(.subheadline.monospacedDigit())
                        .foregroundColor(ColorPalette.text)
                } else {
                    Text("-")
                        .font(.subheadline)
                        .foregroundColor(ColorPalette.text.opacity(0.3))
                }
            }

            // Line 2: Bar graph (full width)
            GeometryReader { geometry in
                let width = geometry.size.width
                let barWidth = ratio != nil ? CGFloat(min(ratio! / maxRatio, 1.0)) * width : 0

                ZStack(alignment: .leading) {
                    // Background track
                    Rectangle()
                        .fill(ColorPalette.text.opacity(0.1))
                        .frame(height: 10)
                        .cornerRadius(5)

                    // Value bar
                    if ratio != nil && barWidth > 0 {
                        Rectangle()
                            .fill(barColor(ratio!))
                            .frame(width: max(barWidth, 4), height: 10)  // Minimum 4pt for visibility
                            .cornerRadius(5)
                    }
                }
            }
            .frame(height: 10)
        }
        .padding(.vertical, 4)
    }

    private func barColor(_ value: Float) -> Color {
        // Color based on ratio percentage
        // Higher values = stronger presence = more green
        if value >= 10 {
            return .green
        } else if value >= 6 {
            return .yellow
        } else if value >= 3 {
            return .orange
        } else {
            return ColorPalette.primary.opacity(0.5)  // Use primary color for low values
        }
    }

    private func formatIntensity(_ value: Float) -> String {
        let sign = value >= 0 ? "+" : ""
        return sign + String(format: "%.1fdB", value)
    }
}
