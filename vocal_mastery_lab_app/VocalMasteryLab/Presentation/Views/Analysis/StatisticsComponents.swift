//
//  StatisticsComponents.swift
//  VocalMasteryLab
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
    @State private var isIntonationExpanded: Bool = false
    @State private var isPitchStabilityExpanded: Bool = false
    @State private var isVocalRangeExpanded: Bool = false
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
                    // Intonation subsection
                    intonationSubsection(stats.intonation)

                    // Pitch Stability subsection
                    pitchStabilitySubsection(stats.pitchStability)

                    // Vocal Range subsection
                    vocalRangeSubsection(stats.vocalRange, detectionRate: stats.detectionRate)

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

    // MARK: - Intonation Subsection (Collapsible)

    private func intonationSubsection(_ intonation: RecordingStatistics.IntonationStatistics) -> some View {
        VStack(spacing: 0) {
            // Header with expand/collapse
            Button(action: { withAnimation { isIntonationExpanded.toggle() } }) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "tuningfork")
                            .foregroundColor(ColorPalette.primary)
                        Text("statistics.intonation".localized)
                            .font(.headline)
                            .foregroundColor(ColorPalette.text)

                        Spacer()

                        Image(systemName: isIntonationExpanded ? "chevron.up" : "chevron.down")
                            .foregroundColor(ColorPalette.text.opacity(0.5))
                    }

                    if !isIntonationExpanded {
                        Text(intonationSummaryText(intonation))
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                            .padding(.leading, 28)
                    }
                }
                .padding()
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("IntonationSectionToggleButton")

            // Expandable content
            if isIntonationExpanded {
                VStack(spacing: 8) {
                    StatisticsRow(
                        label: "statistics.avg_deviation".localized,
                        value: formatCents(intonation.averageDeviationCents),
                        color: deviationColor(intonation.averageDeviationCents)
                    )
                    StatisticsRow(
                        label: "statistics.deviation_stddev".localized,
                        value: "±" + formatCents(intonation.deviationStdDev)
                    )
                    StatisticsRow(
                        label: "statistics.accuracy_rate".localized,
                        value: formatPercent(intonation.accuracyRate),
                        color: accuracyColor(intonation.accuracyRate)
                    )
                    StatisticsRow(
                        label: "statistics.excellent_accuracy".localized,
                        value: formatPercent(intonation.excellentAccuracyRate),
                        color: accuracyColor(intonation.excellentAccuracyRate)
                    )
                }
                .padding(.horizontal)
                .padding(.bottom)
            }
        }
        .background(ColorPalette.background.opacity(0.5))
        .cornerRadius(8)
        .accessibilityIdentifier("IntonationSubsection")
    }

    private func intonationSummaryText(_ intonation: RecordingStatistics.IntonationStatistics) -> String {
        // Format: "正確率: 85%（平均 12.5 cents）"
        let accuracyPercent = String(format: "%.0f%%", intonation.accuracyRate * 100)
        let avgDeviation = String(format: "%.1f", intonation.averageDeviationCents)
        return "statistics.accuracy_rate".localized + ": " + accuracyPercent + "（" + avgDeviation + " " + "statistics.cents".localized + "）"
    }

    // MARK: - Pitch Stability Subsection (Collapsible)

    private func pitchStabilitySubsection(_ stability: RecordingStatistics.PitchStabilityStatistics) -> some View {
        VStack(spacing: 0) {
            // Header with expand/collapse
            Button(action: { withAnimation { isPitchStabilityExpanded.toggle() } }) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "waveform.path")
                            .foregroundColor(ColorPalette.primary)
                        Text("statistics.pitch_stability".localized)
                            .font(.headline)
                            .foregroundColor(ColorPalette.text)

                        Spacer()

                        Image(systemName: isPitchStabilityExpanded ? "chevron.up" : "chevron.down")
                            .foregroundColor(ColorPalette.text.opacity(0.5))
                    }

                    if !isPitchStabilityExpanded {
                        Text(pitchStabilitySummaryText(stability))
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                            .padding(.leading, 28)
                    }
                }
                .padding()
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("PitchStabilitySectionToggleButton")

            // Expandable content
            if isPitchStabilityExpanded {
                if stability.segmentsAnalyzed > 0 {
                    VStack(spacing: 8) {
                        StatisticsRow(
                            label: "statistics.avg_fluctuation".localized,
                            value: formatCents(stability.averageFluctuation),
                            color: fluctuationColor(stability.averageFluctuation)
                        )
                        StatisticsRow(
                            label: "statistics.stability_rate".localized,
                            value: formatPercent(stability.stabilityRate),
                            color: accuracyColor(stability.stabilityRate)
                        )
                        StatisticsRow(
                            label: "statistics.segments_analyzed".localized,
                            value: "\(stability.segmentsAnalyzed)"
                        )
                    }
                    .padding(.horizontal)
                    .padding(.bottom)
                } else {
                    HStack {
                        Text("statistics.stability_no_data".localized)
                            .font(.subheadline)
                            .foregroundColor(ColorPalette.text.opacity(0.5))
                        Spacer()
                    }
                    .padding(.horizontal)
                    .padding(.bottom)
                }
            }
        }
        .background(ColorPalette.background.opacity(0.5))
        .cornerRadius(8)
        .accessibilityIdentifier("PitchStabilitySubsection")
    }

    private func pitchStabilitySummaryText(_ stability: RecordingStatistics.PitchStabilityStatistics) -> String {
        guard stability.segmentsAnalyzed > 0 else {
            return "statistics.stability_no_data".localized
        }
        // Format: "安定率: 75%（揺らぎ 8.5 cents）"
        let stabilityPercent = String(format: "%.0f%%", stability.stabilityRate * 100)
        let fluctuation = String(format: "%.1f", stability.averageFluctuation)
        return "statistics.stability_rate".localized + ": " + stabilityPercent + "（" + fluctuation + " " + "statistics.cents".localized + "）"
    }

    // MARK: - Vocal Range Subsection (Collapsible)

    private func vocalRangeSubsection(_ range: RecordingStatistics.VocalRangeStatistics, detectionRate: Double) -> some View {
        VStack(spacing: 0) {
            // Header with expand/collapse
            Button(action: { withAnimation { isVocalRangeExpanded.toggle() } }) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "music.note.list")
                            .foregroundColor(ColorPalette.primary)
                        Text("statistics.vocal_range".localized)
                            .font(.headline)
                            .foregroundColor(ColorPalette.text)

                        Spacer()

                        Image(systemName: isVocalRangeExpanded ? "chevron.up" : "chevron.down")
                            .foregroundColor(ColorPalette.text.opacity(0.5))
                    }

                    if !isVocalRangeExpanded {
                        Text(vocalRangeSummaryText(range))
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                            .padding(.leading, 28)
                    }
                }
                .padding()
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("VocalRangeSectionToggleButton")

            // Expandable content
            if isVocalRangeExpanded {
                VStack(spacing: 8) {
                    // Vocal range
                    if let lowest = range.lowestNoteName, let highest = range.highestNoteName {
                        StatisticsRow(
                            label: "statistics.range".localized,
                            value: "\(lowest) 〜 \(highest)"
                        )
                    }

                    StatisticsRow(
                        label: "statistics.range_semitones".localized,
                        value: "\(range.rangeSemitones) " + "statistics.semitones".localized
                    )

                    if let center = range.centerNoteName {
                        StatisticsRow(
                            label: "statistics.center_note".localized,
                            value: center
                        )
                    }

                    if let mostUsed = range.mostUsedNote {
                        StatisticsRow(
                            label: "statistics.most_used_note".localized,
                            value: mostUsed
                        )
                    }

                    StatisticsRow(
                        label: "statistics.detection_rate".localized,
                        value: formatPercent(detectionRate)
                    )
                }
                .padding(.horizontal)
                .padding(.bottom)
            }
        }
        .background(ColorPalette.background.opacity(0.5))
        .cornerRadius(8)
        .accessibilityIdentifier("VocalRangeSubsection")
    }

    private func vocalRangeSummaryText(_ range: RecordingStatistics.VocalRangeStatistics) -> String {
        // Format: "C3 〜 G4（16半音）"
        if let lowest = range.lowestNoteName, let highest = range.highestNoteName {
            return "\(lowest) 〜 \(highest)（\(range.rangeSemitones) " + "statistics.semitones".localized + "）"
        }
        return "\(range.rangeSemitones) " + "statistics.semitones".localized
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

    /// Returns color based on accuracy rate (higher is better)
    /// - Green: >= 80% (excellent)
    /// - Yellow: 60-80% (good)
    /// - Orange: 40-60% (needs work)
    /// - Red: < 40% (poor)
    private func accuracyColor(_ rate: Double) -> Color {
        if rate >= 0.8 {
            return .green
        } else if rate >= 0.6 {
            return .yellow
        } else if rate >= 0.4 {
            return .orange
        } else {
            return .red
        }
    }

    /// Returns color based on fluctuation in cents (lower is better)
    /// - Green: < 10 cents (excellent stability)
    /// - Yellow: 10-20 cents (good stability)
    /// - Orange: 20-30 cents (moderate instability)
    /// - Red: > 30 cents (significant wavering)
    private func fluctuationColor(_ fluctuation: Double) -> Color {
        if fluctuation < 10 {
            return .green
        } else if fluctuation < 20 {
            return .yellow
        } else if fluctuation < 30 {
            return .orange
        } else {
            return .red
        }
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
