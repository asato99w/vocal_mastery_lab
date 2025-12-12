import SwiftUI
import VocalisDomain

/// Real-time audio visualization area (spectrum and indicators)
struct RealtimeDisplayArea: View {
    let recordingState: RecordingState
    let isPlayingRecording: Bool
    let targetPitch: DetectedPitch?
    let detectedPitch: DetectedPitch?
    let pitchAccuracy: PitchAccuracy
    let spectrum: [Float]?
    let audioLevel: Float  // dB value (-160 to 0)
    let isSettingsPanelVisible: Bool

    /// Whether to show the realtime display content
    /// Hidden only when settings panel is visible (to save space)
    private var shouldShowContent: Bool {
        !isSettingsPanelVisible
    }

    /// Whether audio visualization is actively running
    private var isActive: Bool {
        recordingState == .recording || isPlayingRecording
    }

    var body: some View {
        if shouldShowContent {
            VStack(spacing: 12) {
                // Frequency spectrum bar chart
                VStack(alignment: .leading, spacing: 6) {
                    Text("recording.realtime_spectrum_title".localized)
                        .font(.subheadline)
                        .fontWeight(.semibold)

                    FrequencySpectrumView(
                        spectrum: spectrum,
                        isActive: isActive
                    )
                    .frame(maxHeight: .infinity)
                }

                Divider()

                // Indicators (audio level + pitch)
                VStack(alignment: .leading, spacing: 6) {
                    Text("recording.indicator_title".localized)
                        .font(.subheadline)
                        .fontWeight(.semibold)

                    IndicatorView(
                        isActive: isActive,
                        targetPitch: targetPitch,
                        detectedPitch: detectedPitch,
                        pitchAccuracy: pitchAccuracy,
                        audioLevel: audioLevel
                    )
                }
            }
            .padding(12)
        } else {
            // Empty state when not recording or playing - minimal space usage
            EmptyView()
        }
    }
}

// MARK: - Frequency Spectrum View

/// Frequency spectrum bar chart view with real-time audio visualization
struct FrequencySpectrumView: View {
    let spectrum: [Float]?
    let isActive: Bool

    private let minFreq: Double = 100.0  // Hz
    private let maxFreq: Double = 800.0  // Hz

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                guard let spectrum = spectrum, !spectrum.isEmpty else {
                    // Draw placeholder when no spectrum data
                    drawPlaceholder(context: context, size: size)
                    return
                }

                let barCount = spectrum.count
                let barWidth = size.width / CGFloat(barCount)
                let maxMagnitude = spectrum.max() ?? 1.0

                for (index, magnitude) in spectrum.enumerated() {
                    let normalizedHeight = maxMagnitude > 0 ? CGFloat(magnitude / maxMagnitude) : 0
                    let barHeight = normalizedHeight * size.height

                    let rect = CGRect(
                        x: CGFloat(index) * barWidth,
                        y: size.height - barHeight,
                        width: max(barWidth - 1, 1),
                        height: barHeight
                    )

                    // Color gradient based on magnitude: blue -> green -> red
                    let color = magnitudeColor(normalizedMagnitude: normalizedHeight)
                    context.fill(Path(rect), with: .color(color))
                }

                // Draw frequency labels
                drawFrequencyLabels(context: context, size: size)
            }
        }
        .background(ColorPalette.background)
        .cornerRadius(8)
    }

    private func drawPlaceholder(context: GraphicsContext, size: CGSize) {
        // Draw subtle grid for inactive state
        let gridColor = ColorPalette.text.opacity(0.1)
        for i in 0..<10 {
            let y = CGFloat(i) * size.height / 10
            var path = Path()
            path.move(to: CGPoint(x: 0, y: y))
            path.addLine(to: CGPoint(x: size.width, y: y))
            context.stroke(path, with: .color(gridColor), lineWidth: 0.5)
        }
    }

    private func drawFrequencyLabels(context: GraphicsContext, size: CGSize) {
        let labelColor = ColorPalette.text.opacity(0.6)
        // Show key frequencies for readability (100, 200, 400, 800 Hz)
        let labeledFrequencies = [100, 200, 400, 800]

        for freq in labeledFrequencies {
            let ratio = (Double(freq) - minFreq) / (maxFreq - minFreq)
            let x = CGFloat(ratio) * size.width

            // Draw tick mark
            var path = Path()
            path.move(to: CGPoint(x: x, y: size.height - 8))
            path.addLine(to: CGPoint(x: x, y: size.height))
            context.stroke(path, with: .color(labelColor), lineWidth: 1)

            // Draw frequency label text
            let labelText = Text("\(freq)")
                .font(.system(size: 9))
                .foregroundColor(labelColor)

            // Position label: left-align first label, right-align last, center others
            let labelPoint = CGPoint(x: x, y: size.height - 12)
            let anchor: UnitPoint
            if freq == labeledFrequencies.first {
                anchor = .bottomLeading  // Left edge: align to left
            } else if freq == labeledFrequencies.last {
                anchor = .bottomTrailing  // Right edge: align to right
            } else {
                anchor = .bottom  // Center labels
            }
            context.draw(labelText, at: labelPoint, anchor: anchor)
        }
    }

    private func magnitudeColor(normalizedMagnitude: CGFloat) -> Color {
        if normalizedMagnitude < 0.33 {
            // Low: Blue
            let ratio = normalizedMagnitude / 0.33
            return Color(
                red: 0,
                green: ratio * 0.5,
                blue: 1.0
            )
        } else if normalizedMagnitude < 0.66 {
            // Medium: Blue -> Green
            let ratio = (normalizedMagnitude - 0.33) / 0.33
            return Color(
                red: 0,
                green: 0.5 + ratio * 0.5,
                blue: 1.0 - ratio
            )
        } else {
            // High: Green -> Red
            let ratio = (normalizedMagnitude - 0.66) / 0.34
            return Color(
                red: ratio,
                green: 1.0 - ratio * 0.5,
                blue: 0
            )
        }
    }
}

// MARK: - Indicator View

/// Combined indicator view displaying audio level and pitch in compact format
struct IndicatorView: View {
    let isActive: Bool
    let targetPitch: DetectedPitch?
    let detectedPitch: DetectedPitch?
    let pitchAccuracy: PitchAccuracy
    let audioLevel: Float  // dB value (-160 to 0)

    var body: some View {
        VStack(spacing: 8) {
            // Audio level row
            audioLevelRow

            // Pitch row (compact: target → detected  diff)
            pitchRow
        }
        .padding(12)
        .frame(maxWidth: .infinity)
        .background(ColorPalette.secondary)
        .cornerRadius(8)
    }

    // MARK: - Audio Level Row

    private var audioLevelRow: some View {
        HStack(spacing: 8) {
            Text("recording.audio_level".localized)
                .font(.caption)
                .foregroundColor(ColorPalette.text.opacity(0.6))
                .frame(width: 50, alignment: .leading)

            // Level meter bar
            AudioLevelMeterView(level: audioLevel)
                .frame(height: 16)

            // dB value
            Text(String(format: "%+.0f dB", audioLevel))
                .font(.caption)
                .foregroundColor(ColorPalette.text.opacity(0.6))
                .frame(width: 50, alignment: .trailing)
        }
    }

    // MARK: - Pitch Row (Compact)

    private var pitchRow: some View {
        HStack(spacing: 8) {
            Text("recording.pitch_label".localized)
                .font(.caption)
                .foregroundColor(ColorPalette.text.opacity(0.6))
                .frame(width: 50, alignment: .leading)

            // Target note (fixed width for stable arrow position)
            Text(targetPitch?.noteName ?? "--")
                .font(.callout)
                .fontWeight(.bold)
                .foregroundColor(targetPitch != nil ? ColorPalette.accent : ColorPalette.text.opacity(0.6))
                .frame(width: 36, alignment: .trailing)
                .accessibilityIdentifier(targetPitch != nil ? "TargetPitchNoteName" : "TargetPitchEmpty")

            Text("→")
                .font(.caption)
                .foregroundColor(ColorPalette.text.opacity(0.4))

            // Detected note (fixed width for consistency)
            if isActive, let detected = detectedPitch {
                HStack(spacing: 4) {
                    Circle()
                        .fill(accuracyColor)
                        .frame(width: 8, height: 8)

                    Text(detected.noteName)
                        .font(.callout)
                        .fontWeight(.bold)
                        .foregroundColor(ColorPalette.text)
                        .frame(width: 36, alignment: .leading)
                        .accessibilityIdentifier("DetectedPitchNoteName")

                    // Cents deviation
                    if let cents = detected.cents {
                        Text(cents >= 0 ? "+\(cents)¢" : "\(cents)¢")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(accuracyColor)
                    }
                }
            } else {
                Text(isActive ? "..." : "--")
                    .font(.callout)
                    .foregroundColor(ColorPalette.text.opacity(0.6))
                    .frame(width: 36, alignment: .leading)
                    .accessibilityIdentifier("DetectedPitchEmpty")
            }

            Spacer()
        }
    }

    // MARK: - Accuracy Color

    private var accuracyColor: Color {
        switch pitchAccuracy {
        case .accurate: return Color.green
        case .slightlyOff: return Color.orange
        case .off: return Color.red
        case .none: return ColorPalette.text.opacity(0.4)
        }
    }
}

// MARK: - Audio Level Meter View

/// Horizontal bar meter for audio level visualization
struct AudioLevelMeterView: View {
    let level: Float  // dB value (-160 to 0)

    // Normalize -60dB to 0dB range to 0.0 to 1.0
    private var normalizedLevel: CGFloat {
        let clamped = max(-60, min(0, level))
        return CGFloat((clamped + 60) / 60)
    }

    private var meterColor: Color {
        if normalizedLevel > 0.9 { return .red }       // -6dB to 0dB: clip danger
        if normalizedLevel > 0.8 { return .yellow }    // -12dB to -6dB: caution
        return .green                                   // below -12dB: normal
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Background
                Rectangle()
                    .fill(Color.gray.opacity(0.3))

                // Level bar
                Rectangle()
                    .fill(meterColor)
                    .frame(width: geometry.size.width * normalizedLevel)
            }
        }
        .cornerRadius(4)
    }
}
