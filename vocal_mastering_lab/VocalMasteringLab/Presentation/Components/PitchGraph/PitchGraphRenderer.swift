import SwiftUI
import VocalisDomain

/// Pitch graph rendering logic
/// Handles all drawing operations for pitch graph visualization
public class PitchGraphRenderer {
    private let coordinateSystem: PitchGraphCoordinateSystem

    /// Note names for frequency to note name conversion
    private static let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    /// Convert frequency to the nearest note name
    /// - Parameter frequency: Frequency in Hz
    /// - Returns: Note name string (e.g., "C4", "G3")
    func frequencyToNoteName(_ frequency: Double) -> String {
        guard frequency > 0 else { return "" }
        // Convert frequency to MIDI note number
        // MIDI note = 12 * log2(freq / 440) + 69
        let midiNote = 12.0 * log2(frequency / 440.0) + 69.0
        let roundedMidi = Int(midiNote.rounded())
        guard roundedMidi >= 0 && roundedMidi <= 127 else { return "" }

        let noteIndex = roundedMidi % 12
        let octave = (roundedMidi / 12) - 1
        return "\(Self.noteNames[noteIndex])\(octave)"
    }

    // MARK: - Initialization

    public init(coordinateSystem: PitchGraphCoordinateSystem = PitchGraphCoordinateSystem()) {
        self.coordinateSystem = coordinateSystem
    }

    // MARK: - Main Drawing

    /// Draw pitch data points and lines with volume-based coloring
    /// - Parameters:
    ///   - context: Graphics context
    ///   - canvasHeight: Canvas height
    ///   - pitchData: Array of (time, frequency, confidence, amplitude) tuples
    ///   - leftPadding: Left padding for canvas
    ///   - targetSegments: Optional array of NoteSegment for pitch accuracy coloring (not used when volume coloring is enabled)
    public func drawPitchData(
        context: GraphicsContext,
        canvasHeight: CGFloat,
        pitchData: [(time: Double, frequency: Double, confidence: Float, amplitude: Float)],
        leftPadding: CGFloat,
        targetSegments: [NoteSegment]? = nil
    ) {
        guard !pitchData.isEmpty else { return }

        // Draw lines connecting pitch points with gap detection
        var path = Path()
        var previousTime: Double?

        for point in pitchData {
            let x = coordinateSystem.timeToCanvasX(time: point.time, leftPadding: leftPadding)
            let y = coordinateSystem.frequencyToCanvasY(frequency: point.frequency, canvasHeight: canvasHeight)

            // Gap detection: start new segment if time gap exceeds threshold
            let shouldStartNewSegment: Bool
            if let prevTime = previousTime {
                shouldStartNewSegment = (point.time - prevTime) > PitchGraphConstants.gapThreshold
            } else {
                shouldStartNewSegment = true  // First point
            }

            if shouldStartNewSegment {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }

            previousTime = point.time
        }

        context.stroke(
            path,
            with: .color(PitchGraphConstants.pitchLineColor),
            lineWidth: PitchGraphConstants.pitchLineWidth
        )

        // Draw dots at each pitch point with color based on volume (spectrogram-style)
        for point in pitchData {
            let x = coordinateSystem.timeToCanvasX(time: point.time, leftPadding: leftPadding)
            let y = coordinateSystem.frequencyToCanvasY(frequency: point.frequency, canvasHeight: canvasHeight)

            // Calculate dot radius based on confidence
            let radius = PitchGraphConstants.minDotRadius +
                (PitchGraphConstants.maxDotRadius - PitchGraphConstants.minDotRadius) * CGFloat(point.confidence)

            // Determine dot color based on volume (spectrogram-style HSB coloring)
            let dotColor = calculateVolumeBasedColor(amplitude: point.amplitude)

            let dotRect = CGRect(
                x: x - radius,
                y: y - radius,
                width: radius * 2,
                height: radius * 2
            )

            context.fill(
                Path(ellipseIn: dotRect),
                with: .color(dotColor)
            )
        }
    }

    /// Calculate color based on volume using spectrogram-style HSB gradient
    /// Low volume = blue-purple, High volume = red-yellow
    /// - Parameter amplitude: Normalized amplitude value (0.0 - 1.0)
    /// - Returns: Color based on amplitude
    private func calculateVolumeBasedColor(amplitude: Float) -> Color {
        let normalizedAmplitude = CGFloat(min(1.0, max(0.0, amplitude)))

        // Gradient: blue-purple (hue ~0.6) for weak → red-yellow (hue ~0.0) for strong
        // Same formula as SpectrogramRenderer
        let hue = PitchGraphConstants.weakestSignalHue - normalizedAmplitude * PitchGraphConstants.weakestSignalHue
        let saturation = PitchGraphConstants.volumeColorSaturation

        // Scale brightness based on amplitude
        let brightness = PitchGraphConstants.volumeMinBrightness +
            (PitchGraphConstants.volumeMaxBrightness - PitchGraphConstants.volumeMinBrightness) * normalizedAmplitude

        return Color(hue: hue, saturation: saturation, brightness: brightness)
    }

    /// Calculate dot color based on pitch accuracy compared to target note
    /// - Parameters:
    ///   - detectedFrequency: Detected pitch frequency
    ///   - time: Time of the detected pitch
    ///   - targetSegments: Array of target note segments
    /// - Returns: Color based on pitch deviation (green: ≤50 cents, yellow: ≤100 cents, cyan: no target or off)
    private func calculateDotColor(
        detectedFrequency: Double,
        time: Double,
        targetSegments: [NoteSegment]?
    ) -> Color {
        guard let segments = targetSegments else {
            return PitchGraphConstants.pitchLineColor  // Default cyan when no target
        }

        // Find target segment that contains this time
        guard let targetSegment = segments.first(where: { time >= $0.startTime && time < $0.endTime }) else {
            return PitchGraphConstants.pitchLineColor  // Default cyan when outside any target
        }

        // Calculate cents deviation
        let targetFrequency = targetSegment.frequency
        let cents = abs(1200 * log2(detectedFrequency / targetFrequency))

        if cents <= 50 {
            // Within 50 cents - excellent (green)
            return Color.green
        } else if cents <= 100 {
            // Within 100 cents - acceptable (yellow)
            return Color.yellow
        } else {
            // Off pitch (red)
            return Color.red
        }
    }

    /// Draw target note bars (karaoke-style rectangles) for each NoteSegment
    /// - Parameters:
    ///   - context: Graphics context
    ///   - canvasHeight: Canvas height
    ///   - segments: Array of NoteSegment representing target notes with timing
    ///   - leftPadding: Left padding for canvas
    ///   - currentPitchFrequency: Currently detected pitch for color highlighting
    public func drawTargetNoteBars(
        context: GraphicsContext,
        canvasHeight: CGFloat,
        segments: [NoteSegment],
        leftPadding: CGFloat,
        currentPitchFrequency: Double? = nil
    ) {
        let barHeight: CGFloat = 20  // Height of each note bar

        for segment in segments {
            let frequency = segment.frequency

            // Check if frequency is within display range
            guard frequency >= PitchGraphConstants.minFrequency &&
                  frequency <= PitchGraphConstants.maxFrequency else { continue }

            // Calculate position
            let y = coordinateSystem.frequencyToCanvasY(frequency: frequency, canvasHeight: canvasHeight)
            let startX = coordinateSystem.timeToCanvasX(time: segment.startTime, leftPadding: leftPadding)
            let endX = coordinateSystem.timeToCanvasX(time: segment.endTime, leftPadding: leftPadding)
            let width = endX - startX

            // Determine color based on pitch accuracy
            let barColor: Color
            if let currentFreq = currentPitchFrequency {
                let cents = abs(1200 * log2(currentFreq / frequency))
                if cents <= 50 {
                    // Within 50 cents - good (green)
                    barColor = Color.green.opacity(0.4)
                } else if cents <= 100 {
                    // Within 100 cents - acceptable (yellow)
                    barColor = Color.yellow.opacity(0.4)
                } else {
                    // Off pitch (default gray)
                    barColor = Color.gray.opacity(0.3)
                }
            } else {
                barColor = Color.gray.opacity(0.3)
            }

            // Draw the bar
            let rect = CGRect(
                x: startX,
                y: y - barHeight / 2,
                width: width,
                height: barHeight
            )

            context.fill(
                Path(roundedRect: rect, cornerRadius: 4),
                with: .color(barColor)
            )

            // Draw border
            context.stroke(
                Path(roundedRect: rect, cornerRadius: 4),
                with: .color(Color.gray.opacity(0.5)),
                lineWidth: 1
            )
        }
    }

    /// Draw target scale lines (reference frequencies)
    /// Note: Labels are drawn separately by drawTargetNoteLabels for fixed positioning
    /// - Parameters:
    ///   - context: Graphics context
    ///   - canvasHeight: Canvas height
    ///   - targetFrequencies: Array of target frequencies in Hz
    ///   - leftPadding: Left padding for canvas
    ///   - canvasWidth: Canvas width
    ///   - highlightedFrequency: Optional frequency to highlight (nearest target will be orange)
    public func drawTargetScaleLines(
        context: GraphicsContext,
        canvasHeight: CGFloat,
        targetFrequencies: [Double],
        leftPadding: CGFloat,
        canvasWidth: CGFloat,
        highlightedFrequency: Double? = nil
    ) {
        // Debug: Log target line drawing
        FileLogger.shared.log(level: "DEBUG", category: "pitch_graph_render", message: "🎨 drawTargetScaleLines called: freqCount=\(targetFrequencies.count), canvasHeight=\(canvasHeight), canvasWidth=\(canvasWidth), leftPadding=\(leftPadding)")

        // Find the nearest target frequency to highlight
        let nearestTarget = findNearestTargetFrequency(
            currentFrequency: highlightedFrequency,
            targetFrequencies: targetFrequencies
        )

        var drawnCount = 0
        for frequency in targetFrequencies {
            // Check if frequency is within display range
            guard frequency >= PitchGraphConstants.minFrequency &&
                  frequency <= PitchGraphConstants.maxFrequency else {
                FileLogger.shared.log(level: "DEBUG", category: "pitch_graph_render", message: "⚠️ freq \(frequency) outside range [\(PitchGraphConstants.minFrequency), \(PitchGraphConstants.maxFrequency)]")
                continue
            }

            let y = coordinateSystem.frequencyToCanvasY(frequency: frequency, canvasHeight: canvasHeight)
            drawnCount += 1
            if drawnCount <= 3 {
                FileLogger.shared.log(level: "DEBUG", category: "pitch_graph_render", message: "📏 Drawing line: freq=\(Int(frequency))Hz, y=\(y)")
            }

            // Determine color based on whether this is the highlighted target
            let isHighlighted = nearestTarget != nil && abs(frequency - nearestTarget!) < 1.0
            let lineColor = isHighlighted ? Color.orange.opacity(0.8) : Color.gray.opacity(0.3)
            let lineWidth = isHighlighted ? PitchGraphConstants.targetLineWidth * 1.5 : PitchGraphConstants.targetLineWidth

            // Draw target line (dashed)
            var path = Path()
            path.move(to: CGPoint(x: leftPadding, y: y))
            path.addLine(to: CGPoint(x: canvasWidth, y: y))

            context.stroke(
                path,
                with: .color(lineColor),
                style: StrokeStyle(
                    lineWidth: lineWidth,
                    dash: [5, 3]  // 5pt line, 3pt gap
                )
            )
        }
    }

    /// Find the nearest target frequency to the current frequency
    /// - Parameters:
    ///   - currentFrequency: Current detected frequency (nil if none)
    ///   - targetFrequencies: Array of target frequencies
    /// - Returns: The nearest target frequency, or nil if no current frequency
    private func findNearestTargetFrequency(
        currentFrequency: Double?,
        targetFrequencies: [Double]
    ) -> Double? {
        guard let current = currentFrequency, current > 0 else { return nil }

        var nearestTarget: Double?
        var minDistance = Double.infinity

        for target in targetFrequencies {
            // Use cents (logarithmic) for musical distance comparison
            let cents = abs(1200 * log2(current / target))
            if cents < minDistance {
                minDistance = cents
                nearestTarget = target
            }
        }

        // Only highlight if within 100 cents (1 semitone)
        return minDistance <= 100 ? nearestTarget : nil
    }

    /// Draw target scale note name labels (fixed at right edge, scrolls with Y)
    /// These labels are fixed in X direction (right edge) but scroll with Y direction
    /// - Parameters:
    ///   - context: Graphics context
    ///   - canvasHeight: Canvas height
    ///   - viewportWidth: Viewport width
    ///   - viewportHeight: Viewport height (for clipping)
    ///   - paperTop: Y-axis scroll position
    ///   - targetFrequencies: Array of target frequencies in Hz
    ///   - highlightedFrequency: Optional frequency to highlight (nearest target will be orange)
    public func drawTargetNoteLabels(
        context: GraphicsContext,
        canvasHeight: CGFloat,
        viewportWidth: CGFloat,
        viewportHeight: CGFloat,
        paperTop: CGFloat,
        targetFrequencies: [Double],
        highlightedFrequency: Double? = nil
    ) {
        let labelWidth: CGFloat = 32
        let labelHeight: CGFloat = 18
        let rightMargin: CGFloat = 8

        // Find the nearest target frequency to highlight
        let nearestTarget = findNearestTargetFrequency(
            currentFrequency: highlightedFrequency,
            targetFrequencies: targetFrequencies
        )

        for frequency in targetFrequencies {
            // Check if frequency is within display range
            guard frequency >= PitchGraphConstants.minFrequency &&
                  frequency <= PitchGraphConstants.maxFrequency else { continue }

            let canvasY = coordinateSystem.frequencyToCanvasY(frequency: frequency, canvasHeight: canvasHeight)
            let viewportY = canvasY + paperTop

            // Skip labels outside viewport (with some margin)
            guard viewportY >= -labelHeight && viewportY <= viewportHeight + labelHeight else { continue }

            let noteName = frequencyToNoteName(frequency)
            guard !noteName.isEmpty else { continue }

            // Determine color based on whether this is the highlighted target
            let isHighlighted = nearestTarget != nil && abs(frequency - nearestTarget!) < 1.0
            let backgroundColor = isHighlighted ? Color.orange : Color.gray.opacity(0.8)
            let textColor = isHighlighted ? Color.white : Color.white

            // Position label at right edge of viewport
            let labelX = viewportWidth - labelWidth - rightMargin
            let labelRect = CGRect(
                x: labelX,
                y: viewportY - labelHeight / 2,
                width: labelWidth,
                height: labelHeight
            )

            // Draw opaque background pill
            context.fill(
                Path(roundedRect: labelRect, cornerRadius: 6),
                with: .color(backgroundColor)
            )

            // Draw note name text
            context.draw(
                Text(noteName)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(textColor),
                at: CGPoint(x: labelX + labelWidth / 2, y: viewportY),
                anchor: .center
            )
        }
    }

    // MARK: - Axis Labels

    /// Draw frequency labels (Y-axis)
    /// These labels are fixed in X direction but scroll with Y direction
    /// - Parameters:
    ///   - context: Graphics context
    ///   - canvasHeight: Canvas height
    ///   - viewportHeight: Viewport height (for clipping)
    ///   - paperTop: Y-axis scroll position
    public func drawFrequencyLabels(
        context: GraphicsContext,
        canvasHeight: CGFloat,
        viewportHeight: CGFloat,
        paperTop: CGFloat
    ) {
        let labelPositions = coordinateSystem.getFrequencyLabelPositions(canvasHeight: canvasHeight)

        for (frequency, canvasY) in labelPositions {
            let viewportY = canvasY + paperTop

            // Skip labels outside viewport
            guard viewportY >= -20 && viewportY <= viewportHeight + 20 else { continue }

            let labelText = "\(Int(frequency))Hz"

            // Draw label at fixed X position (left edge), scrolling Y position
            context.draw(
                Text(labelText)
                    .font(.system(size: 10))
                    .foregroundColor(PitchGraphConstants.frequencyLabelColor),
                at: CGPoint(x: 5, y: viewportY),
                anchor: .leading
            )

            // Draw grid line (optional)
            var gridPath = Path()
            gridPath.move(to: CGPoint(x: PitchGraphConstants.leftMargin - 5, y: viewportY))
            gridPath.addLine(to: CGPoint(x: PitchGraphConstants.leftMargin, y: viewportY))

            context.stroke(
                gridPath,
                with: .color(PitchGraphConstants.frequencyLabelColor.opacity(0.5)),
                lineWidth: 0.5
            )
        }
    }

    /// Draw time labels (X-axis)
    /// These labels are fixed in Y direction but scroll with X direction
    /// - Parameters:
    ///   - context: Graphics context
    ///   - dataDuration: Total duration of audio
    ///   - leftPadding: Left padding for canvas
    ///   - viewportWidth: Viewport width
    ///   - viewportHeight: Viewport height
    ///   - canvasOffsetX: X-axis scroll position
    public func drawTimeLabels(
        context: GraphicsContext,
        dataDuration: Double,
        leftPadding: CGFloat,
        viewportWidth: CGFloat,
        viewportHeight: CGFloat,
        canvasOffsetX: CGFloat
    ) {
        let labelPositions = coordinateSystem.getTimeLabelPositions(dataDuration: dataDuration, leftPadding: leftPadding)

        for (time, canvasX) in labelPositions {
            let viewportX = canvasX + canvasOffsetX

            // Skip labels outside viewport
            guard viewportX >= -30 && viewportX <= viewportWidth + 30 else { continue }

            let labelText = String(format: "%.1fs", time)

            // Draw label at scrolling X position, fixed Y position (bottom)
            let labelY = viewportHeight - PitchGraphConstants.bottomMargin / 2

            context.draw(
                Text(labelText)
                    .font(.system(size: 10))
                    .foregroundColor(PitchGraphConstants.timeLabelColor),
                at: CGPoint(x: viewportX, y: labelY),
                anchor: .center
            )
        }
    }

    // MARK: - Playback Position

    /// Draw playback position line
    /// This line is fully fixed at screen center
    /// - Parameters:
    ///   - context: Graphics context
    ///   - viewportWidth: Viewport width
    ///   - viewportHeight: Viewport height
    public func drawPlaybackPosition(
        context: GraphicsContext,
        viewportWidth: CGFloat,
        viewportHeight: CGFloat
    ) {
        let centerX = viewportWidth / 2

        var path = Path()
        path.move(to: CGPoint(x: centerX, y: 0))
        path.addLine(to: CGPoint(x: centerX, y: viewportHeight - PitchGraphConstants.bottomMargin))

        context.stroke(
            path,
            with: .color(PitchGraphConstants.playbackLineColor),
            lineWidth: PitchGraphConstants.playbackLineWidth
        )
    }

    // MARK: - Placeholder

    /// Draw placeholder when no data available
    /// - Parameters:
    ///   - context: Graphics context
    ///   - size: View size
    public func drawPlaceholder(context: GraphicsContext, size: CGSize) {
        context.draw(
            Text("analysis.no_pitch_data".localized)
                .font(.caption)
                .foregroundColor(.gray),
            at: CGPoint(x: size.width / 2, y: size.height / 2),
            anchor: .center
        )
    }

    // MARK: - Background

    /// Draw graph background with grid
    /// - Parameters:
    ///   - context: Graphics context
    ///   - canvasHeight: Canvas height
    ///   - canvasWidth: Canvas width
    ///   - leftPadding: Left padding
    public func drawBackground(
        context: GraphicsContext,
        canvasHeight: CGFloat,
        canvasWidth: CGFloat,
        leftPadding: CGFloat
    ) {
        // Draw horizontal grid lines at each frequency label position
        let labelPositions = coordinateSystem.getFrequencyLabelPositions(canvasHeight: canvasHeight)

        for (_, canvasY) in labelPositions {
            var gridPath = Path()
            gridPath.move(to: CGPoint(x: leftPadding, y: canvasY))
            gridPath.addLine(to: CGPoint(x: canvasWidth, y: canvasY))

            context.stroke(
                gridPath,
                with: .color(Color.gray.opacity(0.1)),
                lineWidth: 0.5
            )
        }
    }
}
