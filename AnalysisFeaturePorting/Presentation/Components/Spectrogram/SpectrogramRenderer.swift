import SwiftUI
import VocalisDomain

/// Spectrogram rendering engine
/// Handles all drawing operations for spectrogram visualization
public class SpectrogramRenderer {
    // MARK: - Dependencies

    private let coordinateSystem: SpectrogramCoordinateSystem

    // MARK: - Initialization

    public init(coordinateSystem: SpectrogramCoordinateSystem) {
        self.coordinateSystem = coordinateSystem
    }

    // MARK: - Drawing Functions

    /// Draw frequency labels on canvas (canvas coordinate system)
    /// Labels are drawn at fixed intervals across entire canvas
    /// - Parameters:
    ///   - context: Graphics context
    ///   - canvasHeight: Total canvas height
    ///   - maxFreq: Maximum frequency
    ///   - viewportHeight: Viewport height (for clipping calculations)
    ///   - paperTop: Current Y scroll position
    public func drawFrequencyLabels(
        context: GraphicsContext,
        canvasHeight: CGFloat,
        maxFreq: Double,
        viewportHeight: CGFloat,
        paperTop: CGFloat
    ) {
        let textHeight: CGFloat = 16
        let textWidth: CGFloat = 45

        // Clipping margin to prevent labels from being cut off at edges
        let clipMargin = textHeight / 2

        // Start from labelInterval (skip 0Hz label)
        var frequency: Double = SpectrogramConstants.frequencyLabelInterval
        while frequency <= maxFreq {
            // Calculate canvas Y position
            let canvasY = coordinateSystem.frequencyToCanvasY(
                frequency: frequency,
                canvasHeight: canvasHeight,
                maxFreq: maxFreq
            )

            // Clamp label position to prevent cutoff at top/bottom edges
            let clampedY = max(clipMargin, min(canvasHeight - clipMargin, canvasY))

            // Create label text (always in Hz, no abbreviation)
            let labelText = "\(Int(frequency))Hz"

            let text = Text(labelText)
                .font(.caption2)
                .foregroundColor(.white)

            // Draw background with clamped position
            let backgroundRect = CGRect(
                x: 5,
                y: clampedY - textHeight / 2,
                width: textWidth,
                height: textHeight
            )

            context.fill(
                Path(roundedRect: backgroundRect, cornerRadius: 4),
                with: .color(.black.opacity(0.6))
            )

            // Draw text with clamped position
            context.draw(
                text,
                at: CGPoint(x: 5 + textWidth / 2, y: clampedY)
            )

            frequency += SpectrogramConstants.frequencyLabelInterval
        }
    }

    /// Draw spectrogram heatmap on canvas
    /// Each cell represents a time-frequency bin with magnitude-based color
    /// - Parameters:
    ///   - context: Graphics context
    ///   - canvasWidth: Total canvas width
    ///   - canvasHeight: Total canvas height
    ///   - maxFreq: Maximum frequency
    ///   - data: Spectrogram data to visualize
    ///   - leftPadding: Left padding for canvas
    ///   - visibleTimeRange: Optional time range for viewport culling (only draws data within this range)
    public func drawSpectrogram(
        context: GraphicsContext,
        canvasWidth: CGFloat,
        canvasHeight: CGFloat,
        maxFreq: Double,
        data: SpectrogramData,
        leftPadding: CGFloat,
        visibleTimeRange: ClosedRange<Double>? = nil
    ) {
        guard !data.timeStamps.isEmpty else { return }

        let pixelsPerSecond = coordinateSystem.getPixelsPerSecond()
        let maxMagnitude = data.magnitudes.flatMap { $0 }.max() ?? 1.0

        // Calculate cell dimensions
        let cellWidth = pixelsPerSecond * CGFloat(SpectrogramConstants.cellTimeWidthMultiplier)

        // Determine the highest frequency bin we have data for
        let maxDataFreq = data.frequencyBins.last.map { Double($0) } ?? 0.0

        // Calculate number of bins needed to cover entire canvas (0Hz to maxFreq)
        let avgBinWidth: Double
        if data.frequencyBins.count >= 2 {
            avgBinWidth = Double(data.frequencyBins[1] - data.frequencyBins[0])
        } else {
            avgBinWidth = 100.0  // fallback
        }

        let totalBinsNeeded = Int(ceil(maxFreq / avgBinWidth))

        // Draw all frequency bins (including areas with no data)
        for binIndex in 0..<totalBinsNeeded {
            let binFreqLow = Double(binIndex) * avgBinWidth
            let binFreqHigh = Double(binIndex + 1) * avgBinWidth

            // Skip if this bin is entirely above maxFreq
            guard binFreqLow < maxFreq else { break }

            // Find corresponding data bin index (if exists)
            let dataFreqIndex = data.frequencyBins.firstIndex { Double($0) >= binFreqLow }

            // Convert frequency range to canvas Y coordinates
            let yTop = coordinateSystem.frequencyToCanvasY(
                frequency: min(binFreqHigh, maxFreq),
                canvasHeight: canvasHeight,
                maxFreq: maxFreq
            )
            let yBottom = coordinateSystem.frequencyToCanvasY(
                frequency: binFreqLow,
                canvasHeight: canvasHeight,
                maxFreq: maxFreq
            )
            let cellHeight = yBottom - yTop

            // Draw cells for this frequency bin across time
            for (timeIndex, timestamp) in data.timeStamps.enumerated() {
                // Viewport culling: skip data outside visible time range
                if let range = visibleTimeRange {
                    guard range.contains(timestamp) else { continue }
                }

                // X coordinate in Canvas coordinate system
                // x = timestamp × pixelsPerSecond + leftPadding (Canvas absolute coordinate)
                // Data starts at leftPadding to leave space for frequency labels
                let x = CGFloat(timestamp) * pixelsPerSecond + leftPadding

                // Get magnitude from data (if exists)
                let magnitude: Float
                if let dataIdx = dataFreqIndex,
                   timeIndex < data.magnitudes.count,
                   dataIdx < data.magnitudes[timeIndex].count {
                    magnitude = data.magnitudes[timeIndex][dataIdx]
                } else {
                    magnitude = 0.0  // No data - use weakest color
                }

                // Color calculation with proper weakest color
                let normalizedMagnitude = CGFloat(magnitude / maxMagnitude)

                // Gradient: blue-purple (hue ~0.6) for weak → green-yellow (hue ~0.0) for strong
                let hue = SpectrogramConstants.weakestSignalHue - normalizedMagnitude * SpectrogramConstants.weakestSignalHue

                // For magnitude = 0, show visible weak color (not black)
                // saturation: keep constant for consistent color
                // brightness: ensure minimum visibility even at magnitude = 0
                let saturation = SpectrogramConstants.colorSaturation
                let brightness: CGFloat
                if normalizedMagnitude < SpectrogramConstants.minMagnitudeThreshold {
                    // Weakest color: clearly visible dark blue-purple
                    brightness = SpectrogramConstants.minBrightness
                } else {
                    // Scale brightness for stronger signals
                    brightness = SpectrogramConstants.minBrightness + (SpectrogramConstants.maxBrightness - SpectrogramConstants.minBrightness) * normalizedMagnitude
                }

                let color = Color(hue: hue, saturation: saturation, brightness: brightness)

                let rect = CGRect(x: x, y: yTop, width: cellWidth, height: cellHeight)
                context.fill(Path(rect), with: .color(color))
            }
        }
    }

    /// Draw placeholder when no data is available
    /// - Parameters:
    ///   - context: Graphics context
    ///   - size: Canvas size
    public func drawPlaceholder(context: GraphicsContext, size: CGSize) {
        let text = Text("analysis.no_spectrum_data".localized).font(.caption).foregroundColor(.secondary)
        context.draw(text, at: CGPoint(x: size.width / 2, y: size.height / 2))
    }

    /// Draw playback position indicator (red vertical line)
    /// - Parameters:
    ///   - context: Graphics context
    ///   - size: Viewport size
    public func drawPlaybackPosition(context: GraphicsContext, size: CGSize) {
        // Draw playback position line at center
        let centerX = size.width / 2

        context.stroke(
            Path { path in
                path.move(to: CGPoint(x: centerX, y: 0))
                path.addLine(to: CGPoint(x: centerX, y: size.height))
            },
            with: .color(.white),
            lineWidth: 2
        )
    }

    /// Draw time axis labels (X-axis)
    /// Labels are drawn at fixed time intervals
    /// - Parameters:
    ///   - context: Graphics context
    ///   - size: Canvas size
    ///   - leftPadding: Left padding for canvas
    public func drawTimeAxis(
        context: GraphicsContext,
        size: CGSize,
        leftPadding: CGFloat
    ) {
        let pixelsPerSecond = coordinateSystem.getPixelsPerSecond()

        // Calculate recording duration from canvas width (excluding left padding)
        let durationSec = Double((size.width - leftPadding) / pixelsPerSecond)

        // Draw time labels at 0.5-second intervals (0s, 0.5s, 1.0s, 1.5s, 2.0s, ...)
        // X coordinate: timestamp × pixelsPerSecond + leftPadding (Canvas coordinate system - same formula as spectrogram)
        // Y coordinate: size.height - 10 (viewport bottom with 10px padding - lower position)
        let labelCount = Int(ceil(durationSec / SpectrogramConstants.timeLabelInterval))

        for i in 0...labelCount {
            let timestamp = Double(i) * SpectrogramConstants.timeLabelInterval
            // X coordinate in Canvas coordinate system
            // Labels start at leftPadding to align with spectrogram data
            let x = CGFloat(timestamp) * pixelsPerSecond + leftPadding
            let y = size.height - 10  // Fixed at viewport bottom with 10px padding (lower position)

            let text = Text(String(format: "%.1fs", timestamp))
                .font(.caption2)
                .foregroundColor(.white)
            context.draw(text, at: CGPoint(x: x, y: y))
        }
    }
}
