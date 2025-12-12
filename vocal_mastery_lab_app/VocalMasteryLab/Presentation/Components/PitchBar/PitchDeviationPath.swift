import SwiftUI
import VocalisDomain

/// Represents a single pitch detection point with deviation information
public struct PitchDeviationPoint {
    /// Timestamp in seconds since recording start
    public let timestamp: Double

    /// Detected frequency in Hz
    public let frequency: Double

    /// Detection confidence (0.0 - 1.0)
    public let confidence: Float

    /// Target frequency in Hz (nil if no target at this timestamp)
    public let targetFrequency: Double?

    public init(
        timestamp: Double,
        frequency: Double,
        confidence: Float,
        targetFrequency: Double?
    ) {
        self.timestamp = timestamp
        self.frequency = frequency
        self.confidence = confidence
        self.targetFrequency = targetFrequency
    }

    /// Calculate deviation from target in cents
    /// Returns nil if no target frequency is set
    public var deviation: Double? {
        guard let target = targetFrequency else { return nil }
        return PitchBarConstants.calculateDeviation(detected: frequency, expected: target)
    }

    /// Get color based on deviation from target
    /// Returns cyan (default pitch color) if no target
    public var color: Color {
        guard let deviation = deviation else {
            return PitchGraphConstants.pitchLineColor
        }
        return PitchBarConstants.deviationColor(for: deviation)
    }
}

/// Renderer for pitch deviation path in karaoke-style UI
public struct PitchDeviationPathRenderer {

    // MARK: - Position Calculations

    /// Calculate X position for a timestamp
    public static func calculateXPosition(timestamp: Double, leftPadding: CGFloat) -> CGFloat {
        return PitchBarConstants.timeToX(time: timestamp, leftPadding: leftPadding)
    }

    /// Calculate Y position for a frequency (logarithmic scale)
    public static func calculateYPosition(frequency: Double, canvasHeight: CGFloat) -> CGFloat {
        return PitchBarConstants.frequencyToY(frequency: frequency, canvasHeight: canvasHeight)
    }

    // MARK: - Data Conversion

    /// Convert raw pitch data arrays to PitchDeviationPoint array
    public static func convertPitchDataToPoints(
        timestamps: [Double],
        frequencies: [Float],
        confidences: [Float],
        segments: [NoteSegment]
    ) -> [PitchDeviationPoint] {
        var points: [PitchDeviationPoint] = []

        for i in 0..<min(timestamps.count, frequencies.count, confidences.count) {
            let timestamp = timestamps[i]
            let frequency = Double(frequencies[i])
            let confidence = confidences[i]

            // Find target frequency at this timestamp
            let targetFreq = findTargetFrequency(at: timestamp, segments: segments)

            let point = PitchDeviationPoint(
                timestamp: timestamp,
                frequency: frequency,
                confidence: confidence,
                targetFrequency: targetFreq
            )
            points.append(point)
        }

        return points
    }

    /// Find target frequency at a given timestamp by searching segments
    /// Returns nil if no segment covers the timestamp
    public static func findTargetFrequency(at timestamp: Double, segments: [NoteSegment]) -> Double? {
        for segment in segments {
            // Segment range is [startTime, endTime)
            if timestamp >= segment.startTime && timestamp < segment.endTime {
                return segment.frequency
            }
        }
        return nil
    }

    // MARK: - Canvas Drawing

    /// Draw pitch deviation path on canvas
    /// Draws colored line segments based on deviation from target
    public static func drawDeviationPath(
        context: inout GraphicsContext,
        points: [PitchDeviationPoint],
        canvasHeight: CGFloat,
        leftPadding: CGFloat
    ) {
        guard points.count >= 2 else { return }

        // Draw line segments with color based on deviation
        for i in 0..<(points.count - 1) {
            let current = points[i]
            let next = points[i + 1]

            // Skip if frequency is outside display range
            guard current.frequency >= PitchBarConstants.minFrequency &&
                  current.frequency <= PitchBarConstants.maxFrequency &&
                  next.frequency >= PitchBarConstants.minFrequency &&
                  next.frequency <= PitchBarConstants.maxFrequency else {
                continue
            }

            let x1 = calculateXPosition(timestamp: current.timestamp, leftPadding: leftPadding)
            let y1 = calculateYPosition(frequency: current.frequency, canvasHeight: canvasHeight)
            let x2 = calculateXPosition(timestamp: next.timestamp, leftPadding: leftPadding)
            let y2 = calculateYPosition(frequency: next.frequency, canvasHeight: canvasHeight)

            // Use color based on current point's deviation
            let color = current.color

            var path = Path()
            path.move(to: CGPoint(x: x1, y: y1))
            path.addLine(to: CGPoint(x: x2, y: y2))

            context.stroke(
                path,
                with: .color(color),
                lineWidth: PitchBarConstants.pitchLineWidth
            )
        }
    }

    /// Draw dots at each pitch point
    public static func drawDeviationDots(
        context: inout GraphicsContext,
        points: [PitchDeviationPoint],
        canvasHeight: CGFloat,
        leftPadding: CGFloat,
        dotRadius: CGFloat = 3.0
    ) {
        for point in points {
            // Skip if frequency is outside display range
            guard point.frequency >= PitchBarConstants.minFrequency &&
                  point.frequency <= PitchBarConstants.maxFrequency else {
                continue
            }

            let x = calculateXPosition(timestamp: point.timestamp, leftPadding: leftPadding)
            let y = calculateYPosition(frequency: point.frequency, canvasHeight: canvasHeight)

            // Scale radius by confidence
            let radius = dotRadius * CGFloat(point.confidence)

            let dotRect = CGRect(
                x: x - radius,
                y: y - radius,
                width: radius * 2,
                height: radius * 2
            )

            context.fill(
                Path(ellipseIn: dotRect),
                with: .color(point.color)
            )
        }
    }

    /// Draw complete pitch deviation visualization (path + dots)
    public static func drawComplete(
        context: inout GraphicsContext,
        points: [PitchDeviationPoint],
        canvasHeight: CGFloat,
        leftPadding: CGFloat
    ) {
        drawDeviationPath(
            context: &context,
            points: points,
            canvasHeight: canvasHeight,
            leftPadding: leftPadding
        )
        drawDeviationDots(
            context: &context,
            points: points,
            canvasHeight: canvasHeight,
            leftPadding: leftPadding
        )
    }
}

/// SwiftUI View for displaying pitch deviation path
public struct PitchDeviationPathView: View {
    let points: [PitchDeviationPoint]
    let canvasHeight: CGFloat
    let leftPadding: CGFloat

    public init(points: [PitchDeviationPoint], canvasHeight: CGFloat, leftPadding: CGFloat) {
        self.points = points
        self.canvasHeight = canvasHeight
        self.leftPadding = leftPadding
    }

    public var body: some View {
        Canvas { context, size in
            var mutableContext = context
            PitchDeviationPathRenderer.drawComplete(
                context: &mutableContext,
                points: points,
                canvasHeight: canvasHeight,
                leftPadding: leftPadding
            )
        }
    }
}

#if DEBUG
struct PitchDeviationPathView_Previews: PreviewProvider {
    static var previews: some View {
        let samplePoints: [PitchDeviationPoint] = [
            // Perfect match
            PitchDeviationPoint(timestamp: 0.0, frequency: 261.63, confidence: 0.9, targetFrequency: 261.63),
            PitchDeviationPoint(timestamp: 0.1, frequency: 262.0, confidence: 0.85, targetFrequency: 261.63),
            PitchDeviationPoint(timestamp: 0.2, frequency: 265.0, confidence: 0.9, targetFrequency: 261.63),
            // Sharp
            PitchDeviationPoint(timestamp: 0.3, frequency: 280.0, confidence: 0.92, targetFrequency: 261.63),
            PitchDeviationPoint(timestamp: 0.4, frequency: 290.0, confidence: 0.88, targetFrequency: 293.66),
            // Perfect on D4
            PitchDeviationPoint(timestamp: 0.5, frequency: 294.0, confidence: 0.9, targetFrequency: 293.66),
            // Flat
            PitchDeviationPoint(timestamp: 0.6, frequency: 285.0, confidence: 0.85, targetFrequency: 293.66),
            PitchDeviationPoint(timestamp: 0.7, frequency: 270.0, confidence: 0.8, targetFrequency: 293.66)
        ]

        VStack {
            Text("Pitch Deviation Path Preview")
                .font(.headline)

            PitchDeviationPathView(
                points: samplePoints,
                canvasHeight: 300,
                leftPadding: 50
            )
            .frame(width: 400, height: 300)
            .background(Color.black.opacity(0.9))
        }
        .padding()
    }
}
#endif
