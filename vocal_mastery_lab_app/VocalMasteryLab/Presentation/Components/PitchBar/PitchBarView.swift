import SwiftUI
import VocalisDomain

/// ViewModel for PitchBarView
/// Manages pitch deviation data and score calculations
public class PitchBarViewModel: ObservableObject {
    /// Note segments from playback timeline
    public let segments: [NoteSegment]

    /// Pitch deviation points for visualization
    @Published public private(set) var deviationPoints: [PitchDeviationPoint] = []

    /// Note scores for each segment
    @Published public private(set) var noteScores: [NoteScore] = []

    /// Overall accuracy percentage
    @Published public private(set) var overallAccuracy: Double = 0.0

    /// Average deviation in cents
    @Published public private(set) var averageDeviation: Double?

    /// Left padding for canvas
    public let leftPadding: CGFloat = 50.0

    /// Canvas width based on segments
    public var canvasWidth: CGFloat {
        guard let lastSegment = segments.last else { return leftPadding }
        return CGFloat(lastSegment.endTime) * PitchBarConstants.pixelsPerSecond + leftPadding
    }

    /// Canvas height from constants
    public var canvasHeight: CGFloat {
        return PitchBarConstants.calculatedCanvasHeight
    }

    public init(pitchData: PitchAnalysisData, segments: [NoteSegment]) {
        self.segments = segments

        // Convert pitch data to deviation points
        self.deviationPoints = PitchDeviationPathRenderer.convertPitchDataToPoints(
            timestamps: pitchData.timeStamps,
            frequencies: pitchData.frequencies,
            confidences: pitchData.confidences,
            segments: segments
        )

        // Calculate scores
        self.overallAccuracy = DeviationScoreCalculator.calculateOverallAccuracy(points: deviationPoints)
        self.averageDeviation = DeviationScoreCalculator.calculateAverageDeviation(points: deviationPoints)
        self.noteScores = DeviationScoreCalculator.calculateNoteScores(points: deviationPoints, segments: segments)
    }
}

/// Main karaoke-style pitch bar visualization view
public struct PitchBarView: View {
    @StateObject private var viewModel: PitchBarViewModel

    public init(pitchData: PitchAnalysisData, segments: [NoteSegment]) {
        _viewModel = StateObject(wrappedValue: PitchBarViewModel(pitchData: pitchData, segments: segments))
    }

    public var body: some View {
        VStack(spacing: 16) {
            // Score display
            DeviationScoreView(
                accuracy: viewModel.overallAccuracy,
                averageDeviation: viewModel.averageDeviation
            )

            // Pitch bar canvas
            ScrollView(.horizontal, showsIndicators: true) {
                pitchBarCanvas
            }
            .frame(height: viewModel.canvasHeight + 40)  // Extra space for labels

            // Note score list (if we have scores)
            if !viewModel.noteScores.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("ノート別評価")
                        .font(.headline)
                        .padding(.horizontal)

                    NoteScoreListView(noteScores: viewModel.noteScores)
                        .padding(.horizontal)
                }
            }
        }
    }

    private var pitchBarCanvas: some View {
        ZStack(alignment: .topLeading) {
            // Background grid with note labels
            PitchGridBackground(
                canvasHeight: viewModel.canvasHeight,
                canvasWidth: viewModel.canvasWidth,
                leftPadding: viewModel.leftPadding,
                segments: viewModel.segments
            )

            // Target note bars
            TargetNoteBarView(
                segments: viewModel.segments,
                canvasHeight: viewModel.canvasHeight,
                leftPadding: viewModel.leftPadding
            )

            // Pitch deviation path
            PitchDeviationPathView(
                points: viewModel.deviationPoints,
                canvasHeight: viewModel.canvasHeight,
                leftPadding: viewModel.leftPadding
            )
        }
        .frame(width: viewModel.canvasWidth, height: viewModel.canvasHeight)
        .background(Color.black.opacity(0.9))
    }
}

/// Background grid with note labels and time axis
public struct PitchGridBackground: View {
    let canvasHeight: CGFloat
    let canvasWidth: CGFloat
    let leftPadding: CGFloat
    let segments: [NoteSegment]

    public init(
        canvasHeight: CGFloat,
        canvasWidth: CGFloat,
        leftPadding: CGFloat,
        segments: [NoteSegment]
    ) {
        self.canvasHeight = canvasHeight
        self.canvasWidth = canvasWidth
        self.leftPadding = leftPadding
        self.segments = segments
    }

    public var body: some View {
        Canvas { context, size in
            // Draw horizontal grid lines and note labels
            drawNoteLines(context: &context, size: size)

            // Draw vertical time lines
            drawTimeLines(context: &context, size: size)
        }
    }

    private func drawNoteLines(context: inout GraphicsContext, size: CGSize) {
        // Get unique notes from segments
        var drawnNotes: Set<UInt8> = []

        for segment in segments {
            let noteValue = segment.note.value
            guard !drawnNotes.contains(noteValue) else { continue }
            drawnNotes.insert(noteValue)

            let y = PitchBarConstants.frequencyToY(
                frequency: segment.frequency,
                canvasHeight: canvasHeight
            )

            // Draw horizontal line
            var path = Path()
            path.move(to: CGPoint(x: leftPadding, y: y))
            path.addLine(to: CGPoint(x: canvasWidth, y: y))

            context.stroke(
                path,
                with: .color(Color.gray.opacity(0.2)),
                lineWidth: 1
            )

            // Draw note name
            context.draw(
                Text(segment.note.noteName)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.gray),
                at: CGPoint(x: 25, y: y),
                anchor: .center
            )
        }
    }

    private func drawTimeLines(context: inout GraphicsContext, size: CGSize) {
        // Calculate total duration
        guard let lastSegment = segments.last else { return }
        let totalDuration = lastSegment.endTime

        // Draw vertical lines every second
        for second in stride(from: 0, through: totalDuration, by: 1.0) {
            let x = CGFloat(second) * PitchBarConstants.pixelsPerSecond + leftPadding

            // Draw vertical line
            var path = Path()
            path.move(to: CGPoint(x: x, y: 0))
            path.addLine(to: CGPoint(x: x, y: canvasHeight))

            context.stroke(
                path,
                with: .color(Color.gray.opacity(0.15)),
                lineWidth: 1
            )

            // Draw time label at bottom
            context.draw(
                Text(String(format: "%.0fs", second))
                    .font(.system(size: 8))
                    .foregroundColor(.gray),
                at: CGPoint(x: x, y: canvasHeight - 5),
                anchor: .top
            )
        }
    }
}

#if DEBUG
struct PitchBarView_Previews: PreviewProvider {
    static var previews: some View {
        VStack {
            Text("Pitch Bar View Preview")
                .font(.headline)

            PitchBarView(
                pitchData: samplePitchData,
                segments: sampleSegments
            )
        }
        .padding()
        .background(Color(.systemBackground))
    }

    static var sampleSegments: [NoteSegment] {
        do {
            return [
                NoteSegment(note: try MIDINote(60), startTime: 0.0, endTime: 1.0),   // C4
                NoteSegment(note: try MIDINote(62), startTime: 1.0, endTime: 2.0),   // D4
                NoteSegment(note: try MIDINote(64), startTime: 2.0, endTime: 3.0),   // E4
                NoteSegment(note: try MIDINote(65), startTime: 3.0, endTime: 4.0),   // F4
                NoteSegment(note: try MIDINote(67), startTime: 4.0, endTime: 5.0)    // G4
            ]
        } catch {
            return []
        }
    }

    static var samplePitchData: PitchAnalysisData {
        // Generate sample pitch data with some deviation
        var timestamps: [Double] = []
        var frequencies: [Float] = []
        var confidences: [Float] = []
        var amplitudes: [Float] = []

        let notes: [Float] = [261.63, 293.66, 329.63, 349.23, 392.0]  // C4, D4, E4, F4, G4

        for (segmentIndex, baseFreq) in notes.enumerated() {
            let startTime = Double(segmentIndex)
            // Add points within each segment with slight deviations
            for i in 0..<10 {
                let time = startTime + Double(i) * 0.1
                // Add some random-ish deviation
                let deviation = sin(Double(segmentIndex + i)) * 20  // ±20 cents variation
                let freq = baseFreq * Float(pow(2, deviation / 1200.0))

                timestamps.append(time)
                frequencies.append(freq)
                confidences.append(0.85 + Float.random(in: 0...0.1))
                amplitudes.append(0.5 + Float.random(in: 0...0.3))
            }
        }

        return PitchAnalysisData(
            timeStamps: timestamps,
            frequencies: frequencies,
            confidences: confidences,
            amplitudes: amplitudes
        )
    }
}
#endif
