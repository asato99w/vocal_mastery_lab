import SwiftUI
import VocalisDomain

/// Score result for a single note
public struct NoteScore: Equatable, Identifiable {
    public var id: UInt8 { note.value }

    /// The note being evaluated
    public let note: MIDINote

    /// Accuracy percentage (0-100)
    public let accuracy: Double

    /// Average deviation in cents (positive = sharp, negative = flat)
    public let averageDeviation: Double

    /// Number of pitch points evaluated for this note
    public let pointCount: Int

    public init(
        note: MIDINote,
        accuracy: Double,
        averageDeviation: Double,
        pointCount: Int
    ) {
        self.note = note
        self.accuracy = accuracy
        self.averageDeviation = averageDeviation
        self.pointCount = pointCount
    }

    /// Accuracy level based on percentage
    public var accuracyLevel: AccuracyLevel {
        if accuracy >= 90 {
            return .excellent
        } else if accuracy >= 75 {
            return .good
        } else if accuracy >= 50 {
            return .acceptable
        } else {
            return .needsImprovement
        }
    }

    /// Accuracy level enumeration for note scores
    public enum AccuracyLevel: String, Equatable {
        case excellent
        case good
        case acceptable
        case needsImprovement
    }
}

/// Calculator for pitch deviation scores
public struct DeviationScoreCalculator {

    // MARK: - Overall Accuracy

    /// Calculate overall accuracy percentage
    /// A point is considered "accurate" if within ±10 cents (perfect threshold)
    /// - Parameter points: Array of pitch deviation points
    /// - Returns: Accuracy percentage (0-100)
    public static func calculateOverallAccuracy(points: [PitchDeviationPoint]) -> Double {
        // Filter points that have a target frequency
        let validPoints = points.filter { $0.targetFrequency != nil }

        guard !validPoints.isEmpty else { return 0.0 }

        // Count points within perfect threshold (±10 cents)
        let accurateCount = validPoints.filter { point in
            guard let deviation = point.deviation else { return false }
            return abs(deviation) <= PitchBarConstants.perfectThreshold
        }.count

        return Double(accurateCount) / Double(validPoints.count) * 100.0
    }

    // MARK: - Average Deviation

    /// Calculate average deviation across all valid points
    /// - Parameter points: Array of pitch deviation points
    /// - Returns: Average deviation in cents, or nil if no valid points
    public static func calculateAverageDeviation(points: [PitchDeviationPoint]) -> Double? {
        // Get all deviations (only for points with target)
        let deviations = points.compactMap { $0.deviation }

        guard !deviations.isEmpty else { return nil }

        let sum = deviations.reduce(0, +)
        return sum / Double(deviations.count)
    }

    // MARK: - Note Scores

    /// Calculate accuracy scores for each note segment
    /// - Parameters:
    ///   - points: Array of pitch deviation points
    ///   - segments: Array of note segments to evaluate
    /// - Returns: Array of note scores
    public static func calculateNoteScores(
        points: [PitchDeviationPoint],
        segments: [NoteSegment]
    ) -> [NoteScore] {
        var noteScores: [NoteScore] = []

        for segment in segments {
            // Find points within this segment's time range
            let segmentPoints = points.filter { point in
                point.timestamp >= segment.startTime &&
                point.timestamp < segment.endTime &&
                point.targetFrequency != nil
            }

            guard !segmentPoints.isEmpty else { continue }

            // Calculate accuracy for this note
            let accurateCount = segmentPoints.filter { point in
                guard let deviation = point.deviation else { return false }
                return abs(deviation) <= PitchBarConstants.perfectThreshold
            }.count

            let accuracy = Double(accurateCount) / Double(segmentPoints.count) * 100.0

            // Calculate average deviation
            let deviations = segmentPoints.compactMap { $0.deviation }
            let avgDeviation = deviations.isEmpty ? 0.0 : deviations.reduce(0, +) / Double(deviations.count)

            let score = NoteScore(
                note: segment.note,
                accuracy: accuracy,
                averageDeviation: avgDeviation,
                pointCount: segmentPoints.count
            )

            noteScores.append(score)
        }

        return noteScores
    }
}

/// SwiftUI View for displaying overall deviation score
public struct DeviationScoreView: View {
    let accuracy: Double
    let averageDeviation: Double?

    public init(accuracy: Double, averageDeviation: Double?) {
        self.accuracy = accuracy
        self.averageDeviation = averageDeviation
    }

    public var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 24) {
                // Accuracy display
                HStack(spacing: 8) {
                    Image(systemName: "target")
                        .foregroundColor(.green)
                    Text("音程精度:")
                    Text(String(format: "%.0f%%", accuracy))
                        .fontWeight(.bold)
                        .foregroundColor(accuracyColor)
                }

                // Average deviation display
                if let avgDev = averageDeviation {
                    HStack(spacing: 8) {
                        Image(systemName: "chart.bar.fill")
                            .foregroundColor(.blue)
                        Text("平均偏差:")
                        Text(formatDeviation(avgDev))
                            .fontWeight(.bold)
                            .foregroundColor(deviationColor(avgDev))
                    }
                }
            }
            .font(.subheadline)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }

    private var accuracyColor: Color {
        if accuracy >= 90 {
            return .green
        } else if accuracy >= 75 {
            return .blue
        } else if accuracy >= 50 {
            return .yellow
        } else {
            return .red
        }
    }

    private func deviationColor(_ deviation: Double) -> Color {
        let absDeviation = abs(deviation)
        if absDeviation <= PitchBarConstants.perfectThreshold {
            return .green
        } else if absDeviation <= PitchBarConstants.goodThreshold {
            return .blue
        } else if absDeviation <= PitchBarConstants.acceptableThreshold {
            return .yellow
        } else {
            return .red
        }
    }

    private func formatDeviation(_ deviation: Double) -> String {
        let sign = deviation >= 0 ? "+" : ""
        return String(format: "%@%.0f セント", sign, deviation)
    }
}

/// SwiftUI View for displaying individual note scores
public struct NoteScoreListView: View {
    let noteScores: [NoteScore]

    public init(noteScores: [NoteScore]) {
        self.noteScores = noteScores
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(noteScores) { score in
                NoteScoreRow(score: score)
            }
        }
    }
}

/// A single row displaying note score
public struct NoteScoreRow: View {
    let score: NoteScore

    public var body: some View {
        HStack {
            // Note name
            Text(score.note.noteName)
                .font(.system(.body, design: .monospaced))
                .frame(width: 40, alignment: .leading)

            // Progress bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Background
                    Rectangle()
                        .fill(Color.gray.opacity(0.3))
                        .frame(height: 16)
                        .cornerRadius(4)

                    // Progress
                    Rectangle()
                        .fill(progressColor)
                        .frame(width: geometry.size.width * CGFloat(score.accuracy / 100.0), height: 16)
                        .cornerRadius(4)
                }
            }
            .frame(height: 16)

            // Percentage and evaluation
            HStack(spacing: 4) {
                Text(String(format: "%.0f%%", score.accuracy))
                    .font(.caption)
                    .frame(width: 40, alignment: .trailing)

                Text(evaluationText)
                    .font(.caption)
                    .foregroundColor(progressColor)
            }
        }
    }

    private var progressColor: Color {
        switch score.accuracyLevel {
        case .excellent:
            return .green
        case .good:
            return .blue
        case .acceptable:
            return .yellow
        case .needsImprovement:
            return .red
        }
    }

    private var evaluationText: String {
        switch score.accuracyLevel {
        case .excellent:
            return "優秀"
        case .good:
            return "良好"
        case .acceptable:
            return "許容範囲"
        case .needsImprovement:
            return "要改善"
        }
    }
}

#if DEBUG
struct DeviationScoreView_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 20) {
            Text("Deviation Score Preview")
                .font(.headline)

            DeviationScoreView(
                accuracy: 87.0,
                averageDeviation: 8.0
            )

            Divider()

            Text("Note Score List Preview")
                .font(.headline)

            NoteScoreListView(noteScores: sampleNoteScores)
        }
        .padding()
    }

    static var sampleNoteScores: [NoteScore] {
        do {
            return [
                NoteScore(note: try MIDINote(60), accuracy: 95.0, averageDeviation: 3.0, pointCount: 20),
                NoteScore(note: try MIDINote(62), accuracy: 78.0, averageDeviation: -15.0, pointCount: 18),
                NoteScore(note: try MIDINote(64), accuracy: 92.0, averageDeviation: 8.0, pointCount: 22),
                NoteScore(note: try MIDINote(65), accuracy: 65.0, averageDeviation: 35.0, pointCount: 19),
                NoteScore(note: try MIDINote(67), accuracy: 88.0, averageDeviation: -5.0, pointCount: 21)
            ]
        } catch {
            return []
        }
    }
}
#endif
