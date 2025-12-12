import Foundation
import AVFoundation
import VocalisDomain
import Combine

/// Protocol for pitch detector to enable testing
public protocol PitchDetectorProtocol {
    var detectedPitch: DetectedPitch? { get }
    var isDetecting: Bool { get }
    var spectrum: [Float]? { get }
    var detectedPitchPublisher: AnyPublisher<DetectedPitch?, Never> { get }
    func startRealtimeDetection() async throws
    func stopRealtimeDetection()
}

/// Automatic pitch detection evaluator using scale playback
/// Coordinates scale playback with pitch detection to automatically evaluate accuracy
@MainActor
public class AutoPitchEvaluator: ObservableObject {
    @Published public private(set) var isEvaluating: Bool = false
    @Published public private(set) var evaluationResult: EvaluationResult?

    private let scalePlayer: ScalePlayerProtocol
    private let pitchDetector: PitchDetectorProtocol

    private var evaluationTask: Task<Void, Never>?
    private var expectedNotes: [MIDINote] = []
    private var tempo: Tempo?
    private var detectedPitches: [(expected: Double, detected: Double, confidence: Double)] = []

    public init(scalePlayer: ScalePlayerProtocol, pitchDetector: PitchDetectorProtocol) {
        self.scalePlayer = scalePlayer
        self.pitchDetector = pitchDetector
    }

    /// Start automatic evaluation with scale playback
    public func startEvaluation(notes: [MIDINote], tempo: Tempo) async throws {
        guard !isEvaluating else {
            throw EvaluationError.alreadyEvaluating
        }

        self.expectedNotes = notes
        self.tempo = tempo
        self.detectedPitches = []
        self.evaluationResult = nil
        self.isEvaluating = true

        // Load scale into player
        try await scalePlayer.loadScale(notes, tempo: tempo)

        // Start pitch detection
        try await pitchDetector.startRealtimeDetection()

        // Start playback (muted=false for real audio output to microphone)
        try await scalePlayer.play(muted: false)

        // Start monitoring task
        evaluationTask = Task { [weak self] in
            guard let self = self else { return }
            await self.monitorEvaluation()
        }
    }

    /// Stop evaluation and calculate final metrics
    public func stopEvaluation() async {
        guard isEvaluating else { return }

        // Cancel monitoring task
        evaluationTask?.cancel()
        evaluationTask = nil

        // Stop playback and detection
        await scalePlayer.stop()
        pitchDetector.stopRealtimeDetection()

        // Calculate final metrics
        calculateMetrics()

        isEvaluating = false
    }

    /// Monitor pitch detection during scale playback
    private func monitorEvaluation() async {
        guard let tempo = tempo else { return }

        let noteInterval = tempo.secondsPerNote

        for (index, expectedNote) in expectedNotes.enumerated() {
            // Wait for note to be played
            let startTime = Date()

            // Sample pitch detection multiple times during the note duration
            let sampleInterval: TimeInterval = 0.1 // Sample every 100ms
            let samplesPerNote = Int(noteInterval / sampleInterval)

            for _ in 0..<samplesPerNote {
                if Task.isCancelled { return }

                // Get current detected pitch
                if let detected = pitchDetector.detectedPitch {
                    let expectedFreq = expectedNote.frequency
                    detectedPitches.append((
                        expected: expectedFreq,
                        detected: detected.frequency,
                        confidence: detected.confidence
                    ))
                }

                // Wait for next sample
                try? await Task.sleep(nanoseconds: UInt64(sampleInterval * 1_000_000_000))
            }

            // Ensure we spent the full note duration
            let elapsed = Date().timeIntervalSince(startTime)
            if elapsed < noteInterval {
                let remaining = noteInterval - elapsed
                try? await Task.sleep(nanoseconds: UInt64(remaining * 1_000_000_000))
            }
        }

        // Evaluation completed naturally
        await stopEvaluation()
    }

    /// Calculate evaluation metrics from collected pitch data
    private func calculateMetrics() {
        guard !detectedPitches.isEmpty else {
            evaluationResult = nil
            return
        }

        var grossErrors = 0
        var totalCentsError: Double = 0.0
        var octaveErrors = 0
        var totalConfidence: Double = 0.0

        for (expected, detected, confidence) in detectedPitches {
            // Calculate error in cents
            let centsError = abs(1200.0 * log2(detected / expected))

            // GPE: Gross Pitch Error (>50 cents)
            if centsError > 50.0 {
                grossErrors += 1
            }

            // FPE: Fine Pitch Error (accumulate for average)
            totalCentsError += centsError

            // Octave Error: Check if detected is approximately 2x or 0.5x expected
            let ratio = detected / expected
            if abs(ratio - 2.0) < 0.1 || abs(ratio - 0.5) < 0.05 {
                octaveErrors += 1
            }

            totalConfidence += confidence
        }

        let totalSamples = detectedPitches.count

        let gpe = Double(grossErrors) / Double(totalSamples)
        let fpe = totalCentsError / Double(totalSamples)
        let octaveErrorRate = Double(octaveErrors) / Double(totalSamples)
        let averageConfidence = totalConfidence / Double(totalSamples)

        evaluationResult = EvaluationResult(
            gpe: gpe,
            fpe: fpe,
            octaveErrorRate: octaveErrorRate,
            totalNotes: expectedNotes.count,
            totalSamples: totalSamples,
            averageConfidence: averageConfidence
        )
    }
}

// MARK: - Supporting Types

/// Evaluation result containing accuracy metrics
public struct EvaluationResult {
    /// Gross Pitch Error rate (percentage of errors >50 cents)
    public let gpe: Double

    /// Fine Pitch Error (average absolute error in cents)
    public let fpe: Double

    /// Octave error rate (percentage of octave errors)
    public let octaveErrorRate: Double

    /// Total number of notes in scale
    public let totalNotes: Int

    /// Total number of pitch samples collected
    public let totalSamples: Int

    /// Average confidence of detected pitches
    public let averageConfidence: Double
}

/// Errors that can occur during evaluation
public enum EvaluationError: LocalizedError {
    case alreadyEvaluating
    case notEvaluating
    case playbackFailed(String)
    case detectionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .alreadyEvaluating:
            return "Evaluation already in progress"
        case .notEvaluating:
            return "No evaluation in progress"
        case .playbackFailed(let message):
            return "Playback failed: \(message)"
        case .detectionFailed(let message):
            return "Detection failed: \(message)"
        }
    }
}
