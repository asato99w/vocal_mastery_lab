import XCTest
import AVFoundation
import VocalisDomain
@testable import VocalMasteryLab

/// Pitch detection strategy accuracy comparison tests using vocadito singing voice dataset
///
/// Compares YINStrategy vs PYINStrategy accuracy using the same test data
/// This validates that the Strategy pattern implementation works correctly
///
/// FULL FILE ANALYSIS: Analyzes entire audio files to give pYIN's HMM proper context
/// for temporal smoothing. Uses vocadito_1 track (~60s) for representative testing.
///
@available(iOS 13.0, *)
final class PitchStrategyAccuracyComparisonTests: XCTestCase {

    // MARK: - Properties

    private var yinStrategy: YINStrategy!
    private var pyinStrategy: PYINStrategy!
    private let sampleRate: Double = 44100.0

    // MARK: - Setup/Teardown

    override func setUp() async throws {
        try await super.setUp()

        // ⚠️ ACCURACY TESTS DISABLED BY DEFAULT - Comment out this line to enable
        try XCTSkipIf(true, "Accuracy tests take ~60s per track. Enable by commenting out this line.")

        yinStrategy = YINStrategy()
        pyinStrategy = PYINStrategy()
    }

    // MARK: - Full File Analysis Test (Main Comparison)

    /// Compare YIN vs pYIN using full file analysis
    /// This gives pYIN's HMM proper context for temporal smoothing
    func testFullFileAccuracyComparison() async throws {
        // Test with vocadito_1 (representative track)
        let trackName = "vocadito_1"
        let audioFileName = try VocaditoTestDataLoader.getAudioFileName(for: trackName)
        let audioPath = TestResourceLoader.getVocaditoAudioPath(filename: audioFileName)
        let notes = try VocaditoTestDataLoader.getNotes(for: trackName)

        // Load full audio file
        let samples = try loadFullAudioFile(from: audioPath)
        print("\n📊 Full File Analysis: \(trackName)")
        print("   Audio duration: \(String(format: "%.1f", Double(samples.count) / sampleRate))s")
        print("   Sample count: \(samples.count)")
        print("=" .repeated(60))

        // Analyze with both strategies
        let yinFrames = yinStrategy.detectPitch(samples: samples, sampleRate: sampleRate)
        let pyinFrames = pyinStrategy.detectPitch(samples: samples, sampleRate: sampleRate)

        print("   YIN frames: \(yinFrames.count)")
        print("   pYIN frames: \(pyinFrames.count)")
        print("")

        var yinResults: [(name: String, passed: Bool, errorCents: Double, confidence: Float)] = []
        var pyinResults: [(name: String, passed: Bool, errorCents: Double, confidence: Float)] = []

        for (noteIndex, note) in notes.enumerated() {
            let testName = "Note\(noteIndex + 1)"

            let yinResult = evaluateAccuracyFromFrames(
                frames: yinFrames,
                targetTime: note.midTime,
                expectedFrequency: note.frequency
            )
            let pyinResult = evaluateAccuracyFromFrames(
                frames: pyinFrames,
                targetTime: note.midTime,
                expectedFrequency: note.frequency
            )

            yinResults.append((testName, yinResult.passed, yinResult.errorCents, yinResult.confidence))
            pyinResults.append((testName, pyinResult.passed, pyinResult.errorCents, pyinResult.confidence))

            let yinStatus = yinResult.passed ? "✅" : "❌"
            let pyinStatus = pyinResult.passed ? "✅" : "❌"

            print("\(testName) (time: \(String(format: "%.2f", note.midTime))s, expected: \(String(format: "%.1f", note.frequency))Hz):")
            print("  YIN:  \(yinStatus) \(String(format: "%5.1f", yinResult.errorCents)) cents, conf: \(String(format: "%.2f", yinResult.confidence))")
            print("  pYIN: \(pyinStatus) \(String(format: "%5.1f", pyinResult.errorCents)) cents, conf: \(String(format: "%.2f", pyinResult.confidence))")
        }

        // Summary
        let yinPassed = yinResults.filter { $0.passed }.count
        let pyinPassed = pyinResults.filter { $0.passed }.count
        let totalTests = yinResults.count

        let yinAvgError = yinResults.map { $0.errorCents }.reduce(0, +) / Double(totalTests)
        let pyinAvgError = pyinResults.map { $0.errorCents }.reduce(0, +) / Double(totalTests)

        let yinAvgConf = yinResults.map { Double($0.confidence) }.reduce(0, +) / Double(totalTests)
        let pyinAvgConf = pyinResults.map { Double($0.confidence) }.reduce(0, +) / Double(totalTests)

        print("\n" + "=" .repeated(60))
        print("📈 Results for \(trackName):")
        print("   YIN:  \(yinPassed)/\(totalTests) | Avg Error: \(String(format: "%.1f", yinAvgError)) cents | Avg Conf: \(String(format: "%.2f", yinAvgConf))")
        print("   pYIN: \(pyinPassed)/\(totalTests) | Avg Error: \(String(format: "%.1f", pyinAvgError)) cents | Avg Conf: \(String(format: "%.2f", pyinAvgConf))")
    }

    /// Test all 3 tracks with full file analysis
    func testAllTracksFullFileComparison() async throws {
        let trackNames = try VocaditoTestDataLoader.getAllTrackNames()

        var allYinResults: [(track: String, passed: Int, total: Int, avgError: Double, avgConf: Double)] = []
        var allPyinResults: [(track: String, passed: Int, total: Int, avgError: Double, avgConf: Double)] = []

        print("\n📊 Full File Analysis - All Tracks")
        print("=" .repeated(70))

        for trackName in trackNames {
            let audioFileName = try VocaditoTestDataLoader.getAudioFileName(for: trackName)
            let audioPath = TestResourceLoader.getVocaditoAudioPath(filename: audioFileName)
            let notes = try VocaditoTestDataLoader.getNotes(for: trackName)

            let samples = try loadFullAudioFile(from: audioPath)

            print("\n📁 \(trackName) (\(String(format: "%.1f", Double(samples.count) / sampleRate))s)")

            let yinFrames = yinStrategy.detectPitch(samples: samples, sampleRate: sampleRate)
            let pyinFrames = pyinStrategy.detectPitch(samples: samples, sampleRate: sampleRate)

            var yinPassed = 0
            var pyinPassed = 0
            var yinErrors: [Double] = []
            var pyinErrors: [Double] = []
            var yinConfs: [Float] = []
            var pyinConfs: [Float] = []

            for (noteIndex, note) in notes.enumerated() {
                let yinResult = evaluateAccuracyFromFrames(frames: yinFrames, targetTime: note.midTime, expectedFrequency: note.frequency)
                let pyinResult = evaluateAccuracyFromFrames(frames: pyinFrames, targetTime: note.midTime, expectedFrequency: note.frequency)

                if yinResult.passed { yinPassed += 1 }
                if pyinResult.passed { pyinPassed += 1 }

                yinErrors.append(yinResult.errorCents)
                pyinErrors.append(pyinResult.errorCents)
                yinConfs.append(yinResult.confidence)
                pyinConfs.append(pyinResult.confidence)

                let yinStatus = yinResult.passed ? "✅" : "❌"
                let pyinStatus = pyinResult.passed ? "✅" : "❌"
                print("  Note\(noteIndex + 1): YIN \(yinStatus) \(String(format: "%5.1f", yinResult.errorCents))c | pYIN \(pyinStatus) \(String(format: "%5.1f", pyinResult.errorCents))c")
            }

            let yinAvgError = yinErrors.reduce(0, +) / Double(notes.count)
            let pyinAvgError = pyinErrors.reduce(0, +) / Double(notes.count)
            let yinAvgConf = Double(yinConfs.reduce(0, +)) / Double(notes.count)
            let pyinAvgConf = Double(pyinConfs.reduce(0, +)) / Double(notes.count)

            allYinResults.append((trackName, yinPassed, notes.count, yinAvgError, yinAvgConf))
            allPyinResults.append((trackName, pyinPassed, notes.count, pyinAvgError, pyinAvgConf))
        }

        // Overall summary
        let totalYinPassed = allYinResults.map { $0.passed }.reduce(0, +)
        let totalPyinPassed = allPyinResults.map { $0.passed }.reduce(0, +)
        let totalTests = allYinResults.map { $0.total }.reduce(0, +)

        let overallYinError = allYinResults.map { $0.avgError }.reduce(0, +) / Double(allYinResults.count)
        let overallPyinError = allPyinResults.map { $0.avgError }.reduce(0, +) / Double(allPyinResults.count)
        let overallYinConf = allYinResults.map { $0.avgConf }.reduce(0, +) / Double(allYinResults.count)
        let overallPyinConf = allPyinResults.map { $0.avgConf }.reduce(0, +) / Double(allPyinResults.count)

        print("\n" + "=" .repeated(70))
        print("📈 OVERALL RESULTS (Full File Analysis):")
        print("   YIN:  \(totalYinPassed)/\(totalTests) (\(String(format: "%.1f", Double(totalYinPassed) / Double(totalTests) * 100))%) | Avg Error: \(String(format: "%.1f", overallYinError)) cents | Avg Conf: \(String(format: "%.2f", overallYinConf))")
        print("   pYIN: \(totalPyinPassed)/\(totalTests) (\(String(format: "%.1f", Double(totalPyinPassed) / Double(totalTests) * 100))%) | Avg Error: \(String(format: "%.1f", overallPyinError)) cents | Avg Conf: \(String(format: "%.2f", overallPyinConf))")
    }

    // MARK: - Helper Methods

    private struct AccuracyResult {
        let passed: Bool
        let errorCents: Double
        let confidence: Float
    }

    private func loadFullAudioFile(from path: String) throws -> [Float] {
        let audioURL = URL(fileURLWithPath: path)
        let audioFile = try AVAudioFile(forReading: audioURL)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw NSError(domain: "TestError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate buffer"])
        }

        try audioFile.read(into: buffer)

        guard let channelData = buffer.floatChannelData else {
            throw NSError(domain: "TestError", code: 2, userInfo: [NSLocalizedDescriptionKey: "No channel data"])
        }

        return Array(UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))
    }

    private func evaluateAccuracyFromFrames(
        frames: [PitchFrame],
        targetTime: Double,
        expectedFrequency: Double
    ) -> AccuracyResult {
        guard !frames.isEmpty else {
            return AccuracyResult(passed: false, errorCents: 999, confidence: 0)
        }

        // Find frame closest to target time
        var closestFrame: PitchFrame?
        var minTimeDiff = Double.infinity

        for frame in frames {
            let timeDiff = abs(frame.timestamp - targetTime)
            if timeDiff < minTimeDiff {
                minTimeDiff = timeDiff
                closestFrame = frame
            }
        }

        guard minTimeDiff < 0.1, let frame = closestFrame, let freq = frame.frequency else {
            return AccuracyResult(passed: false, errorCents: 999, confidence: 0)
        }

        let errorCents = abs(1200.0 * log2(Double(freq) / expectedFrequency))
        let passed = errorCents < 50.0 && frame.confidence > 0.5

        return AccuracyResult(passed: passed, errorCents: errorCents, confidence: frame.confidence)
    }
}

// MARK: - String Extension for Repeat

private extension String {
    func repeated(_ times: Int) -> String {
        return String(repeating: self, count: times)
    }
}
