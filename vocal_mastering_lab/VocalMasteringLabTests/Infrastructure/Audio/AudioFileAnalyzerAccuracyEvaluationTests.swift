import XCTest
import AVFoundation
import VocalisDomain
@testable import VocalMasteringLab

/// AudioFileAnalyzer accuracy evaluation tests using vocadito singing voice dataset
///
/// Uses TestNotes.json for test data with 3 tracks (vocadito_1, vocadito_4, vocadito_7)
/// Each track has 3 notes for pitch detection accuracy testing
///
/// AudioFileAnalyzer analyzes entire audio files and returns PitchAnalysisData with time-series data
/// Tests extract pitch values at note center times and compare against expected frequencies
///
/// ⚠️ SLOW-RUNNING TESTS (execution time: ~450 seconds) - DISABLED BY DEFAULT
/// To enable: Comment out the `try XCTSkipIf(true)` line in setUp()
///
@available(iOS 13.0, *)
final class AudioFileAnalyzerAccuracyEvaluationTests: XCTestCase {

    // MARK: - Properties

    private var audioFileAnalyzer: AudioFileAnalyzer!

    // MARK: - Setup/Teardown

    override func setUp() async throws {
        try await super.setUp()

        // ⚠️ ACCURACY TESTS DISABLED BY DEFAULT - Comment out this line to enable
        try XCTSkipIf(true, "Accuracy tests are slow (~450s). Enable by commenting out this line.")

        // Initialize AudioFileAnalyzer
        audioFileAnalyzer = AudioFileAnalyzer()
    }

    // MARK: - Resource Loading Tests

    /// Test loading test resources from TestNotes.json
    func testLoadTestResources() throws {
        // Load test data from JSON
        let testData = try VocaditoTestDataLoader.loadTestData()

        XCTAssertEqual(testData.tracks.count, 3, "Should have 3 tracks")
        XCTAssertTrue(testData.tracks.keys.contains("vocadito_1"), "Should contain vocadito_1")
        XCTAssertTrue(testData.tracks.keys.contains("vocadito_4"), "Should contain vocadito_4")
        XCTAssertTrue(testData.tracks.keys.contains("vocadito_7"), "Should contain vocadito_7")

        // Verify audio files exist
        let audioPath1 = TestResourceLoader.getVocaditoAudioPath(filename: "vocadito_1.wav")
        XCTAssertTrue(FileManager.default.fileExists(atPath: audioPath1), "vocadito_1.wav should exist")

        let audioPath4 = TestResourceLoader.getVocaditoAudioPath(filename: "vocadito_4.wav")
        XCTAssertTrue(FileManager.default.fileExists(atPath: audioPath4), "vocadito_4.wav should exist")

        let audioPath7 = TestResourceLoader.getVocaditoAudioPath(filename: "vocadito_7.wav")
        XCTAssertTrue(FileManager.default.fileExists(atPath: audioPath7), "vocadito_7.wav should exist")

        print("✅ All test resources loaded successfully")
    }

    // MARK: - Single Note Accuracy Tests

    /// Test pitch detection accuracy for a single note from vocadito using full-file analysis
    func testSingleNoteAccuracy() async throws {
        // Load note data from JSON
        let notes = try VocaditoTestDataLoader.getNotes(for: "vocadito_1")

        guard let firstNote = notes.first else {
            XCTFail("No notes found for vocadito_1")
            return
        }

        // Load audio file
        let audioPath = TestResourceLoader.getVocaditoAudioPath(filename: "vocadito_1.wav")
        let audioURL = URL(fileURLWithPath: audioPath)

        // Analyze entire file
        let (pitchData, _) = try await audioFileAnalyzer.analyze(fileURL: audioURL) { _ in }

        // Find pitch data closest to note center time
        let analysisTime = firstNote.midTime
        guard let detectedPitch = findPitchAtTime(pitchData: pitchData, targetTime: analysisTime) else {
            XCTFail("No pitch data found near time \(analysisTime)s")
            return
        }

        // Verify detection
        let expectedFreq = firstNote.frequency
        let errorCents = abs(1200.0 * log2(Double(detectedPitch.frequency) / expectedFreq))

        print("🎵 Note at \(String(format: "%.2f", firstNote.startTime))s")
        print("   Expected: \(String(format: "%.2f", expectedFreq)) Hz")
        print("   Detected: \(String(format: "%.2f", detectedPitch.frequency)) Hz")
        print("   Error: \(String(format: "%.1f", errorCents)) cents")
        print("   Confidence: \(String(format: "%.3f", detectedPitch.confidence))")

        // Verify accuracy (within 50 cents = half semitone)
        XCTAssertLessThan(errorCents, 50.0, "Pitch detection error exceeds 50 cents")
        XCTAssertGreaterThan(detectedPitch.confidence, 0.5, "Confidence too low")
    }

    // MARK: - Multi-Note Accuracy Tests

    /// Test pitch detection accuracy for Track 1 (vocadito_1), first 3 notes
    func testTrack1_Note1() async throws {
        try await testNoteAccuracy(trackName: "vocadito_1", noteIndex: 0, testName: "Track1_Note1")
    }

    func testTrack1_Note2() async throws {
        try await testNoteAccuracy(trackName: "vocadito_1", noteIndex: 1, testName: "Track1_Note2")
    }

    func testTrack1_Note3() async throws {
        try await testNoteAccuracy(trackName: "vocadito_1", noteIndex: 2, testName: "Track1_Note3")
    }

    /// Test pitch detection accuracy for Track 4 (vocadito_4), first 3 notes
    func testTrack4_Note1() async throws {
        try await testNoteAccuracy(trackName: "vocadito_4", noteIndex: 0, testName: "Track4_Note1")
    }

    func testTrack4_Note2() async throws {
        try await testNoteAccuracy(trackName: "vocadito_4", noteIndex: 1, testName: "Track4_Note2")
    }

    func testTrack4_Note3() async throws {
        try await testNoteAccuracy(trackName: "vocadito_4", noteIndex: 2, testName: "Track4_Note3")
    }

    /// Test pitch detection accuracy for Track 7 (vocadito_7), first 3 notes
    func testTrack7_Note1() async throws {
        try await testNoteAccuracy(trackName: "vocadito_7", noteIndex: 0, testName: "Track7_Note1")
    }

    func testTrack7_Note2() async throws {
        try await testNoteAccuracy(trackName: "vocadito_7", noteIndex: 1, testName: "Track7_Note2")
    }

    func testTrack7_Note3() async throws {
        try await testNoteAccuracy(trackName: "vocadito_7", noteIndex: 2, testName: "Track7_Note3")
    }

    // MARK: - Overall Accuracy Tests (Regression Detection)

    /// Overall accuracy test: Pass if 100% (9/9) of individual note tests pass
    /// This test detects regression in AudioFileAnalyzer pitch detection accuracy
    /// All tests must pass to ensure no regression in pitch detection performance
    func testOverallAccuracy_shouldMaintainPerfectSuccessRate() async throws {
        var passedTests = 0
        var failedTests = 0
        let totalTests = 9  // 3 tracks × 3 notes each
        let requiredPasses = 9  // 9/9 = 100% (all tests must pass for regression detection)

        // Test all notes from all tracks
        let trackNames = try VocaditoTestDataLoader.getAllTrackNames()

        for trackName in trackNames {
            let notes = try VocaditoTestDataLoader.getNotes(for: trackName)

            for (noteIndex, _) in notes.enumerated() {
                let testName = "\(trackName)_Note\(noteIndex + 1)"

                do {
                    let passed = try await checkNoteAccuracy(trackName: trackName, noteIndex: noteIndex)
                    if passed {
                        passedTests += 1
                        print("✅ \(testName) passed")
                    } else {
                        failedTests += 1
                        print("❌ \(testName) failed (accuracy below threshold)")
                    }
                } catch {
                    failedTests += 1
                    print("❌ \(testName) failed with error: \(error)")
                }
            }
        }

        let successRate = Double(passedTests) / Double(totalTests) * 100.0

        print("\n📊 Overall Accuracy Results (AudioFileAnalyzer):")
        print("   Passed: \(passedTests)/\(totalTests)")
        print("   Failed: \(failedTests)/\(totalTests)")
        print("   Success Rate: \(String(format: "%.1f", successRate))%")
        print("   Required: \(requiredPasses)/\(totalTests) passes (100% - all tests must pass)")

        // Require 100% success rate (all 9 tests must pass for regression detection)
        XCTAssertGreaterThanOrEqual(passedTests, requiredPasses,
            "Overall accuracy test failed: Only \(passedTests)/\(totalTests) tests passed (required: \(requiredPasses)/\(totalTests)). Success rate: \(String(format: "%.1f", successRate))%")
    }

    // MARK: - Helper Methods

    /// Check accuracy for a specific note without throwing on failure
    /// Returns true if note passes accuracy thresholds, false otherwise
    private func checkNoteAccuracy(trackName: String, noteIndex: Int) async throws -> Bool {
        let notes = try VocaditoTestDataLoader.getNotes(for: trackName)

        guard noteIndex < notes.count else {
            return false
        }

        let note = notes[noteIndex]
        let audioFileName = try VocaditoTestDataLoader.getAudioFileName(for: trackName)
        let audioPath = TestResourceLoader.getVocaditoAudioPath(filename: audioFileName)
        let audioURL = URL(fileURLWithPath: audioPath)

        // Analyze entire file
        let (pitchData, _) = try await audioFileAnalyzer.analyze(fileURL: audioURL) { _ in }

        // Find pitch data closest to note center time
        let analysisTime = note.midTime
        guard let detectedPitch = findPitchAtTime(pitchData: pitchData, targetTime: analysisTime) else {
            return false
        }

        let expectedFreq = note.frequency
        let errorCents = abs(1200.0 * log2(Double(detectedPitch.frequency) / expectedFreq))

        // Return true if both thresholds are met
        return errorCents < 50.0 && detectedPitch.confidence > 0.5
    }

    /// Test accuracy for a specific note by index
    private func testNoteAccuracy(trackName: String, noteIndex: Int, testName: String) async throws {
        // Load note data from JSON
        let notes = try VocaditoTestDataLoader.getNotes(for: trackName)

        guard noteIndex < notes.count else {
            XCTFail("Note index \(noteIndex) out of range (total: \(notes.count) notes)")
            return
        }

        let note = notes[noteIndex]

        // Load audio file
        let audioFileName = try VocaditoTestDataLoader.getAudioFileName(for: trackName)
        let audioPath = TestResourceLoader.getVocaditoAudioPath(filename: audioFileName)
        let audioURL = URL(fileURLWithPath: audioPath)

        print("\n[DEBUG] Analyzing \(testName) at time \(String(format: "%.3f", note.midTime))s")
        print("[DEBUG] Expected frequency: \(String(format: "%.2f", note.frequency)) Hz")

        // Analyze entire file
        let (pitchData, _) = try await audioFileAnalyzer.analyze(fileURL: audioURL) { _ in }

        print("[DEBUG] Analysis returned \(pitchData.dataPointCount) data points")

        // Find pitch data closest to note center time
        let analysisTime = note.midTime
        guard let detectedPitch = findPitchAtTime(pitchData: pitchData, targetTime: analysisTime) else {
            XCTFail("No pitch data found near time \(String(format: "%.3f", analysisTime))s for \(testName)")
            return
        }

        print("[DEBUG] Found pitch data at time \(String(format: "%.3f", detectedPitch.timestamp))s")

        let expectedFreq = note.frequency
        let errorCents = abs(1200.0 * log2(Double(detectedPitch.frequency) / expectedFreq))

        let endTime = note.startTime + note.duration
        print("\n🎵 \(testName) (\(String(format: "%.2f", note.startTime))s - \(String(format: "%.2f", endTime))s)")
        print("   Expected: \(String(format: "%.2f", expectedFreq)) Hz")
        print("   Detected: \(String(format: "%.2f", detectedPitch.frequency)) Hz")
        print("   Error: \(String(format: "%.1f", errorCents)) cents")
        print("   Confidence: \(String(format: "%.3f", detectedPitch.confidence))")

        // Verify accuracy
        print("[DEBUG] Checking errorCents (\(String(format: "%.1f", errorCents))) < 50.0")
        print("[DEBUG] Checking confidence (\(String(format: "%.3f", detectedPitch.confidence))) > 0.5")

        XCTAssertLessThan(errorCents, 50.0, "\(testName): Pitch error exceeds 50 cents (actual: \(String(format: "%.1f", errorCents)) cents)")
        XCTAssertGreaterThan(detectedPitch.confidence, 0.5, "\(testName): Confidence too low (actual: \(String(format: "%.3f", detectedPitch.confidence)))")
    }

    /// Find pitch data point closest to target time
    /// Returns detected pitch with timestamp, frequency, and confidence
    private func findPitchAtTime(pitchData: PitchAnalysisData, targetTime: Double) -> (timestamp: Double, frequency: Float, confidence: Float)? {
        guard pitchData.dataPointCount > 0 else {
            return nil
        }

        // Find index of timestamp closest to target time
        var closestIndex = 0
        var minTimeDiff = abs(pitchData.timeStamps[0] - targetTime)

        for i in 1..<pitchData.dataPointCount {
            let timeDiff = abs(pitchData.timeStamps[i] - targetTime)
            if timeDiff < minTimeDiff {
                minTimeDiff = timeDiff
                closestIndex = i
            }
        }

        // Verify we're within reasonable time range (within 100ms)
        guard minTimeDiff < 0.1 else {
            print("[WARNING] Closest pitch data is \(String(format: "%.3f", minTimeDiff * 1000))ms away from target time")
            return nil
        }

        return (
            timestamp: pitchData.timeStamps[closestIndex],
            frequency: pitchData.frequencies[closestIndex],
            confidence: pitchData.confidences[closestIndex]
        )
    }
}
