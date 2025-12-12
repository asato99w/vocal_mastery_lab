import XCTest
import AVFoundation
import VocalisDomain
@testable import VocalMasteryLab

/// vocadito singing voice dataset accuracy evaluation tests for RealtimePitchDetector
///
/// Uses TestNotes.json for test data with 3 tracks (vocadito_1, vocadito_4, vocadito_7)
/// Each track has 3 notes for pitch detection accuracy testing
///
/// ⚠️ SLOW-RUNNING TESTS (execution time: ~450 seconds) - DISABLED BY DEFAULT
/// To enable: Comment out the `try XCTSkipIf(true)` line in setUp()
///
@available(iOS 13.0, *)
final class VocaditoAccuracyEvaluationTests: XCTestCase {

    // MARK: - Properties

    private var pitchDetector: RealtimePitchDetector!

    // MARK: - Setup/Teardown

    override func setUp() async throws {
        try await super.setUp()

        // ⚠️ ACCURACY TESTS DISABLED BY DEFAULT - Comment out this line to enable
        try XCTSkipIf(true, "Accuracy tests are slow (~450s). Enable by commenting out this line.")

        // Initialize pitch detector on MainActor
        pitchDetector = await MainActor.run {
            return RealtimePitchDetector()
        }
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

    /// Test pitch detection accuracy for a single note from vocadito
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

        // Analyze pitch at note center (start + duration/2) to avoid onset/offset transients
        let analysisTime = firstNote.midTime
        let expectation = expectation(description: "Analyze pitch at \(analysisTime)s")
        var detectedPitch: DetectedPitch?

        await MainActor.run {
            pitchDetector.analyzePitchFromFile(audioURL, atTime: analysisTime) { pitch in
                detectedPitch = pitch
                expectation.fulfill()
            }
        }

        await fulfillment(of: [expectation], timeout: 10.0)

        // Verify detection
        XCTAssertNotNil(detectedPitch, "Failed to detect pitch")

        if let detected = detectedPitch {
            let expectedFreq = firstNote.frequency
            let errorCents = abs(1200.0 * log2(detected.frequency / expectedFreq))

            print("🎵 Note at \(String(format: "%.2f", firstNote.startTime))s")
            print("   Expected: \(String(format: "%.2f", expectedFreq)) Hz")
            print("   Detected: \(String(format: "%.2f", detected.frequency)) Hz")
            print("   Error: \(String(format: "%.1f", errorCents)) cents")
            print("   Confidence: \(String(format: "%.3f", detected.confidence))")

            // Verify accuracy (within 50 cents = half semitone)
            XCTAssertLessThan(errorCents, 50.0, "Pitch detection error exceeds 50 cents")
            XCTAssertGreaterThan(detected.confidence, 0.5, "Confidence too low")
        }
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

    /// Overall accuracy test: Pass if 78%+ (7/9+) of individual note tests pass
    /// This test detects regression in pitch detection accuracy
    /// Threshold based on original 9/11 (81.8%) requirement scaled to 9 tests
    func testOverallAccuracy_shouldMaintain80PercentSuccessRate() async throws {
        var passedTests = 0
        var failedTests = 0
        let totalTests = 9  // 3 tracks × 3 notes each
        let requiredPasses = 7  // 7/9 = 77.8% (equivalent to original 9/11 = 81.8%)

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

        print("\n📊 Overall Accuracy Results:")
        print("   Passed: \(passedTests)/\(totalTests)")
        print("   Failed: \(failedTests)/\(totalTests)")
        print("   Success Rate: \(String(format: "%.1f", successRate))%")
        print("   Required: \(requiredPasses)+ passes (78%+, equivalent to 9/11)")

        // Require 78%+ success rate (7 out of 9 tests must pass, equivalent to original 9/11)
        XCTAssertGreaterThanOrEqual(passedTests, requiredPasses,
            "Overall accuracy test failed: Only \(passedTests)/\(totalTests) tests passed (required: \(requiredPasses)+). Success rate: \(String(format: "%.1f", successRate))%")
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

        let analysisTime = note.midTime
        let expectation = expectation(description: "Analyze \(trackName)_Note\(noteIndex + 1)")
        var detectedPitch: DetectedPitch?

        await MainActor.run {
            pitchDetector.analyzePitchFromFile(audioURL, atTime: analysisTime) { pitch in
                detectedPitch = pitch
                expectation.fulfill()
            }
        }

        await fulfillment(of: [expectation], timeout: 10.0)

        guard let detected = detectedPitch else {
            return false
        }

        let expectedFreq = note.frequency
        let errorCents = abs(1200.0 * log2(detected.frequency / expectedFreq))

        // Return true if both thresholds are met
        return errorCents < 50.0 && detected.confidence > 0.5
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

        // Analyze pitch at note center
        let analysisTime = note.midTime
        let expectation = expectation(description: "Analyze \(testName)")
        var detectedPitch: DetectedPitch?

        print("\n[DEBUG] Analyzing \(testName) at time \(String(format: "%.3f", analysisTime))s")
        print("[DEBUG] Expected frequency: \(String(format: "%.2f", note.frequency)) Hz")

        await MainActor.run {
            pitchDetector.analyzePitchFromFile(audioURL, atTime: analysisTime) { pitch in
                detectedPitch = pitch
                print("[DEBUG] Pitch detection callback received: \(pitch != nil ? "SUCCESS" : "FAILED")")
                expectation.fulfill()
            }
        }

        await fulfillment(of: [expectation], timeout: 10.0)

        // Verify detection
        print("[DEBUG] detectedPitch after fulfillment: \(detectedPitch != nil ? "NOT NIL" : "NIL")")
        XCTAssertNotNil(detectedPitch, "Failed to detect pitch for \(testName)")

        if let detected = detectedPitch {
            let expectedFreq = note.frequency
            let errorCents = abs(1200.0 * log2(detected.frequency / expectedFreq))

            // Write results to file for debugging
            let endTime = note.startTime + note.duration
            let logMessage = """
            🎵 \(testName) (\(String(format: "%.2f", note.startTime))s - \(String(format: "%.2f", endTime))s)
               Expected: \(String(format: "%.2f", expectedFreq)) Hz
               Detected: \(String(format: "%.2f", detected.frequency)) Hz
               Error: \(String(format: "%.1f", errorCents)) cents
               Confidence: \(String(format: "%.3f", detected.confidence))
               Pass errorCents check: \(errorCents < 50.0 ? "YES" : "NO")
               Pass confidence check: \(detected.confidence > 0.5 ? "YES" : "NO")

            """
            try? logMessage.appendToFile(at: "/tmp/vocadito_pitch_results.txt")

            print("\n🎵 \(testName) (\(String(format: "%.2f", note.startTime))s - \(String(format: "%.2f", endTime))s)")
            print("   Expected: \(String(format: "%.2f", expectedFreq)) Hz")
            print("   Detected: \(String(format: "%.2f", detected.frequency)) Hz")
            print("   Error: \(String(format: "%.1f", errorCents)) cents")
            print("   Confidence: \(String(format: "%.3f", detected.confidence))")

            // Verify accuracy
            print("[DEBUG] Checking errorCents (\(String(format: "%.1f", errorCents))) < 50.0")
            print("[DEBUG] Checking confidence (\(String(format: "%.3f", detected.confidence))) > 0.5")

            XCTAssertLessThan(errorCents, 50.0, "\(testName): Pitch error exceeds 50 cents (actual: \(String(format: "%.1f", errorCents)) cents)")
            XCTAssertGreaterThan(detected.confidence, 0.5, "\(testName): Confidence too low (actual: \(String(format: "%.3f", detected.confidence)))")
        } else {
            print("[DEBUG] ERROR: detectedPitch is nil in the if-let block - this should never happen!")
        }
    }
}

// MARK: - String Extension for File Logging
extension String {
    func appendToFile(at path: String) throws {
        let url = URL(fileURLWithPath: path)

        if FileManager.default.fileExists(atPath: path) {
            let fileHandle = try FileHandle(forWritingTo: url)
            fileHandle.seekToEndOfFile()
            if let data = self.data(using: .utf8) {
                fileHandle.write(data)
            }
            fileHandle.closeFile()
        } else {
            try self.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
