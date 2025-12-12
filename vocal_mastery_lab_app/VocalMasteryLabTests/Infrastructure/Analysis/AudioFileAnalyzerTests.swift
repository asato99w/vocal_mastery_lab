import XCTest
import AVFoundation
import VocalisDomain
@testable import VocalMasteryLab

/// Unit tests for AudioFileAnalyzer pitch detection accuracy
///
/// These tests verify that AudioFileAnalyzer can accurately detect pitch
/// from audio files using the YIN algorithm implementation.
///
/// Test categories:
/// 1. Single frequency accuracy tests (pure tones with harmonics)
/// 2. Scale progression tests (Do-Re-Mi-Fa-Sol sequence)
/// 3. Confidence score validation
///
/// Accuracy criteria:
/// - Frequency error: ±10Hz for single notes (strict)
/// - Frequency error: ±20Hz for scale transitions (moderate, due to onset/offset)
/// - Confidence score: >0.7 for clean synthetic audio
final class AudioFileAnalyzerTests: XCTestCase {

    // MARK: - Properties

    private var sut: AudioFileAnalyzer!

    // MARK: - Setup/Teardown

    override func setUp() async throws {
        try await super.setUp()
        sut = AudioFileAnalyzer()
    }

    override func tearDown() async throws {
        sut = nil
        try await super.tearDown()
    }

    // MARK: - Single Frequency Accuracy Tests

    /// Test C4 (261.63 Hz) detection accuracy
    func testAnalyze_C4PureTone_DetectsCorrectFrequency() async throws {
        // Given: C4 (261.63 Hz) audio file with harmonics
        let expectedNote = try MIDINote(60) // C4
        let expectedFreq = expectedNote.frequency
        let fileURL = try createTestAudioFile(duration: 1.0, frequency: expectedFreq)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        // When: Analyze the audio file
        let (pitchData, _) = try await sut.analyze(fileURL: fileURL) { _ in }

        // Then: Detected frequencies should be close to expected (±10Hz)
        XCTAssertFalse(pitchData.frequencies.isEmpty, "Should detect pitch data")

        let avgFrequency = pitchData.frequencies.reduce(0, +) / Float(pitchData.frequencies.count)
        let errorHz = abs(avgFrequency - Float(expectedFreq))

        print("C4 Test Results:")
        print("  Expected: \(String(format: "%.2f", expectedFreq)) Hz")
        print("  Detected: \(String(format: "%.2f", avgFrequency)) Hz")
        print("  Error: \(String(format: "%.2f", errorHz)) Hz")
        print("  Avg Confidence: \(String(format: "%.3f", pitchData.confidences.reduce(0, +) / Float(pitchData.confidences.count)))")

        XCTAssertLessThan(errorHz, 10.0, "Frequency error should be within ±10Hz")
        XCTAssertGreaterThan(pitchData.confidences.min() ?? 0, 0.7, "Confidence should be >0.7 for synthetic audio")
    }

    /// Test A4 (440 Hz) detection accuracy - standard tuning reference
    func testAnalyze_A4PureTone_DetectsCorrectFrequency() async throws {
        // Given: A4 (440 Hz) audio file
        let expectedNote = try MIDINote(69) // A4
        let expectedFreq = expectedNote.frequency
        let fileURL = try createTestAudioFile(duration: 1.0, frequency: expectedFreq)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        // When: Analyze the audio file
        let (pitchData, _) = try await sut.analyze(fileURL: fileURL) { _ in }

        // Then: Verify accuracy
        XCTAssertFalse(pitchData.frequencies.isEmpty, "Should detect pitch data")

        let avgFrequency = pitchData.frequencies.reduce(0, +) / Float(pitchData.frequencies.count)
        let errorHz = abs(avgFrequency - Float(expectedFreq))

        print("A4 Test Results:")
        print("  Expected: \(String(format: "%.2f", expectedFreq)) Hz")
        print("  Detected: \(String(format: "%.2f", avgFrequency)) Hz")
        print("  Error: \(String(format: "%.2f", errorHz)) Hz")

        XCTAssertLessThan(errorHz, 10.0, "Frequency error should be within ±10Hz")
    }

    /// Test G4 (392 Hz) detection accuracy
    func testAnalyze_G4PureTone_DetectsCorrectFrequency() async throws {
        // Given: G4 (392 Hz) audio file
        let expectedNote = try MIDINote(67) // G4
        let expectedFreq = expectedNote.frequency
        let fileURL = try createTestAudioFile(duration: 1.0, frequency: expectedFreq)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        // When: Analyze the audio file
        let (pitchData, _) = try await sut.analyze(fileURL: fileURL) { _ in }

        // Then: Verify accuracy
        XCTAssertFalse(pitchData.frequencies.isEmpty, "Should detect pitch data")

        let avgFrequency = pitchData.frequencies.reduce(0, +) / Float(pitchData.frequencies.count)
        let errorHz = abs(avgFrequency - Float(expectedFreq))

        print("G4 Test Results:")
        print("  Expected: \(String(format: "%.2f", expectedFreq)) Hz")
        print("  Detected: \(String(format: "%.2f", avgFrequency)) Hz")
        print("  Error: \(String(format: "%.2f", errorHz)) Hz")

        XCTAssertLessThan(errorHz, 10.0, "Frequency error should be within ±10Hz")
    }

    // MARK: - Scale Progression Tests

    /// Test 5-note scale (Do-Re-Mi-Fa-Sol) progression detection
    func testAnalyze_FiveNoteScale_DetectsAllNotes() async throws {
        // Given: 5-note scale audio file (C4-D4-E4-F4-G4, 1 second each)
        let scale = [
            try MIDINote(60), // C4 (Do)
            try MIDINote(62), // D4 (Re)
            try MIDINote(64), // E4 (Mi)
            try MIDINote(65), // F4 (Fa)
            try MIDINote(67), // G4 (Sol)
        ]

        let fileURL = try createScaleAudioFile(notes: scale, notesDuration: 1.0)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        // When: Analyze the scale audio file
        let (pitchData, _) = try await sut.analyze(fileURL: fileURL) { _ in }

        // Then: Should detect pitch throughout the file
        XCTAssertFalse(pitchData.frequencies.isEmpty, "Should detect pitch data")
        XCTAssertGreaterThanOrEqual(pitchData.timeStamps.count, 50, "Should have substantial pitch data points")

        print("\nScale Progression Test Results:")
        print("  Total data points: \(pitchData.timeStamps.count)")
        print("  Duration: \(String(format: "%.2f", pitchData.timeStamps.last ?? 0)) seconds")

        // Verify each note segment (with tolerance for transitions)
        for (index, note) in scale.enumerated() {
            let segmentStart = Double(index) * 1.0 + 0.3  // Skip 300ms onset
            let segmentEnd = Double(index) * 1.0 + 0.7    // Skip 300ms offset

            // Find pitch data points in this time segment
            let segmentIndices = pitchData.timeStamps.enumerated().filter { idx, timestamp in
                timestamp >= segmentStart && timestamp <= segmentEnd
            }.map { $0.offset }

            guard !segmentIndices.isEmpty else {
                XCTFail("No pitch data found for note \(index + 1) (expected around \(String(format: "%.1f", Double(index) * 1.0 + 0.5))s)")
                continue
            }

            let segmentFrequencies = segmentIndices.map { pitchData.frequencies[$0] }
            let avgFreq = segmentFrequencies.reduce(0, +) / Float(segmentFrequencies.count)
            let expectedFreq = note.frequency
            let errorHz = abs(avgFreq - Float(expectedFreq))

            print("  Note \(index + 1) (\(note.noteName)): Expected \(String(format: "%.2f", expectedFreq)) Hz, " +
                  "Detected \(String(format: "%.2f", avgFreq)) Hz, " +
                  "Error \(String(format: "%.2f", errorHz)) Hz")

            // More lenient threshold for scale progression (±20Hz) due to onset/offset effects
            XCTAssertLessThan(errorHz, 20.0, "Note \(index + 1) error should be within ±20Hz")
        }
    }

    // MARK: - Confidence Score Tests

    /// Test that confidence scores are reasonable for clean synthetic audio
    func testAnalyze_CleanAudio_ReturnsHighConfidence() async throws {
        // Given: Clean C4 audio with strong harmonics
        let expectedNote = try MIDINote(60)
        let fileURL = try createTestAudioFile(duration: 1.0, frequency: expectedNote.frequency)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        // When: Analyze the audio file
        let (pitchData, _) = try await sut.analyze(fileURL: fileURL) { _ in }

        // Then: Confidence should be consistently high
        let avgConfidence = pitchData.confidences.reduce(0, +) / Float(pitchData.confidences.count)
        let minConfidence = pitchData.confidences.min() ?? 0

        print("Confidence Score Test Results:")
        print("  Average Confidence: \(String(format: "%.3f", avgConfidence))")
        print("  Minimum Confidence: \(String(format: "%.3f", minConfidence))")
        print("  Confidence Distribution:")
        let highConfCount = pitchData.confidences.filter { $0 > 0.8 }.count
        let medConfCount = pitchData.confidences.filter { $0 > 0.5 && $0 <= 0.8 }.count
        let lowConfCount = pitchData.confidences.filter { $0 <= 0.5 }.count
        print("    High (>0.8): \(highConfCount) (\(String(format: "%.1f", Double(highConfCount) / Double(pitchData.confidences.count) * 100))%)")
        print("    Medium (0.5-0.8): \(medConfCount)")
        print("    Low (≤0.5): \(lowConfCount)")

        XCTAssertGreaterThan(avgConfidence, 0.7, "Average confidence should be >0.7 for clean synthetic audio")
        XCTAssertGreaterThan(minConfidence, 0.5, "Minimum confidence should be >0.5")
    }

    // MARK: - Progress Reporting Tests

    /// Test that progress callback is called during analysis
    func testAnalyze_ReportsProgress() async throws {
        // Given: Test audio file
        let expectedNote = try MIDINote(60)
        let fileURL = try createTestAudioFile(duration: 1.0, frequency: expectedNote.frequency)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        var progressUpdates: [Double] = []

        // When: Analyze with progress tracking
        _ = try await sut.analyze(fileURL: fileURL) { progress in
            progressUpdates.append(progress)
        }

        // Then: Progress should be reported
        XCTAssertFalse(progressUpdates.isEmpty, "Progress callback should be called")
        XCTAssertTrue(progressUpdates.contains(0.0), "Progress should start at 0.0")
        XCTAssertTrue(progressUpdates.contains(1.0), "Progress should end at 1.0")
        XCTAssertEqual(progressUpdates.first, 0.0, "First progress update should be 0.0")
        XCTAssertEqual(progressUpdates.last, 1.0, "Last progress update should be 1.0")

        print("Progress Updates:")
        print("  Count: \(progressUpdates.count)")
        print("  Values: \(progressUpdates.prefix(10).map { String(format: "%.1f", $0 * 100) }.joined(separator: "%, "))%...")
    }

    // MARK: - Helper Methods

    /// Create a test audio file with a specific frequency
    /// - Parameters:
    ///   - duration: Duration in seconds
    ///   - frequency: Fundamental frequency in Hz
    /// - Returns: URL of the created audio file
    private func createTestAudioFile(duration: TimeInterval, frequency: Double) throws -> URL {
        let sampleRate = 44100.0
        let frameCount = AVAudioFrameCount(duration * sampleRate)

        let format = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: 1
        )!

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameCount
        ) else {
            throw NSError(domain: "AudioFileAnalyzerTests", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffer"])
        }

        buffer.frameLength = frameCount

        // Generate sine wave with harmonics (to simulate natural sound)
        guard let floatChannelData = buffer.floatChannelData else {
            throw NSError(domain: "AudioFileAnalyzerTests", code: -2, userInfo: [NSLocalizedDescriptionKey: "No channel data"])
        }

        let samples = floatChannelData[0]
        for frame in 0..<Int(frameCount) {
            let time = Double(frame) / sampleRate

            // Fundamental + 3 harmonics (2nd, 3rd, 4th) with decreasing amplitudes
            let fundamental = sin(2.0 * .pi * frequency * time)
            let harmonic2 = 0.5 * sin(2.0 * .pi * frequency * 2.0 * time)
            let harmonic3 = 0.25 * sin(2.0 * .pi * frequency * 3.0 * time)
            let harmonic4 = 0.125 * sin(2.0 * .pi * frequency * 4.0 * time)

            let value = Float(fundamental + harmonic2 + harmonic3 + harmonic4)
            samples[frame] = value * 0.3  // Scale to avoid clipping
        }

        // Write to temporary file
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("test_audio_\(UUID().uuidString).wav")

        let audioFile = try AVAudioFile(
            forWriting: fileURL,
            settings: [
                AVFormatIDKey: Int(kAudioFormatLinearPCM) as Any,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsBigEndianKey: false
            ]
        )

        try audioFile.write(from: buffer)

        return fileURL
    }

    /// Create a test audio file with a scale progression
    /// - Parameters:
    ///   - notes: Array of MIDI notes to generate
    ///   - notesDuration: Duration of each note in seconds
    /// - Returns: URL of the created audio file
    private func createScaleAudioFile(notes: [MIDINote], notesDuration: TimeInterval) throws -> URL {
        let sampleRate = 44100.0
        let totalDuration = TimeInterval(notes.count) * notesDuration
        let frameCount = AVAudioFrameCount(totalDuration * sampleRate)

        let format = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: 1
        )!

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameCount
        ) else {
            throw NSError(domain: "AudioFileAnalyzerTests", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffer"])
        }

        buffer.frameLength = frameCount

        guard let floatChannelData = buffer.floatChannelData else {
            throw NSError(domain: "AudioFileAnalyzerTests", code: -2, userInfo: [NSLocalizedDescriptionKey: "No channel data"])
        }

        let samples = floatChannelData[0]

        for frame in 0..<Int(frameCount) {
            let time = Double(frame) / sampleRate
            let noteIndex = Int(time / notesDuration)

            guard noteIndex < notes.count else { break }

            let frequency = notes[noteIndex].frequency

            // Generate with harmonics
            let fundamental = sin(2.0 * .pi * frequency * time)
            let harmonic2 = 0.5 * sin(2.0 * .pi * frequency * 2.0 * time)
            let harmonic3 = 0.25 * sin(2.0 * .pi * frequency * 3.0 * time)
            let harmonic4 = 0.125 * sin(2.0 * .pi * frequency * 4.0 * time)

            let value = Float(fundamental + harmonic2 + harmonic3 + harmonic4)
            samples[frame] = value * 0.3
        }

        // Write to temporary file
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("test_scale_\(UUID().uuidString).wav")

        let audioFile = try AVAudioFile(
            forWriting: fileURL,
            settings: [
                AVFormatIDKey: Int(kAudioFormatLinearPCM) as Any,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsBigEndianKey: false
            ]
        )

        try audioFile.write(from: buffer)

        return fileURL
    }
}
