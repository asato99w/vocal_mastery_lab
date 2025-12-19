import XCTest
@testable import VocalisDomain

final class OctaveCorrectionServiceTests: XCTestCase {

    var sut: OctaveCorrectionService!

    override func setUp() {
        super.setUp()
        sut = OctaveCorrectionService()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    // MARK: - Single Frequency Correction Tests

    /// Test: Frequency one octave below target should be corrected up
    func testCorrectFrequency_oneOctaveBelow_shouldCorrectUp() throws {
        // Given: A4 (440Hz) as target, detected at A3 (220Hz) - one octave below
        let targetNote = try MIDINote(69)  // A4 = 440Hz
        let detectedFreq: Float = 220.0     // A3 = one octave below

        // When
        let corrected = sut.correctFrequency(detectedFreq, targetNote: targetNote)

        // Then: Should correct to 440Hz (±5%)
        XCTAssertEqual(corrected, 440.0, accuracy: 22.0,
            "220Hz detected against A4 target should correct to ~440Hz, got \(corrected)")
    }

    /// Test: Frequency one octave above target should be corrected down
    func testCorrectFrequency_oneOctaveAbove_shouldCorrectDown() throws {
        // Given: A4 (440Hz) as target, detected at A5 (880Hz) - one octave above
        let targetNote = try MIDINote(69)  // A4 = 440Hz
        let detectedFreq: Float = 880.0     // A5 = one octave above

        // When
        let corrected = sut.correctFrequency(detectedFreq, targetNote: targetNote)

        // Then: Should correct to 440Hz (±5%)
        XCTAssertEqual(corrected, 440.0, accuracy: 22.0,
            "880Hz detected against A4 target should correct to ~440Hz, got \(corrected)")
    }

    /// Test: Frequency already in correct octave should not change
    func testCorrectFrequency_alreadyCorrect_shouldNotChange() throws {
        // Given: A4 (440Hz) as target, detected at 445Hz - slightly sharp but correct octave
        let targetNote = try MIDINote(69)  // A4 = 440Hz
        let detectedFreq: Float = 445.0     // slightly sharp

        // When
        let corrected = sut.correctFrequency(detectedFreq, targetNote: targetNote)

        // Then: Should remain unchanged
        XCTAssertEqual(corrected, 445.0, accuracy: 0.1,
            "445Hz should remain unchanged, got \(corrected)")
    }

    /// Test: Two octaves below should correct properly
    func testCorrectFrequency_twoOctavesBelow_shouldCorrectUp() throws {
        // Given: A4 (440Hz) as target, detected at A2 (110Hz) - two octaves below
        let targetNote = try MIDINote(69)  // A4 = 440Hz
        let detectedFreq: Float = 110.0     // A2 = two octaves below

        // When
        let corrected = sut.correctFrequency(detectedFreq, targetNote: targetNote)

        // Then: Should correct to 440Hz (±5%)
        XCTAssertEqual(corrected, 440.0, accuracy: 22.0,
            "110Hz detected against A4 target should correct to ~440Hz, got \(corrected)")
    }

    /// Test: C4 target with C3 detection
    func testCorrectFrequency_C4TargetC3Detected_shouldCorrect() throws {
        // Given: C4 (261.63Hz) as target, detected at C3 (130.81Hz)
        let targetNote = try MIDINote(60)  // C4 ≈ 261.63Hz
        let detectedFreq: Float = 130.81    // C3 = one octave below

        // When
        let corrected = sut.correctFrequency(detectedFreq, targetNote: targetNote)

        // Then: Should correct to ~261.63Hz
        XCTAssertEqual(corrected, 261.63, accuracy: 15.0,
            "130.81Hz (C3) detected against C4 target should correct to ~261.63Hz, got \(corrected)")
    }

    // MARK: - PitchAnalysisData Correction Tests

    /// Test: Apply correction to PitchAnalysisData with segments
    func testApplyCorrection_withSegments_shouldCorrectFrequencies() throws {
        // Given: PitchData with one octave error
        let timestamps = [0.5, 1.5]
        let frequencies: [Float] = [220.0, 440.0]  // A3 (wrong), A4 (correct)
        let confidences: [Float] = [0.9, 0.9]
        let targetNote = try MIDINote(69)  // A4 = 440Hz

        let pitchData = PitchAnalysisData(
            timeStamps: timestamps,
            frequencies: frequencies,
            confidences: confidences,
            targetNotes: [nil, nil],
            amplitudes: [0.8, 0.8]
        )

        // Segment covering both timestamps with A4 target
        let segment = NoteSegment(
            note: targetNote,
            startTime: 0.0,
            endTime: 2.0
        )

        // When
        let corrected = sut.applyCorrection(to: pitchData, segments: [segment])

        // Then: First frequency should be corrected, second unchanged
        XCTAssertEqual(corrected.frequencies[0], 440.0, accuracy: 22.0,
            "220Hz should be corrected to ~440Hz")
        XCTAssertEqual(corrected.frequencies[1], 440.0, accuracy: 22.0,
            "440Hz should remain ~440Hz")

        // Timestamps and confidences should be unchanged
        XCTAssertEqual(corrected.timeStamps, timestamps)
        XCTAssertEqual(corrected.confidences, confidences)
    }

    /// Test: Frequencies outside segments should not be corrected
    func testApplyCorrection_outsideSegment_shouldNotCorrect() throws {
        // Given: PitchData with timestamp outside segment
        let timestamps = [0.5, 2.5]  // Second is outside segment
        let frequencies: [Float] = [220.0, 220.0]  // Both A3
        let confidences: [Float] = [0.9, 0.9]
        let targetNote = try MIDINote(69)  // A4 = 440Hz

        let pitchData = PitchAnalysisData(
            timeStamps: timestamps,
            frequencies: frequencies,
            confidences: confidences,
            targetNotes: [nil, nil],
            amplitudes: [0.8, 0.8]
        )

        // Segment only covers first timestamp
        let segment = NoteSegment(
            note: targetNote,
            startTime: 0.0,
            endTime: 1.0  // Ends before second timestamp
        )

        // When
        let corrected = sut.applyCorrection(to: pitchData, segments: [segment])

        // Then: First should be corrected, second unchanged
        XCTAssertEqual(corrected.frequencies[0], 440.0, accuracy: 22.0,
            "220Hz in segment should be corrected to ~440Hz")
        XCTAssertEqual(corrected.frequencies[1], 220.0, accuracy: 0.1,
            "220Hz outside segment should remain unchanged")
    }

    /// Test: Empty segments should return unchanged data
    func testApplyCorrection_emptySegments_shouldNotChange() {
        // Given: PitchData with no segments
        let pitchData = PitchAnalysisData(
            timeStamps: [0.5],
            frequencies: [220.0],
            confidences: [0.9],
            targetNotes: [nil],
            amplitudes: [0.8]
        )

        // When
        let corrected = sut.applyCorrection(to: pitchData, segments: [])

        // Then: Should be unchanged
        XCTAssertEqual(corrected.frequencies[0], 220.0, accuracy: 0.1)
    }

    /// Test: Multiple segments with different target notes
    func testApplyCorrection_multipleSegments_shouldCorrectBasedOnSegment() throws {
        // Given: Two segments with different target notes
        let timestamps = [0.5, 1.5]
        let frequencies: [Float] = [220.0, 130.81]  // A3, C3
        let confidences: [Float] = [0.9, 0.9]

        let pitchData = PitchAnalysisData(
            timeStamps: timestamps,
            frequencies: frequencies,
            confidences: confidences,
            targetNotes: [nil, nil],
            amplitudes: [0.8, 0.8]
        )

        // First segment: A4 target (0-1s)
        // Second segment: C4 target (1-2s)
        let segments = [
            NoteSegment(note: try MIDINote(69), startTime: 0.0, endTime: 1.0),  // A4
            NoteSegment(note: try MIDINote(60), startTime: 1.0, endTime: 2.0)   // C4
        ]

        // When
        let corrected = sut.applyCorrection(to: pitchData, segments: segments)

        // Then: Each should be corrected to its target octave
        XCTAssertEqual(corrected.frequencies[0], 440.0, accuracy: 22.0,
            "220Hz (A3) should correct to A4 (440Hz)")
        XCTAssertEqual(corrected.frequencies[1], 261.63, accuracy: 15.0,
            "130.81Hz (C3) should correct to C4 (261.63Hz)")
    }
}
