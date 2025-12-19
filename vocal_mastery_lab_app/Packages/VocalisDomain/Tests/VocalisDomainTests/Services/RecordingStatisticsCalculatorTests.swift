import XCTest
@testable import VocalisDomain

final class RecordingStatisticsCalculatorTests: XCTestCase {

    var sut: RecordingStatisticsCalculator!

    override func setUp() {
        super.setUp()
        sut = RecordingStatisticsCalculator()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    // MARK: - Helper Functions

    private func createTimeline(segments: [(note: MIDINote, startTime: TimeInterval, endTime: TimeInterval)]) -> ScalePlaybackTimeline {
        var events: [ScalePlaybackEvent] = []
        for segment in segments {
            events.append(ScalePlaybackEvent(timestamp: segment.startTime, note: segment.note, eventType: .noteStart))
            events.append(ScalePlaybackEvent(timestamp: segment.endTime, note: segment.note, eventType: .noteEnd))
        }
        return ScalePlaybackTimeline(events: events, recordingStartTime: Date())
    }

    private func createPitchData(
        frequencies: [Float],
        confidences: [Float],
        timeStamps: [Double]
    ) -> PitchAnalysisData {
        let targetNotes: [MIDINote?] = Array(repeating: nil, count: timeStamps.count)
        return PitchAnalysisData(
            timeStamps: timeStamps,
            frequencies: frequencies,
            confidences: confidences,
            targetNotes: targetNotes
        )
    }

    // MARK: - Empty Data Tests

    func testCalculate_WithEmptyPitchData_ShouldReturnNil() {
        // Given
        let pitchData = createPitchData(
            frequencies: [],
            confidences: [],
            timeStamps: []
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: nil,
            scaleSettings: nil
        )

        // Then
        XCTAssertNil(result)
    }

    // MARK: - Overall Statistics Tests

    func testCalculate_WithValidData_ShouldReturnOverallStatistics() throws {
        // Given: Pitch data with C4 (261.63 Hz) detected
        let c4Frequency: Float = 261.63
        let pitchData = createPitchData(
            frequencies: [c4Frequency, c4Frequency, c4Frequency],
            confidences: [0.9, 0.9, 0.9],
            timeStamps: [0.0, 0.5, 1.0]
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: nil,
            scaleSettings: nil
        )

        // Then
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.overall.totalSamples, 3)
    }

    func testCalculate_WithLowConfidenceSamples_ShouldExcludeThem() throws {
        // Given: Some samples below confidence threshold (0.5)
        let pitchData = createPitchData(
            frequencies: [261.63, 261.63, 261.63, 261.63],
            confidences: [0.9, 0.3, 0.8, 0.2],  // 2 below threshold
            timeStamps: [0.0, 0.5, 1.0, 1.5]
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: nil,
            scaleSettings: nil
        )

        // Then
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.overall.totalSamples, 2)  // Only high confidence samples
    }

    func testCalculate_WithVocalRange_ShouldReturnLowestAndHighestFrequency() throws {
        // Given: Range from C4 to C5
        let c4: Float = 261.63
        let e4: Float = 329.63
        let g4: Float = 392.00
        let c5: Float = 523.25

        let pitchData = createPitchData(
            frequencies: [c4, e4, g4, c5],
            confidences: [0.9, 0.9, 0.9, 0.9],
            timeStamps: [0.0, 1.0, 2.0, 3.0]
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: nil,
            scaleSettings: nil
        )

        // Then
        XCTAssertNotNil(result?.overall.lowestFrequency)
        XCTAssertNotNil(result?.overall.highestFrequency)
        XCTAssertEqual(result!.overall.lowestFrequency!, Double(c4), accuracy: 0.1)
        XCTAssertEqual(result!.overall.highestFrequency!, Double(c5), accuracy: 0.1)
    }

    // MARK: - Deviation Calculation Tests

    func testCalculate_WithPerfectPitch_ShouldReturnZeroDeviation() throws {
        // Given: Detected pitch exactly matches target
        let c4Note = try MIDINote(60)
        let c4Frequency: Float = 261.63

        let timeline = createTimeline(segments: [(note: c4Note, startTime: 0.0, endTime: 2.0)])

        let pitchData = createPitchData(
            frequencies: [c4Frequency],
            confidences: [0.9],
            timeStamps: [1.0]  // Within segment
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: timeline,
            scaleSettings: nil
        )

        // Then
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.overall.averageDeviationCents ?? 100, 0.0, accuracy: 1.0)
    }

    func testCalculate_WithSharpPitch_ShouldReturnPositiveDeviation() throws {
        // Given: Detected pitch is sharp (higher than target)
        let c4Note = try MIDINote(60)
        let targetFrequency: Double = 261.63  // C4
        let sharpFrequency: Float = Float(targetFrequency * pow(2.0, 50.0/1200.0))  // 50 cents sharp

        let timeline = createTimeline(segments: [(note: c4Note, startTime: 0.0, endTime: 2.0)])

        let pitchData = createPitchData(
            frequencies: [sharpFrequency],
            confidences: [0.9],
            timeStamps: [1.0]
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: timeline,
            scaleSettings: nil
        )

        // Then
        XCTAssertNotNil(result)
        // Overall deviation is absolute value
        XCTAssertEqual(result?.overall.averageDeviationCents ?? 0, 50.0, accuracy: 1.0)
    }

    // MARK: - Position Statistics Tests

    /// Position statistics should return ALL positions based on playbackPattern.count
    /// fiveToneScale: playbackPattern = [0, 2, 4, 5, 7, 5, 4, 2, 0] = 9 positions
    /// Statistics are calculated from scale settings, NOT from detection results
    func testCalculate_WithScalePattern_ShouldReturnAllPositions() throws {
        // Given: 5-tone scale pattern has 9 positions (playbackPattern.count)
        let scaleSettings = ScaleSettings(
            startNote: try MIDINote(60),
            endNote: try MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: try Tempo(secondsPerNote: 0.5),
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 1,
            descendingKeyCount: 0,
            keyStepInterval: 1
        )

        // Create segments for full scale pattern (9 positions)
        // playbackPattern = [0, 2, 4, 5, 7, 5, 4, 2, 0]
        let playbackPattern = [0, 2, 4, 5, 7, 5, 4, 2, 0]
        var segments: [(note: MIDINote, startTime: TimeInterval, endTime: TimeInterval)] = []
        for (index, interval) in playbackPattern.enumerated() {
            let note = try MIDINote(60 + UInt8(interval))
            let startTime = Double(index) * 0.5
            segments.append((note: note, startTime: startTime, endTime: startTime + 0.5))
        }

        let timeline = createTimeline(segments: segments)

        // Pitch data matching each segment (C4, D4, E4, F4, G4, F4, E4, D4, C4)
        let pitchData = createPitchData(
            frequencies: [261.63, 293.66, 329.63, 349.23, 392.00, 349.23, 329.63, 293.66, 261.63],
            confidences: [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            timeStamps: [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25]
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: timeline,
            scaleSettings: scaleSettings
        )

        // Then: Should have 9 positions (playbackPattern.count)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.positionStatistics.count, 9, "fiveToneScale should have 9 positions")

        // Positions should be 1 through 9
        let positions = result?.positionStatistics.map { $0.position } ?? []
        XCTAssertEqual(positions.sorted(), [1, 2, 3, 4, 5, 6, 7, 8, 9])
    }

    /// Position statistics should return ALL positions even when some have no detection data
    /// This tests that positions are determined by scale settings, not by detection
    func testCalculate_WithPartialDetection_ShouldStillReturnAllPositions() throws {
        // Given: 5-tone scale but only detecting first 3 positions
        let scaleSettings = ScaleSettings(
            startNote: try MIDINote(60),
            endNote: try MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: try Tempo(secondsPerNote: 0.5),
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 1,
            descendingKeyCount: 0,
            keyStepInterval: 1
        )

        // Only create segments for first 3 positions
        let segments: [(note: MIDINote, startTime: TimeInterval, endTime: TimeInterval)] = [
            (note: try MIDINote(60), startTime: 0.0, endTime: 0.5),  // Position 1
            (note: try MIDINote(62), startTime: 0.5, endTime: 1.0),  // Position 2
            (note: try MIDINote(64), startTime: 1.0, endTime: 1.5),  // Position 3
        ]

        let timeline = createTimeline(segments: segments)

        let pitchData = createPitchData(
            frequencies: [261.63, 293.66, 329.63],
            confidences: [0.9, 0.9, 0.9],
            timeStamps: [0.25, 0.75, 1.25]
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: timeline,
            scaleSettings: scaleSettings
        )

        // Then: Should STILL have 9 positions (all positions from scale settings)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.positionStatistics.count, 9, "Should return ALL 9 positions even with partial detection")
    }

    func testCalculate_WithoutScaleSettings_ShouldReturnEmptyPositionStatistics() throws {
        // Given: No scale settings
        let pitchData = createPitchData(
            frequencies: [261.63, 293.66, 329.63],
            confidences: [0.9, 0.9, 0.9],
            timeStamps: [0.0, 0.5, 1.0]
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: nil,
            scaleSettings: nil
        )

        // Then
        XCTAssertNotNil(result)
        XCTAssertTrue(result?.positionStatistics.isEmpty ?? false)
    }

    // MARK: - Pitch Statistics Tests

    /// Pitch statistics should return ALL unique notes from scale settings
    /// calculated across all key changes, NOT from detection results
    /// MVP Default: fiveToneScale, ascendingKeyCount=3, keyStepInterval=1
    /// Keys: [C4, C#4, D4, D#4] (ascending) + [D4, C#4, C4] (descending, peak excluded)
    /// Unique notes across all keys: C4, C#4, D4, D#4, E4, F4, F#4, G4, G#4, A4, A#4 (11 notes)
    func testCalculate_WithKeyProgression_ShouldReturnAllUniqueNotes() throws {
        // Given: MVP default-like settings (ascendingThenDescending, 3 keys up)
        let scaleSettings = ScaleSettings(
            startNote: try MIDINote(60),  // C4
            endNote: try MIDINote(72),
            notePattern: .fiveToneScale,  // intervals = [0, 2, 4, 5, 7]
            tempo: try Tempo(secondsPerNote: 0.5),
            keyProgressionPattern: .ascendingThenDescending,
            ascendingKeyCount: 3,
            descendingKeyCount: 3,
            keyStepInterval: 1  // semitone
        )

        // Minimal pitch data (actual detection doesn't matter for expected notes)
        let pitchData = createPitchData(
            frequencies: [261.63],
            confidences: [0.9],
            timeStamps: [0.5]
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: nil,
            scaleSettings: scaleSettings
        )

        // Then: Should have 11 unique notes from all key changes
        // Key C4:  C4(60), D4(62), E4(64), F4(65), G4(67)
        // Key C#4: C#4(61), D#4(63), F4(65), F#4(66), G#4(68)
        // Key D4:  D4(62), E4(64), F#4(66), G4(67), A4(69)
        // Key D#4: D#4(63), F4(65), G4(67), G#4(68), A#4(70)
        // Unique MIDI: 60,61,62,63,64,65,66,67,68,69,70 = 11 notes
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.pitchStatistics.count, 11, "Should have 11 unique notes across all key changes")
    }

    func testCalculate_WithMultipleNotes_ShouldReturnPitchStatistics() throws {
        // Given: Multiple different notes played (without scale settings)
        let c4Note = try MIDINote(60)
        let e4Note = try MIDINote(64)
        let g4Note = try MIDINote(67)

        let segments: [(note: MIDINote, startTime: TimeInterval, endTime: TimeInterval)] = [
            (note: c4Note, startTime: 0.0, endTime: 1.0),
            (note: e4Note, startTime: 1.0, endTime: 2.0),
            (note: g4Note, startTime: 2.0, endTime: 3.0),
        ]

        let timeline = createTimeline(segments: segments)

        let pitchData = createPitchData(
            frequencies: [261.63, 329.63, 392.00],
            confidences: [0.9, 0.9, 0.9],
            timeStamps: [0.5, 1.5, 2.5]
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: timeline,
            scaleSettings: nil
        )

        // Then: Without scale settings, returns detected notes only
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.pitchStatistics.count, 3)
    }

    func testCalculate_PitchStatistics_ShouldBeSortedByFrequencyDescending() throws {
        // Given: Notes C4, E4, G4
        let c4Note = try MIDINote(60)
        let e4Note = try MIDINote(64)
        let g4Note = try MIDINote(67)

        let segments: [(note: MIDINote, startTime: TimeInterval, endTime: TimeInterval)] = [
            (note: c4Note, startTime: 0.0, endTime: 1.0),
            (note: e4Note, startTime: 1.0, endTime: 2.0),
            (note: g4Note, startTime: 2.0, endTime: 3.0),
        ]

        let timeline = createTimeline(segments: segments)

        let pitchData = createPitchData(
            frequencies: [261.63, 329.63, 392.00],
            confidences: [0.9, 0.9, 0.9],
            timeStamps: [0.5, 1.5, 2.5]
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: timeline,
            scaleSettings: nil
        )

        // Then: Should be sorted G4, E4, C4 (high to low)
        XCTAssertNotNil(result)
        let frequencies = result?.pitchStatistics.map { $0.frequency } ?? []
        XCTAssertEqual(frequencies, frequencies.sorted(by: >))
    }

    func testCalculate_PitchStatistics_ShouldCountOccurrences() throws {
        // Given: Same note played multiple times
        let c4Note = try MIDINote(60)

        let segments: [(note: MIDINote, startTime: TimeInterval, endTime: TimeInterval)] = [
            (note: c4Note, startTime: 0.0, endTime: 1.0),
            (note: c4Note, startTime: 1.0, endTime: 2.0),
            (note: c4Note, startTime: 2.0, endTime: 3.0),
        ]

        let timeline = createTimeline(segments: segments)

        let pitchData = createPitchData(
            frequencies: [261.63, 261.63, 261.63],
            confidences: [0.9, 0.9, 0.9],
            timeStamps: [0.5, 1.5, 2.5]
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: timeline,
            scaleSettings: nil
        )

        // Then
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.pitchStatistics.count, 1)  // Only one unique note
        XCTAssertEqual(result?.pitchStatistics.first?.occurrenceCount, 3)
    }

    // MARK: - Detection Rate Tests

    func testCalculate_DetectionRate_ShouldBeCorrect() throws {
        // Given: 3 samples in segment, 1 sample outside
        let c4Note = try MIDINote(60)
        let timeline = createTimeline(segments: [(note: c4Note, startTime: 1.0, endTime: 4.0)])

        let pitchData = createPitchData(
            frequencies: [261.63, 261.63, 261.63, 261.63],
            confidences: [0.9, 0.9, 0.9, 0.9],
            timeStamps: [0.5, 1.5, 2.5, 3.5]  // First one outside segment
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: timeline,
            scaleSettings: nil
        )

        // Then: 3 out of 4 samples in segment = 75%
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.overall.detectionRate ?? 0, 0.75, accuracy: 0.01)
    }

    // MARK: - MIDINote Extension Tests

    func testMIDINoteNoteName_ForA4_ShouldReturnA4() {
        // Given
        let frequency: Double = 440.0  // A4

        // When
        let noteName = MIDINote.noteName(forFrequency: frequency)

        // Then
        XCTAssertEqual(noteName, "A4")
    }

    func testMIDINoteNoteName_ForMiddleC_ShouldReturnC4() {
        // Given
        let frequency: Double = 261.63  // C4

        // When
        let noteName = MIDINote.noteName(forFrequency: frequency)

        // Then
        XCTAssertEqual(noteName, "C4")
    }

    func testMIDINoteNoteName_ForInvalidFrequency_ShouldReturnNil() {
        // Given
        let frequency: Double = 0.0

        // When
        let noteName = MIDINote.noteName(forFrequency: frequency)

        // Then
        XCTAssertNil(noteName)
    }

    // MARK: - Standard Deviation Tests

    func testCalculate_WithVariedPitches_ShouldHaveNonZeroStdDev() throws {
        // Given: Varied pitch detections (some sharp, some flat)
        let c4Note = try MIDINote(60)
        let targetFrequency: Double = 261.63
        let sharpFrequency: Float = Float(targetFrequency * pow(2.0, 30.0/1200.0))  // 30 cents sharp
        let flatFrequency: Float = Float(targetFrequency * pow(2.0, -30.0/1200.0))  // 30 cents flat

        let timeline = createTimeline(segments: [(note: c4Note, startTime: 0.0, endTime: 3.0)])

        let pitchData = createPitchData(
            frequencies: [sharpFrequency, flatFrequency, 261.63],  // +30, -30, 0
            confidences: [0.9, 0.9, 0.9],
            timeStamps: [0.5, 1.5, 2.5]
        )

        // When
        let result = sut.calculate(
            pitchData: pitchData,
            playbackTimeline: timeline,
            scaleSettings: nil
        )

        // Then
        XCTAssertNotNil(result)
        XCTAssertGreaterThan(result?.overall.deviationStdDev ?? 0, 0)
    }
}
