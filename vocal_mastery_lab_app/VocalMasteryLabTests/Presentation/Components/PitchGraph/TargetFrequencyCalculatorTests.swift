import XCTest
import VocalisDomain
@testable import VocalMasteryLab

final class TargetFrequencyCalculatorTests: XCTestCase {
    var sut: TargetFrequencyCalculator!

    override func setUp() {
        super.setUp()
        sut = TargetFrequencyCalculator()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    // MARK: - Key Progression Tests

    /// Test: Target frequencies should include all notes from key progression
    /// Given: C3 start, 3 ascending keys with semitone interval, five-tone scale
    /// Expected: Frequencies from C3, C#3, and D3 keys should all be included
    ///
    /// Key C3 (root=48): C3(48), D3(50), E3(52), F3(53), G3(55)
    /// Key C#3 (root=49): C#3(49), D#3(51), F3(53), F#3(54), G#3(56)
    /// Key D3 (root=50): D3(50), E3(52), F#3(54), G3(55), A3(57)
    ///
    /// Current bug: Only C3 key notes are calculated (octave-based iteration)
    func testCalculateTargetFrequencies_WithKeyProgression_ShouldIncludeAllKeyNotes() {
        // Given: 3 ascending keys with semitone interval
        let settings = ScaleSettings(
            startNote: try! MIDINote(48),  // C3
            endNote: try! MIDINote(72),    // C5
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 1.0),
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 3,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 1  // Semitone
        )

        // When
        let frequencies = sut.calculateTargetFrequencies(from: settings)

        // Then: Should include notes from all 3 keys
        // C#3 frequency (MIDI 49) = 138.59 Hz
        let cSharp3Frequency = try! MIDINote(49).frequency

        // D#3 frequency (MIDI 51) = 155.56 Hz
        let dSharp3Frequency = try! MIDINote(51).frequency

        // A3 frequency (MIDI 57) = 220.00 Hz
        let a3Frequency = try! MIDINote(57).frequency

        // These notes should be included (from C#3 and D3 keys)
        XCTAssertTrue(
            frequencies.contains { abs($0 - cSharp3Frequency) < 0.1 },
            "Should include C#3 (\(cSharp3Frequency) Hz) from key C#3. Got: \(frequencies)"
        )

        XCTAssertTrue(
            frequencies.contains { abs($0 - dSharp3Frequency) < 0.1 },
            "Should include D#3 (\(dSharp3Frequency) Hz) from key C#3. Got: \(frequencies)"
        )

        XCTAssertTrue(
            frequencies.contains { abs($0 - a3Frequency) < 0.1 },
            "Should include A3 (\(a3Frequency) Hz) from key D3. Got: \(frequencies)"
        )

        // Should NOT include notes from keys that are not played
        // C4 key (MIDI 60) is NOT in the progression (only C3, C#3, D3)
        let c4Frequency = try! MIDINote(60).frequency  // 261.63 Hz
        let d4Frequency = try! MIDINote(62).frequency  // 293.66 Hz

        XCTAssertFalse(
            frequencies.contains { abs($0 - c4Frequency) < 0.1 },
            "Should NOT include C4 (\(c4Frequency) Hz) - not in key progression"
        )

        XCTAssertFalse(
            frequencies.contains { abs($0 - d4Frequency) < 0.1 },
            "Should NOT include D4 (\(d4Frequency) Hz) - not in key progression"
        )
    }

    /// Test: With whole-tone key progression, frequencies should follow the progression
    /// Given: C3 start, 3 ascending keys with whole-tone (2 semitone) interval
    /// Expected: Frequencies from C3, D3, and E3 keys
    func testCalculateTargetFrequencies_WithWholeToneProgression_ShouldIncludeAllKeyNotes() {
        // Given: 3 ascending keys with whole-tone interval
        let settings = ScaleSettings(
            startNote: try! MIDINote(48),  // C3
            endNote: try! MIDINote(72),    // C5
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 1.0),
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 3,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 2  // Whole tone
        )

        // When
        let frequencies = sut.calculateTargetFrequencies(from: settings)

        // Then: Should include notes from C3, D3, E3 keys
        // Key E3 (root=52): E3(52), F#3(54), G#3(56), A3(57), B3(59)

        // G#3 frequency (MIDI 56) = 207.65 Hz (from E3 key's 4th scale degree)
        let gSharp3Frequency = try! MIDINote(56).frequency

        // B3 frequency (MIDI 59) = 246.94 Hz (from E3 key's 5th scale degree)
        let b3Frequency = try! MIDINote(59).frequency

        XCTAssertTrue(
            frequencies.contains { abs($0 - gSharp3Frequency) < 0.1 },
            "Should include G#3 (\(gSharp3Frequency) Hz) from key E3. Got: \(frequencies)"
        )

        XCTAssertTrue(
            frequencies.contains { abs($0 - b3Frequency) < 0.1 },
            "Should include B3 (\(b3Frequency) Hz) from key E3. Got: \(frequencies)"
        )
    }

    /// Test: Ascending then descending pattern should include all keys
    func testCalculateTargetFrequencies_WithAscendingThenDescending_ShouldIncludeAllKeyNotes() {
        // Given: Ascending 2 keys, then descending 2 keys
        let settings = ScaleSettings(
            startNote: try! MIDINote(48),  // C3
            endNote: try! MIDINote(72),    // C5
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 1.0),
            keyProgressionPattern: .ascendingThenDescending,
            ascendingKeyCount: 2,
            descendingKeyCount: 2,
            ascendingKeyStepInterval: 1,
            descendingKeyStepInterval: 1
        )

        // When
        let frequencies = sut.calculateTargetFrequencies(from: settings)

        // Then: Should include notes from all keys in the progression
        // Ascending: C3(48) -> C#3(49)
        // Descending from C#3: C3(48) -> B2(47)

        // C#3 (root of second ascending key)
        let cSharp3Frequency = try! MIDINote(49).frequency

        XCTAssertTrue(
            frequencies.contains { abs($0 - cSharp3Frequency) < 0.1 },
            "Should include C#3 from ascending keys. Got: \(frequencies)"
        )
    }

    // MARK: - Basic Functionality Tests (these should pass)

    /// Test: Basic case - single key should return correct frequencies
    func testCalculateTargetFrequencies_SingleKey_ReturnsCorrectNotes() {
        // Given: Single key (1 ascending)
        let settings = ScaleSettings(
            startNote: try! MIDINote(60),  // C4
            endNote: try! MIDINote(72),    // C5
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 1.0),
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 1,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 1
        )

        // When
        let frequencies = sut.calculateTargetFrequencies(from: settings)

        // Then: Should include C4, D4, E4, F4, G4
        let c4Frequency = try! MIDINote(60).frequency  // 261.63 Hz
        let d4Frequency = try! MIDINote(62).frequency  // 293.66 Hz
        let e4Frequency = try! MIDINote(64).frequency  // 329.63 Hz
        let f4Frequency = try! MIDINote(65).frequency  // 349.23 Hz
        let g4Frequency = try! MIDINote(67).frequency  // 392.00 Hz

        XCTAssertTrue(frequencies.contains { abs($0 - c4Frequency) < 0.1 }, "Should include C4")
        XCTAssertTrue(frequencies.contains { abs($0 - d4Frequency) < 0.1 }, "Should include D4")
        XCTAssertTrue(frequencies.contains { abs($0 - e4Frequency) < 0.1 }, "Should include E4")
        XCTAssertTrue(frequencies.contains { abs($0 - f4Frequency) < 0.1 }, "Should include F4")
        XCTAssertTrue(frequencies.contains { abs($0 - g4Frequency) < 0.1 }, "Should include G4")
    }

    /// Test: Zero ascending count returns only start note scale
    /// "0 ascending" means 0 key changes = just the start note (1 key)
    func testCalculateTargetFrequencies_ZeroAscendingCount_ReturnsStartNoteScale() {
        // Given: Zero ascending count (only start key plays)
        let settings = ScaleSettings(
            startNote: try! MIDINote(60),  // C4
            endNote: try! MIDINote(72),    // C5
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 1.0),
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 0,  // No key changes = just start key
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 1
        )

        // When
        let frequencies = sut.calculateTargetFrequencies(from: settings)

        // Then: Should return start key's scale notes (C4 five-tone scale)
        // C4, D4, E4, F4, G4
        XCTAssertEqual(frequencies.count, 5, "Should return 5 notes for five-tone scale from start key")

        let c4Frequency = try! MIDINote(60).frequency
        XCTAssertTrue(
            frequencies.contains { abs($0 - c4Frequency) < 0.1 },
            "Should include C4 (start note)"
        )
    }

    // MARK: - Bug Detection Tests: Boundary Values and Edge Cases

    /// BUG DETECTION TEST: Low MIDI start note with descending pattern
    /// Should handle cases where descending would go below MIDI 0
    func testCalculateTargetFrequencies_LowStartNote_DescendingPattern_ShouldNotCrash() {
        // Given: Low start note with descending pattern
        let settings = ScaleSettings(
            startNote: try! MIDINote(36),  // C2 - low note
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 1.0),
            keyProgressionPattern: .descendingOnly,
            ascendingKeyCount: 0,
            descendingKeyCount: 3,  // Descending from C2
            ascendingKeyStepInterval: 1,
            descendingKeyStepInterval: 12  // Octave steps - would underflow!
        )

        // When: This should not crash
        let frequencies = sut.calculateTargetFrequencies(from: settings)

        // Then: Should return valid frequencies (positive, within human hearing range)
        for freq in frequencies {
            XCTAssertGreaterThan(freq, 0, "Frequency should be positive")
            XCTAssertLessThan(freq, 20000, "Frequency should be below 20kHz")
            XCTAssertFalse(freq.isNaN, "Frequency should not be NaN")
            XCTAssertFalse(freq.isInfinite, "Frequency should not be infinite")
        }
    }

    /// BUG DETECTION TEST: High MIDI start note with ascending pattern
    /// Should handle cases where ascending would exceed MIDI 127
    func testCalculateTargetFrequencies_HighStartNote_AscendingPattern_ShouldNotCrash() {
        // Given: High start note with ascending pattern
        let settings = ScaleSettings(
            startNote: try! MIDINote(120),  // Very high note
            endNote: try! MIDINote(127),
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 1.0),
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 3,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 5  // Would exceed MIDI 127
        )

        // When
        let frequencies = sut.calculateTargetFrequencies(from: settings)

        // Then: Should return valid frequencies
        for freq in frequencies {
            XCTAssertGreaterThan(freq, 0, "Frequency should be positive")
            XCTAssertFalse(freq.isNaN, "Frequency should not be NaN")
        }
    }

    /// BUG DETECTION TEST: Empty result should be handled gracefully
    func testCalculateTargetFrequencies_ValidSettings_ShouldNotReturnEmpty() {
        // Given: Valid settings that should produce results
        let settings = ScaleSettings(
            startNote: try! MIDINote(60),
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 1.0),
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 1,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 1
        )

        // When
        let frequencies = sut.calculateTargetFrequencies(from: settings)

        // Then: Should not be empty
        XCTAssertFalse(frequencies.isEmpty, "Valid settings should produce frequencies")
    }

    /// BUG DETECTION TEST: Frequencies should be unique (no duplicates)
    func testCalculateTargetFrequencies_ShouldReturnUniqueFrequencies() {
        // Given: Settings that might produce duplicate frequencies
        let settings = ScaleSettings(
            startNote: try! MIDINote(60),
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 1.0),
            keyProgressionPattern: .ascendingThenDescending,
            ascendingKeyCount: 3,
            descendingKeyCount: 3,
            ascendingKeyStepInterval: 1,
            descendingKeyStepInterval: 1
        )

        // When
        let frequencies = sut.calculateTargetFrequencies(from: settings)

        // Then: Frequencies should be unique (Set count == Array count)
        let uniqueFreqs = Set(frequencies.map { Int($0 * 100) })  // Round to avoid floating point issues
        XCTAssertEqual(uniqueFreqs.count, frequencies.count,
            "Frequencies should be unique. Got \(frequencies.count) total but \(uniqueFreqs.count) unique")
    }

    /// BUG DETECTION TEST: Frequencies should be sorted
    func testCalculateTargetFrequencies_ShouldBeSortedAscending() {
        // Given
        let settings = ScaleSettings(
            startNote: try! MIDINote(60),
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 1.0),
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 2,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 1
        )

        // When
        let frequencies = sut.calculateTargetFrequencies(from: settings)

        // Then: Should be sorted in ascending order
        let sorted = frequencies.sorted()
        XCTAssertEqual(frequencies, sorted, "Frequencies should be sorted ascending")
    }

    /// BUG DETECTION TEST: Large key count should not cause performance issues
    func testCalculateTargetFrequencies_LargeKeyCount_ShouldCompleteInReasonableTime() {
        // Given: Large key count
        let settings = ScaleSettings(
            startNote: try! MIDINote(48),
            endNote: try! MIDINote(84),
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 1.0),
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 24,  // Two octaves worth
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 1
        )

        // When: Measure execution time
        let startTime = Date()
        let frequencies = sut.calculateTargetFrequencies(from: settings)
        let elapsed = Date().timeIntervalSince(startTime)

        // Then: Should complete quickly (under 1 second) and produce results
        XCTAssertLessThan(elapsed, 1.0, "Should complete in under 1 second")
        XCTAssertFalse(frequencies.isEmpty, "Should produce results")
    }

    /// BUG DETECTION TEST: All frequencies should be within piano range
    func testCalculateTargetFrequencies_ShouldBeWithinMusicalRange() {
        // Given
        let settings = ScaleSettings(
            startNote: try! MIDINote(60),
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 1.0),
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 5,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 1
        )

        // When
        let frequencies = sut.calculateTargetFrequencies(from: settings)

        // Then: All frequencies should be within reasonable musical range
        // Piano range: ~27.5 Hz (A0) to ~4186 Hz (C8)
        for freq in frequencies {
            XCTAssertGreaterThanOrEqual(freq, 20.0,
                "Frequency \(freq) Hz is below audible range")
            XCTAssertLessThanOrEqual(freq, 5000.0,
                "Frequency \(freq) Hz is above typical vocal/piano range")
        }
    }
}
