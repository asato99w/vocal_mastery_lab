import XCTest
@testable import VocalisDomain

final class MIDINoteTests: XCTestCase {
    func testInit_ValidValue_Success() throws {
        // Given & When
        let note = try MIDINote(60)
        
        // Then
        XCTAssertEqual(note.value, 60)
    }
    
    func testInit_MaxValue_Success() throws {
        // Given & When
        let note = try MIDINote(127)
        
        // Then
        XCTAssertEqual(note.value, 127)
    }
    
    func testInit_MinValue_Success() throws {
        // Given & When
        let note = try MIDINote(0)
        
        // Then
        XCTAssertEqual(note.value, 0)
    }
    
    func testInit_OutOfRange_ThrowsError() {
        // Given & When & Then
        XCTAssertThrowsError(try MIDINote(128)) { error in
            XCTAssertTrue(error is MIDINoteError)
        }
    }
    
    func testMiddleC_CorrectValue() {
        // Given & When
        let middleC = MIDINote.middleC
        
        // Then
        XCTAssertEqual(middleC.value, 60)
    }
    
    func testHiC_CorrectValue() {
        // Given & When
        let hiC = MIDINote.hiC
        
        // Then
        XCTAssertEqual(hiC.value, 72)
    }
    
    func testComparable_LessThan() throws {
        // Given
        let c4 = try MIDINote(60)
        let c5 = try MIDINote(72)
        
        // When & Then
        XCTAssertTrue(c4 < c5)
        XCTAssertFalse(c5 < c4)
    }
    
    func testEquatable() throws {
        // Given
        let note1 = try MIDINote(60)
        let note2 = try MIDINote(60)
        let note3 = try MIDINote(61)

        // When & Then
        XCTAssertEqual(note1, note2)
        XCTAssertNotEqual(note1, note3)
    }

    // MARK: - Frequency Calculation Tests

    func testFrequency_A4_Returns440Hz() throws {
        // Given: A4 (MIDI 69) is the standard tuning reference
        let a4 = try MIDINote(69)

        // When
        let frequency = a4.frequency

        // Then: A4 = 440 Hz (standard tuning)
        XCTAssertEqual(frequency, 440.0, accuracy: 0.01)
    }

    func testFrequency_MiddleC_ReturnsCorrectFrequency() {
        // Given: C4 (MIDI 60)
        let c4 = MIDINote.middleC

        // When
        let frequency = c4.frequency

        // Then: C4 ≈ 261.63 Hz
        XCTAssertEqual(frequency, 261.63, accuracy: 0.01)
    }

    func testFrequency_A3_ReturnsCorrectFrequency() throws {
        // Given: A3 (MIDI 57) - one octave below A4
        let a3 = try MIDINote(57)

        // When
        let frequency = a3.frequency

        // Then: A3 = 220 Hz (half of A4)
        XCTAssertEqual(frequency, 220.0, accuracy: 0.01)
    }

    func testFrequency_A5_ReturnsCorrectFrequency() throws {
        // Given: A5 (MIDI 81) - one octave above A4
        let a5 = try MIDINote(81)

        // When
        let frequency = a5.frequency

        // Then: A5 = 880 Hz (double of A4)
        XCTAssertEqual(frequency, 880.0, accuracy: 0.01)
    }

    func testFrequency_OctaveRelationship() throws {
        // Given: Two notes one octave apart
        let c4 = try MIDINote(60)
        let c5 = try MIDINote(72)

        // When
        let freqC4 = c4.frequency
        let freqC5 = c5.frequency

        // Then: C5 frequency should be exactly double C4
        XCTAssertEqual(freqC5 / freqC4, 2.0, accuracy: 0.0001)
    }

    // MARK: - Note Name Tests

    func testNoteName_MiddleC_ReturnsC4() {
        // Given
        let c4 = MIDINote.middleC

        // When
        let name = c4.noteName

        // Then
        XCTAssertEqual(name, "C4")
    }

    func testNoteName_A4_ReturnsA4() throws {
        // Given
        let a4 = try MIDINote(69)

        // When
        let name = a4.noteName

        // Then
        XCTAssertEqual(name, "A4")
    }

    func testNoteName_CSharp5_ReturnsCSharp5() throws {
        // Given: C#5 (MIDI 73)
        let cSharp5 = try MIDINote(73)

        // When
        let name = cSharp5.noteName

        // Then
        XCTAssertEqual(name, "C#5")
    }

    func testNoteName_LowNote_ReturnsCorrectOctave() throws {
        // Given: C0 (MIDI 12)
        let c0 = try MIDINote(12)

        // When
        let name = c0.noteName

        // Then
        XCTAssertEqual(name, "C0")
    }

    func testNoteName_HighNote_ReturnsCorrectOctave() throws {
        // Given: G9 (MIDI 127)
        let g9 = try MIDINote(127)

        // When
        let name = g9.noteName

        // Then
        XCTAssertEqual(name, "G9")
    }

    func testNoteName_StaticMethod() {
        // Given & When
        let name = MIDINote.noteName(for: 60)

        // Then
        XCTAssertEqual(name, "C4")
    }

    // MARK: - Bug Detection Tests: Boundary Values

    /// BUG DETECTION TEST: Static constants should not use force-unwrap
    /// Verify that static properties don't crash the app on launch
    func testStaticConstants_ShouldBeValid() {
        // These should not crash when accessed
        let middleC = MIDINote.middleC
        let hiC = MIDINote.hiC

        XCTAssertEqual(middleC.value, 60)
        XCTAssertEqual(hiC.value, 72)
    }

    /// BUG DETECTION TEST: Frequency calculation at boundary values
    func testFrequency_BoundaryValues_ShouldNotCrash() throws {
        // Given: Boundary MIDI values
        let minNote = try MIDINote(0)
        let maxNote = try MIDINote(127)

        // When: Calculate frequencies
        let minFreq = minNote.frequency
        let maxFreq = maxNote.frequency

        // Then: Should return valid frequencies (not NaN, not Inf)
        XCTAssertFalse(minFreq.isNaN, "Min frequency should not be NaN")
        XCTAssertFalse(minFreq.isInfinite, "Min frequency should not be infinite")
        XCTAssertFalse(maxFreq.isNaN, "Max frequency should not be NaN")
        XCTAssertFalse(maxFreq.isInfinite, "Max frequency should not be infinite")

        // MIDI 0 ≈ 8.18 Hz, MIDI 127 ≈ 12543.85 Hz
        XCTAssertGreaterThan(minFreq, 0, "Min frequency should be positive")
        XCTAssertLessThan(maxFreq, 20000, "Max frequency should be below 20kHz")
    }

    /// BUG DETECTION TEST: noteName at boundary values
    func testNoteName_BoundaryValues_ShouldNotCrash() throws {
        // Given: Boundary MIDI values
        let minNote = try MIDINote(0)
        let maxNote = try MIDINote(127)

        // When: Get note names
        let minName = minNote.noteName
        let maxName = maxNote.noteName

        // Then: Should return valid names
        XCTAssertFalse(minName.isEmpty, "Min note name should not be empty")
        XCTAssertFalse(maxName.isEmpty, "Max note name should not be empty")

        // MIDI 0 = C-1, MIDI 127 = G9
        XCTAssertTrue(minName.hasPrefix("C"), "MIDI 0 should be C")
        XCTAssertTrue(maxName.hasPrefix("G"), "MIDI 127 should be G")
    }

    /// BUG DETECTION TEST: Invalid MIDI values should throw, not crash
    func testInit_ValuesAbove127_ShouldThrow() {
        // Given: Values above valid MIDI range
        let invalidValues: [UInt8] = [128, 255]  // Note: UInt8 max is 255

        for value in invalidValues {
            // When/Then: Should throw MIDINoteError
            XCTAssertThrowsError(try MIDINote(value), "Value \(value) should throw") { error in
                XCTAssertTrue(error is MIDINoteError,
                    "Expected MIDINoteError, got \(type(of: error))")
            }
        }
    }

    /// BUG DETECTION TEST: Codable round-trip at boundaries
    func testCodable_BoundaryValues_ShouldRoundTrip() throws {
        // Given: Boundary notes
        let minNote = try MIDINote(0)
        let maxNote = try MIDINote(127)

        // When: Encode and decode
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        let minData = try encoder.encode(minNote)
        let maxData = try encoder.encode(maxNote)

        let decodedMin = try decoder.decode(MIDINote.self, from: minData)
        let decodedMax = try decoder.decode(MIDINote.self, from: maxData)

        // Then: Should preserve values
        XCTAssertEqual(decodedMin, minNote)
        XCTAssertEqual(decodedMax, maxNote)
    }

    /// BUG DETECTION TEST: Arithmetic near boundaries
    func testArithmetic_NearBoundaries_ShouldNotOverflow() throws {
        // Given: Notes near boundaries
        let nearMax = try MIDINote(120)
        let nearMin = try MIDINote(7)

        // When: Perform common operations (like adding intervals)
        // These are internal operations that might be used in scale generation
        let majorThirdUp = nearMax.value + 4  // 124 - still valid
        let perfectFifthUp = nearMax.value + 7  // 127 - still valid

        let majorThirdDown = Int(nearMin.value) - 4  // 3 - still valid
        let perfectFifthDown = Int(nearMin.value) - 7  // 0 - still valid

        // Then: Results should be within valid range
        XCTAssertLessThanOrEqual(majorThirdUp, 127)
        XCTAssertLessThanOrEqual(perfectFifthUp, 127)
        XCTAssertGreaterThanOrEqual(majorThirdDown, 0)
        XCTAssertGreaterThanOrEqual(perfectFifthDown, 0)

        // But these would overflow/underflow:
        let overflowCase = nearMax.value + 8  // 128 - INVALID
        let underflowCase = Int(nearMin.value) - 8  // -1 - INVALID

        XCTAssertGreaterThan(overflowCase, 127, "Should detect overflow case")
        XCTAssertLessThan(underflowCase, 0, "Should detect underflow case")
    }
}
