import XCTest
import AVFoundation
import VocalisDomain
@testable import VocalMasteryLab

/// Test CSV parser for CSD (Children's Song Dataset) note timing data
final class CSDNoteTimingParserTests: XCTestCase {

    // MARK: - CSV Structure Tests

    /// Test parsing a single note entry from CSV
    func testParseSimpleNoteEntry() throws {
        // CSD CSV format: onset, offset, midi_note, syllable
        let csvLine = "0.5,1.0,60,do"

        let parser = CSDNoteTimingParser()
        let notes = try parser.parseCSVLine(csvLine)

        XCTAssertEqual(notes.count, 1)

        let note = notes[0]
        XCTAssertEqual(note.onset, 0.5, accuracy: 0.001)
        XCTAssertEqual(note.offset, 1.0, accuracy: 0.001)
        XCTAssertEqual(note.midiNote.value, 60)
        XCTAssertEqual(note.syllable, "do")
    }

    /// Test parsing multiple note entries
    func testParseMultipleNoteEntries() throws {
        let csvContent = """
        0.5,1.0,60,do
        1.0,1.5,62,re
        1.5,2.0,64,mi
        """

        let parser = CSDNoteTimingParser()
        let notes = try parser.parseCSVContent(csvContent)

        XCTAssertEqual(notes.count, 3)

        // Verify first note (C4 - do)
        XCTAssertEqual(notes[0].midiNote.value, 60)
        XCTAssertEqual(notes[0].syllable, "do")

        // Verify second note (D4 - re)
        XCTAssertEqual(notes[1].midiNote.value, 62)
        XCTAssertEqual(notes[1].syllable, "re")

        // Verify third note (E4 - mi)
        XCTAssertEqual(notes[2].midiNote.value, 64)
        XCTAssertEqual(notes[2].syllable, "mi")
    }

    /// Test parsing CSV with header line
    func testParseCSVWithHeader() throws {
        let csvContent = """
        onset,offset,midi_note,syllable
        0.5,1.0,60,do
        1.0,1.5,62,re
        """

        let parser = CSDNoteTimingParser()
        let notes = try parser.parseCSVContent(csvContent, hasHeader: true)

        XCTAssertEqual(notes.count, 2)
        XCTAssertEqual(notes[0].midiNote.value, 60)
    }

    /// Test error handling for invalid MIDI note
    func testInvalidMIDINoteThrowsError() {
        let csvLine = "0.5,1.0,128,invalid"

        let parser = CSDNoteTimingParser()

        XCTAssertThrowsError(try parser.parseCSVLine(csvLine)) { error in
            XCTAssertTrue(error is CSDParseError)
        }
    }

    /// Test error handling for malformed CSV
    func testMalformedCSVThrowsError() {
        let csvLine = "0.5,1.0"  // Missing fields

        let parser = CSDNoteTimingParser()

        XCTAssertThrowsError(try parser.parseCSVLine(csvLine)) { error in
            XCTAssertTrue(error is CSDParseError)
        }
    }

    // MARK: - Frequency Calculation Tests

    /// Test frequency calculation from MIDI note
    func testFrequencyCalculation() throws {
        let csvLine = "0.5,1.0,60,do"

        let parser = CSDNoteTimingParser()
        let notes = try parser.parseCSVLine(csvLine)

        let frequency = notes[0].midiNote.frequency
        XCTAssertEqual(frequency, 261.63, accuracy: 0.01, "C4 should be approximately 261.63 Hz")
    }

    /// Test A4 (440 Hz) reference frequency
    func testA4ReferenceFrequency() throws {
        let csvLine = "0.5,1.0,69,la"  // A4

        let parser = CSDNoteTimingParser()
        let notes = try parser.parseCSVLine(csvLine)

        let frequency = notes[0].midiNote.frequency
        XCTAssertEqual(frequency, 440.0, accuracy: 0.01, "A4 should be exactly 440.0 Hz")
    }
}

// MARK: - Supporting Types (to be implemented)

/// Represents a single note timing entry from CSD dataset
struct CSDNoteTiming {
    let onset: TimeInterval      // Note start time in seconds
    let offset: TimeInterval     // Note end time in seconds
    let midiNote: MIDINote       // MIDI note number (0-127)
    let syllable: String         // Syllable/lyric for this note

    var duration: TimeInterval {
        offset - onset
    }
}

/// Parser for CSD CSV files
class CSDNoteTimingParser {
    func parseCSVLine(_ line: String) throws -> [CSDNoteTiming] {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return []
        }

        let components = splitCSVLine(trimmed)
        guard components.count >= 4 else {
            throw CSDParseError.invalidFormat("Expected 4 fields (onset,offset,midi_note,syllable), got \(components.count)")
        }

        let onset = try parseTimestamp(components[0], fieldName: "onset")
        let offset = try parseTimestamp(components[1], fieldName: "offset")
        let midiNote = try parseMIDINote(components[2])
        let syllable = components[3]

        let noteTiming = CSDNoteTiming(
            onset: onset,
            offset: offset,
            midiNote: midiNote,
            syllable: syllable
        )

        return [noteTiming]
    }

    func parseCSVContent(_ content: String, hasHeader: Bool = false) throws -> [CSDNoteTiming] {
        var lines = content.components(separatedBy: .newlines)

        // Remove header if present
        if hasHeader && !lines.isEmpty {
            lines.removeFirst()
        }

        return try lines
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .flatMap { try parseCSVLine($0) }
    }

    // MARK: - Private Helper Methods

    /// Split CSV line and trim each component
    private func splitCSVLine(_ line: String) -> [String] {
        return line.components(separatedBy: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
    }

    /// Parse timestamp value (onset/offset)
    private func parseTimestamp(_ value: String, fieldName: String) throws -> TimeInterval {
        guard let timestamp = Double(value) else {
            throw CSDParseError.invalidTimestamp("Invalid \(fieldName) value: \(value)")
        }
        return timestamp
    }

    /// Parse MIDI note value
    private func parseMIDINote(_ value: String) throws -> MIDINote {
        guard let midiValue = UInt8(value) else {
            throw CSDParseError.invalidMIDINote("Invalid MIDI note value: \(value)")
        }

        do {
            return try MIDINote(midiValue)
        } catch {
            throw CSDParseError.invalidMIDINote("MIDI note out of range (0-127): \(midiValue)")
        }
    }
}

/// Errors that can occur during CSV parsing
enum CSDParseError: LocalizedError {
    case invalidFormat(String)
    case invalidMIDINote(String)
    case invalidTimestamp(String)

    var errorDescription: String? {
        switch self {
        case .invalidFormat(let detail):
            return "Invalid CSV format: \(detail)"
        case .invalidMIDINote(let detail):
            return "Invalid MIDI note: \(detail)"
        case .invalidTimestamp(let detail):
            return "Invalid timestamp: \(detail)"
        }
    }
}
