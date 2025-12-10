import XCTest
import AVFoundation
import VocalisDomain
@testable import VocalMasteringLab

/// Test CSV parser for vocadito singing voice dataset annotations
final class VocaditoAnnotationParserTests: XCTestCase {

    // MARK: - F0 Annotation Parsing Tests

    /// Test parsing F0 annotation line
    func testParseF0Line() throws {
        // vocadito F0 format: timestamp, f0_value
        let csvLine = "0.661768707,143.742"

        let parser = VocaditoF0Parser()
        let f0Point = try parser.parseF0Line(csvLine)

        XCTAssertEqual(f0Point.timestamp, 0.661768707, accuracy: 0.0001)
        XCTAssertEqual(f0Point.frequency, 143.742, accuracy: 0.001)
        XCTAssertTrue(f0Point.isVoiced)
    }

    /// Test parsing F0 line with unvoiced region (0.0 Hz)
    func testParseF0Line_Unvoiced() throws {
        let csvLine = "0.0,0.0"

        let parser = VocaditoF0Parser()
        let f0Point = try parser.parseF0Line(csvLine)

        XCTAssertEqual(f0Point.timestamp, 0.0, accuracy: 0.0001)
        XCTAssertEqual(f0Point.frequency, 0.0, accuracy: 0.001)
        XCTAssertFalse(f0Point.isVoiced)
    }

    /// Test parsing multiple F0 entries
    func testParseF0Content() throws {
        let csvContent = """
        0.0,0.0
        0.005804988662131519,0.0
        0.661768707,143.742
        0.667573696,144.125
        """

        let parser = VocaditoF0Parser()
        let f0Points = try parser.parseF0Content(csvContent)

        XCTAssertEqual(f0Points.count, 4)
        XCTAssertFalse(f0Points[0].isVoiced)
        XCTAssertFalse(f0Points[1].isVoiced)
        XCTAssertTrue(f0Points[2].isVoiced)
        XCTAssertEqual(f0Points[2].frequency, 143.742, accuracy: 0.001)
    }

    // MARK: - Note Annotation Parsing Tests

    /// Test parsing note annotation line
    func testParseNoteLine() throws {
        // vocadito note format: start_time, pitch_hz, duration
        let csvLine = "0.661768707,143.742,0.290249433"

        let parser = VocaditoNoteParser()
        let note = try parser.parseNoteLine(csvLine)

        XCTAssertEqual(note.startTime, 0.661768707, accuracy: 0.0001)
        XCTAssertEqual(note.frequency, 143.742, accuracy: 0.001)
        XCTAssertEqual(note.duration, 0.290249433, accuracy: 0.0001)
    }

    /// Test parsing multiple note entries
    func testParseNoteContent() throws {
        let csvContent = """
        0.661768707,143.742,0.290249433
        1.010068027,158.441,0.301859410
        1.317732426,174.841,0.847528345
        """

        let parser = VocaditoNoteParser()
        let notes = try parser.parseNoteContent(csvContent)

        XCTAssertEqual(notes.count, 3)

        // Verify first note
        XCTAssertEqual(notes[0].startTime, 0.661768707, accuracy: 0.0001)
        XCTAssertEqual(notes[0].frequency, 143.742, accuracy: 0.001)
        XCTAssertEqual(notes[0].duration, 0.290249433, accuracy: 0.0001)

        // Verify second note
        XCTAssertEqual(notes[1].frequency, 158.441, accuracy: 0.001)
    }

    /// Test calculating end time from start time and duration
    func testNoteEndTime() throws {
        let csvLine = "0.661768707,143.742,0.290249433"

        let parser = VocaditoNoteParser()
        let note = try parser.parseNoteLine(csvLine)

        let expectedEndTime = 0.661768707 + 0.290249433
        XCTAssertEqual(note.endTime, expectedEndTime, accuracy: 0.0001)
    }

    // MARK: - Error Handling Tests

    /// Test error handling for invalid F0 line
    func testInvalidF0LineThrowsError() {
        let csvLine = "0.5"  // Missing frequency field

        let parser = VocaditoF0Parser()

        XCTAssertThrowsError(try parser.parseF0Line(csvLine)) { error in
            XCTAssertTrue(error is VocaditoParseError)
        }
    }

    /// Test error handling for invalid note line
    func testInvalidNoteLineThrowsError() {
        let csvLine = "0.5,143.742"  // Missing duration field

        let parser = VocaditoNoteParser()

        XCTAssertThrowsError(try parser.parseNoteLine(csvLine)) { error in
            XCTAssertTrue(error is VocaditoParseError)
        }
    }

    /// Test error handling for non-numeric values
    func testNonNumericValuesThrowError() {
        let csvLine = "abc,def"

        let parser = VocaditoF0Parser()

        XCTAssertThrowsError(try parser.parseF0Line(csvLine)) { error in
            XCTAssertTrue(error is VocaditoParseError)
        }
    }

    // MARK: - Empty Line Handling Tests

    /// Test that empty lines are skipped
    func testEmptyLinesAreSkipped() throws {
        let csvContent = """
        0.661768707,143.742,0.290249433

        1.010068027,158.441,0.301859410
        """

        let parser = VocaditoNoteParser()
        let notes = try parser.parseNoteContent(csvContent)

        XCTAssertEqual(notes.count, 2)
    }
}

// MARK: - Supporting Types

/// Represents a single F0 data point from vocadito dataset
struct VocaditoF0Point {
    let timestamp: TimeInterval  // Time in seconds
    let frequency: Double        // F0 value in Hz (0.0 = unvoiced)

    var isVoiced: Bool {
        frequency > 0.0
    }
}

/// Represents a single note annotation from vocadito dataset
struct VocaditoNote {
    let startTime: TimeInterval  // Note start time in seconds
    let frequency: Double        // Note pitch in Hz
    let duration: TimeInterval   // Note duration in seconds

    var endTime: TimeInterval {
        startTime + duration
    }
}

/// Parser for vocadito F0 annotations
class VocaditoF0Parser {
    func parseF0Line(_ line: String) throws -> VocaditoF0Point {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw VocaditoParseError.emptyLine
        }

        let components = trimmed.components(separatedBy: ",")
        guard components.count == 2 else {
            throw VocaditoParseError.invalidFormat("Expected 2 fields (timestamp,f0), got \(components.count)")
        }

        guard let timestamp = Double(components[0].trimmingCharacters(in: .whitespaces)) else {
            throw VocaditoParseError.invalidNumber("Invalid timestamp: \(components[0])")
        }

        guard let frequency = Double(components[1].trimmingCharacters(in: .whitespaces)) else {
            throw VocaditoParseError.invalidNumber("Invalid frequency: \(components[1])")
        }

        return VocaditoF0Point(timestamp: timestamp, frequency: frequency)
    }

    func parseF0Content(_ content: String) throws -> [VocaditoF0Point] {
        return try content
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map { try parseF0Line($0) }
    }
}

/// Parser for vocadito note annotations
class VocaditoNoteParser {
    func parseNoteLine(_ line: String) throws -> VocaditoNote {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw VocaditoParseError.emptyLine
        }

        let components = trimmed.components(separatedBy: ",")
        guard components.count == 3 else {
            throw VocaditoParseError.invalidFormat("Expected 3 fields (start_time,pitch_hz,duration), got \(components.count)")
        }

        guard let startTime = Double(components[0].trimmingCharacters(in: .whitespaces)) else {
            throw VocaditoParseError.invalidNumber("Invalid start time: \(components[0])")
        }

        guard let frequency = Double(components[1].trimmingCharacters(in: .whitespaces)) else {
            throw VocaditoParseError.invalidNumber("Invalid frequency: \(components[1])")
        }

        guard let duration = Double(components[2].trimmingCharacters(in: .whitespaces)) else {
            throw VocaditoParseError.invalidNumber("Invalid duration: \(components[2])")
        }

        return VocaditoNote(startTime: startTime, frequency: frequency, duration: duration)
    }

    func parseNoteContent(_ content: String) throws -> [VocaditoNote] {
        return try content
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map { try parseNoteLine($0) }
    }
}

/// Errors that can occur during vocadito annotation parsing
enum VocaditoParseError: LocalizedError {
    case emptyLine
    case invalidFormat(String)
    case invalidNumber(String)

    var errorDescription: String? {
        switch self {
        case .emptyLine:
            return "Empty line encountered"
        case .invalidFormat(let detail):
            return "Invalid CSV format: \(detail)"
        case .invalidNumber(let detail):
            return "Invalid number: \(detail)"
        }
    }
}
