import Foundation
import VocalisDomain

/// Mock implementation of LoggerProtocol for testing
class MockLogger: LoggerProtocol {

    // MARK: - Captured Log Data

    struct LogEntry {
        let level: LogLevel
        let message: String
        let category: String
    }

    enum LogLevel {
        case debug
        case info
        case warning
        case error
    }

    private(set) var logEntries: [LogEntry] = []

    // MARK: - LoggerProtocol Implementation

    func debug(_ message: String, category: String) {
        logEntries.append(LogEntry(level: .debug, message: message, category: category))
    }

    func info(_ message: String, category: String) {
        logEntries.append(LogEntry(level: .info, message: message, category: category))
    }

    func warning(_ message: String, category: String) {
        logEntries.append(LogEntry(level: .warning, message: message, category: category))
    }

    func error(_ message: String, category: String) {
        logEntries.append(LogEntry(level: .error, message: message, category: category))
    }

    // MARK: - Test Helper Methods

    /// Clear all logged entries
    func reset() {
        logEntries.removeAll()
    }

    /// Check if a message containing the given text was logged
    func contains(_ text: String) -> Bool {
        return logEntries.contains { $0.message.contains(text) }
    }

    /// Check if a message at the given level was logged
    func hasLevel(_ level: LogLevel) -> Bool {
        return logEntries.contains { $0.level == level }
    }

    /// Check if a message with the given category was logged
    func hasCategory(_ category: String) -> Bool {
        return logEntries.contains { $0.category == category }
    }

    /// Get all messages at the given level
    func messages(level: LogLevel) -> [String] {
        return logEntries.filter { $0.level == level }.map { $0.message }
    }

    /// Get all messages with the given category
    func messages(category: String) -> [String] {
        return logEntries.filter { $0.category == category }.map { $0.message }
    }

    /// Count of log entries at the given level
    func count(level: LogLevel) -> Int {
        return logEntries.filter { $0.level == level }.count
    }

    /// Total count of all log entries
    var count: Int {
        return logEntries.count
    }
}
