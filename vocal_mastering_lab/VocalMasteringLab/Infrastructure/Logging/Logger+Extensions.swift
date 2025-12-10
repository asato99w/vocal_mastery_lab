import Foundation
import OSLog

/// Centralized logging configuration for VocalMasteringLab
///
/// Logs are sent to both OSLog (system) and file (debug builds only).
/// File logs are stored in Documents/logs/ directory.
///
/// Usage:
/// ```
/// Logger.recording.info("Recording started")
/// Logger.pitchDetection.debug("Frequency: \(frequency)")
/// Logger.audio.error("Audio session error: \(error)")
/// ```
extension Logger {
    /// Subsystem identifier for the app
    private static let subsystem = Bundle.main.bundleIdentifier ?? "com.vocalis.studio"

    // MARK: - Application Layer Loggers

    /// Logger for recording operations (start, stop, session management)
    static let recording = Logger(subsystem: subsystem, category: "recording")

    /// Logger for use case execution and business logic
    static let useCase = Logger(subsystem: subsystem, category: "usecase")

    // MARK: - Infrastructure Layer Loggers

    /// Logger for audio session and device management
    static let audio = Logger(subsystem: subsystem, category: "audio")

    /// Logger for pitch detection and analysis
    static let pitchDetection = Logger(subsystem: subsystem, category: "pitch")

    /// Logger for scale playback operations
    static let scalePlayer = Logger(subsystem: subsystem, category: "scale")

    /// Logger for file system and data persistence operations
    static let storage = Logger(subsystem: subsystem, category: "storage")

    // MARK: - Presentation Layer Loggers

    /// Logger for UI events and user interactions
    static let ui = Logger(subsystem: subsystem, category: "ui")

    /// Logger for ViewModel state changes
    static let viewModel = Logger(subsystem: subsystem, category: "viewmodel")
}

// MARK: - Logging Helpers

extension Logger {
    /// Log an error with file and function context
    func logError(_ error: Error, file: String = #file, function: String = #function, line: Int = #line) {
        let fileName = (file as NSString).lastPathComponent
        let message = "[\(fileName):\(line)] \(function) - Error: \(error.localizedDescription)"
        self.error("\(message)")

        // Also log to file in debug builds
        FileLogger.shared.log(level: "ERROR", category: self.category, message: message, file: file, function: function, line: line)
    }

    /// Log a critical failure that should never happen
    func logCritical(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        let fileName = (file as NSString).lastPathComponent
        let fullMessage = "[\(fileName):\(line)] \(function) - CRITICAL: \(message)"
        self.critical("\(fullMessage)")

        // Also log to file in debug builds
        FileLogger.shared.log(level: "CRITICAL", category: self.category, message: fullMessage, file: file, function: function, line: line)
    }

    /// Get the category name from the logger
    private var category: String {
        // Extract category from logger description
        // Logger description format: "OSLog(subsystem: com.kazuasato.VocalMasteringLab, category: recording)"
        let description = String(describing: self)
        if let categoryRange = description.range(of: "category: "),
           let endRange = description.range(of: ")", range: categoryRange.upperBound..<description.endIndex) {
            return String(description[categoryRange.upperBound..<endRange.lowerBound])
        }
        return "unknown"
    }
}

// MARK: - File Logging Helpers
//
// Note: OSLog methods (info, debug, warning, error) automatically log to both
// system log and file in debug builds through OSLog observation.
// Use the methods below for explicit file logging when needed.

extension Logger {
    /// Explicitly log to file with custom message
    func logToFile(level: String, message: String, file: String = #file, function: String = #function, line: Int = #line) {
        FileLogger.shared.log(level: level, category: self.category, message: message, file: file, function: function, line: line)
    }
}
