import Foundation

/// File-based logger for development and debugging
/// Logs are written to Documents/logs/ directory with automatic rotation
class FileLogger {
    static let shared = FileLogger()

    private let fileManager = FileManager.default
    private let logDirectory: URL
    private let currentLogFile: URL
    private let dateFormatter: DateFormatter
    private let queue = DispatchQueue(label: "com.vocalis.studio.filelogger", qos: .utility)

    // Configuration
    private let maxFileSize: Int = 5 * 1024 * 1024 // 5MB
    private let maxLogFiles: Int = 5

    private init() {
        // Setup log directory
        let documentsPath = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        logDirectory = documentsPath.appendingPathComponent("logs", isDirectory: true)

        // Create log directory if needed
        try? fileManager.createDirectory(at: logDirectory, withIntermediateDirectories: true)

        // Current log file
        let timestamp = Date().formatted(.iso8601.year().month().day().time(includingFractionalSeconds: false))
        currentLogFile = logDirectory.appendingPathComponent("vocalis_\(timestamp).log")

        // Date formatter for log entries
        dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        dateFormatter.locale = Locale(identifier: "en_US_POSIX")
        dateFormatter.timeZone = TimeZone.current

        // Initial log entry
        log(level: "INFO", category: "system", message: "FileLogger initialized")
        log(level: "INFO", category: "system", message: "Log file: \(currentLogFile.lastPathComponent)")
    }

    /// Log a message to file
    func log(level: String, category: String, message: String, file: String = #file, function: String = #function, line: Int = #line) {
        #if DEBUG
        queue.async { [weak self] in
            guard let self = self else { return }

            let timestamp = self.dateFormatter.string(from: Date())
            let fileName = (file as NSString).lastPathComponent
            let logEntry = "\(timestamp) [\(level)] [\(category)] \(message) [\(fileName):\(line)]\n"

            self.writeToFile(logEntry)
            self.rotateLogsIfNeeded()
        }
        #endif
    }

    /// Write log entry to current log file
    private func writeToFile(_ entry: String) {
        guard let data = entry.data(using: .utf8) else { return }

        if fileManager.fileExists(atPath: currentLogFile.path) {
            // Append to existing file
            if let fileHandle = try? FileHandle(forWritingTo: currentLogFile) {
                fileHandle.seekToEndOfFile()
                fileHandle.write(data)
                try? fileHandle.close()
            }
        } else {
            // Create new file
            try? data.write(to: currentLogFile)
        }
    }

    /// Rotate logs if current file exceeds max size
    private func rotateLogsIfNeeded() {
        guard let attributes = try? fileManager.attributesOfItem(atPath: currentLogFile.path),
              let fileSize = attributes[.size] as? Int,
              fileSize > maxFileSize else {
            return
        }

        // Get all log files sorted by modification date
        guard let logFiles = try? fileManager.contentsOfDirectory(
            at: logDirectory,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: .skipsHiddenFiles
        ).filter({ $0.lastPathComponent.hasPrefix("vocalis_") && $0.pathExtension == "log" })
            .sorted(by: { lhs, rhs in
                let lhsDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? Date.distantPast
                let rhsDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? Date.distantPast
                return lhsDate > rhsDate
            }) else {
            return
        }

        // Delete old log files if exceeding max count
        if logFiles.count >= maxLogFiles {
            let filesToDelete = logFiles.dropFirst(maxLogFiles - 1)
            for file in filesToDelete {
                try? fileManager.removeItem(at: file)
            }
        }
    }

    /// Get path to current log file (for debugging)
    var currentLogPath: String {
        return currentLogFile.path
    }

    /// Get all log file paths
    var allLogPaths: [String] {
        guard let logFiles = try? fileManager.contentsOfDirectory(
            at: logDirectory,
            includingPropertiesForKeys: nil,
            options: .skipsHiddenFiles
        ).filter({ $0.lastPathComponent.hasPrefix("vocalis_") && $0.pathExtension == "log" })
            .sorted(by: { lhs, rhs in
                let lhsDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? Date.distantPast
                let rhsDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? Date.distantPast
                return lhsDate > rhsDate
            }) else {
            return []
        }

        return logFiles.map { $0.path }
    }
}
