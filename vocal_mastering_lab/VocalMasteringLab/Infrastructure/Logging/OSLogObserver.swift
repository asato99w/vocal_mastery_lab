import Foundation
import OSLog

/// Observes OSLog messages and automatically writes them to file
/// Only active in DEBUG builds
class OSLogObserver {
    static let shared = OSLogObserver()

    private var observerTask: Task<Void, Never>?
    private let subsystem: String

    private init() {
        self.subsystem = Bundle.main.bundleIdentifier ?? "com.vocalis.studio"
    }

    /// Start observing OSLog messages
    func startObserving() {
        #if DEBUG
        observerTask = Task.detached { [weak self] in
            guard let self = self else { return }

            do {
                // Get the OSLogStore for the current process
                let logStore = try OSLogStore(scope: .currentProcessIdentifier)

                // Get position to start reading from
                let position = logStore.position(timeIntervalSinceLatestBoot: 0)

                // Create predicate to filter only our app's logs
                let predicate = NSPredicate(format: "subsystem == %@", self.subsystem)

                // Read logs
                let entries = try logStore.getEntries(at: position, matching: predicate)

                // Process each log entry
                for entry in entries {
                    guard let logEntry = entry as? OSLogEntryLog else { continue }

                    // Extract log information
                    let level = self.levelString(from: logEntry.level)
                    let category = logEntry.category
                    let message = logEntry.composedMessage

                    // Write to file
                    FileLogger.shared.log(
                        level: level,
                        category: category,
                        message: message
                    )
                }
            } catch {
                // Failed to observe logs, fall back to manual logging only
                FileLogger.shared.log(
                    level: "ERROR",
                    category: "system",
                    message: "Failed to start OSLog observation: \(error.localizedDescription)"
                )
            }
        }
        #endif
    }

    /// Stop observing OSLog messages
    func stopObserving() {
        observerTask?.cancel()
        observerTask = nil
    }

    /// Convert OSLogEntryLog.Level to string
    private func levelString(from level: OSLogEntryLog.Level) -> String {
        switch level {
        case .undefined:
            return "UNDEFINED"
        case .debug:
            return "DEBUG"
        case .info:
            return "INFO"
        case .notice:
            return "NOTICE"
        case .error:
            return "ERROR"
        case .fault:
            return "FAULT"
        @unknown default:
            return "UNKNOWN"
        }
    }
}
