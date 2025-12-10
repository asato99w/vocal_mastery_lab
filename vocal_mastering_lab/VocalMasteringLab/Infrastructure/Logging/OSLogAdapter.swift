import Foundation
import OSLog
import VocalisDomain

/// Adapter that bridges VocalisDomain's LoggerProtocol to OSLog and FileLogger
///
/// This adapter implements the logging interface defined in the domain layer,
/// providing concrete logging capabilities using iOS's OSLog framework and
/// file-based logging for development.
///
/// Following Clean Architecture principles, this adapter lives in the Infrastructure
/// layer and allows the Application layer to depend only on the LoggerProtocol
/// abstraction from the Domain layer.
public final class OSLogAdapter: LoggerProtocol {

    // MARK: - Properties

    private let osLogger: Logger
    private let fileLogger: FileLogger

    // MARK: - Initialization

    /// Initialize with a specific OSLog category
    /// - Parameter category: The logging category (e.g., "useCase", "recording")
    public init(category: String) {
        let subsystem = Bundle.main.bundleIdentifier ?? "com.vocalis.studio"
        self.osLogger = Logger(subsystem: subsystem, category: category)
        self.fileLogger = FileLogger.shared
    }

    // MARK: - LoggerProtocol Implementation

    public func debug(_ message: String, category: String) {
        osLogger.debug("\(message)")
        fileLogger.log(level: "DEBUG", category: category, message: message)
    }

    public func info(_ message: String, category: String) {
        osLogger.info("\(message)")
        fileLogger.log(level: "INFO", category: category, message: message)
    }

    public func warning(_ message: String, category: String) {
        osLogger.warning("\(message)")
        fileLogger.log(level: "WARNING", category: category, message: message)
    }

    public func error(_ message: String, category: String) {
        osLogger.error("\(message)")
        fileLogger.log(level: "ERROR", category: category, message: message)
    }
}

// MARK: - Convenience Initializers for Common Categories

public extension OSLogAdapter {

    /// Logger for use case execution and business logic
    static var useCase: OSLogAdapter {
        OSLogAdapter(category: "useCase")
    }

    /// Logger for recording operations
    static var recording: OSLogAdapter {
        OSLogAdapter(category: "recording")
    }

    /// Logger for audio session and device management
    static var audio: OSLogAdapter {
        OSLogAdapter(category: "audio")
    }

    /// Logger for pitch detection and analysis
    static var pitchDetection: OSLogAdapter {
        OSLogAdapter(category: "pitch")
    }

    /// Logger for scale playback operations
    static var scalePlayer: OSLogAdapter {
        OSLogAdapter(category: "scale")
    }

    /// Logger for file system and data persistence operations
    static var storage: OSLogAdapter {
        OSLogAdapter(category: "storage")
    }

    /// Logger for UI events and user interactions
    static var ui: OSLogAdapter {
        OSLogAdapter(category: "ui")
    }

    /// Logger for ViewModel state changes
    static var viewModel: OSLogAdapter {
        OSLogAdapter(category: "viewModel")
    }
}
