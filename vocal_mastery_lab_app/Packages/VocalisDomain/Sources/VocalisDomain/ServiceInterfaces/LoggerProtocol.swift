import Foundation

/// Domain layer logging abstraction
///
/// This protocol defines the interface for logging services in the domain layer.
/// Following the Dependency Inversion Principle, Application layer depends on this
/// abstraction rather than concrete logging implementations.
public protocol LoggerProtocol {

    /// Log a debug message
    /// - Parameters:
    ///   - message: The message to log
    ///   - category: Optional category for organizing logs (default: "Default")
    func debug(_ message: String, category: String)

    /// Log an informational message
    /// - Parameters:
    ///   - message: The message to log
    ///   - category: Optional category for organizing logs (default: "Default")
    func info(_ message: String, category: String)

    /// Log a warning message
    /// - Parameters:
    ///   - message: The message to log
    ///   - category: Optional category for organizing logs (default: "Default")
    func warning(_ message: String, category: String)

    /// Log an error message
    /// - Parameters:
    ///   - message: The message to log
    ///   - category: Optional category for organizing logs (default: "Default")
    func error(_ message: String, category: String)
}

// MARK: - Default implementations for convenience

public extension LoggerProtocol {

    func debug(_ message: String) {
        debug(message, category: "Default")
    }

    func info(_ message: String) {
        info(message, category: "Default")
    }

    func warning(_ message: String) {
        warning(message, category: "Default")
    }

    func error(_ message: String) {
        error(message, category: "Default")
    }
}
