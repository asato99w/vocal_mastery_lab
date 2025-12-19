import Foundation

/// User identifier value object
///
/// Ensures user identity is immutable and type-safe
public struct UserId: Hashable, Codable, Sendable {
    public let value: UUID

    /// Create a new unique user ID
    public init() {
        self.value = UUID()
    }

    /// Create user ID from existing UUID
    /// - Parameter value: The UUID value
    public init(value: UUID) {
        self.value = value
    }

    /// Create user ID from UUID string
    /// - Parameter string: UUID string representation
    /// - Returns: UserId if string is valid UUID, nil otherwise
    public init?(string: String) {
        guard let uuid = UUID(uuidString: string) else {
            return nil
        }
        self.value = uuid
    }
}

// MARK: - Identifiable Support

extension UserId: Identifiable {
    public var id: UUID { value }
}

// MARK: - CustomStringConvertible

extension UserId: CustomStringConvertible {
    public var description: String {
        value.uuidString
    }
}
