import Foundation

/// Tempo value object representing duration per note
public struct Tempo: Equatable, Hashable {
    public let secondsPerNote: Double

    public init(secondsPerNote: Double) throws {
        guard secondsPerNote > 0 else {
            throw TempoError.invalidValue(secondsPerNote)
        }
        self.secondsPerNote = secondsPerNote
    }

    /// Standard tempo: 1 second per note
    public static let standard = try! Tempo(secondsPerNote: 1.0)
}

public enum TempoError: LocalizedError {
    case invalidValue(Double)

    public var errorDescription: String? {
        switch self {
        case .invalidValue(let value):
            return "Tempo value \(value) must be greater than 0"
        }
    }
}
