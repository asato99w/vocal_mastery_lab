import Foundation
import VocalisDomain

/// Calculator for target frequencies based on scale settings
/// Extracts frequency calculation logic for testability
public struct TargetFrequencyCalculator {

    public init() {}

    /// Calculate target frequencies from scale settings
    /// Returns all frequencies that should be displayed as target lines on the pitch graph
    /// - Parameter settings: Scale settings containing key progression and note pattern
    /// - Returns: Array of frequencies in Hz, sorted in ascending order (deduplicated)
    public func calculateTargetFrequencies(from settings: ScaleSettings) -> [Double] {
        var frequencySet: Set<Double> = []

        // Get all root notes from key progression
        let keyRoots = settings.generateKeyRoots()

        // For each key root, add scale notes based on note pattern intervals
        for root in keyRoots {
            for interval in settings.notePattern.intervals {
                let noteValue = root + UInt8(interval)
                if let note = try? MIDINote(noteValue) {
                    frequencySet.insert(note.frequency)
                }
            }
        }

        return Array(frequencySet).sorted()
    }
}
