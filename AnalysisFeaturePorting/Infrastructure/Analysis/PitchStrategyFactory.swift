import Foundation
import VocalisDomain

/// Factory to create PitchDetectionStrategy instances from PitchDetectionAlgorithm settings
enum PitchStrategyFactory {

    /// Create a pitch detection strategy based on the algorithm setting
    /// - Parameter algorithm: The selected pitch detection algorithm
    /// - Returns: A configured PitchDetectionStrategy instance
    static func createStrategy(for algorithm: PitchDetectionAlgorithm) -> PitchDetectionStrategy {
        switch algorithm {
        case .yin:
            return YINStrategy()

        case .pyinDefault:
            return PYINStrategy(configuration: .default, name: "pYIN")

        case .pyinHighDetection:
            return PYINStrategy(configuration: .highDetection, name: "pYIN-highDetection")

        case .pyinBalanced:
            return PYINStrategy(configuration: .balanced, name: "pYIN-balanced")

        case .pyinAggressive:
            return PYINStrategy(configuration: .aggressive, name: "pYIN-aggressive")

        case .fcpe:
            return FCPEStrategy()
        }
    }
}
