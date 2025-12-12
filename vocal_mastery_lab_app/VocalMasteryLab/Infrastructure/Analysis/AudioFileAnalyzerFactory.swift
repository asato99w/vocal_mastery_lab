import Foundation
import VocalisDomain

/// Factory that creates AudioFileAnalyzer instances with the currently configured pitch algorithm
/// This allows the algorithm to be changed in settings and applied to subsequent analyses
public class AudioFileAnalyzerFactory: AudioFileAnalyzerFactoryProtocol {
    private let settingsRepository: AudioSettingsRepositoryProtocol

    public init(settingsRepository: AudioSettingsRepositoryProtocol) {
        self.settingsRepository = settingsRepository
    }

    /// Create a new audio file analyzer with the current pitch algorithm setting
    /// - Returns: AudioFileAnalyzer configured with the selected strategy
    public func makeAnalyzer() -> AudioFileAnalyzerProtocol {
        let settings = settingsRepository.get()
        let strategy = PitchStrategyFactory.createStrategy(for: settings.pitchAlgorithm)
        return AudioFileAnalyzer(pitchStrategy: strategy)
    }
}
