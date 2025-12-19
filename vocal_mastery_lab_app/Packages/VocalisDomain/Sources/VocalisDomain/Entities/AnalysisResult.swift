import Foundation

/// Analysis result for a recording
/// Contains both pitch analysis and spectrogram data
public struct AnalysisResult: Equatable {
    /// Pitch analysis data (detected frequencies over time)
    public let pitchData: PitchAnalysisData

    /// Spectrogram data (frequency spectrum over time)
    public let spectrogramData: SpectrogramData

    public init(
        pitchData: PitchAnalysisData,
        spectrogramData: SpectrogramData
    ) {
        self.pitchData = pitchData
        self.spectrogramData = spectrogramData
    }

    public static func == (lhs: AnalysisResult, rhs: AnalysisResult) -> Bool {
        return lhs.pitchData == rhs.pitchData &&
               lhs.spectrogramData == rhs.spectrogramData
    }
}
