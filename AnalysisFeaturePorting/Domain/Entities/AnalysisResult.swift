import Foundation

/// Analysis result for a recording
/// Contains both pitch analysis and spectrogram data
public struct AnalysisResult: Equatable {
    /// Pitch analysis data (detected frequencies over time)
    public let pitchData: PitchAnalysisData

    /// Spectrogram data (frequency spectrum over time)
    public let spectrogramData: SpectrogramData

    /// Original scale settings used during recording (nil for free recordings)
    public let scaleSettings: ScaleSettings?

    public init(
        pitchData: PitchAnalysisData,
        spectrogramData: SpectrogramData,
        scaleSettings: ScaleSettings?
    ) {
        self.pitchData = pitchData
        self.spectrogramData = spectrogramData
        self.scaleSettings = scaleSettings
    }

    public static func == (lhs: AnalysisResult, rhs: AnalysisResult) -> Bool {
        return lhs.pitchData == rhs.pitchData &&
               lhs.spectrogramData == rhs.spectrogramData &&
               lhs.scaleSettings == rhs.scaleSettings
    }
}
