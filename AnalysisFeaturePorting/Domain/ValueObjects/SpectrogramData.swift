import Foundation

/// Spectrogram data for a recording
/// Contains frequency spectrum information over time
public struct SpectrogramData: Equatable {
    /// Time stamps for each frame (in seconds)
    public let timeStamps: [Double]

    /// Frequency bins (in Hz)
    public let frequencyBins: [Float]

    /// Magnitudes for each time-frequency bin
    /// [timeIndex][frequencyIndex] = magnitude (0.0 - 1.0)
    public let magnitudes: [[Float]]

    /// Number of time frames in this spectrogram
    public var timeFrameCount: Int {
        return timeStamps.count
    }

    /// Number of frequency bins
    public var frequencyBinCount: Int {
        return frequencyBins.count
    }

    public init(
        timeStamps: [Double],
        frequencyBins: [Float],
        magnitudes: [[Float]]
    ) {
        self.timeStamps = timeStamps
        self.frequencyBins = frequencyBins
        self.magnitudes = magnitudes
    }

    public static func == (lhs: SpectrogramData, rhs: SpectrogramData) -> Bool {
        return lhs.timeStamps == rhs.timeStamps &&
               lhs.frequencyBins == rhs.frequencyBins &&
               lhs.magnitudes == rhs.magnitudes
    }
}
