import Foundation
import Combine

/// Protocol for real-time pitch detection
public protocol PitchDetectorProtocol {
    /// Publisher that emits detected pitch values
    var detectedPitchPublisher: AnyPublisher<DetectedPitch?, Never> { get }

    /// Start real-time pitch detection from microphone
    func startRealtimeDetection() async throws

    /// Stop real-time pitch detection
    func stopRealtimeDetection()
}
