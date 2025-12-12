import Foundation
import VocalisDomain
import Combine
@testable import VocalMasteryLab

@MainActor
final class MockRealtimePitchDetector: ObservableObject, PitchDetectorProtocol {
    var startRealtimeDetectionCalled = false
    var stopRealtimeDetectionCalled = false

    @Published var detectedPitch: DetectedPitch?
    @Published var isDetecting: Bool = false
    @Published var spectrum: [Float]?

    // Publisher for protocol conformance
    var detectedPitchPublisher: AnyPublisher<DetectedPitch?, Never> {
        $detectedPitch.eraseToAnyPublisher()
    }

    var mockDetectedPitchSequence: [DetectedPitch] = []
    private var sequenceIndex = 0

    // 単一のピッチを設定するための互換性プロパティ
    var mockDetectedPitch: DetectedPitch? {
        didSet {
            detectedPitch = mockDetectedPitch
        }
    }

    // ピッチ検出のシミュレーション用タイマー
    private var pitchSimulationTimer: Timer?

    func startRealtimeDetection() async throws {
        startRealtimeDetectionCalled = true
        isDetecting = true

        // ピッチ検出をシミュレート（100msごとにピッチを更新）
        pitchSimulationTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self else { return }

            Task { @MainActor in
                if !self.mockDetectedPitchSequence.isEmpty && self.sequenceIndex < self.mockDetectedPitchSequence.count {
                    self.detectedPitch = self.mockDetectedPitchSequence[self.sequenceIndex]
                    self.sequenceIndex += 1
                } else {
                    // シーケンスが終わったらループ
                    self.sequenceIndex = 0
                }
            }
        }
    }

    func stopRealtimeDetection() {
        stopRealtimeDetectionCalled = true
        isDetecting = false
        pitchSimulationTimer?.invalidate()
        pitchSimulationTimer = nil
        sequenceIndex = 0
        detectedPitch = nil
        spectrum = nil
    }

    func reset() {
        startRealtimeDetectionCalled = false
        stopRealtimeDetectionCalled = false
        detectedPitch = nil
        mockDetectedPitchSequence = []
        sequenceIndex = 0
        isDetecting = false
        spectrum = nil
        pitchSimulationTimer?.invalidate()
        pitchSimulationTimer = nil
    }
}
