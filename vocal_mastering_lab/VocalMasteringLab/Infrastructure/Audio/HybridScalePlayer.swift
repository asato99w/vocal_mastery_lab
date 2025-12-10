import Foundation
import VocalisDomain
import AVFoundation

/// Engine type used for scale playback
public enum ScaleEngineType: Equatable {
    /// AVAudioUnitSampler-based engine (SF2 sounds)
    case sampler
    /// AVAudioPlayerNode-based engine (synthesized sounds)
    case playerNode
}

/// Hybrid scale player that automatically switches between Sampler and PlayerNode engines
/// based on the selected ScaleSoundType.
///
/// Engine Selection Logic:
/// - `midiProgram != nil` → AVAudioUnitSampler (SF2 sounds: Piano, Marimba, etc.)
/// - `midiProgram == nil` → AVAudioPlayerNode (synthesized sounds: Sine Wave)
///
/// This allows seamless integration with existing sound selection UI while providing
/// the timing accuracy benefits of AVAudioPlayerNode for synthesized sounds.
public class HybridScalePlayer: ScalePlayerProtocol {
    private let settingsRepository: AudioSettingsRepositoryProtocol

    // Lazy-initialized engines (created on demand)
    private var _samplerEngine: AVAudioEngineScalePlayer?
    private var _playerNodeEngine: AVAudioPlayerNodeScalePlayer?

    // Current active engine
    private var activeEngine: ScalePlayerProtocol?
    private var _currentEngineType: ScaleEngineType?

    /// Current engine type being used (nil if no scale loaded)
    public var currentEngineType: ScaleEngineType? {
        _currentEngineType
    }

    // MARK: - ScalePlayerProtocol Properties (delegated)

    public var isPlaying: Bool {
        activeEngine?.isPlaying ?? false
    }

    public var currentNoteIndex: Int {
        activeEngine?.currentNoteIndex ?? 0
    }

    public var progress: Double {
        activeEngine?.progress ?? 0.0
    }

    public var currentScaleElement: ScaleElement? {
        activeEngine?.currentScaleElement
    }

    // MARK: - Initialization

    public init(settingsRepository: AudioSettingsRepositoryProtocol) {
        self.settingsRepository = settingsRepository
    }

    // MARK: - Engine Management

    private func getSamplerEngine() -> AVAudioEngineScalePlayer {
        if _samplerEngine == nil {
            _samplerEngine = AVAudioEngineScalePlayer(settingsRepository: settingsRepository)
        }
        return _samplerEngine!
    }

    private func getPlayerNodeEngine() -> AVAudioPlayerNodeScalePlayer {
        if _playerNodeEngine == nil {
            _playerNodeEngine = AVAudioPlayerNodeScalePlayer(settingsRepository: settingsRepository)
        }
        return _playerNodeEngine!
    }

    private func selectEngine() -> ScalePlayerProtocol {
        let settings = settingsRepository.get()
        let soundType = settings.scaleSoundType

        // Engine selection based on midiProgram:
        // - nil (sineWave) → PlayerNode for accurate timing
        // - non-nil (SF2 sounds) → Sampler for rich instrument sounds
        if soundType.midiProgram == nil {
            _currentEngineType = .playerNode
            return getPlayerNodeEngine()
        } else {
            _currentEngineType = .sampler
            return getSamplerEngine()
        }
    }

    // MARK: - ScalePlayerProtocol Methods

    public func loadScale(_ notes: [MIDINote], tempo: Tempo) async throws {
        activeEngine = selectEngine()
        try await activeEngine?.loadScale(notes, tempo: tempo)
    }

    public func loadScaleElements(_ elements: [ScaleElement], tempo: Tempo) async throws {
        activeEngine = selectEngine()
        try await activeEngine?.loadScaleElements(elements, tempo: tempo)
    }

    public func play(muted: Bool = false) async throws {
        guard let engine = activeEngine else {
            throw ScalePlayerError.notLoaded
        }
        try await engine.play(muted: muted)
    }

    public func stop() async {
        await activeEngine?.stop()
    }

    // MARK: - Timestamp Recording (delegated)

    public func startTimestampRecording(recordingStartTime: Date) {
        activeEngine?.startTimestampRecording(recordingStartTime: recordingStartTime)
    }

    public func stopTimestampRecording() {
        activeEngine?.stopTimestampRecording()
    }

    public func getPlaybackTimeline() -> ScalePlaybackTimeline? {
        return activeEngine?.getPlaybackTimeline()
    }
}
