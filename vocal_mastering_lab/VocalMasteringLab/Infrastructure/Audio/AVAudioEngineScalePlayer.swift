import Foundation
import VocalisDomain
import AVFoundation

/// Scale player implementation using AVAudioEngine and AVAudioUnitSampler
/// Now supports ScaleElement for chord playback
public class AVAudioEngineScalePlayer: ScalePlayerProtocol {
    private let engine: AVAudioEngine
    private let sampler: AVAudioUnitSampler
    private let settingsRepository: AudioSettingsRepositoryProtocol
    private var scale: [MIDINote] = []  // Legacy support
    private var scaleElements: [ScaleElement] = []  // New chord-enabled playback
    private var tempo: Tempo?
    private var playbackTask: Task<Void, Error>?
    private var _currentNoteIndex: Int = 0
    private var _isPlaying: Bool = false

    // MARK: - Timestamp Strategy
    private var timestampStrategy: ScaleTimestampStrategy

    public var isPlaying: Bool {
        _isPlaying
    }

    public var currentNoteIndex: Int {
        _currentNoteIndex
    }

    public var progress: Double {
        let totalCount = scaleElements.isEmpty ? scale.count : scaleElements.count
        guard totalCount > 0 else { return 0.0 }
        // Progress is 0.0 at start, 1.0 at completion
        // During playback: currentNoteIndex ranges from 0 to totalCount-1
        // After completion: currentNoteIndex = totalCount
        return min(1.0, Double(_currentNoteIndex) / Double(totalCount))
    }

    public var currentScaleElement: ScaleElement? {
        guard _isPlaying else { return nil }
        guard _currentNoteIndex >= 0 else { return nil }

        if !scaleElements.isEmpty {
            guard _currentNoteIndex < scaleElements.count else { return nil }
            return scaleElements[_currentNoteIndex]
        } else if !scale.isEmpty {
            guard _currentNoteIndex < scale.count else { return nil }
            return ScaleElement.scaleNote(scale[_currentNoteIndex])
        }
        return nil
    }

    /// Initialize with default TapBased timestamp strategy for accurate timing
    public init(settingsRepository: AudioSettingsRepositoryProtocol) {
        self.settingsRepository = settingsRepository
        engine = AVAudioEngine()
        sampler = AVAudioUnitSampler()

        // Use TapBasedTimestampStrategy by default for accurate timing
        // ImmediateTimestampStrategy has ~50ms offset due to AVAudioUnitSampler latency
        let tapStrategy = TapBasedTimestampStrategy()
        tapStrategy.configureSampler(sampler)
        self.timestampStrategy = tapStrategy

        // Connect sampler to engine's main mixer
        engine.attach(sampler)
        engine.connect(sampler, to: engine.mainMixerNode, format: nil)

        // Set volume to maximum for playback (will be adjusted in play method)
        engine.mainMixerNode.outputVolume = 1.0
    }

    /// Initialize with custom timestamp strategy
    public init(settingsRepository: AudioSettingsRepositoryProtocol, timestampStrategy: ScaleTimestampStrategy) {
        self.settingsRepository = settingsRepository
        self.timestampStrategy = timestampStrategy
        engine = AVAudioEngine()
        sampler = AVAudioUnitSampler()

        // Connect sampler to engine's main mixer
        engine.attach(sampler)
        engine.connect(sampler, to: engine.mainMixerNode, format: nil)

        // Set volume to maximum for playback (will be adjusted in play method)
        engine.mainMixerNode.outputVolume = 1.0

        // Configure tap-based strategy if needed
        if let tapStrategy = timestampStrategy as? TapBasedTimestampStrategy {
            tapStrategy.configureSampler(sampler)
        }
    }

    /// Switch timestamp strategy at runtime
    public func setTimestampStrategy(_ strategy: ScaleTimestampStrategy) {
        // Stop current recording if active
        if timestampStrategy.isRecording {
            timestampStrategy.stopRecording()
        }

        // Configure new strategy
        timestampStrategy = strategy

        if let tapStrategy = strategy as? TapBasedTimestampStrategy {
            tapStrategy.configureSampler(sampler)
        }
    }

    /// Get current timestamp strategy (for testing/debugging)
    public func getTimestampStrategy() -> ScaleTimestampStrategy {
        return timestampStrategy
    }

    public func loadScale(_ notes: [MIDINote], tempo: Tempo) async throws {
        self.scale = notes
        self.scaleElements = []  // Clear new format
        self.tempo = tempo
        self._currentNoteIndex = 0

        try await loadSoundBank()
    }

    /// Load scale elements with chord support (new format)
    public func loadScaleElements(_ elements: [ScaleElement], tempo: Tempo) async throws {
        self.scaleElements = elements
        self.scale = []  // Clear legacy format
        self.tempo = tempo
        self._currentNoteIndex = 0

        try await loadSoundBank()
    }

    /// Load SF2 sound bank for specified sound type
    ///
    /// Platform Notes:
    /// - All platforms now use SF2 file (GeneralUserGS.sf2) for consistency
    /// - SF2 file must be added to project as Bundle Resource
    /// - Supports all General MIDI instruments via Program Number
    private func loadSoundBank() async throws {
        // Get current scale sound type from settings
        let settings = settingsRepository.get()
        let soundType = settings.scaleSoundType

        // Get MIDI program number (fallback to Acoustic Grand Piano if nil, e.g., sine wave)
        let program = soundType.midiProgram ?? 0

        // Load SF2 file from Bundle (works on all platforms)
        guard let sf2URL = Bundle.main.url(forResource: "GeneralUserGS", withExtension: "sf2") else {
            throw ScalePlayerError.soundBankNotFound
        }

        do {
            try sampler.loadSoundBankInstrument(
                at: sf2URL,
                program: program,
                bankMSB: UInt8(kAUSampler_DefaultMelodicBankMSB),
                bankLSB: UInt8(kAUSampler_DefaultBankLSB)
            )
        } catch {
            throw ScalePlayerError.soundBankLoadFailed(error.localizedDescription)
        }
    }

    public func play(muted: Bool = false) async throws {
        guard tempo != nil else {
            throw ScalePlayerError.notLoaded
        }

        guard !_isPlaying else {
            throw ScalePlayerError.alreadyPlaying
        }

        // Load current settings and apply scale playback volume
        let settings = settingsRepository.get()
        engine.mainMixerNode.outputVolume = muted ? 0.0 : settings.scalePlaybackVolume

        // Choose playback mode based on what's loaded
        if !scaleElements.isEmpty {
            try await playScaleElements()
        } else if !scale.isEmpty {
            try await playLegacyScale()
        } else {
            // Empty scale completes immediately
            _currentNoteIndex = 0
            return
        }
    }

    /// Play scale elements with chord support (new format)
    private func playScaleElements() async throws {
        _isPlaying = true

        do {
            // Ensure audio session is active before starting engine
            try AudioSessionManager.shared.activateIfNeeded()

            try engine.start()

            playbackTask = Task { [weak self] in
                guard let self = self else { return }

                // Synchronize timestamp recording start time with actual playback loop start
                // This fixes the ~60ms offset caused by engine initialization + Task creation delay
                self.timestampStrategy.updateRecordingStartTime(Date())

                do {
                    for (index, element) in scaleElements.enumerated() {
                        try Task.checkCancellation()
                        self._currentNoteIndex = index

                        // Calculate durations based on tempo for consistent 4/4 rhythm
                        // Pattern: [だん 1拍] [ダーン 2拍] [間 1拍] = 4拍
                        let beat = self.tempo!.secondsPerNote

                        switch element {
                        case .chordShort(let notes):
                            // "Dan" - 1 beat
                            try await self.playChord(notes, duration: beat)

                        case .chordLong(let notes):
                            // "Daan" - 2 beats
                            try await self.playChord(notes, duration: beat * 2)

                        case .scaleNote(let note):
                            // Scale note - 1 beat
                            try await self.playNote(note, duration: beat)

                        case .silence:
                            // Gap before scale - 1 beat
                            try await Task.sleep(nanoseconds: UInt64(beat * 1_000_000_000))
                        }
                    }

                    // Playback completed
                    self._currentNoteIndex = scaleElements.count
                    self._isPlaying = false
                    self.engine.stop()
                } catch {
                    // Task cancelled or other error
                    self._isPlaying = false
                    self.engine.stop()
                }
            }

            // Don't wait for playback to complete - return immediately
        } catch is CancellationError {
            _isPlaying = false
        } catch {
            _isPlaying = false
            throw ScalePlayerError.playbackFailed(error.localizedDescription)
        }
    }

    /// Play legacy scale format (single notes only)
    private func playLegacyScale() async throws {
        _isPlaying = true

        do {
            // Ensure audio session is active before starting engine
            try AudioSessionManager.shared.activateIfNeeded()

            try engine.start()

            playbackTask = Task { [weak self] in
                guard let self = self else { return }

                // Synchronize timestamp recording start time with actual playback loop start
                // This fixes the ~60ms offset caused by engine initialization + Task creation delay
                self.timestampStrategy.updateRecordingStartTime(Date())

                do {
                    for (index, note) in scale.enumerated() {
                        try Task.checkCancellation()
                        self._currentNoteIndex = index

                        // Prepare for note start using strategy
                        self.timestampStrategy.prepareForNoteStart(note)

                        // Record timestamp immediately before MIDI command for accurate timing
                        if let timestamp = self.timestampStrategy.getNoteStartTimestamp() {
                            if let samplerStrategy = self.timestampStrategy as? SamplerTimestampStrategy {
                                let event = ScalePlaybackEvent(timestamp: timestamp, note: note, eventType: .noteStart)
                                samplerStrategy.appendEventDirectly(event)
                            }
                        }

                        // Play note (legato: stop previous note just before next one plays)
                        self.sampler.startNote(note.value, withVelocity: 64, onChannel: 0)

                        // Most of the note duration
                        try await Task.sleep(nanoseconds: UInt64(self.tempo!.secondsPerNote * 0.9 * 1_000_000_000))

                        self.sampler.stopNote(note.value, onChannel: 0)

                        // Record note end event
                        self.timestampStrategy.recordNoteEnd(note)

                        // Small gap between notes
                        try await Task.sleep(nanoseconds: UInt64(self.tempo!.secondsPerNote * 0.1 * 1_000_000_000))
                    }

                    // Playback completed
                    self._currentNoteIndex = scale.count
                    self._isPlaying = false
                    self.engine.stop()
                } catch {
                    // Task cancelled or other error
                    self._isPlaying = false
                    self.engine.stop()
                }
            }

            // Don't wait for playback to complete - return immediately
        } catch is CancellationError {
            _isPlaying = false
        } catch {
            _isPlaying = false
            throw ScalePlayerError.playbackFailed(error.localizedDescription)
        }
    }

    /// Play a single note with specified duration
    private func playNote(_ note: MIDINote, duration: TimeInterval) async throws {
        // Prepare for note start using strategy
        timestampStrategy.prepareForNoteStart(note)

        // Record timestamp immediately before MIDI command for accurate timing
        if let timestamp = timestampStrategy.getNoteStartTimestamp() {
            if let samplerStrategy = timestampStrategy as? SamplerTimestampStrategy {
                let event = ScalePlaybackEvent(timestamp: timestamp, note: note, eventType: .noteStart)
                samplerStrategy.appendEventDirectly(event)
            }
        }

        // Start note (MIDI command sent to sampler)
        sampler.startNote(note.value, withVelocity: 64, onChannel: 0)

        // Play for most of the duration (90%)
        try await Task.sleep(nanoseconds: UInt64(duration * 0.9 * 1_000_000_000))

        // Stop note
        sampler.stopNote(note.value, onChannel: 0)

        // Record note end event
        timestampStrategy.recordNoteEnd(note)

        // Small gap (10%)
        try await Task.sleep(nanoseconds: UInt64(duration * 0.1 * 1_000_000_000))
    }

    /// Play multiple notes simultaneously (chord)
    /// Note: Chords are key-change indicators, not target notes for singing
    /// Therefore, we do NOT record chord timestamps for pitch bar visualization
    private func playChord(_ notes: [MIDINote], duration: TimeInterval) async throws {
        // Start all notes simultaneously
        for note in notes {
            sampler.startNote(note.value, withVelocity: 64, onChannel: 0)
        }

        // Play for most of the duration (90%)
        try await Task.sleep(nanoseconds: UInt64(duration * 0.9 * 1_000_000_000))

        // Stop all notes simultaneously
        for note in notes {
            sampler.stopNote(note.value, onChannel: 0)
        }

        // Small gap (10%)
        try await Task.sleep(nanoseconds: UInt64(duration * 0.1 * 1_000_000_000))
    }

    public func stop() async {
        playbackTask?.cancel()
        playbackTask = nil
        _isPlaying = false
        engine.stop()

        // Send All Notes Off (MIDI CC 123) on channel 0 - much more efficient than stopping each note
        sampler.sendController(123, withValue: 0, onChannel: 0)
    }

    // MARK: - Timestamp Recording (delegated to strategy)

    public func startTimestampRecording(recordingStartTime: Date) {
        timestampStrategy.startRecording(recordingStartTime: recordingStartTime)
        // Note: Tap is no longer installed as timestamps are recorded directly
        // at MIDI send time via appendEventDirectly() for accurate timing
    }

    public func stopTimestampRecording() {
        timestampStrategy.stopRecording()
    }

    public func getPlaybackTimeline() -> ScalePlaybackTimeline? {
        guard timestampStrategy.isRecording, let startTime = timestampStrategy.recordingStartTime else {
            return nil
        }
        return ScalePlaybackTimeline(events: timestampStrategy.getRecordedEvents(), recordingStartTime: startTime)
    }
}
