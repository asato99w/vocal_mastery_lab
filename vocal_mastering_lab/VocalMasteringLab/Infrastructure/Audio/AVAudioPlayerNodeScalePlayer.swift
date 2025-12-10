import Foundation
import VocalisDomain
import AVFoundation

/// Scale player implementation using AVAudioEngine and AVAudioPlayerNode
/// Plays synthesized sine wave sounds using PCM buffers for predictable timing
/// Alternative to AVAudioEngineScalePlayer (Sampler-based) with lower latency
public class AVAudioPlayerNodeScalePlayer: ScalePlayerProtocol {
    private let engine: AVAudioEngine
    private let playerNode: AVAudioPlayerNode
    private let settingsRepository: AudioSettingsRepositoryProtocol
    private let sineWaveGenerator: SineWaveGenerator

    private var scale: [MIDINote] = []
    private var scaleElements: [ScaleElement] = []
    private var tempo: Tempo?
    private var playbackTask: Task<Void, Error>?
    private var _currentNoteIndex: Int = 0
    private var _isPlaying: Bool = false

    // MARK: - Timestamp Strategy
    private var timestampStrategy: ScaleTimestampStrategy

    // MARK: - Audio Format
    private let sampleRate: Double = 44100

    public var isPlaying: Bool {
        _isPlaying
    }

    public var currentNoteIndex: Int {
        _currentNoteIndex
    }

    public var progress: Double {
        let totalCount = scaleElements.isEmpty ? scale.count : scaleElements.count
        guard totalCount > 0 else { return 0.0 }
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

    // MARK: - Initialization

    public init(settingsRepository: AudioSettingsRepositoryProtocol) {
        self.settingsRepository = settingsRepository
        engine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        sineWaveGenerator = SineWaveGenerator()

        // Use SynthesizerTimestampStrategy for AVAudioPlayerNode with synthesized audio
        // This strategy applies outputLatency compensation only (no sampler offset)
        // because synthesized audio has much lower and predictable internal latency than SF2 sampler
        let synthesizerStrategy = SynthesizerTimestampStrategy()
        self.timestampStrategy = synthesizerStrategy

        // Connect player node to engine's main mixer
        engine.attach(playerNode)

        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        engine.connect(playerNode, to: engine.mainMixerNode, format: format)

        engine.mainMixerNode.outputVolume = 1.0
    }

    // MARK: - ScalePlayerProtocol

    public func loadScale(_ notes: [MIDINote], tempo: Tempo) async throws {
        self.scale = notes
        self.scaleElements = []
        self.tempo = tempo
        self._currentNoteIndex = 0
    }

    public func loadScaleElements(_ elements: [ScaleElement], tempo: Tempo) async throws {
        self.scaleElements = elements
        self.scale = []
        self.tempo = tempo
        self._currentNoteIndex = 0
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

    // MARK: - Playback Methods

    private func playScaleElements() async throws {
        _isPlaying = true

        do {
            try AudioSessionManager.shared.activateIfNeeded()
            try engine.start()
            playerNode.play()

            playbackTask = Task { [weak self] in
                guard let self = self else { return }

                self.timestampStrategy.updateRecordingStartTime(Date())

                do {
                    for (index, element) in scaleElements.enumerated() {
                        try Task.checkCancellation()
                        self._currentNoteIndex = index

                        let beat = self.tempo!.secondsPerNote

                        switch element {
                        case .chordShort(let notes):
                            try await self.playChord(notes, duration: beat)

                        case .chordLong(let notes):
                            try await self.playChord(notes, duration: beat * 2)

                        case .scaleNote(let note):
                            try await self.playNote(note, duration: beat)

                        case .silence(let duration):
                            try await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
                        }
                    }

                    self._currentNoteIndex = scaleElements.count
                    self._isPlaying = false
                    self.playerNode.stop()
                    self.engine.stop()
                } catch {
                    self._isPlaying = false
                    self.playerNode.stop()
                    self.engine.stop()
                }
            }
        } catch is CancellationError {
            _isPlaying = false
        } catch {
            _isPlaying = false
            throw ScalePlayerError.playbackFailed(error.localizedDescription)
        }
    }

    private func playLegacyScale() async throws {
        _isPlaying = true

        do {
            try AudioSessionManager.shared.activateIfNeeded()
            try engine.start()
            playerNode.play()

            playbackTask = Task { [weak self] in
                guard let self = self else { return }

                self.timestampStrategy.updateRecordingStartTime(Date())

                do {
                    for (index, note) in scale.enumerated() {
                        try Task.checkCancellation()
                        self._currentNoteIndex = index

                        try await self.playNote(note, duration: self.tempo!.secondsPerNote)
                    }

                    self._currentNoteIndex = scale.count
                    self._isPlaying = false
                    self.playerNode.stop()
                    self.engine.stop()
                } catch {
                    self._isPlaying = false
                    self.playerNode.stop()
                    self.engine.stop()
                }
            }
        } catch is CancellationError {
            _isPlaying = false
        } catch {
            _isPlaying = false
            throw ScalePlayerError.playbackFailed(error.localizedDescription)
        }
    }

    private func playNote(_ note: MIDINote, duration: TimeInterval) async throws {
        // Prepare for note start using strategy
        timestampStrategy.prepareForNoteStart(note)

        // Record timestamp before playback
        if let timestamp = timestampStrategy.getNoteStartTimestamp() {
            if let synthesizerStrategy = timestampStrategy as? SynthesizerTimestampStrategy {
                synthesizerStrategy.appendNoteStartEvent(note, timestamp: timestamp)
            }
        }

        // Generate and schedule buffer
        let buffer = sineWaveGenerator.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)
        playerNode.scheduleBuffer(buffer, completionHandler: nil)

        // Wait for note duration
        try await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))

        // Record note end event
        timestampStrategy.recordNoteEnd(note)
    }

    private func playChord(_ notes: [MIDINote], duration: TimeInterval) async throws {
        // For chords, we mix multiple sine waves into a single buffer
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        let frameCount = AVAudioFrameCount(duration * sampleRate)
        let mixedBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        mixedBuffer.frameLength = frameCount

        let mixedData = mixedBuffer.floatChannelData![0]

        // Initialize with zeros
        for i in 0..<Int(frameCount) {
            mixedData[i] = 0.0
        }

        // Mix all notes together with proper scaling
        // Use sqrt(count) instead of count to maintain perceived loudness
        // (power scales linearly, but loudness perception is roughly sqrt)
        let scaleFactor = Float(1.0 / sqrt(Double(notes.count)))
        for note in notes {
            let noteBuffer = sineWaveGenerator.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)
            if let noteData = noteBuffer.floatChannelData?[0] {
                for i in 0..<Int(frameCount) {
                    mixedData[i] += noteData[i] * scaleFactor
                }
            }
        }

        playerNode.scheduleBuffer(mixedBuffer, completionHandler: nil)

        // Wait for chord duration
        try await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
    }

    public func stop() async {
        playbackTask?.cancel()
        playbackTask = nil
        _isPlaying = false
        playerNode.stop()
        engine.stop()
    }

    // MARK: - Timestamp Recording

    public func startTimestampRecording(recordingStartTime: Date) {
        timestampStrategy.startRecording(recordingStartTime: recordingStartTime)
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
