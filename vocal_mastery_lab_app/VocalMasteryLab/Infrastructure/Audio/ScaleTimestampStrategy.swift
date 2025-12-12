import Foundation
import AVFoundation
import VocalisDomain

// MARK: - Protocol

/// Strategy protocol for recording scale note timestamps
/// Allows switching between different timing approaches:
/// - Immediate: Record timestamp when startNote() is called (simple, but has latency offset)
/// - TapBased: Record timestamp when audio is actually detected (accurate, but more complex)
public protocol ScaleTimestampStrategy: AnyObject {
    /// Called when timestamp recording should begin
    func startRecording(recordingStartTime: Date)

    /// Called when timestamp recording should end
    func stopRecording()

    /// Prepare to record timestamp for the next note
    /// Called immediately before startNote()
    func prepareForNoteStart(_ note: MIDINote)

    /// Get the timestamp for the note start event
    /// For immediate strategy: returns timestamp immediately
    /// For tap-based strategy: returns timestamp when audio is detected
    func getNoteStartTimestamp() -> TimeInterval?

    /// Record a note end timestamp
    /// Note end uses immediate timing (no significant latency for stopNote)
    func recordNoteEnd(_ note: MIDINote)

    /// Get all recorded events
    func getRecordedEvents() -> [ScalePlaybackEvent]

    /// Check if recording is active
    var isRecording: Bool { get }

    /// Get recording start time
    var recordingStartTime: Date? { get }

    /// Update recording start time after engine initialization
    /// This allows synchronizing the start time with actual audio engine start
    func updateRecordingStartTime(_ newStartTime: Date)
}

// MARK: - Raw Timestamp Strategy (No Latency Compensation)

/// Records raw timestamps without any latency compensation
/// Use for testing or when latency compensation is handled elsewhere
/// Also aliased as ImmediateTimestampStrategy for backwards compatibility
public class RawTimestampStrategy: ScaleTimestampStrategy {
    private var _isRecording: Bool = false
    private var _recordingStartTime: Date?
    private var recordedEvents: [ScalePlaybackEvent] = []

    public var isRecording: Bool { _isRecording }
    public var recordingStartTime: Date? { _recordingStartTime }

    public init() {}

    public func startRecording(recordingStartTime: Date) {
        _isRecording = true
        _recordingStartTime = recordingStartTime
        recordedEvents = []
    }

    public func updateRecordingStartTime(_ newStartTime: Date) {
        guard _isRecording else { return }
        _recordingStartTime = newStartTime
    }

    public func stopRecording() {
        _isRecording = false
        _recordingStartTime = nil
    }

    public func prepareForNoteStart(_ note: MIDINote) {
        // No preparation needed for immediate strategy
    }

    public func getNoteStartTimestamp() -> TimeInterval? {
        guard _isRecording, let startTime = _recordingStartTime else { return nil }
        return Date().timeIntervalSince(startTime)
    }

    public func recordNoteEnd(_ note: MIDINote) {
        guard _isRecording, let startTime = _recordingStartTime else { return }
        let timestamp = Date().timeIntervalSince(startTime)
        let event = ScalePlaybackEvent(timestamp: timestamp, note: note, eventType: .noteEnd)
        recordedEvents.append(event)
    }

    public func getRecordedEvents() -> [ScalePlaybackEvent] {
        return recordedEvents
    }

    /// Internal method to append note start event (called by ScalePlayer)
    func appendNoteStartEvent(_ note: MIDINote, timestamp: TimeInterval) {
        let event = ScalePlaybackEvent(timestamp: timestamp, note: note, eventType: .noteStart)
        recordedEvents.append(event)
    }
}

/// Backwards compatibility alias for RawTimestampStrategy
public typealias ImmediateTimestampStrategy = RawTimestampStrategy

// MARK: - Synthesizer Timestamp Strategy (OutputLatency Only)

/// Timestamp strategy optimized for synthesized audio (e.g., AVAudioPlayerNode with PCM buffers)
/// Applies outputLatency compensation but NOT sampler-specific offset
/// because synthesized audio has much lower and predictable internal latency compared to SF2 sampler
/// Also aliased as PlayerNodeTimestampStrategy for backwards compatibility
public class SynthesizerTimestampStrategy: ScaleTimestampStrategy {
    private var _isRecording: Bool = false
    private var _recordingStartTime: Date?
    private var recordedEvents: [ScalePlaybackEvent] = []

    public var isRecording: Bool { _isRecording }
    public var recordingStartTime: Date? { _recordingStartTime }

    // Output latency compensation - cached at recording start to avoid per-note API calls
    // Calling AVAudioSession.sharedInstance().outputLatency per note can cause rhythm issues on real devices
    private var cachedOutputLatency: TimeInterval = 0

    public init() {}

    public func startRecording(recordingStartTime: Date) {
        _isRecording = true
        _recordingStartTime = recordingStartTime
        recordedEvents = []
        // Cache outputLatency at recording start to avoid per-note API calls
        cachedOutputLatency = AVAudioSession.sharedInstance().outputLatency
    }

    public func updateRecordingStartTime(_ newStartTime: Date) {
        guard _isRecording else { return }
        _recordingStartTime = newStartTime
    }

    public func stopRecording() {
        _isRecording = false
        _recordingStartTime = nil
        cachedOutputLatency = 0
    }

    public func prepareForNoteStart(_ note: MIDINote) {
        // No preparation needed for PlayerNode strategy
    }

    public func getNoteStartTimestamp() -> TimeInterval? {
        guard _isRecording, let startTime = _recordingStartTime else { return nil }
        let rawTimestamp = Date().timeIntervalSince(startTime)
        // Add cached outputLatency only - PlayerNode does not have the additional sampler latency
        return rawTimestamp + cachedOutputLatency
    }

    public func recordNoteEnd(_ note: MIDINote) {
        guard _isRecording, let startTime = _recordingStartTime else { return }
        let rawTimestamp = Date().timeIntervalSince(startTime)
        // Consistent with noteStart: add cached outputLatency only
        let timestamp = rawTimestamp + cachedOutputLatency
        let event = ScalePlaybackEvent(timestamp: timestamp, note: note, eventType: .noteEnd)
        recordedEvents.append(event)
    }

    public func getRecordedEvents() -> [ScalePlaybackEvent] {
        return recordedEvents
    }

    /// Internal method to append note start event (called by ScalePlayer)
    func appendNoteStartEvent(_ note: MIDINote, timestamp: TimeInterval) {
        let event = ScalePlaybackEvent(timestamp: timestamp, note: note, eventType: .noteStart)
        recordedEvents.append(event)
    }
}

/// Backwards compatibility alias for SynthesizerTimestampStrategy
public typealias PlayerNodeTimestampStrategy = SynthesizerTimestampStrategy

// MARK: - Sampler Timestamp Strategy (SF2 with Full Latency Compensation)

/// Timestamp strategy optimized for SF2 sampler audio (e.g., AVAudioUnitSampler with SoundFont)
/// Applies full latency compensation: outputLatency + samplerLatencyOffset (80ms)
/// because SF2 sampler has significant internal processing delay
/// Also aliased as TapBasedTimestampStrategy for backwards compatibility
public class SamplerTimestampStrategy: ScaleTimestampStrategy {
    private var _isRecording: Bool = false
    private var _recordingStartTime: Date?
    private var recordedEvents: [ScalePlaybackEvent] = []

    // Tap-related state
    private var pendingNote: MIDINote?
    private var detectedTimestamp: TimeInterval?
    private var tapInstalled: Bool = false
    private weak var samplerNode: AVAudioNode?

    // Thread-safe access using a lock-free approach
    // We use a simple atomic flag pattern for the real-time audio thread
    private var audioDetected: Bool = false
    private var audioDetectionWallTime: Date?

    // Output latency compensation - cached at recording start to avoid per-note API calls
    // When audio is detected in the tap, the sound actually reaches the speaker/headphone
    // after an additional outputLatency delay. We add this offset to timestamps so that
    // scale bars appear at the time the user actually hears them.
    // Calling AVAudioSession.sharedInstance().outputLatency per note can cause rhythm issues on real devices
    private var cachedOutputLatency: TimeInterval = 0

    // Sampler internal latency compensation (in seconds)
    // Measured offset between ScaleBarTime and PitchTime:
    // - Piano: -100.1ms, Marimba: -102.3ms, SineWave: -113.6ms
    // - SF2 sounds average: ~101ms
    // We add this offset to ScaleBarTime to align with detected pitch times.
    // Testing with 80ms first as a conservative value.
    private let samplerLatencyOffset: TimeInterval = 0.080

    public var isRecording: Bool { _isRecording }
    public var recordingStartTime: Date? { _recordingStartTime }

    public init() {}

    public func startRecording(recordingStartTime: Date) {
        _isRecording = true
        _recordingStartTime = recordingStartTime
        recordedEvents = []
        // Cache outputLatency at recording start to avoid per-note API calls
        cachedOutputLatency = AVAudioSession.sharedInstance().outputLatency
    }

    /// Update recording start time after engine initialization
    /// This allows synchronizing the start time with actual audio engine start
    public func updateRecordingStartTime(_ newStartTime: Date) {
        guard _isRecording else { return }
        _recordingStartTime = newStartTime
    }

    public func stopRecording() {
        _isRecording = false
        _recordingStartTime = nil
        cachedOutputLatency = 0
        removeTap()
    }

    /// Configure the sampler node for tap installation
    public func configureSampler(_ sampler: AVAudioUnitSampler) {
        self.samplerNode = sampler
    }

    /// Install tap on sampler to detect audio output
    public func installTap() {
        guard let sampler = samplerNode as? AVAudioUnitSampler, !tapInstalled else { return }

        let bufferSize: AVAudioFrameCount = 512
        let format = sampler.outputFormat(forBus: 0)

        sampler.installTap(onBus: 0, bufferSize: bufferSize, format: format) { [weak self] buffer, _ in
            guard let self = self else { return }

            // Only process if we have a pending note and haven't detected audio yet
            guard self.pendingNote != nil && !self.audioDetected else { return }

            // Check for non-zero audio samples
            let channelData = buffer.floatChannelData?[0]
            let frameLength = Int(buffer.frameLength)
            var hasAudio = false

            if let data = channelData {
                for i in 0..<frameLength {
                    if abs(data[i]) > 0.001 {
                        hasAudio = true
                        break
                    }
                }
            }

            if hasAudio {
                // Mark audio as detected and capture wall time
                // Note: Date() is not strictly real-time safe, but acceptable for our use case
                self.audioDetected = true
                self.audioDetectionWallTime = Date()
            }
        }

        tapInstalled = true
    }

    /// Remove tap from sampler
    public func removeTap() {
        guard let sampler = samplerNode as? AVAudioUnitSampler, tapInstalled else { return }
        sampler.removeTap(onBus: 0)
        tapInstalled = false
        pendingNote = nil
        audioDetected = false
        audioDetectionWallTime = nil
    }

    public func prepareForNoteStart(_ note: MIDINote) {
        pendingNote = note
        audioDetected = false
        audioDetectionWallTime = nil
        detectedTimestamp = nil
        // Note: Tap is no longer used - timestamps are recorded directly at MIDI send time
    }

    public func getNoteStartTimestamp() -> TimeInterval? {
        guard _isRecording, let startTime = _recordingStartTime else { return nil }

        // If audio was detected, use the detected timestamp
        if audioDetected, let detectionTime = audioDetectionWallTime {
            let rawTimestamp = detectionTime.timeIntervalSince(startTime)
            // Add cached outputLatency: the user hears the sound outputLatency seconds after detection
            // Add samplerLatencyOffset: compensate for sampler internal processing delay
            let compensatedTimestamp = rawTimestamp + cachedOutputLatency + samplerLatencyOffset
            return compensatedTimestamp
        }

        // Fallback to immediate timestamp if audio not yet detected
        // This handles edge cases where tap doesn't fire quickly enough
        let rawTimestamp = Date().timeIntervalSince(startTime)
        return rawTimestamp + cachedOutputLatency + samplerLatencyOffset
    }

    public func recordNoteEnd(_ note: MIDINote) {
        guard _isRecording, let startTime = _recordingStartTime else { return }
        // Note end uses immediate timing but still needs cached outputLatency and samplerLatency compensation
        // to stay consistent with noteStart timestamps
        let rawTimestamp = Date().timeIntervalSince(startTime)
        let timestamp = rawTimestamp + cachedOutputLatency + samplerLatencyOffset
        let event = ScalePlaybackEvent(timestamp: timestamp, note: note, eventType: .noteEnd)
        recordedEvents.append(event)
    }

    public func getRecordedEvents() -> [ScalePlaybackEvent] {
        return recordedEvents
    }

    /// Append an event directly without waiting for audio detection
    /// Used when recording timestamp at MIDI send time for more accurate timing
    public func appendEventDirectly(_ event: ScalePlaybackEvent) {
        recordedEvents.append(event)
    }
}

/// Backwards compatibility alias for SamplerTimestampStrategy
public typealias TapBasedTimestampStrategy = SamplerTimestampStrategy
