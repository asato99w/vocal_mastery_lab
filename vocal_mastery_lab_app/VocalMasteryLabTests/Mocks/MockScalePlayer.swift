import Foundation
import VocalisDomain
@testable import VocalMasteryLab

final class MockScalePlayer: ScalePlayerProtocol {
    var loadScaleCalled = false
    var playCalled = false
    var stopCalled = false

    var loadedNotes: [MIDINote]?
    var loadedTempo: Tempo?

    var loadScaleShouldFail = false
    var playShouldFail = false

    var loadScaleCallTime: Date?
    var playCallTime: Date?
    var stopCallTime: Date?

    var _isPlaying = false
    var _currentNoteIndex = 0
    var playMuted: Bool?  // Capture muted parameter from play(muted:)

    var isPlaying: Bool {
        _isPlaying
    }

    var currentNoteIndex: Int {
        _currentNoteIndex
    }

    var progress: Double {
        0.0
    }

    var currentScaleElement: ScaleElement?

    func loadScale(_ notes: [MIDINote], tempo: Tempo) async throws {
        loadScaleCalled = true
        loadScaleCallTime = Date()
        loadedNotes = notes
        loadedTempo = tempo

        if loadScaleShouldFail {
            throw ScalePlayerError.notLoaded
        }
    }

    func loadScaleElements(_ elements: [ScaleElement], tempo: Tempo) async throws {
        loadScaleCalled = true
        loadScaleCallTime = Date()
        loadedTempo = tempo

        // Extract MIDI notes from scale elements for test verification
        // This allows tests to verify the loaded content using loadedNotes
        var notes: [MIDINote] = []
        for element in elements {
            switch element {
            case .scaleNote(let note):
                notes.append(note)
            case .chordShort(let chordNotes), .chordLong(let chordNotes):
                // For chords, add all notes
                notes.append(contentsOf: chordNotes)
            case .silence:
                // Silence doesn't contribute notes
                break
            }
        }
        loadedNotes = notes

        if loadScaleShouldFail {
            throw ScalePlayerError.notLoaded
        }
    }

    func play(muted: Bool) async throws {
        playCalled = true
        playCallTime = Date()
        playMuted = muted  // Capture muted parameter
        _isPlaying = true

        if playShouldFail {
            _isPlaying = false
            throw ScalePlayerError.playbackFailed("Mock play error")
        }
    }

    func stop() async {
        stopCalled = true
        stopCallTime = Date()
        _isPlaying = false
    }

    // MARK: - Timestamp Recording

    var startTimestampRecordingCalled = false
    var stopTimestampRecordingCalled = false
    var getPlaybackTimelineCalled = false
    var recordingStartTime: Date?
    var mockPlaybackTimeline: ScalePlaybackTimeline?

    func startTimestampRecording(recordingStartTime: Date) {
        startTimestampRecordingCalled = true
        self.recordingStartTime = recordingStartTime
    }

    func stopTimestampRecording() {
        stopTimestampRecordingCalled = true
        recordingStartTime = nil
        mockPlaybackTimeline = nil
    }

    func getPlaybackTimeline() -> ScalePlaybackTimeline? {
        getPlaybackTimelineCalled = true
        return mockPlaybackTimeline
    }

    func reset() {
        loadScaleCalled = false
        playCalled = false
        stopCalled = false
        loadedNotes = nil
        loadedTempo = nil
        loadScaleShouldFail = false
        playShouldFail = false
        loadScaleCallTime = nil
        playCallTime = nil
        stopCallTime = nil
        playMuted = nil
        _isPlaying = false
        _currentNoteIndex = 0
        startTimestampRecordingCalled = false
        stopTimestampRecordingCalled = false
        getPlaybackTimelineCalled = false
        recordingStartTime = nil
        mockPlaybackTimeline = nil
    }
}
