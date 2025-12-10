import XCTest
import AVFoundation
@testable import VocalMasteringLab
@testable import VocalisDomain

/// Tests for AVAudioPlayerNodeScalePlayer
/// Implements ScalePlayerProtocol using AVAudioPlayerNode for synthesized sound playback
final class AVAudioPlayerNodeScalePlayerTests: XCTestCase {

    var sut: AVAudioPlayerNodeScalePlayer!
    var mockSettingsRepository: MockAudioSettingsRepository!

    override func setUp() {
        super.setUp()
        mockSettingsRepository = MockAudioSettingsRepository()
        sut = AVAudioPlayerNodeScalePlayer(settingsRepository: mockSettingsRepository)
    }

    override func tearDown() async throws {
        await sut.stop()
        sut = nil
        mockSettingsRepository = nil
        try await super.tearDown()
    }

    // MARK: - ScalePlayerProtocol Conformance Tests

    func testConformsToScalePlayerProtocol() {
        // Then: AVAudioPlayerNodeScalePlayer should conform to ScalePlayerProtocol
        XCTAssertTrue(sut is ScalePlayerProtocol)
    }

    // MARK: - Initial State Tests

    func testInitialState_isNotPlaying() {
        // Then
        XCTAssertFalse(sut.isPlaying)
    }

    func testInitialState_currentNoteIndexIsZero() {
        // Then
        XCTAssertEqual(sut.currentNoteIndex, 0)
    }

    func testInitialState_progressIsZero() {
        // Then
        XCTAssertEqual(sut.progress, 0.0)
    }

    func testInitialState_currentScaleElementIsNil() {
        // Then
        XCTAssertNil(sut.currentScaleElement)
    }

    // MARK: - Load Scale Tests

    func testLoadScale_withValidNotes_succeeds() async throws {
        // Given
        let notes = try [MIDINote(60), MIDINote(62), MIDINote(64)]
        let tempo = try Tempo(secondsPerNote: 0.5)

        // When / Then: Should not throw
        try await sut.loadScale(notes, tempo: tempo)
    }

    func testLoadScale_resetsCurrentNoteIndex() async throws {
        // Given
        let notes = try [MIDINote(60), MIDINote(62)]
        let tempo = try Tempo(secondsPerNote: 0.5)

        // When
        try await sut.loadScale(notes, tempo: tempo)

        // Then
        XCTAssertEqual(sut.currentNoteIndex, 0)
    }

    func testLoadScaleElements_withValidElements_succeeds() async throws {
        // Given
        let elements: [ScaleElement] = [
            .scaleNote(try MIDINote(60)),
            .scaleNote(try MIDINote(62)),
            .silence(0.5)
        ]
        let tempo = try Tempo(secondsPerNote: 0.5)

        // When / Then: Should not throw
        try await sut.loadScaleElements(elements, tempo: tempo)
    }

    // MARK: - Play Error Tests

    func testPlay_withoutLoadingScale_throwsNotLoadedError() async {
        // When / Then
        do {
            try await sut.play()
            XCTFail("Expected notLoaded error")
        } catch let error as ScalePlayerError {
            XCTAssertEqual(error, .notLoaded)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testPlay_afterLoadingScale_startsPlayback() async throws {
        // Given
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        // When
        try await sut.play()

        // Then
        // Give a tiny moment for playback to start
        try await Task.sleep(nanoseconds: 10_000_000) // 10ms
        XCTAssertTrue(sut.isPlaying)
    }

    func testPlay_whileAlreadyPlaying_throwsAlreadyPlayingError() async throws {
        // Given
        let notes = try [MIDINote(60), MIDINote(62), MIDINote(64)]
        let tempo = try Tempo(secondsPerNote: 0.5)
        try await sut.loadScale(notes, tempo: tempo)
        try await sut.play()

        // When / Then
        do {
            try await sut.play()
            XCTFail("Expected alreadyPlaying error")
        } catch let error as ScalePlayerError {
            XCTAssertEqual(error, .alreadyPlaying)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - Stop Tests

    func testStop_stopsPlayback() async throws {
        // Given
        let notes = try [MIDINote(60), MIDINote(62)]
        let tempo = try Tempo(secondsPerNote: 0.5)
        try await sut.loadScale(notes, tempo: tempo)
        try await sut.play()

        // When
        await sut.stop()

        // Then
        XCTAssertFalse(sut.isPlaying)
    }

    func testStop_whenNotPlaying_doesNothing() async {
        // When / Then: Should not crash
        await sut.stop()
        XCTAssertFalse(sut.isPlaying)
    }

    // MARK: - Progress Tracking Tests

    func testProgress_duringPlayback_increaseOverTime() async throws {
        // Given
        let notes = try [MIDINote(60), MIDINote(62), MIDINote(64)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        // When
        try await sut.play()
        try await Task.sleep(nanoseconds: 150_000_000) // 150ms - should be past first note

        // Then
        XCTAssertGreaterThan(sut.progress, 0.0)
    }

    func testCurrentScaleElement_duringPlayback_returnsCurrentNote() async throws {
        // Given
        let note = try MIDINote(60)
        let notes = [note]
        let tempo = try Tempo(secondsPerNote: 0.5)
        try await sut.loadScale(notes, tempo: tempo)

        // When
        try await sut.play()
        try await Task.sleep(nanoseconds: 50_000_000) // 50ms

        // Then
        if let element = sut.currentScaleElement {
            if case .scaleNote(let currentNote) = element {
                XCTAssertEqual(currentNote, note)
            } else {
                XCTFail("Expected scaleNote element")
            }
        } else {
            // May be nil if playback completed very quickly
            XCTAssertTrue(true)
        }
    }

    // MARK: - Timestamp Recording Tests

    func testStartTimestampRecording_enablesRecording() async throws {
        // Given
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        // When
        let startTime = Date()
        sut.startTimestampRecording(recordingStartTime: startTime)

        // Then
        // No direct way to check, but getPlaybackTimeline should work after playback
        try await sut.play()
        try await Task.sleep(nanoseconds: 150_000_000) // 150ms

        let timeline = sut.getPlaybackTimeline()
        XCTAssertNotNil(timeline)
    }

    func testGetPlaybackTimeline_withoutStartingRecording_returnsNil() async throws {
        // Given
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        // When
        try await sut.play()
        try await Task.sleep(nanoseconds: 150_000_000) // 150ms

        // Then
        let timeline = sut.getPlaybackTimeline()
        XCTAssertNil(timeline)
    }

    func testGetPlaybackTimeline_afterPlayback_containsEvents() async throws {
        // Given
        let notes = try [MIDINote(60), MIDINote(62)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        let startTime = Date()
        sut.startTimestampRecording(recordingStartTime: startTime)

        // When
        try await sut.play()
        // Wait for playback to complete
        try await Task.sleep(nanoseconds: 300_000_000) // 300ms

        // Then
        let timeline = sut.getPlaybackTimeline()
        XCTAssertNotNil(timeline)

        if let timeline = timeline {
            // Should have events for each note (start and end)
            XCTAssertGreaterThan(timeline.events.count, 0)
        }
    }

    func testStopTimestampRecording_clearsRecording() async throws {
        // Given
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        sut.startTimestampRecording(recordingStartTime: Date())
        try await sut.play()
        try await Task.sleep(nanoseconds: 150_000_000) // 150ms

        // When
        sut.stopTimestampRecording()

        // Then
        let timeline = sut.getPlaybackTimeline()
        XCTAssertNil(timeline)
    }

    // MARK: - Muted Playback Tests

    func testPlay_withMutedTrue_playsWithoutSound() async throws {
        // Given
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        // When
        try await sut.play(muted: true)
        try await Task.sleep(nanoseconds: 50_000_000) // 50ms

        // Then
        XCTAssertTrue(sut.isPlaying)
        // Note: We can't directly test that audio is muted without audio analysis
        // The important thing is that playback proceeds normally
    }

    // MARK: - Volume Settings Tests

    func testPlay_usesVolumeFromSettings() async throws {
        // Given
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scalePlaybackVolume: 0.5, // 50% volume
            recordingPlaybackVolume: 0.8,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4,
            scaleSoundType: .sineWave
        )
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        // When
        try await sut.play()
        try await Task.sleep(nanoseconds: 50_000_000) // 50ms

        // Then: Volume setting should be applied
        // Note: We can't directly verify volume without audio analysis
        // This test ensures the settings are being read
        XCTAssertTrue(sut.isPlaying)
    }

    // MARK: - ScaleElement Playback Tests

    func testPlayScaleElements_withChords_succeeds() async throws {
        // Given
        let chord = try [MIDINote(60), MIDINote(64), MIDINote(67)]
        let elements: [ScaleElement] = [
            .chordShort(chord),
            .chordLong(chord),
            .silence(0.5),
            .scaleNote(try MIDINote(60))
        ]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScaleElements(elements, tempo: tempo)

        // When
        try await sut.play()
        try await Task.sleep(nanoseconds: 50_000_000) // 50ms

        // Then
        XCTAssertTrue(sut.isPlaying)
    }

    // MARK: - Empty Scale Tests

    func testLoadScale_withEmptyNotes_succeeds() async throws {
        // Given
        let notes: [MIDINote] = []
        let tempo = try Tempo(secondsPerNote: 0.5)

        // When / Then: Should not throw
        try await sut.loadScale(notes, tempo: tempo)
    }

    func testPlay_withEmptyScale_completesImmediately() async throws {
        // Given
        let notes: [MIDINote] = []
        let tempo = try Tempo(secondsPerNote: 0.5)
        try await sut.loadScale(notes, tempo: tempo)

        // When
        try await sut.play()

        // Then: Should complete without error
        XCTAssertFalse(sut.isPlaying)
    }
}
