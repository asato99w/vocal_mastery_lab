import XCTest
import AVFoundation
import VocalisDomain
@testable import VocalMasteringLab

final class AVAudioEngineScalePlayerTests: XCTestCase {

    var sut: AVAudioEngineScalePlayer!
    var mockSettingsRepository: MockAudioSettingsRepository!

    override func setUp() async throws {
        try await super.setUp()
        mockSettingsRepository = MockAudioSettingsRepository()
        sut = AVAudioEngineScalePlayer(settingsRepository: mockSettingsRepository)
    }

    override func tearDown() async throws {
        await sut.stop()
        sut = nil
        try await super.tearDown()
    }

    // MARK: - Initialization Tests

    func testInit_DefaultState_NotPlaying() {
        XCTAssertFalse(sut.isPlaying)
        XCTAssertEqual(sut.currentNoteIndex, 0)
        XCTAssertEqual(sut.progress, 0.0)
    }

    // MARK: - loadScale Tests

    func testLoadScale_ValidNotesAndTempo_Success() async throws {
        let notes = [
            try MIDINote(60), // C4
            try MIDINote(62), // D4
            try MIDINote(64)  // E4
        ]
        let tempo = try Tempo(secondsPerNote: 0.5)

        try await sut.loadScale(notes, tempo: tempo)

        // No error should be thrown
        XCTAssertFalse(sut.isPlaying)
    }

    func testLoadScale_EmptyNotes_Success() async throws {
        let notes: [MIDINote] = []
        let tempo = Tempo.standard

        try await sut.loadScale(notes, tempo: tempo)

        // Should not throw error for empty notes
    }

    // MARK: - play Tests

    func testPlay_WithoutLoad_ThrowsError() async {
        do {
            try await sut.play()
            XCTFail("Expected to throw ScalePlayerError.notLoaded")
        } catch let error as ScalePlayerError {
            XCTAssertEqual(error, ScalePlayerError.notLoaded)
        } catch {
            XCTFail("Expected ScalePlayerError.notLoaded, got \(error)")
        }
    }

    func testPlay_AfterLoad_StartsPlayback() async throws {
        let notes = [try MIDINote(60)]
        try await sut.loadScale(notes, tempo: .standard)

        let playTask = Task {
            try await sut.play()
        }

        // Give it a moment to start
        try await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds

        XCTAssertTrue(sut.isPlaying)

        await sut.stop()
        playTask.cancel()
    }

    func testPlay_WhilePlaying_ThrowsError() async throws {
        let notes = [try MIDINote(60)]
        try await sut.loadScale(notes, tempo: .standard)

        let playTask = Task {
            try await sut.play()
        }

        // Give it a moment to start
        try await Task.sleep(nanoseconds: 100_000_000)

        // Try to play again while already playing
        do {
            try await sut.play()
            XCTFail("Expected to throw ScalePlayerError.alreadyPlaying")
        } catch let error as ScalePlayerError {
            XCTAssertEqual(error, ScalePlayerError.alreadyPlaying)
        }

        await sut.stop()
        playTask.cancel()
    }

    func testPlay_UpdatesCurrentNoteIndex() async throws {
        let notes = [
            try MIDINote(60),
            try MIDINote(62),
            try MIDINote(64)
        ]
        let tempo = try Tempo(secondsPerNote: 0.2)

        try await sut.loadScale(notes, tempo: tempo)

        let playTask = Task {
            try await sut.play()
        }

        // Wait for first note
        try await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
        let firstIndex = sut.currentNoteIndex
        XCTAssertGreaterThanOrEqual(firstIndex, 0)

        // Wait for progression
        try await Task.sleep(nanoseconds: 300_000_000) // 0.3 seconds
        let secondIndex = sut.currentNoteIndex
        XCTAssertGreaterThan(secondIndex, firstIndex)

        await sut.stop()
        playTask.cancel()
    }

    func testProgress_Calculation_CorrectValues() {
        // Test progress calculation without actual playback
        // This avoids timing issues
        XCTAssertEqual(sut.progress, 0.0) // No scale loaded

        // Note: We can't directly test progress during playback without timing dependencies
        // The progress property is tested implicitly through testPlay_CompletesSuccessfully
    }

    func testPlay_CompletesSuccessfully() async throws {
        let notes = [try MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)

        try await sut.loadScale(notes, tempo: tempo)

        try await sut.play()

        // Wait for playback to complete (note duration + small buffer)
        try await Task.sleep(nanoseconds: 200_000_000) // 0.2 seconds

        // After completion, should not be playing
        XCTAssertFalse(sut.isPlaying)
        XCTAssertEqual(sut.progress, 1.0)
    }

    // MARK: - stop Tests

    func testStop_WhilePlaying_StopsPlayback() async throws {
        let notes = [try MIDINote(60), try MIDINote(62)]
        let tempo = try Tempo(secondsPerNote: 1.0)

        try await sut.loadScale(notes, tempo: tempo)

        let playTask = Task {
            try await sut.play()
        }

        // Start playing
        try await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertTrue(sut.isPlaying)

        // Stop
        await sut.stop()

        // Should stop immediately
        XCTAssertFalse(sut.isPlaying)

        playTask.cancel()
    }

    func testStop_WhenNotPlaying_DoesNothing() async {
        // Should not crash or throw
        await sut.stop()
        XCTAssertFalse(sut.isPlaying)
    }

    // MARK: - Multiple playback cycle tests

    func testMultipleCycles_LoadPlayStopMultipleTimes_Success() async throws {
        for i in 0..<3 {
            let note = try MIDINote(UInt8(60 + i))
            try await sut.loadScale([note], tempo: .standard)

            let playTask = Task {
                try await sut.play()
            }

            try await Task.sleep(nanoseconds: 100_000_000)
            await sut.stop()
            playTask.cancel()

            XCTAssertFalse(sut.isPlaying, "Cycle \(i) should be stopped")
        }
    }

    // MARK: - Timestamp Recording Tests

    func testStartTimestampRecording_SetsRecordingStartTime() {
        // Given
        let startTime = Date()

        // When
        sut.startTimestampRecording(recordingStartTime: startTime)

        // Then - timeline should exist but be empty
        let timeline = sut.getPlaybackTimeline()
        XCTAssertNotNil(timeline)
        XCTAssertTrue(timeline?.events.isEmpty ?? false)
    }

    func testGetPlaybackTimeline_WithoutRecording_ReturnsNil() {
        // Given - no recording started

        // When
        let timeline = sut.getPlaybackTimeline()

        // Then
        XCTAssertNil(timeline)
    }

    func testStopTimestampRecording_ClearsRecordingState() {
        // Given
        sut.startTimestampRecording(recordingStartTime: Date())

        // When
        sut.stopTimestampRecording()

        // Then
        let timeline = sut.getPlaybackTimeline()
        XCTAssertNil(timeline)
    }

    func testPlayback_WithTimestampRecording_RecordsNoteEvents() async throws {
        // Given
        let notes = [try MIDINote(60), try MIDINote(62)]
        let tempo = try Tempo(secondsPerNote: 0.2)
        try await sut.loadScale(notes, tempo: tempo)

        let startTime = Date()
        sut.startTimestampRecording(recordingStartTime: startTime)

        // When - play the scale
        try await sut.play()

        // Wait for playback to complete
        try await Task.sleep(nanoseconds: 600_000_000) // 0.6 seconds (2 notes * 0.2s + buffer)

        // Then
        let timeline = sut.getPlaybackTimeline()
        XCTAssertNotNil(timeline)

        // Should have 4 events (2 noteStart + 2 noteEnd)
        XCTAssertEqual(timeline?.events.count, 4)

        // Verify event structure
        let events = timeline?.events ?? []

        // First note start
        XCTAssertEqual(events[0].eventType, .noteStart)
        XCTAssertEqual(events[0].note.value, 60)

        // First note end
        XCTAssertEqual(events[1].eventType, .noteEnd)
        XCTAssertEqual(events[1].note.value, 60)

        // Second note start
        XCTAssertEqual(events[2].eventType, .noteStart)
        XCTAssertEqual(events[2].note.value, 62)

        // Second note end
        XCTAssertEqual(events[3].eventType, .noteEnd)
        XCTAssertEqual(events[3].note.value, 62)

        // Verify timestamps are in order
        for i in 1..<events.count {
            XCTAssertGreaterThanOrEqual(events[i].timestamp, events[i - 1].timestamp)
        }
    }

    func testPlaybackScaleElements_WithTimestampRecording_RecordsNoteEvents() async throws {
        // Given
        let elements: [ScaleElement] = [
            .scaleNote(try MIDINote(60)),
            .scaleNote(try MIDINote(62))
        ]
        let tempo = try Tempo(secondsPerNote: 0.2)
        try await sut.loadScaleElements(elements, tempo: tempo)

        let startTime = Date()
        sut.startTimestampRecording(recordingStartTime: startTime)

        // When - play the scale
        try await sut.play()

        // Wait for playback to complete
        try await Task.sleep(nanoseconds: 600_000_000) // 0.6 seconds

        // Then
        let timeline = sut.getPlaybackTimeline()
        XCTAssertNotNil(timeline)

        // Should have 4 events (2 noteStart + 2 noteEnd)
        XCTAssertEqual(timeline?.events.count, 4)
    }

    func testPlayback_WithoutTimestampRecording_DoesNotRecordEvents() async throws {
        // Given - no timestamp recording started
        let notes = [try MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        // When - play the scale
        try await sut.play()
        try await Task.sleep(nanoseconds: 200_000_000)

        // Then - no timeline recorded
        let timeline = sut.getPlaybackTimeline()
        XCTAssertNil(timeline)
    }

    func testPlayback_ChordElements_DoesNotRecordTimestamps() async throws {
        // Given
        // Chords are key-change indicators, not target notes for singing
        // Therefore, chord timestamps are NOT recorded for pitch bar visualization
        let chord: [MIDINote] = [try MIDINote(60), try MIDINote(64), try MIDINote(67)] // C Major chord
        let elements: [ScaleElement] = [.chordShort(chord)]
        let tempo = Tempo.standard
        try await sut.loadScaleElements(elements, tempo: tempo)

        let startTime = Date()
        sut.startTimestampRecording(recordingStartTime: startTime)

        // When - play the chord
        try await sut.play()
        try await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds

        // Then
        let timeline = sut.getPlaybackTimeline()
        XCTAssertNotNil(timeline)

        // Chords should NOT be recorded - they are not target notes for singing
        XCTAssertEqual(timeline?.events.count, 0, "Chord events should not be recorded for pitch bar visualization")
    }
}
