import XCTest
import AVFoundation
@testable import VocalMasteryLab
@testable import VocalisDomain

/// Tests for HybridScalePlayer
/// Automatically switches between Sampler (SF2) and PlayerNode (synthesized) engines
/// based on ScaleSoundType.midiProgram (nil = PlayerNode, non-nil = Sampler)
final class HybridScalePlayerTests: XCTestCase {

    var sut: HybridScalePlayer!
    var mockSettingsRepository: MockAudioSettingsRepository!

    override func setUp() {
        super.setUp()
        mockSettingsRepository = MockAudioSettingsRepository()
        sut = HybridScalePlayer(settingsRepository: mockSettingsRepository)
    }

    override func tearDown() async throws {
        await sut.stop()
        sut = nil
        mockSettingsRepository = nil
        try await super.tearDown()
    }

    // MARK: - ScalePlayerProtocol Conformance Tests

    func testConformsToScalePlayerProtocol() {
        XCTAssertTrue(sut is ScalePlayerProtocol)
    }

    // MARK: - Engine Selection Tests

    func testLoadScale_withSineWaveSound_usesPlayerNodeEngine() async throws {
        // Given: Settings with sineWave (midiProgram = nil)
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .sineWave
        )
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.5)

        // When
        try await sut.loadScale(notes, tempo: tempo)

        // Then: Should use PlayerNode engine
        XCTAssertEqual(sut.currentEngineType, .playerNode)
    }

    func testLoadScale_withPianoSound_usesSamplerEngine() async throws {
        // Given: Settings with acousticGrandPiano (midiProgram = 0)
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .acousticGrandPiano
        )
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.5)

        // When
        try await sut.loadScale(notes, tempo: tempo)

        // Then: Should use Sampler engine
        XCTAssertEqual(sut.currentEngineType, .sampler)
    }

    func testLoadScale_withMarimbaSound_usesSamplerEngine() async throws {
        // Given: Settings with marimba (midiProgram = 12)
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .marimba
        )
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.5)

        // When
        try await sut.loadScale(notes, tempo: tempo)

        // Then: Should use Sampler engine
        XCTAssertEqual(sut.currentEngineType, .sampler)
    }

    func testLoadScale_engineSwitchesWhenSoundTypeChanges() async throws {
        // Given: Start with piano
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .acousticGrandPiano
        )
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.5)
        try await sut.loadScale(notes, tempo: tempo)
        XCTAssertEqual(sut.currentEngineType, .sampler)

        // When: Change to sine wave
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .sineWave
        )
        try await sut.loadScale(notes, tempo: tempo)

        // Then: Engine should switch to PlayerNode
        XCTAssertEqual(sut.currentEngineType, .playerNode)
    }

    // MARK: - Initial State Tests

    func testInitialState_isNotPlaying() {
        XCTAssertFalse(sut.isPlaying)
    }

    func testInitialState_currentNoteIndexIsZero() {
        XCTAssertEqual(sut.currentNoteIndex, 0)
    }

    func testInitialState_progressIsZero() {
        XCTAssertEqual(sut.progress, 0.0)
    }

    func testInitialState_currentScaleElementIsNil() {
        XCTAssertNil(sut.currentScaleElement)
    }

    // MARK: - Load Scale Tests

    func testLoadScale_withValidNotes_succeeds() async throws {
        let notes = try [MIDINote(60), MIDINote(62), MIDINote(64)]
        let tempo = try Tempo(secondsPerNote: 0.5)

        try await sut.loadScale(notes, tempo: tempo)
    }

    func testLoadScaleElements_withValidElements_succeeds() async throws {
        let elements: [ScaleElement] = [
            .scaleNote(try MIDINote(60)),
            .scaleNote(try MIDINote(62)),
            .silence(0.5)
        ]
        let tempo = try Tempo(secondsPerNote: 0.5)

        try await sut.loadScaleElements(elements, tempo: tempo)
    }

    // MARK: - Play Error Tests

    func testPlay_withoutLoadingScale_throwsNotLoadedError() async {
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
        // Given: Sine wave for faster test
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .sineWave
        )
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        // When
        try await sut.play()
        try await Task.sleep(nanoseconds: 10_000_000) // 10ms

        // Then
        XCTAssertTrue(sut.isPlaying)
    }

    func testPlay_whileAlreadyPlaying_throwsAlreadyPlayingError() async throws {
        // Given
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .sineWave
        )
        let notes = try [MIDINote(60), MIDINote(62)]
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
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .sineWave
        )
        let notes = try [MIDINote(60), MIDINote(62)]
        let tempo = try Tempo(secondsPerNote: 0.5)
        try await sut.loadScale(notes, tempo: tempo)
        try await sut.play()

        await sut.stop()

        XCTAssertFalse(sut.isPlaying)
    }

    // MARK: - Playback with Different Engines Tests

    func testPlay_withSineWave_playsSuccessfully() async throws {
        // Given
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .sineWave
        )
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        // When
        try await sut.play()
        try await Task.sleep(nanoseconds: 50_000_000) // 50ms

        // Then
        XCTAssertTrue(sut.isPlaying)
        XCTAssertEqual(sut.currentEngineType, .playerNode)
    }

    func testPlay_withPiano_playsSuccessfully() async throws {
        // Given
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .acousticGrandPiano
        )
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        // When
        try await sut.play()
        try await Task.sleep(nanoseconds: 50_000_000) // 50ms

        // Then
        XCTAssertTrue(sut.isPlaying)
        XCTAssertEqual(sut.currentEngineType, .sampler)
    }

    // MARK: - Timestamp Recording Tests

    func testStartTimestampRecording_worksWithBothEngines() async throws {
        // Test with PlayerNode
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .sineWave
        )
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        sut.startTimestampRecording(recordingStartTime: Date())
        try await sut.play()
        try await Task.sleep(nanoseconds: 150_000_000) // 150ms

        let timeline = sut.getPlaybackTimeline()
        XCTAssertNotNil(timeline)
    }

    func testStopTimestampRecording_clearsRecording() async throws {
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .sineWave
        )
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        sut.startTimestampRecording(recordingStartTime: Date())
        try await sut.play()
        try await Task.sleep(nanoseconds: 150_000_000) // 150ms
        sut.stopTimestampRecording()

        let timeline = sut.getPlaybackTimeline()
        XCTAssertNil(timeline)
    }

    // MARK: - Muted Playback Tests

    func testPlay_withMutedTrue_playsWithoutSound() async throws {
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .sineWave
        )
        let notes = try [MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        try await sut.play(muted: true)
        try await Task.sleep(nanoseconds: 50_000_000) // 50ms

        XCTAssertTrue(sut.isPlaying)
    }

    // MARK: - Progress and State Delegation Tests

    func testProgress_delegatesToActiveEngine() async throws {
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .sineWave
        )
        let notes = try [MIDINote(60), MIDINote(62), MIDINote(64)]
        let tempo = try Tempo(secondsPerNote: 0.1)
        try await sut.loadScale(notes, tempo: tempo)

        try await sut.play()
        try await Task.sleep(nanoseconds: 150_000_000) // 150ms

        XCTAssertGreaterThan(sut.progress, 0.0)
    }

    func testCurrentScaleElement_delegatesToActiveEngine() async throws {
        mockSettingsRepository.settingsToReturn = AudioDetectionSettings(
            scaleSoundType: .sineWave
        )
        let note = try MIDINote(60)
        let notes = [note]
        let tempo = try Tempo(secondsPerNote: 0.5)
        try await sut.loadScale(notes, tempo: tempo)

        try await sut.play()
        try await Task.sleep(nanoseconds: 50_000_000) // 50ms

        if let element = sut.currentScaleElement {
            if case .scaleNote(let currentNote) = element {
                XCTAssertEqual(currentNote, note)
            } else {
                XCTFail("Expected scaleNote element")
            }
        }
    }
}
