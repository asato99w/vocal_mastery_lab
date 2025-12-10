import XCTest
import VocalisDomain
@testable import VocalMasteringLab

@MainActor
final class ScalePlaybackCoordinatorTests: XCTestCase {
    var sut: ScalePlaybackCoordinator!
    var mockScalePlayer: MockScalePlayer!

    override func setUp() async throws {
        try await super.setUp()
        mockScalePlayer = MockScalePlayer()
        sut = ScalePlaybackCoordinator(scalePlayer: mockScalePlayer)
    }

    override func tearDown() async throws {
        sut = nil
        mockScalePlayer = nil
        try await super.tearDown()
    }

    // MARK: - Test startMutedPlayback()

    func testStartMutedPlayback_shouldLoadAndPlayScaleInMutedMode() async throws {
        // Given: Scale settings with specific configuration
        let settings = ScaleSettings(
            startNote: try MIDINote(60),
            endNote: try MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: .standard
        )

        // When: Start muted playback
        try await sut.startMutedPlayback(settings: settings)

        // Wait for background task to execute
        try await Task.sleep(nanoseconds: 50_000_000) // 0.05 seconds

        // Then: ScalePlayer should be loaded and playing in muted mode
        XCTAssertTrue(mockScalePlayer.loadScaleCalled, "loadScaleElements should be called")
        XCTAssertTrue(mockScalePlayer.playCalled, "play should be called")
        XCTAssertEqual(mockScalePlayer.playMuted, true, "play should be called with muted=true")
    }

    func testStartMutedPlayback_shouldStoreCurrentSettings() async throws {
        // Given: Scale settings
        let settings = ScaleSettings(
            startNote: try MIDINote(48),
            endNote: try MIDINote(60),
            notePattern: .fiveToneScale,
            tempo: .standard
        )

        // When: Start muted playback
        try await sut.startMutedPlayback(settings: settings)

        // Then: Current settings should be stored
        XCTAssertNotNil(sut.currentSettings, "currentSettings should be stored")
        XCTAssertEqual(sut.currentSettings?.tempo, .standard)
    }

    // MARK: - Test stopPlayback()

    func testStopPlayback_shouldStopScalePlayerAndClearSettings() async {
        // Given: Coordinator with ongoing playback
        let settings = ScaleSettings(
            startNote: try! MIDINote(60),
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: .standard
        )
        try! await sut.startMutedPlayback(settings: settings)
        mockScalePlayer.stopCalled = false  // Reset for this test

        // When: Stop playback
        await sut.stopPlayback()

        // Then: ScalePlayer should be stopped and settings cleared
        XCTAssertTrue(mockScalePlayer.stopCalled, "stop should be called on scalePlayer")
        XCTAssertNil(sut.currentSettings, "currentSettings should be cleared")
    }

    // MARK: - Test currentScaleElement

    func testCurrentScaleElement_shouldReturnScalePlayerCurrentElement() async {
        // Given: Mock scale player with current element
        let expectedElement = ScaleElement.scaleNote(try! MIDINote(64))
        mockScalePlayer.currentScaleElement = expectedElement

        // When: Get current scale element from coordinator
        let actualElement = sut.currentScaleElement

        // Then: Should return player's current element
        XCTAssertEqual(actualElement, expectedElement, "Should return scalePlayer's currentScaleElement")
    }

    func testCurrentScaleElement_whenNil_shouldReturnNil() async {
        // Given: Mock scale player with nil current element
        mockScalePlayer.currentScaleElement = nil

        // When: Get current scale element from coordinator
        let actualElement = sut.currentScaleElement

        // Then: Should return nil
        XCTAssertNil(actualElement, "Should return nil when scalePlayer has no current element")
    }
}
