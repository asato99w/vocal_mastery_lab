import XCTest
@testable import VocalisDomain

final class NotePatternTests: XCTestCase {
    func testFiveToneScale_Intervals() {
        // Given
        let pattern = NotePattern.fiveToneScale
        
        // When
        let intervals = pattern.intervals
        
        // Then
        XCTAssertEqual(intervals, [0, 2, 4, 5, 7])
    }
    
    func testFiveToneScale_AscendingDescending() {
        // Given
        let pattern = NotePattern.fiveToneScale

        // When
        let ascDesc = pattern.ascendingDescending()

        // Then
        // Expected: [0, 2, 4, 5, 7, 5, 4, 2, 0] (C-D-E-F-G-F-E-D-C)
        XCTAssertEqual(ascDesc.count, 9)
        XCTAssertEqual(ascDesc, [0, 2, 4, 5, 7, 5, 4, 2, 0])
    }

    func testFiveToneScale_PlaybackPattern() {
        // Given
        let pattern = NotePattern.fiveToneScale

        // When
        let playback = pattern.playbackPattern

        // Then
        // playbackPattern should return the same as ascendingDescending for fiveToneScale
        XCTAssertEqual(playback, [0, 2, 4, 5, 7, 5, 4, 2, 0])
    }

    func testFiveToneScale_DisplayName() {
        // Given
        let pattern = NotePattern.fiveToneScale

        // When
        let displayName = pattern.displayName

        // Then: displayName is non-localized (English) for logging/debugging
        XCTAssertEqual(displayName, "Five-Tone Scale")
    }

    func testFiveToneScale_DisplayNameKey() {
        // Given
        let pattern = NotePattern.fiveToneScale

        // When
        let displayNameKey = pattern.displayNameKey

        // Then: displayNameKey should be the localization key
        XCTAssertEqual(displayNameKey, "scale.fiveTone")
    }

    // MARK: - OctaveRepeat Tests

    func testOctaveRepeat_Intervals() {
        // Given
        let pattern = NotePattern.octaveRepeat

        // When
        let intervals = pattern.intervals

        // Then
        // Major triad + octave: C, E, G, C
        XCTAssertEqual(intervals, [0, 4, 7, 12])
    }

    func testOctaveRepeat_PlaybackPattern() {
        // Given
        let pattern = NotePattern.octaveRepeat

        // When
        let playback = pattern.playbackPattern

        // Then
        // Ascending + top 4 times + descending
        XCTAssertEqual(playback, [0, 4, 7, 12, 12, 12, 12, 7, 4, 0])
    }

    func testOctaveRepeat_DisplayName() {
        // Given
        let pattern = NotePattern.octaveRepeat

        // When
        let displayName = pattern.displayName

        // Then: displayName is non-localized (English) for logging/debugging
        XCTAssertEqual(displayName, "Octave Repeat")
    }

    func testOctaveRepeat_DisplayNameKey() {
        // Given
        let pattern = NotePattern.octaveRepeat

        // When
        let displayNameKey = pattern.displayNameKey

        // Then: displayNameKey should be the localization key
        XCTAssertEqual(displayNameKey, "scale.octaveRepeat")
    }

    // MARK: - BrokenScale Tests

    func testBrokenScale_Intervals() {
        // Given
        let pattern = NotePattern.brokenScale

        // When
        let intervals = pattern.intervals

        // Then
        // Chord tones: Root, 3rd, 5th, Octave
        XCTAssertEqual(intervals, [0, 4, 7, 12])
    }

    func testBrokenScale_PlaybackPattern() {
        // Given
        let pattern = NotePattern.brokenScale

        // When
        let playback = pattern.playbackPattern

        // Then
        // Pattern: 1→5→3→8→5→3→1 (x2) = 13 notes
        // Semitones: 0→7→4→12→7→4→0→7→4→12→7→4→0
        XCTAssertEqual(playback.count, 13)
        XCTAssertEqual(playback, [0, 7, 4, 12, 7, 4, 0, 7, 4, 12, 7, 4, 0])
    }

    func testBrokenScale_DisplayName() {
        // Given
        let pattern = NotePattern.brokenScale

        // When
        let displayName = pattern.displayName

        // Then: displayName is non-localized (English) for logging/debugging
        XCTAssertEqual(displayName, "Broken Scale")
    }

    func testBrokenScale_DisplayNameKey() {
        // Given
        let pattern = NotePattern.brokenScale

        // When
        let displayNameKey = pattern.displayNameKey

        // Then: displayNameKey should be the localization key
        XCTAssertEqual(displayNameKey, "scale.brokenScale")
    }

    // MARK: - RossiniScale Tests

    func testRossiniScale_Intervals() {
        // Given
        let pattern = NotePattern.rossiniScale

        // When
        let intervals = pattern.intervals

        // Then
        // 1.5 octave scale tones: Root, 3rd, 5th, Octave, 10th, 12th
        XCTAssertEqual(intervals, [0, 4, 7, 12, 16, 19])
    }

    func testRossiniScale_PlaybackPattern() {
        // Given
        let pattern = NotePattern.rossiniScale

        // When
        let playback = pattern.playbackPattern

        // Then
        // Pattern: 1→3→5→8→10→12→11→9→7→5→4→2→1 = 13 notes
        // Semitones: 0→4→7→12→16→19→17→14→12→7→5→2→0
        XCTAssertEqual(playback.count, 13)
        XCTAssertEqual(playback, [0, 4, 7, 12, 16, 19, 17, 14, 12, 7, 5, 2, 0])
    }

    func testRossiniScale_DisplayName() {
        // Given
        let pattern = NotePattern.rossiniScale

        // When
        let displayName = pattern.displayName

        // Then: displayName is non-localized (English) for logging/debugging
        XCTAssertEqual(displayName, "Rossini Scale")
    }

    func testRossiniScale_DisplayNameKey() {
        // Given
        let pattern = NotePattern.rossiniScale

        // When
        let displayNameKey = pattern.displayNameKey

        // Then: displayNameKey should be the localization key
        XCTAssertEqual(displayNameKey, "scale.rossiniScale")
    }
}
