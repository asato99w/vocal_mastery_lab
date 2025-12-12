//
//  PitchGraphRendererTests.swift
//  VocalMasteryLabTests
//
//  Tests for PitchGraphRenderer note name conversion
//

import XCTest
@testable import VocalMasteryLab

final class PitchGraphRendererTests: XCTestCase {

    var sut: PitchGraphRenderer!

    override func setUp() {
        super.setUp()
        sut = PitchGraphRenderer()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    // MARK: - frequencyToNoteName Tests

    func testFrequencyToNoteName_A4_returns_A4() {
        // Given: A4 = 440Hz (standard tuning reference)
        let frequency = 440.0

        // When
        let noteName = sut.frequencyToNoteName(frequency)

        // Then
        XCTAssertEqual(noteName, "A4")
    }

    func testFrequencyToNoteName_C4_returns_C4() {
        // Given: C4 (Middle C) = 261.63Hz
        let frequency = 261.63

        // When
        let noteName = sut.frequencyToNoteName(frequency)

        // Then
        XCTAssertEqual(noteName, "C4")
    }

    func testFrequencyToNoteName_C3_returns_C3() {
        // Given: C3 = 130.81Hz
        let frequency = 130.81

        // When
        let noteName = sut.frequencyToNoteName(frequency)

        // Then
        XCTAssertEqual(noteName, "C3")
    }

    func testFrequencyToNoteName_G4_returns_G4() {
        // Given: G4 = 392.00Hz
        let frequency = 392.0

        // When
        let noteName = sut.frequencyToNoteName(frequency)

        // Then
        XCTAssertEqual(noteName, "G4")
    }

    func testFrequencyToNoteName_zeroFrequency_returnsEmptyString() {
        // Given: Invalid frequency
        let frequency = 0.0

        // When
        let noteName = sut.frequencyToNoteName(frequency)

        // Then
        XCTAssertEqual(noteName, "")
    }

    func testFrequencyToNoteName_negativeFrequency_returnsEmptyString() {
        // Given: Negative frequency
        let frequency = -100.0

        // When
        let noteName = sut.frequencyToNoteName(frequency)

        // Then
        XCTAssertEqual(noteName, "")
    }

    func testFrequencyToNoteName_veryHighFrequency_returnsValidNote() {
        // Given: C8 = 4186Hz (upper limit of piano)
        let frequency = 4186.0

        // When
        let noteName = sut.frequencyToNoteName(frequency)

        // Then
        XCTAssertEqual(noteName, "C8")
    }

    func testFrequencyToNoteName_sharpNote_FSharp4_returns_FSharp4() {
        // Given: F#4 = 369.99Hz
        let frequency = 369.99

        // When
        let noteName = sut.frequencyToNoteName(frequency)

        // Then
        XCTAssertEqual(noteName, "F#4")
    }
}
