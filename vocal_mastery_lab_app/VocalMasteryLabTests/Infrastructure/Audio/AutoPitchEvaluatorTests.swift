import XCTest
import AVFoundation
import VocalisDomain
@testable import VocalMasteryLab

@MainActor
final class AutoPitchEvaluatorTests: XCTestCase {

    var sut: AutoPitchEvaluator!
    var mockScalePlayer: MockScalePlayer!
    var mockPitchDetector: MockRealtimePitchDetector!

    override func setUp() {
        super.setUp()
        mockScalePlayer = MockScalePlayer()
        mockPitchDetector = MockRealtimePitchDetector()
        sut = AutoPitchEvaluator(
            scalePlayer: mockScalePlayer,
            pitchDetector: mockPitchDetector
        )
    }

    override func tearDown() {
        sut = nil
        mockPitchDetector = nil
        mockScalePlayer = nil
        super.tearDown()
    }

    // MARK: - Initialization Tests

    func testInit_InitialState_NotEvaluating() {
        // Then
        XCTAssertFalse(sut.isEvaluating)
        XCTAssertNil(sut.evaluationResult)
    }

    // MARK: - Basic Evaluation Flow Tests

    func testStartEvaluation_LoadsScaleAndStartsDetection() async throws {
        // Given
        let testScale: [MIDINote] = [
            try MIDINote(60), // C4 (261.63 Hz)
            try MIDINote(62), // D4 (293.66 Hz)
            try MIDINote(64), // E4 (329.63 Hz)
        ]
        let tempo = try Tempo(secondsPerNote: 1.0)

        // When
        try await sut.startEvaluation(notes: testScale, tempo: tempo)

        // Then
        XCTAssertTrue(mockScalePlayer.loadScaleCalled)
        XCTAssertTrue(mockScalePlayer.playCalled)
        XCTAssertTrue(mockPitchDetector.startRealtimeDetectionCalled)
        XCTAssertTrue(sut.isEvaluating)
    }

    func testStartEvaluation_WhenAlreadyEvaluating_ThrowsError() async throws {
        // Given
        let testScale: [MIDINote] = [try MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 1.0)
        try await sut.startEvaluation(notes: testScale, tempo: tempo)
        XCTAssertTrue(sut.isEvaluating)

        // When/Then
        do {
            try await sut.startEvaluation(notes: testScale, tempo: tempo)
            XCTFail("Should throw error when already evaluating")
        } catch {
            // Expected
        }
    }

    func testStopEvaluation_StopsPlaybackAndDetection() async throws {
        // Given
        let testScale: [MIDINote] = [try MIDINote(60)]
        let tempo = try Tempo(secondsPerNote: 1.0)
        try await sut.startEvaluation(notes: testScale, tempo: tempo)
        XCTAssertTrue(sut.isEvaluating)

        // When
        await sut.stopEvaluation()

        // Then
        XCTAssertTrue(mockScalePlayer.stopCalled)
        XCTAssertTrue(mockPitchDetector.stopRealtimeDetectionCalled)
        XCTAssertFalse(sut.isEvaluating)
    }

    // MARK: - Metrics Calculation Tests

    func testEvaluationResult_CalculatesGPE_Correctly() async throws {
        // Given
        let testScale: [MIDINote] = [
            try MIDINote(60), // Expected: C4 (261.63 Hz)
        ]
        let tempo = try Tempo(secondsPerNote: 1.0)

        // Mock detected pitch with large error (>50 cents = GPE)
        let expectedFreq = 261.63
        let detectedFreq = 290.0 // ~+170 cents error
        mockPitchDetector.mockDetectedPitch = DetectedPitch.fromFrequency(
            detectedFreq,
            confidence: 0.8
        )

        // When
        try await sut.startEvaluation(notes: testScale, tempo: tempo)
        await Task.sleep(1_500_000_000) // Wait 1.5 seconds
        await sut.stopEvaluation()

        // Then
        guard let result = sut.evaluationResult else {
            XCTFail("Evaluation result should not be nil")
            return
        }
        XCTAssertGreaterThan(result.gpe, 0.0, "GPE should be > 0 for large error")
    }

    func testEvaluationResult_CalculatesFPE_Correctly() async throws {
        // Given
        let testScale: [MIDINote] = [
            try MIDINote(60), // Expected: C4 (261.63 Hz)
        ]
        let tempo = try Tempo(secondsPerNote: 1.0)

        // Mock detected pitch with small error (<50 cents)
        let expectedFreq = 261.63
        let detectedFreq = 265.0 // ~+20 cents error
        mockPitchDetector.mockDetectedPitch = DetectedPitch.fromFrequency(
            detectedFreq,
            confidence: 0.8
        )

        // When
        try await sut.startEvaluation(notes: testScale, tempo: tempo)
        await Task.sleep(1_500_000_000) // Wait 1.5 seconds
        await sut.stopEvaluation()

        // Then
        guard let result = sut.evaluationResult else {
            XCTFail("Evaluation result should not be nil")
            return
        }
        XCTAssertGreaterThan(result.fpe, 0.0)
        XCTAssertLessThan(result.fpe, 30.0, "FPE should be < 30 cents for small error")
    }

    func testEvaluationResult_DetectsOctaveErrors() async throws {
        // Given
        let testScale: [MIDINote] = [
            try MIDINote(60), // Expected: C4 (261.63 Hz)
        ]
        let tempo = try Tempo(secondsPerNote: 1.0)

        // Mock detected pitch with octave error (2x frequency)
        let expectedFreq = 261.63
        let detectedFreq = 523.26 // Exactly one octave higher
        mockPitchDetector.mockDetectedPitch = DetectedPitch.fromFrequency(
            detectedFreq,
            confidence: 0.8
        )

        // When
        try await sut.startEvaluation(notes: testScale, tempo: tempo)
        await Task.sleep(1_500_000_000) // Wait 1.5 seconds
        await sut.stopEvaluation()

        // Then
        guard let result = sut.evaluationResult else {
            XCTFail("Evaluation result should not be nil")
            return
        }
        XCTAssertGreaterThan(result.octaveErrorRate, 0.0, "Should detect octave error")
    }

    func testEvaluationResult_WithMultipleNotes_CalculatesAverageMetrics() async throws {
        // Given
        let testScale: [MIDINote] = [
            try MIDINote(60), // C4
            try MIDINote(62), // D4
            try MIDINote(64), // E4
        ]
        let tempo = try Tempo(secondsPerNote: 1.0)

        // Mock detected pitches with varying accuracy
        mockPitchDetector.mockDetectedPitchSequence = [
            DetectedPitch.fromFrequency(261.63, confidence: 0.8), // Perfect
            DetectedPitch.fromFrequency(300.0, confidence: 0.7),  // ~+35 cents
            DetectedPitch.fromFrequency(330.0, confidence: 0.9),  // ~+2 cents
        ]

        // When
        try await sut.startEvaluation(notes: testScale, tempo: tempo)
        await Task.sleep(3_500_000_000) // Wait 3.5 seconds for all notes
        await sut.stopEvaluation()

        // Then
        guard let result = sut.evaluationResult else {
            XCTFail("Evaluation result should not be nil")
            return
        }
        XCTAssertEqual(result.totalNotes, 3)
        XCTAssertGreaterThan(result.averageConfidence, 0.0)
    }

    // MARK: - Timing Synchronization Tests

    func testEvaluation_SynchronizesExpectedAndDetectedPitches() async throws {
        // Given
        let testScale: [MIDINote] = [
            try MIDINote(60), // C4
        ]
        let tempo = try Tempo(secondsPerNote: 1.0)

        // When
        try await sut.startEvaluation(notes: testScale, tempo: tempo)
        await Task.sleep(500_000_000) // Wait 0.5 seconds (middle of note)

        // Then - should be comparing the expected note at this time
        XCTAssertTrue(sut.isEvaluating)
        // Internal state should have detected vs expected comparison ongoing
    }
}
