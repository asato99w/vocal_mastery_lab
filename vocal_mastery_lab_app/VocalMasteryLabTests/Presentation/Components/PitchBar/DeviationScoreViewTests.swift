import XCTest
import SwiftUI
@testable import VocalMasteryLab
@testable import VocalisDomain

final class DeviationScoreViewTests: XCTestCase {

    // MARK: - DeviationScore Calculation Tests

    func testDeviationScoreCalculator_calculateOverallAccuracy_perfect() {
        // Given: All points have perfect deviation (0 cents)
        let points = [
            PitchDeviationPoint(timestamp: 0.0, frequency: 440.0, confidence: 0.9, targetFrequency: 440.0),
            PitchDeviationPoint(timestamp: 0.1, frequency: 440.0, confidence: 0.9, targetFrequency: 440.0),
            PitchDeviationPoint(timestamp: 0.2, frequency: 440.0, confidence: 0.9, targetFrequency: 440.0)
        ]

        // When
        let accuracy = DeviationScoreCalculator.calculateOverallAccuracy(points: points)

        // Then: Should be 100%
        XCTAssertEqual(accuracy, 100.0, accuracy: 0.1)
    }

    func testDeviationScoreCalculator_calculateOverallAccuracy_mixed() {
        // Given: Mix of accurate and inaccurate points
        let targetFreq = 440.0
        // Perfect (within ±10 cents)
        let perfectFreq = targetFreq  // 0 cents
        // Poor (100 cents off = one semitone)
        let poorFreq = targetFreq * pow(2, 1.0 / 12.0)  // +100 cents

        let points = [
            PitchDeviationPoint(timestamp: 0.0, frequency: perfectFreq, confidence: 0.9, targetFrequency: targetFreq),
            PitchDeviationPoint(timestamp: 0.1, frequency: perfectFreq, confidence: 0.9, targetFrequency: targetFreq),
            PitchDeviationPoint(timestamp: 0.2, frequency: poorFreq, confidence: 0.9, targetFrequency: targetFreq),
            PitchDeviationPoint(timestamp: 0.3, frequency: poorFreq, confidence: 0.9, targetFrequency: targetFreq)
        ]

        // When
        let accuracy = DeviationScoreCalculator.calculateOverallAccuracy(points: points)

        // Then: 2/4 perfect = 50%
        XCTAssertEqual(accuracy, 50.0, accuracy: 0.1)
    }

    func testDeviationScoreCalculator_calculateOverallAccuracy_noTargetSkipped() {
        // Given: Some points have no target (should be skipped in calculation)
        let points = [
            PitchDeviationPoint(timestamp: 0.0, frequency: 440.0, confidence: 0.9, targetFrequency: 440.0),
            PitchDeviationPoint(timestamp: 0.1, frequency: 440.0, confidence: 0.9, targetFrequency: nil),  // Skipped
            PitchDeviationPoint(timestamp: 0.2, frequency: 440.0, confidence: 0.9, targetFrequency: 440.0)
        ]

        // When
        let accuracy = DeviationScoreCalculator.calculateOverallAccuracy(points: points)

        // Then: 2/2 perfect = 100% (nil target point skipped)
        XCTAssertEqual(accuracy, 100.0, accuracy: 0.1)
    }

    func testDeviationScoreCalculator_calculateOverallAccuracy_emptyPoints() {
        // Given: Empty points array
        let points: [PitchDeviationPoint] = []

        // When
        let accuracy = DeviationScoreCalculator.calculateOverallAccuracy(points: points)

        // Then: Should return 0 for empty data
        XCTAssertEqual(accuracy, 0.0)
    }

    // MARK: - Average Deviation Tests

    func testDeviationScoreCalculator_calculateAverageDeviation_perfect() {
        // Given: All points have 0 deviation
        let points = [
            PitchDeviationPoint(timestamp: 0.0, frequency: 440.0, confidence: 0.9, targetFrequency: 440.0),
            PitchDeviationPoint(timestamp: 0.1, frequency: 440.0, confidence: 0.9, targetFrequency: 440.0)
        ]

        // When
        let avgDeviation = DeviationScoreCalculator.calculateAverageDeviation(points: points)

        // Then
        XCTAssertNotNil(avgDeviation)
        XCTAssertEqual(avgDeviation!, 0.0, accuracy: 0.1)
    }

    func testDeviationScoreCalculator_calculateAverageDeviation_sharp() {
        // Given: All points are sharp by 50 cents
        let targetFreq = 440.0
        // 50 cents sharp: freq = target * 2^(50/1200)
        let sharpFreq = targetFreq * pow(2, 50.0 / 1200.0)

        let points = [
            PitchDeviationPoint(timestamp: 0.0, frequency: sharpFreq, confidence: 0.9, targetFrequency: targetFreq),
            PitchDeviationPoint(timestamp: 0.1, frequency: sharpFreq, confidence: 0.9, targetFrequency: targetFreq)
        ]

        // When
        let avgDeviation = DeviationScoreCalculator.calculateAverageDeviation(points: points)

        // Then: Should be +50 cents
        XCTAssertNotNil(avgDeviation)
        XCTAssertEqual(avgDeviation!, 50.0, accuracy: 0.5)
    }

    func testDeviationScoreCalculator_calculateAverageDeviation_flat() {
        // Given: All points are flat by 30 cents
        let targetFreq = 440.0
        // 30 cents flat: freq = target * 2^(-30/1200)
        let flatFreq = targetFreq * pow(2, -30.0 / 1200.0)

        let points = [
            PitchDeviationPoint(timestamp: 0.0, frequency: flatFreq, confidence: 0.9, targetFrequency: targetFreq),
            PitchDeviationPoint(timestamp: 0.1, frequency: flatFreq, confidence: 0.9, targetFrequency: targetFreq)
        ]

        // When
        let avgDeviation = DeviationScoreCalculator.calculateAverageDeviation(points: points)

        // Then: Should be -30 cents
        XCTAssertNotNil(avgDeviation)
        XCTAssertEqual(avgDeviation!, -30.0, accuracy: 0.5)
    }

    func testDeviationScoreCalculator_calculateAverageDeviation_noValidPoints() {
        // Given: All points have no target
        let points = [
            PitchDeviationPoint(timestamp: 0.0, frequency: 440.0, confidence: 0.9, targetFrequency: nil),
            PitchDeviationPoint(timestamp: 0.1, frequency: 440.0, confidence: 0.9, targetFrequency: nil)
        ]

        // When
        let avgDeviation = DeviationScoreCalculator.calculateAverageDeviation(points: points)

        // Then: Should be nil (no valid points)
        XCTAssertNil(avgDeviation)
    }

    // MARK: - Note Score Tests

    func testDeviationScoreCalculator_calculateNoteScores_singleNote() throws {
        // Given: Points for a single note (C4)
        let c4Freq = try MIDINote(60).frequency
        let points = [
            PitchDeviationPoint(timestamp: 0.0, frequency: c4Freq, confidence: 0.9, targetFrequency: c4Freq),
            PitchDeviationPoint(timestamp: 0.1, frequency: c4Freq, confidence: 0.9, targetFrequency: c4Freq),
            PitchDeviationPoint(timestamp: 0.2, frequency: c4Freq, confidence: 0.9, targetFrequency: c4Freq)
        ]

        let segment = NoteSegment(note: try MIDINote(60), startTime: 0.0, endTime: 1.0)
        let segments = [segment]

        // When
        let noteScores = DeviationScoreCalculator.calculateNoteScores(points: points, segments: segments)

        // Then: Should have one note score with 100% accuracy
        XCTAssertEqual(noteScores.count, 1)
        XCTAssertEqual(noteScores[0].note.value, 60)
        XCTAssertEqual(noteScores[0].accuracy, 100.0, accuracy: 0.1)
    }

    func testDeviationScoreCalculator_calculateNoteScores_multipleNotes() throws {
        // Given: Points for multiple notes
        let c4 = try MIDINote(60)
        let d4 = try MIDINote(62)

        // C4 perfect, D4 with 50 cents sharp
        let points = [
            PitchDeviationPoint(timestamp: 0.5, frequency: c4.frequency, confidence: 0.9, targetFrequency: c4.frequency),
            PitchDeviationPoint(timestamp: 1.5, frequency: d4.frequency * pow(2, 50.0/1200.0), confidence: 0.9, targetFrequency: d4.frequency)
        ]

        let segments = [
            NoteSegment(note: c4, startTime: 0.0, endTime: 1.0),
            NoteSegment(note: d4, startTime: 1.0, endTime: 2.0)
        ]

        // When
        let noteScores = DeviationScoreCalculator.calculateNoteScores(points: points, segments: segments)

        // Then: Should have two note scores
        XCTAssertEqual(noteScores.count, 2)

        // C4 should be 100% (perfect)
        let c4Score = noteScores.first { $0.note.value == 60 }
        XCTAssertNotNil(c4Score)
        XCTAssertEqual(c4Score!.accuracy, 100.0, accuracy: 0.1)

        // D4 should be lower (50 cents off is outside ±10 threshold but within ±50)
        let d4Score = noteScores.first { $0.note.value == 62 }
        XCTAssertNotNil(d4Score)
        // 50 cents is outside perfect (±10) so accuracy should be 0% using strict threshold
        XCTAssertEqual(d4Score!.accuracy, 0.0, accuracy: 0.1)
    }

    // MARK: - NoteScore Model Tests

    func testNoteScore_accuracyLevel_perfect() throws {
        // Given
        let noteScore = NoteScore(
            note: try MIDINote(60),
            accuracy: 95.0,
            averageDeviation: 5.0,
            pointCount: 10
        )

        // Then
        XCTAssertEqual(noteScore.accuracyLevel, .excellent)
    }

    func testNoteScore_accuracyLevel_good() throws {
        // Given
        let noteScore = NoteScore(
            note: try MIDINote(60),
            accuracy: 80.0,
            averageDeviation: 15.0,
            pointCount: 10
        )

        // Then
        XCTAssertEqual(noteScore.accuracyLevel, .good)
    }

    func testNoteScore_accuracyLevel_acceptable() throws {
        // Given
        let noteScore = NoteScore(
            note: try MIDINote(60),
            accuracy: 65.0,
            averageDeviation: 35.0,
            pointCount: 10
        )

        // Then
        XCTAssertEqual(noteScore.accuracyLevel, .acceptable)
    }

    func testNoteScore_accuracyLevel_needsImprovement() throws {
        // Given
        let noteScore = NoteScore(
            note: try MIDINote(60),
            accuracy: 40.0,
            averageDeviation: 80.0,
            pointCount: 10
        )

        // Then
        XCTAssertEqual(noteScore.accuracyLevel, .needsImprovement)
    }
}
