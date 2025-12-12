import XCTest
import AVFoundation
@testable import VocalMasteryLab
@testable import VocalisDomain

/// Tests for SynthesizerTimestampStrategy
/// Optimized for synthesized audio (e.g., AVAudioPlayerNode with PCM buffers)
/// Only applies outputLatency compensation (no sampler-specific offset needed)
final class SynthesizerTimestampStrategyTests: XCTestCase {

    var sut: SynthesizerTimestampStrategy!

    override func setUp() {
        super.setUp()
        sut = SynthesizerTimestampStrategy()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    // MARK: - Protocol Conformance

    func testConformsToScaleTimestampStrategy() {
        XCTAssertTrue(sut is ScaleTimestampStrategy)
    }

    // MARK: - Initial State

    func testInitialState_isNotRecording() {
        XCTAssertFalse(sut.isRecording)
    }

    func testInitialState_recordingStartTimeIsNil() {
        XCTAssertNil(sut.recordingStartTime)
    }

    // MARK: - Start Recording

    func testStartRecording_setsIsRecordingTrue() {
        sut.startRecording(recordingStartTime: Date())
        XCTAssertTrue(sut.isRecording)
    }

    func testStartRecording_storesRecordingStartTime() {
        let startTime = Date()
        sut.startRecording(recordingStartTime: startTime)
        XCTAssertEqual(sut.recordingStartTime, startTime)
    }

    // MARK: - Stop Recording

    func testStopRecording_setsIsRecordingFalse() {
        sut.startRecording(recordingStartTime: Date())
        sut.stopRecording()
        XCTAssertFalse(sut.isRecording)
    }

    func testStopRecording_clearsRecordingStartTime() {
        sut.startRecording(recordingStartTime: Date())
        sut.stopRecording()
        XCTAssertNil(sut.recordingStartTime)
    }

    // MARK: - Update Recording Start Time

    func testUpdateRecordingStartTime_updatesWhileRecording() {
        let initialTime = Date()
        sut.startRecording(recordingStartTime: initialTime)

        let newTime = Date().addingTimeInterval(1.0)
        sut.updateRecordingStartTime(newTime)

        XCTAssertEqual(sut.recordingStartTime, newTime)
    }

    func testUpdateRecordingStartTime_doesNothingWhenNotRecording() {
        let newTime = Date()
        sut.updateRecordingStartTime(newTime)
        XCTAssertNil(sut.recordingStartTime)
    }

    // MARK: - Get Note Start Timestamp

    func testGetNoteStartTimestamp_returnsNilWhenNotRecording() {
        XCTAssertNil(sut.getNoteStartTimestamp())
    }

    func testGetNoteStartTimestamp_returnsTimestampWhenRecording() {
        sut.startRecording(recordingStartTime: Date())
        let timestamp = sut.getNoteStartTimestamp()
        XCTAssertNotNil(timestamp)
        XCTAssertGreaterThanOrEqual(timestamp!, 0.0)
    }

    func testGetNoteStartTimestamp_includesOutputLatencyCompensation() {
        // Given: Start recording
        let startTime = Date()
        sut.startRecording(recordingStartTime: startTime)

        // When: Get timestamp after a short delay
        Thread.sleep(forTimeInterval: 0.05) // 50ms delay
        let timestamp = sut.getNoteStartTimestamp()

        // Then: Timestamp should include outputLatency
        // The raw elapsed time is ~50ms, plus outputLatency (typically 5-20ms)
        XCTAssertNotNil(timestamp)

        // Expected: timestamp >= elapsed time (50ms)
        // Because outputLatency is added, timestamp should be > raw elapsed time
        let rawElapsed = Date().timeIntervalSince(startTime)
        let outputLatency = AVAudioSession.sharedInstance().outputLatency

        // Allow some tolerance for timing variations
        // Timestamp should be approximately rawElapsed + outputLatency
        let expectedMin = rawElapsed + outputLatency - 0.01 // 10ms tolerance
        XCTAssertGreaterThanOrEqual(timestamp!, expectedMin)
    }

    // MARK: - Record Note End

    func testRecordNoteEnd_addsEventToRecordedEvents() throws {
        let note = try MIDINote(60)
        sut.startRecording(recordingStartTime: Date())

        sut.recordNoteEnd(note)

        let events = sut.getRecordedEvents()
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].note, note)
        XCTAssertEqual(events[0].eventType, .noteEnd)
    }

    func testRecordNoteEnd_doesNothingWhenNotRecording() throws {
        let note = try MIDINote(60)
        sut.recordNoteEnd(note)

        let events = sut.getRecordedEvents()
        XCTAssertTrue(events.isEmpty)
    }

    // MARK: - Get Recorded Events

    func testGetRecordedEvents_returnsEmptyArrayInitially() {
        sut.startRecording(recordingStartTime: Date())
        let events = sut.getRecordedEvents()
        XCTAssertTrue(events.isEmpty)
    }

    func testGetRecordedEvents_returnsAllRecordedEvents() throws {
        let note1 = try MIDINote(60)
        let note2 = try MIDINote(62)

        sut.startRecording(recordingStartTime: Date())
        sut.recordNoteEnd(note1)
        sut.recordNoteEnd(note2)

        let events = sut.getRecordedEvents()
        XCTAssertEqual(events.count, 2)
    }

    // MARK: - Append Note Start Event

    func testAppendNoteStartEvent_addsNoteStartEvent() throws {
        let note = try MIDINote(60)
        sut.startRecording(recordingStartTime: Date())

        sut.appendNoteStartEvent(note, timestamp: 0.5)

        let events = sut.getRecordedEvents()
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].note, note)
        XCTAssertEqual(events[0].eventType, .noteStart)
        XCTAssertEqual(events[0].timestamp, 0.5)
    }

    // MARK: - Comparison with RawTimestampStrategy

    func testOutputLatencyCompensation_isGreaterThanRaw() {
        // Given: Both strategies start at the same time
        let startTime = Date()
        let rawStrategy = RawTimestampStrategy()

        sut.startRecording(recordingStartTime: startTime)
        rawStrategy.startRecording(recordingStartTime: startTime)

        // When: Get timestamps from both
        Thread.sleep(forTimeInterval: 0.01) // Small delay
        let synthesizerTimestamp = sut.getNoteStartTimestamp()!
        let rawTimestamp = rawStrategy.getNoteStartTimestamp()!

        // Then: Synthesizer timestamp should be greater due to outputLatency
        let outputLatency = AVAudioSession.sharedInstance().outputLatency

        // Synthesizer strategy adds outputLatency, so it should be larger
        // (unless outputLatency is 0, which is rare on real devices)
        if outputLatency > 0 {
            XCTAssertGreaterThan(synthesizerTimestamp, rawTimestamp)
        }
    }

    // MARK: - No Sampler Latency Offset

    func testNoSamplerLatencyOffset_onlyOutputLatency() {
        // This test verifies that SynthesizerTimestampStrategy does NOT include
        // the 80ms sampler latency offset that SamplerTimestampStrategy uses.
        // Synthesized audio has much lower internal latency than SF2 sampler.

        let startTime = Date()
        sut.startRecording(recordingStartTime: startTime)

        Thread.sleep(forTimeInterval: 0.01) // 10ms delay
        let timestamp = sut.getNoteStartTimestamp()!

        let rawElapsed = Date().timeIntervalSince(startTime)
        let outputLatency = AVAudioSession.sharedInstance().outputLatency

        // Timestamp should be approximately rawElapsed + outputLatency
        // NOT rawElapsed + outputLatency + 0.080 (sampler offset)
        let expectedWithSamplerOffset = rawElapsed + outputLatency + 0.080

        // Allow 20ms tolerance for timing variations
        // If sampler offset was included, timestamp would be ~80ms higher
        XCTAssertLessThan(timestamp, expectedWithSamplerOffset - 0.060)
    }

    // MARK: - OutputLatency Investigation

    /// Measure the cost and variability of outputLatency retrieval
    /// This test helps diagnose performance issues on real devices
    func testOutputLatency_measureRetrievalCostAndVariability() {
        let iterations = 100

        var latencyValues: [TimeInterval] = []
        var retrievalTimes: [TimeInterval] = []

        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let latency = AVAudioSession.sharedInstance().outputLatency
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            latencyValues.append(latency)
            retrievalTimes.append(elapsed)
        }

        // Calculate statistics for latency values
        let minLatency = latencyValues.min()! * 1000
        let maxLatency = latencyValues.max()! * 1000
        let avgLatency = (latencyValues.reduce(0, +) / Double(iterations)) * 1000
        let latencyRange = maxLatency - minLatency

        // Calculate statistics for retrieval times
        let minRetrieval = retrievalTimes.min()! * 1_000_000 // microseconds
        let maxRetrieval = retrievalTimes.max()! * 1_000_000
        let avgRetrieval = (retrievalTimes.reduce(0, +) / Double(iterations)) * 1_000_000
        let totalRetrieval = retrievalTimes.reduce(0, +) * 1000 // ms

        let report = """

        ========== OUTPUT LATENCY INVESTIGATION ==========
        Iterations: \(iterations)

        LATENCY VALUES:
        - Min: \(String(format: "%.2f", minLatency)) ms
        - Max: \(String(format: "%.2f", maxLatency)) ms
        - Avg: \(String(format: "%.2f", avgLatency)) ms
        - Range (variability): \(String(format: "%.2f", latencyRange)) ms

        RETRIEVAL COST:
        - Min: \(String(format: "%.1f", minRetrieval)) µs
        - Max: \(String(format: "%.1f", maxRetrieval)) µs
        - Avg: \(String(format: "%.1f", avgRetrieval)) µs
        - Total for \(iterations) calls: \(String(format: "%.2f", totalRetrieval)) ms
        ==================================================

        """

        // Write to file for reliable retrieval
        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent("outputLatency_investigation.txt")
        try? report.write(to: fileURL, atomically: true, encoding: .utf8)

        // Add as XCTest attachment for xcresult access
        let attachment = XCTAttachment(string: report)
        attachment.name = "OutputLatency_Investigation_Report"
        attachment.lifetime = .keepAlways
        add(attachment)

        // Also use XCTContext for visibility
        XCTContext.runActivity(named: "OutputLatency Investigation Results") { _ in
            // Force output through assertion message
            XCTAssertTrue(true, report)
        }

        // Assertions to verify test ran
        XCTAssertEqual(latencyValues.count, iterations)
        XCTAssertEqual(retrievalTimes.count, iterations)

        // Note: On simulator, outputLatency is often 0
        // On real device, expect 5-20ms typical values
    }
}
