import XCTest
import VocalisDomain
@testable import VocalMasteringLab

final class AnalyzeRecordingUseCaseTests: XCTestCase {
    var sut: AnalyzeRecordingUseCase!
    fileprivate var mockAnalyzer: MockAudioFileAnalyzer!
    fileprivate var mockAnalyzerFactory: MockAudioFileAnalyzerFactory!
    fileprivate var mockCache: MockAnalysisCache!
    fileprivate var mockPitchDataCache: MockPitchDataCache!
    fileprivate var mockAudioSettingsRepository: MockAudioSettingsRepository!
    fileprivate var mockRecordingRepository: MockRecordingRepository!
    var mockLogger: MockLogger!
    var testRecording: Recording!

    @MainActor
    override func setUp() {
        super.setUp()
        mockAnalyzer = MockAudioFileAnalyzer()
        mockAnalyzerFactory = MockAudioFileAnalyzerFactory(analyzer: mockAnalyzer)
        mockCache = MockAnalysisCache()
        mockPitchDataCache = MockPitchDataCache()
        mockAudioSettingsRepository = MockAudioSettingsRepository()
        mockRecordingRepository = MockRecordingRepository()
        mockLogger = MockLogger()
        sut = AnalyzeRecordingUseCase(
            analyzerFactory: mockAnalyzerFactory,
            analysisCache: mockCache,
            pitchDataCache: mockPitchDataCache,
            audioSettingsRepository: mockAudioSettingsRepository,
            recordingRepository: mockRecordingRepository,
            logger: mockLogger
        )
        testRecording = createTestRecording()
    }

    override func tearDown() {
        sut = nil
        mockAnalyzer = nil
        mockAnalyzerFactory = nil
        mockCache = nil
        mockPitchDataCache = nil
        mockAudioSettingsRepository = nil
        mockRecordingRepository = nil
        mockLogger = nil
        testRecording = nil
        super.tearDown()
    }

    // MARK: - Test Helpers

    private func createTestRecording() -> Recording {
        let scaleSettings = ScaleSettings(
            startNote: try! MIDINote(60),
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 0.5)
        )
        return Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0),
            scaleSettings: scaleSettings
        )
    }

    private func createTestAnalysisResult() -> AnalysisResult {
        let pitchData = PitchAnalysisData(
            timeStamps: [0.0, 0.05],
            frequencies: [261.6, 262.3],
            confidences: [0.85, 0.92],
            targetNotes: [nil, nil]
        )

        let spectrogramData = SpectrogramData(
            timeStamps: [0.0, 0.1],
            frequencyBins: [80, 180],
            magnitudes: [[0.1, 0.3], [0.2, 0.4]]
        )

        return AnalysisResult(
            pitchData: pitchData,
            spectrogramData: spectrogramData,
            scaleSettings: nil
        )
    }

    // MARK: - Cache Hit Tests

    @MainActor
    func testExecute_WithCachedResult_ReturnsCachedResult() async throws {
        // Given: Cached result exists
        let cachedResult = createTestAnalysisResult()
        mockCache.cachedResults[testRecording.id] = cachedResult

        // When: Executing analysis
        let result = try await sut.execute(recording: testRecording)

        // Then: Should return cached result without analyzing
        XCTAssertEqual(result, cachedResult)
        XCTAssertEqual(mockCache.getCallCount, 1)
        XCTAssertEqual(mockAnalyzer.analyzeCallCount, 0)
    }

    // MARK: - Cache Miss Tests

    @MainActor
    func testExecute_WithoutCachedResult_PerformsAnalysis() async throws {
        // Given: No cached result
        let expectedResult = createTestAnalysisResult()
        mockAnalyzer.resultToReturn = (
            pitchData: expectedResult.pitchData,
            spectrogramData: expectedResult.spectrogramData
        )

        // When: Executing analysis
        let result = try await sut.execute(recording: testRecording)

        // Then: Should analyze and cache result
        XCTAssertEqual(mockAnalyzer.analyzeCallCount, 1)
        XCTAssertEqual(mockAnalyzer.lastAnalyzedURL, testRecording.fileURL)
        XCTAssertEqual(mockCache.setCallCount, 1)
        XCTAssertEqual(result.pitchData, expectedResult.pitchData)
        XCTAssertEqual(result.spectrogramData, expectedResult.spectrogramData)
    }

    @MainActor
    func testExecute_CachesAnalysisResult() async throws {
        // Given: No cached result
        let expectedResult = createTestAnalysisResult()
        mockAnalyzer.resultToReturn = (
            pitchData: expectedResult.pitchData,
            spectrogramData: expectedResult.spectrogramData
        )

        // When: Executing analysis
        _ = try await sut.execute(recording: testRecording)

        // Then: Result should be cached
        XCTAssertEqual(mockCache.setCallCount, 1)
        XCTAssertNotNil(mockCache.cachedResults[testRecording.id])
    }

    // MARK: - Error Handling Tests

    @MainActor
    func testExecute_AnalyzerThrowsError_PropagatesError() async {
        // Given: Analyzer will throw error
        mockAnalyzer.shouldThrowError = true

        // When: Executing analysis
        do {
            _ = try await sut.execute(recording: testRecording)
            XCTFail("Should have thrown error")
        } catch {
            // Then: Error should be propagated
            XCTAssertTrue(error is MockError)
        }
    }

    @MainActor
    func testExecute_AnalyzerThrowsError_DoesNotCache() async {
        // Given: Analyzer will throw error
        mockAnalyzer.shouldThrowError = true

        // When: Executing analysis (catching error)
        do {
            _ = try await sut.execute(recording: testRecording)
        } catch {
            // Expected error
        }

        // Then: Should not cache result
        XCTAssertEqual(mockCache.setCallCount, 0)
    }

    // MARK: - Octave Correction Tests

    @MainActor
    func testExecute_WithPlaybackTimeline_AppliesOctaveCorrection() async throws {
        // Given: Recording with playback timeline (A4 target)
        let targetNote = try! MIDINote(69)  // A4 = 440Hz
        let timeline = ScalePlaybackTimeline(
            events: [
                ScalePlaybackEvent(timestamp: 0.0, note: targetNote, eventType: .noteStart),
                ScalePlaybackEvent(timestamp: 1.0, note: targetNote, eventType: .noteEnd)
            ],
            recordingStartTime: Date()
        )

        let recordingWithTimeline = Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 1.0),
            scaleSettings: ScaleSettings(
                startNote: targetNote,
                endNote: targetNote,
                notePattern: .fiveToneScale,
                tempo: try! Tempo(secondsPerNote: 1.0)
            ),
            playbackTimeline: timeline
        )

        // Analyzer returns pitch one octave below target (220Hz instead of 440Hz)
        let pitchDataWithOctaveError = PitchAnalysisData(
            timeStamps: [0.5],
            frequencies: [220.0],  // A3 - one octave below A4
            confidences: [0.9],
            targetNotes: [nil],
            amplitudes: [0.8]
        )

        let spectrogramData = SpectrogramData(
            timeStamps: [0.0],
            frequencyBins: [100],
            magnitudes: [[0.5]]
        )

        mockAnalyzer.resultToReturn = (pitchData: pitchDataWithOctaveError, spectrogramData: spectrogramData)

        // When: Executing analysis
        let result = try await sut.execute(recording: recordingWithTimeline)

        // Then: Frequency should be corrected to ~440Hz (not 220Hz)
        XCTAssertEqual(result.pitchData.frequencies[0], 440.0, accuracy: 22.0,
            "220Hz should be corrected to ~440Hz, got \(result.pitchData.frequencies[0])")
    }

    @MainActor
    func testExecute_WithoutPlaybackTimeline_NoOctaveCorrection() async throws {
        // Given: Recording without playback timeline
        let recordingWithoutTimeline = Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 1.0),
            scaleSettings: nil,
            playbackTimeline: nil
        )

        let pitchData = PitchAnalysisData(
            timeStamps: [0.5],
            frequencies: [220.0],  // Should remain unchanged
            confidences: [0.9],
            targetNotes: [nil],
            amplitudes: [0.8]
        )

        let spectrogramData = SpectrogramData(
            timeStamps: [0.0],
            frequencyBins: [100],
            magnitudes: [[0.5]]
        )

        mockAnalyzer.resultToReturn = (pitchData: pitchData, spectrogramData: spectrogramData)

        // When: Executing analysis
        let result = try await sut.execute(recording: recordingWithoutTimeline)

        // Then: Frequency should remain unchanged (no correction without timeline)
        XCTAssertEqual(result.pitchData.frequencies[0], 220.0, accuracy: 0.1,
            "Without timeline, frequency should remain unchanged")
    }

    // MARK: - Scale Settings Tests

    @MainActor
    func testExecute_PreservesScaleSettings() async throws {
        // Given: Recording with scale settings
        let scaleSettings = ScaleSettings(
            startNote: try! MIDINote(60),
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: try! Tempo(secondsPerNote: 0.5)
        )
        let recordingWithScale = Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0),
            scaleSettings: scaleSettings
        )

        let expectedResult = createTestAnalysisResult()
        mockAnalyzer.resultToReturn = (
            pitchData: expectedResult.pitchData,
            spectrogramData: expectedResult.spectrogramData
        )

        // When: Executing analysis
        let result = try await sut.execute(recording: recordingWithScale)

        // Then: Scale settings should be preserved
        XCTAssertEqual(result.scaleSettings, scaleSettings)
    }

    // MARK: - Algorithm Change Tests

    @MainActor
    func testExecute_WhenAlgorithmChanged_InvalidatesCache() async throws {
        // Given: Recording was analyzed with YIN, but settings now use pYIN
        let recordingId = RecordingId()
        let recordingWithOldAlgorithm = Recording(
            id: recordingId,
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0),
            scaleSettings: nil,
            analysisAlgorithm: .yin  // Previously analyzed with YIN
        )

        // Settings now use pYIN
        let newSettings = AudioDetectionSettings(pitchAlgorithm: .pyinDefault)
        mockAudioSettingsRepository.settingsToReturn = newSettings

        // Cache has old data
        mockCache.cachedResults[recordingId] = createTestAnalysisResult()
        let oldPitchData = PitchAnalysisData(
            timeStamps: [0.0],
            frequencies: [440.0],
            confidences: [0.9],
            targetNotes: [nil]
        )
        mockPitchDataCache.cachedData[recordingId] = oldPitchData

        // Set up new analysis result
        let newResult = createTestAnalysisResult()
        mockAnalyzer.resultToReturn = (
            pitchData: newResult.pitchData,
            spectrogramData: newResult.spectrogramData
        )
        mockRecordingRepository.savedRecordings = [recordingWithOldAlgorithm]

        // When: Executing analysis
        _ = try await sut.execute(recording: recordingWithOldAlgorithm)

        // Then: Cache should be cleared and full analysis performed
        XCTAssertEqual(mockCache.clearCallCount, 1, "In-memory cache should be cleared")
        XCTAssertEqual(mockPitchDataCache.deleteCallCount, 1, "File cache should be deleted")
        XCTAssertEqual(mockPitchDataCache.lastDeletedId, recordingId, "Correct recording cache should be deleted")
        XCTAssertEqual(mockAnalyzer.analyzeCallCount, 1, "Full analysis should be performed")
    }

    @MainActor
    func testExecute_WhenAlgorithmSame_UsesCachedData() async throws {
        // Given: Recording was analyzed with YIN, settings still use YIN
        let recordingId = RecordingId()
        let recordingWithSameAlgorithm = Recording(
            id: recordingId,
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0),
            scaleSettings: nil,
            analysisAlgorithm: .yin  // Previously analyzed with YIN
        )

        // Settings still use YIN
        let settings = AudioDetectionSettings(pitchAlgorithm: .yin)
        mockAudioSettingsRepository.settingsToReturn = settings

        // Cache has data
        let cachedResult = createTestAnalysisResult()
        mockCache.cachedResults[recordingId] = cachedResult

        // When: Executing analysis
        let result = try await sut.execute(recording: recordingWithSameAlgorithm)

        // Then: Should use cached result without clearing
        XCTAssertEqual(mockCache.clearCallCount, 0, "Cache should NOT be cleared")
        XCTAssertEqual(mockAnalyzer.analyzeCallCount, 0, "No analysis should be performed")
        XCTAssertEqual(result, cachedResult, "Cached result should be returned")
    }

    @MainActor
    func testExecute_UpdatesRecordingWithAlgorithm() async throws {
        // Given: Recording without analysisAlgorithm (never analyzed)
        let recordingId = RecordingId()
        let newRecording = Recording(
            id: recordingId,
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0),
            scaleSettings: nil,
            analysisAlgorithm: nil  // Never analyzed
        )

        let settings = AudioDetectionSettings(pitchAlgorithm: .pyinDefault)
        mockAudioSettingsRepository.settingsToReturn = settings

        let expectedResult = createTestAnalysisResult()
        mockAnalyzer.resultToReturn = (
            pitchData: expectedResult.pitchData,
            spectrogramData: expectedResult.spectrogramData
        )
        mockRecordingRepository.savedRecordings = [newRecording]

        // When: Executing analysis
        _ = try await sut.execute(recording: newRecording)

        // Then: Recording should be updated with algorithm
        XCTAssertTrue(mockRecordingRepository.updateCalled, "Repository update should be called")
        if let updatedRecording = mockRecordingRepository.savedRecordings.first {
            XCTAssertEqual(updatedRecording.analysisAlgorithm, .pyinDefault, "Algorithm should be saved to recording")
        }
    }

    // MARK: - hasCachedData Tests

    @MainActor
    func testHasCachedData_WhenAlgorithmMatches_ReturnsTrue() {
        // Given: Recording analyzed with YIN, settings use YIN
        let recordingId = RecordingId()
        let recording = Recording(
            id: recordingId,
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0),
            scaleSettings: nil,
            analysisAlgorithm: .yin
        )

        let settings = AudioDetectionSettings(pitchAlgorithm: .yin)
        mockAudioSettingsRepository.settingsToReturn = settings

        // Cache has data
        mockPitchDataCache.cachedData[recordingId] = PitchAnalysisData(
            timeStamps: [0.0],
            frequencies: [440.0],
            confidences: [0.9],
            targetNotes: [nil]
        )

        // When: Checking cached data
        let hasCached = sut.hasCachedData(for: recording)

        // Then: Should return true
        XCTAssertTrue(hasCached, "Should return true when algorithm matches and cache exists")
    }

    @MainActor
    func testHasCachedData_WhenAlgorithmDiffers_ReturnsFalse() {
        // Given: Recording analyzed with YIN, but settings now use pYIN
        let recordingId = RecordingId()
        let recording = Recording(
            id: recordingId,
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0),
            scaleSettings: nil,
            analysisAlgorithm: .yin  // Was analyzed with YIN
        )

        let settings = AudioDetectionSettings(pitchAlgorithm: .pyinDefault)  // Now using pYIN
        mockAudioSettingsRepository.settingsToReturn = settings

        // Cache has data (but for old algorithm)
        mockPitchDataCache.cachedData[recordingId] = PitchAnalysisData(
            timeStamps: [0.0],
            frequencies: [440.0],
            confidences: [0.9],
            targetNotes: [nil]
        )

        // When: Checking cached data
        let hasCached = sut.hasCachedData(for: recording)

        // Then: Should return false (algorithm mismatch)
        XCTAssertFalse(hasCached, "Should return false when algorithm differs even if cache exists")
    }

    @MainActor
    func testHasCachedData_WhenNeverAnalyzed_ReturnsBasedOnCache() {
        // Given: Recording never analyzed (analysisAlgorithm is nil)
        let recordingId = RecordingId()
        let recording = Recording(
            id: recordingId,
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0),
            scaleSettings: nil,
            analysisAlgorithm: nil  // Never analyzed
        )

        let settings = AudioDetectionSettings(pitchAlgorithm: .yin)
        mockAudioSettingsRepository.settingsToReturn = settings

        // No cache
        // When: Checking cached data
        let hasCachedWithoutData = sut.hasCachedData(for: recording)
        XCTAssertFalse(hasCachedWithoutData, "Should return false when no cache exists")

        // With cache
        mockPitchDataCache.cachedData[recordingId] = PitchAnalysisData(
            timeStamps: [0.0],
            frequencies: [440.0],
            confidences: [0.9],
            targetNotes: [nil]
        )
        let hasCachedWithData = sut.hasCachedData(for: recording)
        XCTAssertTrue(hasCachedWithData, "Should return true when cache exists for never-analyzed recording")
    }
}

// MARK: - Mock Objects

fileprivate enum MockError: Error {
    case testError
}

@MainActor
fileprivate class MockAudioFileAnalyzerFactory: AudioFileAnalyzerFactoryProtocol {
    private let analyzer: AudioFileAnalyzerProtocol

    init(analyzer: AudioFileAnalyzerProtocol) {
        self.analyzer = analyzer
    }

    func makeAnalyzer() -> AudioFileAnalyzerProtocol {
        return analyzer
    }
}

@MainActor
fileprivate class MockAudioFileAnalyzer: AudioFileAnalyzerProtocol {
    var analyzeCallCount = 0
    var analyzeSpectrogramOnlyCallCount = 0
    var lastAnalyzedURL: URL?
    var shouldThrowError = false
    var resultToReturn: (pitchData: PitchAnalysisData, spectrogramData: SpectrogramData)?
    var spectrogramResultToReturn: SpectrogramData?

    func analyze(fileURL: URL, progress: @escaping @MainActor (Double) async -> Void) async throws -> (pitchData: PitchAnalysisData, spectrogramData: SpectrogramData) {
        analyzeCallCount += 1
        lastAnalyzedURL = fileURL

        // Simulate progress updates
        await progress(0.0)
        await progress(0.5)
        await progress(1.0)

        if shouldThrowError {
            throw MockError.testError
        }

        guard let result = resultToReturn else {
            throw MockError.testError
        }

        return result
    }

    func analyzeSpectrogramOnly(fileURL: URL, progress: @escaping @MainActor (Double) async -> Void) async throws -> SpectrogramData {
        analyzeSpectrogramOnlyCallCount += 1
        lastAnalyzedURL = fileURL

        // Simulate progress updates
        await progress(0.0)
        await progress(0.5)
        await progress(1.0)

        if shouldThrowError {
            throw MockError.testError
        }

        guard let result = spectrogramResultToReturn else {
            throw MockError.testError
        }

        return result
    }
}

fileprivate class MockAnalysisCache: AnalysisCacheProtocol {
    var cachedResults: [RecordingId: AnalysisResult] = [:]
    var getCallCount = 0
    var setCallCount = 0
    var clearCallCount = 0

    func get(_ id: RecordingId) -> AnalysisResult? {
        getCallCount += 1
        return cachedResults[id]
    }

    func set(_ id: RecordingId, result: AnalysisResult) {
        setCallCount += 1
        cachedResults[id] = result
    }

    func clear() {
        clearCallCount += 1
        cachedResults.removeAll()
    }
}
