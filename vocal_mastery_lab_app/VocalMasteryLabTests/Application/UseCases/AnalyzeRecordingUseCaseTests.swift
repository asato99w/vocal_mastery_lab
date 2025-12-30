import XCTest
import VocalisDomain
@testable import VocalMasteryLab

// MARK: - Mock Implementations

/// Mock implementation of AudioFileAnalyzerProtocol
final class MockAudioFileAnalyzer: AudioFileAnalyzerProtocol {
    var analyzeCalled = false
    var analyzeSpectrogramOnlyCalled = false
    var analyzeCallCount = 0
    var analyzeSpectrogramOnlyCallCount = 0
    var lastAnalyzedURL: URL?
    var shouldThrowError = false
    var errorToThrow: Error = NSError(domain: "MockError", code: 1)

    var pitchDataToReturn = PitchAnalysisData(
        timeStamps: [0.0, 0.1, 0.2],
        frequencies: [440.0, 442.0, 438.0],
        confidences: [0.9, 0.85, 0.92],
        amplitudes: [0.5, 0.6, 0.55]
    )

    var spectrogramDataToReturn = SpectrogramData(
        timeStamps: [0.0, 0.1, 0.2],
        frequencyBins: [100.0, 200.0, 300.0],
        magnitudes: [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
    )

    func analyze(fileURL: URL, progress: @escaping @MainActor (Double) async -> Void) async throws -> (pitchData: PitchAnalysisData, spectrogramData: SpectrogramData) {
        analyzeCalled = true
        analyzeCallCount += 1
        lastAnalyzedURL = fileURL

        if shouldThrowError {
            throw errorToThrow
        }

        await progress(0.5)
        await progress(1.0)

        return (pitchDataToReturn, spectrogramDataToReturn)
    }

    func analyzeSpectrogramOnly(fileURL: URL, progress: @escaping @MainActor (Double) async -> Void) async throws -> SpectrogramData {
        analyzeSpectrogramOnlyCalled = true
        analyzeSpectrogramOnlyCallCount += 1
        lastAnalyzedURL = fileURL

        if shouldThrowError {
            throw errorToThrow
        }

        await progress(0.5)
        await progress(1.0)

        return spectrogramDataToReturn
    }

    func reset() {
        analyzeCalled = false
        analyzeSpectrogramOnlyCalled = false
        analyzeCallCount = 0
        analyzeSpectrogramOnlyCallCount = 0
        lastAnalyzedURL = nil
        shouldThrowError = false
    }
}

/// Mock implementation of AnalysisCacheProtocol
final class MockAnalysisCache: AnalysisCacheProtocol {
    var cachedResults: [RecordingId: AnalysisResult] = [:]
    var getCalled = false
    var setCalled = false
    var clearCalled = false
    var getCallCount = 0
    var setCallCount = 0
    var clearCallCount = 0

    func get(_ id: RecordingId) -> AnalysisResult? {
        getCalled = true
        getCallCount += 1
        return cachedResults[id]
    }

    func set(_ id: RecordingId, result: AnalysisResult) {
        setCalled = true
        setCallCount += 1
        cachedResults[id] = result
    }

    func clear() {
        clearCalled = true
        clearCallCount += 1
        cachedResults.removeAll()
    }

    func reset() {
        cachedResults = [:]
        getCalled = false
        setCalled = false
        clearCalled = false
        getCallCount = 0
        setCallCount = 0
        clearCallCount = 0
    }
}

/// Mock implementation of AudioFileAnalyzerFactoryProtocol
final class MockAudioFileAnalyzerFactory: AudioFileAnalyzerFactoryProtocol {
    let mockAnalyzer: MockAudioFileAnalyzer
    var makeAnalyzerCalled = false
    var makeAnalyzerCallCount = 0

    init(mockAnalyzer: MockAudioFileAnalyzer) {
        self.mockAnalyzer = mockAnalyzer
    }

    func makeAnalyzer() -> AudioFileAnalyzerProtocol {
        makeAnalyzerCalled = true
        makeAnalyzerCallCount += 1
        return mockAnalyzer
    }
}

// MARK: - Test Class

@MainActor
final class AnalyzeRecordingUseCaseTests: XCTestCase {

    var sut: AnalyzeRecordingUseCase!
    var mockAnalyzer: MockAudioFileAnalyzer!
    var mockAnalyzerFactory: MockAudioFileAnalyzerFactory!
    var mockAnalysisCache: MockAnalysisCache!
    var mockPitchDataCache: MockPitchDataCache!
    var mockAudioSettingsRepository: MockAudioSettingsRepository!
    var mockRecordingRepository: MockRecordingRepository!
    var mockLogger: MockLogger!

    override func setUp() async throws {
        try await super.setUp()

        mockAnalyzer = MockAudioFileAnalyzer()
        mockAnalyzerFactory = MockAudioFileAnalyzerFactory(mockAnalyzer: mockAnalyzer)
        mockAnalysisCache = MockAnalysisCache()
        mockPitchDataCache = MockPitchDataCache()
        mockAudioSettingsRepository = MockAudioSettingsRepository()
        mockRecordingRepository = MockRecordingRepository()
        mockLogger = MockLogger()

        sut = AnalyzeRecordingUseCase(
            analyzerFactory: mockAnalyzerFactory,
            analysisCache: mockAnalysisCache,
            pitchDataCache: mockPitchDataCache,
            audioSettingsRepository: mockAudioSettingsRepository,
            recordingRepository: mockRecordingRepository,
            logger: mockLogger
        )
    }

    override func tearDown() async throws {
        sut = nil
        mockAnalyzer = nil
        mockAnalyzerFactory = nil
        mockAnalysisCache = nil
        mockPitchDataCache = nil
        mockAudioSettingsRepository = nil
        mockRecordingRepository = nil
        mockLogger = nil
        try await super.tearDown()
    }

    // MARK: - Helper Methods

    private func createTestRecording(algorithm: PitchDetectionAlgorithm? = nil) -> Recording {
        var recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test_recording.m4a"),
            duration: Duration(seconds: 30.0)
        )
        recording.analysisAlgorithm = algorithm
        return recording
    }

    private func createSettings(algorithm: PitchDetectionAlgorithm) -> AudioDetectionSettings {
        return AudioDetectionSettings(
            recordingPlaybackVolume: 0.8,
            rmsSilenceThreshold: 0.02,
            confidenceThreshold: 0.4,
            pitchAlgorithm: algorithm
        )
    }

    private func createTestPitchData() -> PitchAnalysisData {
        return PitchAnalysisData(
            timeStamps: [0.0, 0.1, 0.2],
            frequencies: [440.0, 442.0, 438.0],
            confidences: [0.9, 0.85, 0.92],
            amplitudes: [0.5, 0.6, 0.55]
        )
    }

    private func createTestSpectrogramData() -> SpectrogramData {
        return SpectrogramData(
            timeStamps: [0.0, 0.1, 0.2],
            frequencyBins: [100.0, 200.0, 300.0],
            magnitudes: [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
        )
    }

    private func createTestAnalysisResult() -> AnalysisResult {
        return AnalysisResult(
            pitchData: createTestPitchData(),
            spectrogramData: createTestSpectrogramData()
        )
    }

    // MARK: - In-Memory Cache Hit Tests

    func testExecute_InMemoryCacheHit_ReturnsCachedResult() async throws {
        // Given
        let recording = createTestRecording(algorithm: .yin)
        let cachedResult = createTestAnalysisResult()
        mockAnalysisCache.cachedResults[recording.id] = cachedResult
        mockAudioSettingsRepository.settingsToReturn = createSettings(algorithm: .yin)

        var progressValues: [Double] = []

        // When
        let result = try await sut.execute(recording: recording) { progress in
            progressValues.append(progress)
        }

        // Then
        XCTAssertEqual(result.pitchData.timeStamps, cachedResult.pitchData.timeStamps)
        XCTAssertFalse(mockAnalyzer.analyzeCalled, "Analyzer should not be called on cache hit")
        XCTAssertFalse(mockAnalyzer.analyzeSpectrogramOnlyCalled)
        XCTAssertTrue(mockAnalysisCache.getCalled)
        XCTAssertEqual(progressValues.last, 1.0, "Progress should be set to 1.0")
    }

    func testExecute_InMemoryCacheHit_DoesNotInvokeAnalyzer() async throws {
        // Given
        let recording = createTestRecording(algorithm: .yin)
        mockAnalysisCache.cachedResults[recording.id] = createTestAnalysisResult()
        mockAudioSettingsRepository.settingsToReturn = createSettings(algorithm: .yin)

        // When
        _ = try await sut.execute(recording: recording)

        // Then
        XCTAssertEqual(mockAnalyzer.analyzeCallCount, 0)
        XCTAssertEqual(mockAnalyzer.analyzeSpectrogramOnlyCallCount, 0)
    }

    // MARK: - File Cache Hit Tests (Spectrogram Only Analysis)

    func testExecute_FileCacheHit_AnalyzesSpectrogramOnly() async throws {
        // Given
        let recording = createTestRecording(algorithm: .yin)
        let cachedPitchData = createTestPitchData()
        mockPitchDataCache.cachedData[recording.id] = cachedPitchData
        mockAudioSettingsRepository.settingsToReturn = createSettings(algorithm: .yin)
        // In-memory cache is empty, but file cache has pitch data

        // When
        let result = try await sut.execute(recording: recording)

        // Then
        XCTAssertTrue(mockAnalyzer.analyzeSpectrogramOnlyCalled, "Should call spectrogram-only analysis")
        XCTAssertFalse(mockAnalyzer.analyzeCalled, "Should not call full analysis")
        XCTAssertEqual(result.pitchData.timeStamps, cachedPitchData.timeStamps)
        XCTAssertTrue(mockAnalysisCache.setCalled, "Should cache the combined result")
    }

    func testExecute_FileCacheHit_UpdatesInMemoryCache() async throws {
        // Given
        let recording = createTestRecording(algorithm: .yin)
        mockPitchDataCache.cachedData[recording.id] = createTestPitchData()
        mockAudioSettingsRepository.settingsToReturn = createSettings(algorithm: .yin)

        // When
        _ = try await sut.execute(recording: recording)

        // Then
        XCTAssertTrue(mockAnalysisCache.setCalled)
        XCTAssertNotNil(mockAnalysisCache.cachedResults[recording.id])
    }

    // MARK: - Full Analysis Tests (Cache Miss)

    func testExecute_CacheMiss_PerformsFullAnalysis() async throws {
        // Given
        let recording = createTestRecording()
        mockAudioSettingsRepository.settingsToReturn = createSettings(algorithm: .yin)
        mockRecordingRepository.savedRecordings = [recording]

        // When
        let result = try await sut.execute(recording: recording)

        // Then
        XCTAssertTrue(mockAnalyzer.analyzeCalled, "Should perform full analysis")
        XCTAssertEqual(mockAnalyzer.analyzeCallCount, 1)
        XCTAssertEqual(mockAnalyzer.lastAnalyzedURL, recording.fileURL)
        XCTAssertEqual(result.pitchData.timeStamps, mockAnalyzer.pitchDataToReturn.timeStamps)
    }

    func testExecute_CacheMiss_CachesBothLayers() async throws {
        // Given
        let recording = createTestRecording()
        mockAudioSettingsRepository.settingsToReturn = createSettings(algorithm: .yin)
        mockRecordingRepository.savedRecordings = [recording]

        // When
        _ = try await sut.execute(recording: recording)

        // Then
        XCTAssertTrue(mockAnalysisCache.setCalled, "Should cache in memory")
        XCTAssertNotNil(mockAnalysisCache.cachedResults[recording.id])
        XCTAssertEqual(mockPitchDataCache.setCallCount, 1, "Should cache pitch data to file")
        XCTAssertNotNil(mockPitchDataCache.cachedData[recording.id])
    }

    func testExecute_CacheMiss_UpdatesRecordingAlgorithm() async throws {
        // Given
        let recording = createTestRecording()
        mockAudioSettingsRepository.settingsToReturn = createSettings(algorithm: .pyinDefault)
        mockRecordingRepository.savedRecordings = [recording]

        // When
        _ = try await sut.execute(recording: recording)

        // Then
        XCTAssertTrue(mockRecordingRepository.updateCalled, "Should update recording with algorithm")
    }

    func testExecute_UsesCustomAudioURL() async throws {
        // Given
        let recording = createTestRecording()
        let customURL = URL(fileURLWithPath: "/tmp/custom_audio.m4a")
        mockRecordingRepository.savedRecordings = [recording]

        // When
        _ = try await sut.execute(recording: recording, audioURL: customURL)

        // Then
        XCTAssertEqual(mockAnalyzer.lastAnalyzedURL, customURL)
    }

    // MARK: - Algorithm Change Tests

    func testExecute_AlgorithmChanged_InvalidatesCache() async throws {
        // Given
        let recording = createTestRecording(algorithm: .yin)
        mockAnalysisCache.cachedResults[recording.id] = createTestAnalysisResult()
        mockPitchDataCache.cachedData[recording.id] = createTestPitchData()
        // Change to different algorithm
        mockAudioSettingsRepository.settingsToReturn = createSettings(algorithm: .pyinDefault)
        mockRecordingRepository.savedRecordings = [recording]

        // When
        _ = try await sut.execute(recording: recording)

        // Then
        XCTAssertTrue(mockAnalysisCache.clearCalled, "Should clear in-memory cache")
        XCTAssertEqual(mockPitchDataCache.deleteCallCount, 1, "Should delete file cache")
        XCTAssertTrue(mockAnalyzer.analyzeCalled, "Should perform full analysis")
    }

    func testExecute_AlgorithmChanged_PerformsFullAnalysis() async throws {
        // Given
        let recording = createTestRecording(algorithm: .yin)
        mockAnalysisCache.cachedResults[recording.id] = createTestAnalysisResult()
        mockAudioSettingsRepository.settingsToReturn = createSettings(algorithm: .pyinDefault)
        mockRecordingRepository.savedRecordings = [recording]

        // When
        _ = try await sut.execute(recording: recording)

        // Then
        XCTAssertTrue(mockAnalyzer.analyzeCalled)
        XCTAssertEqual(mockAnalyzer.analyzeCallCount, 1)
    }

    func testExecute_SameAlgorithm_UsesCachedData() async throws {
        // Given
        let recording = createTestRecording(algorithm: .yin)
        mockAnalysisCache.cachedResults[recording.id] = createTestAnalysisResult()
        mockAudioSettingsRepository.settingsToReturn = createSettings(algorithm: .yin)

        // When
        _ = try await sut.execute(recording: recording)

        // Then
        XCTAssertFalse(mockAnalysisCache.clearCalled)
        XCTAssertFalse(mockAnalyzer.analyzeCalled)
    }

    // MARK: - Error Handling Tests

    func testExecute_AnalyzerThrowsError_PropagatesError() async throws {
        // Given
        let recording = createTestRecording()
        mockAnalyzer.shouldThrowError = true
        mockAnalyzer.errorToThrow = NSError(domain: "TestError", code: 42)

        // When/Then
        do {
            _ = try await sut.execute(recording: recording)
            XCTFail("Should throw error")
        } catch {
            XCTAssertEqual((error as NSError).code, 42)
        }
    }

    func testExecute_UpdateRecordingFails_ContinuesSuccessfully() async throws {
        // Given
        let recording = createTestRecording()
        mockRecordingRepository.updateShouldFail = true
        mockRecordingRepository.savedRecordings = [recording]

        // When
        let result = try await sut.execute(recording: recording)

        // Then - Should complete analysis despite update failure
        XCTAssertNotNil(result)
        XCTAssertTrue(mockLogger.hasLevel(.error))
    }

    // MARK: - hasCachedData Tests

    func testHasCachedData_WithRecording_ReturnsTrueForInMemoryCache() {
        // Given
        let recording = createTestRecording(algorithm: .yin)
        mockAnalysisCache.cachedResults[recording.id] = createTestAnalysisResult()
        mockAudioSettingsRepository.settingsToReturn = createSettings(algorithm: .yin)

        // When
        let hasCached = sut.hasCachedData(for: recording)

        // Then
        XCTAssertTrue(hasCached)
    }

    func testHasCachedData_WithRecording_ReturnsTrueForFileCache() {
        // Given
        let recording = createTestRecording(algorithm: .yin)
        mockPitchDataCache.cachedData[recording.id] = createTestPitchData()
        mockAudioSettingsRepository.settingsToReturn = createSettings(algorithm: .yin)

        // When
        let hasCached = sut.hasCachedData(for: recording)

        // Then
        XCTAssertTrue(hasCached)
    }

    func testHasCachedData_WithRecording_ReturnsFalseWhenAlgorithmChanged() {
        // Given
        let recording = createTestRecording(algorithm: .yin)
        mockAnalysisCache.cachedResults[recording.id] = createTestAnalysisResult()
        mockAudioSettingsRepository.settingsToReturn = createSettings(algorithm: .pyinDefault)

        // When
        let hasCached = sut.hasCachedData(for: recording)

        // Then
        XCTAssertFalse(hasCached, "Cache should be invalid when algorithm changed")
    }

    func testHasCachedData_WithRecording_ReturnsFalseWhenNoCache() {
        // Given
        let recording = createTestRecording()

        // When
        let hasCached = sut.hasCachedData(for: recording)

        // Then
        XCTAssertFalse(hasCached)
    }

    func testHasCachedData_WithRecordingId_ReturnsTrueForInMemoryCache() {
        // Given
        let recording = createTestRecording()
        mockAnalysisCache.cachedResults[recording.id] = createTestAnalysisResult()

        // When
        let hasCached = sut.hasCachedData(for: recording.id)

        // Then
        XCTAssertTrue(hasCached)
    }

    func testHasCachedData_WithRecordingId_ReturnsTrueForFileCache() {
        // Given
        let recording = createTestRecording()
        mockPitchDataCache.cachedData[recording.id] = createTestPitchData()

        // When
        let hasCached = sut.hasCachedData(for: recording.id)

        // Then
        XCTAssertTrue(hasCached)
    }

    func testHasCachedData_WithRecordingId_ReturnsFalseWhenNoCache() {
        // Given
        let recordingId = RecordingId()

        // When
        let hasCached = sut.hasCachedData(for: recordingId)

        // Then
        XCTAssertFalse(hasCached)
    }

    // MARK: - Legacy Initializer Tests

    func testLegacyInitializer_CreatesWorkingUseCase() async throws {
        // Given
        let legacySut = AnalyzeRecordingUseCase(
            audioFileAnalyzer: mockAnalyzer,
            analysisCache: mockAnalysisCache,
            pitchDataCache: mockPitchDataCache,
            logger: mockLogger
        )
        let recording = createTestRecording()

        // When
        let result = try await legacySut.execute(recording: recording)

        // Then
        XCTAssertNotNil(result)
        XCTAssertTrue(mockAnalyzer.analyzeCalled)
    }

    // MARK: - Progress Callback Tests

    func testExecute_ReportsProgressCorrectly() async throws {
        // Given
        let recording = createTestRecording()
        var progressValues: [Double] = []

        // When
        _ = try await sut.execute(recording: recording) { progress in
            progressValues.append(progress)
        }

        // Then
        XCTAssertFalse(progressValues.isEmpty)
        XCTAssertEqual(progressValues.last, 1.0)
    }

    // MARK: - Without Optional Dependencies Tests

    func testExecute_WithoutPitchDataCache_SkipsFileCaching() async throws {
        // Given
        let sutWithoutFileCache = AnalyzeRecordingUseCase(
            analyzerFactory: mockAnalyzerFactory,
            analysisCache: mockAnalysisCache,
            pitchDataCache: nil,
            audioSettingsRepository: mockAudioSettingsRepository,
            recordingRepository: mockRecordingRepository,
            logger: mockLogger
        )
        let recording = createTestRecording()

        // When
        _ = try await sutWithoutFileCache.execute(recording: recording)

        // Then
        XCTAssertTrue(mockAnalyzer.analyzeCalled)
        XCTAssertEqual(mockPitchDataCache.setCallCount, 0)
    }

    func testExecute_WithoutRecordingRepository_SkipsRecordingUpdate() async throws {
        // Given
        let sutWithoutRepo = AnalyzeRecordingUseCase(
            analyzerFactory: mockAnalyzerFactory,
            analysisCache: mockAnalysisCache,
            pitchDataCache: mockPitchDataCache,
            audioSettingsRepository: mockAudioSettingsRepository,
            recordingRepository: nil,
            logger: mockLogger
        )
        let recording = createTestRecording()

        // When
        _ = try await sutWithoutRepo.execute(recording: recording)

        // Then
        XCTAssertTrue(mockAnalyzer.analyzeCalled)
        XCTAssertFalse(mockRecordingRepository.updateCalled)
    }
}
