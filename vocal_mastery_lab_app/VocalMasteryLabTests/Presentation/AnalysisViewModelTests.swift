import XCTest
import VocalisDomain
@testable import VocalMasteryLab

final class AnalysisViewModelTests: XCTestCase {
    var sut: AnalysisViewModel!
    var mockAudioPlayer: MockAudioPlayer!
    fileprivate var mockUseCase: MockAnalyzeRecordingUseCase!
    var testRecording: Recording!

    @MainActor
    override func setUp() {
        super.setUp()
        mockAudioPlayer = MockAudioPlayer()
        mockUseCase = MockAnalyzeRecordingUseCase()
        mockUseCase.progressCallbacks = []  // Reset progress callbacks
        testRecording = createTestRecording()
        sut = AnalysisViewModel(
            recording: testRecording,
            audioPlayer: mockAudioPlayer,
            analyzeRecordingUseCase: mockUseCase
        )
    }

    override func tearDown() {
        sut = nil
        mockAudioPlayer = nil
        mockUseCase = nil
        testRecording = nil
        super.tearDown()
    }

    // MARK: - Test Helpers

    private func createTestRecording() -> Recording {
        return Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0)
        )
    }

    private func createTestAnalysisResult() -> AnalysisResult {
        let pitchData = PitchAnalysisData(
            timeStamps: [0.0, 0.05],
            frequencies: [261.6, 262.3],
            confidences: [0.85, 0.92],
            amplitudes: [0.5, 0.6]
        )

        let spectrogramData = SpectrogramData(
            timeStamps: [0.0, 0.1],
            frequencyBins: [80, 180],
            magnitudes: [[0.1, 0.3], [0.2, 0.4]]
        )

        return AnalysisResult(
            pitchData: pitchData,
            spectrogramData: spectrogramData
        )
    }

    // MARK: - Initial State Tests

    @MainActor
    func testInitialState_IsLoading() {
        // Then: Initial state should be loading with 0% progress
        XCTAssertEqual(sut.state, .loading(progress: 0.0))
        XCTAssertFalse(sut.isPlaying)
        XCTAssertEqual(sut.currentTime, 0.0)
    }

    // MARK: - Analysis Tests

    @MainActor
    func testStartAnalysis_Success_UpdatesStateToReady() async {
        // Given: Successful analysis
        let expectedResult = createTestAnalysisResult()
        mockUseCase.resultToReturn = expectedResult

        // When: Starting analysis
        await sut.startAnalysis()

        // Then: State should be ready
        if case .ready(let result) = sut.state {
            XCTAssertEqual(result, expectedResult)
        } else {
            XCTFail("Expected ready state, got \(sut.state)")
        }
    }

    @MainActor
    func testStartAnalysis_Failure_UpdatesStateToError() async {
        // Given: Analysis will fail
        mockUseCase.shouldThrowError = true

        // When: Starting analysis
        await sut.startAnalysis()

        // Then: State should be error
        if case .error(let message) = sut.state {
            XCTAssertFalse(message.isEmpty)
        } else {
            XCTFail("Expected error state, got \(sut.state)")
        }
    }

    @MainActor
    func testStartAnalysis_CallsUseCase() async {
        // Given: Successful analysis
        mockUseCase.resultToReturn = createTestAnalysisResult()

        // When: Starting analysis
        await sut.startAnalysis()

        // Then: Use case should be called
        XCTAssertEqual(mockUseCase.executeCallCount, 1)
        XCTAssertEqual(mockUseCase.lastRecording?.id, testRecording.id)
    }

    /// Test: When vocalURL is provided, it should be passed to the use case for analysis
    /// This ensures the analysis uses the extracted vocal audio, not the original recording
    @MainActor
    func testStartAnalysis_WithVocalURL_PassesVocalURLToUseCase() async {
        // Given: A vocal URL is provided
        let vocalURL = URL(fileURLWithPath: "/tmp/extracted_vocal.wav")
        mockUseCase.resultToReturn = createTestAnalysisResult()

        // Create ViewModel with vocalURL
        let viewModelWithVocal = AnalysisViewModel(
            recording: testRecording,
            vocalURL: vocalURL,
            audioPlayer: mockAudioPlayer,
            analyzeRecordingUseCase: mockUseCase
        )

        // When: Starting analysis
        await viewModelWithVocal.startAnalysis()

        // Then: Use case should receive the vocalURL as audioURL parameter
        XCTAssertEqual(mockUseCase.lastAudioURL, vocalURL,
            "Analysis should use extracted vocal URL, not original recording")
    }

    /// Test: When vocalURL is nil, audioURL passed to use case should also be nil
    @MainActor
    func testStartAnalysis_WithoutVocalURL_PassesNilAudioURL() async {
        // Given: No vocal URL (default setup)
        mockUseCase.resultToReturn = createTestAnalysisResult()

        // When: Starting analysis
        await sut.startAnalysis()

        // Then: audioURL should be nil (use case will use recording.fileURL)
        XCTAssertNil(mockUseCase.lastAudioURL,
            "Without vocalURL, audioURL should be nil")
    }

    @MainActor
    func testStartAnalysis_ReportsProgressUpdates() async {
        // Given: Successful analysis
        mockUseCase.resultToReturn = createTestAnalysisResult()

        // When: Starting analysis
        await sut.startAnalysis()

        // Then: Progress should have been reported
        XCTAssertEqual(mockUseCase.progressCallbacks.count, 3, "Should report 0%, 50%, and 100% progress")
        XCTAssertEqual(mockUseCase.progressCallbacks[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(mockUseCase.progressCallbacks[1], 0.5, accuracy: 0.01)
        XCTAssertEqual(mockUseCase.progressCallbacks[2], 1.0, accuracy: 0.01)
    }

    @MainActor
    func testStartAnalysis_UpdatesStateWithProgress() async {
        // Given: Successful analysis
        mockUseCase.resultToReturn = createTestAnalysisResult()

        // Track state changes
        var stateChanges: [AnalysisState] = []
        let cancellable = sut.$state
            .sink { state in
                stateChanges.append(state)
            }

        // When: Starting analysis
        await sut.startAnalysis()

        // Then: State should transition through loading states and end in ready
        XCTAssertGreaterThanOrEqual(stateChanges.count, 2, "Should have at least initial and final state")

        // Initial state
        if case .loading(let progress) = stateChanges[0] {
            XCTAssertEqual(progress, 0.0, accuracy: 0.01)
        } else {
            XCTFail("Initial state should be loading with 0% progress")
        }

        // Final state should be ready
        if case .ready = stateChanges.last {
            // Success
        } else {
            XCTFail("Final state should be ready, got \(stateChanges.last!)")
        }

        cancellable.cancel()
    }

    // MARK: - Playback Control Tests

    @MainActor
    func testTogglePlayback_WhenNotPlaying_StartsPlayback() async {
        // Given: Not playing and ready state
        mockUseCase.resultToReturn = createTestAnalysisResult()
        await sut.startAnalysis()

        // When: Toggling playback
        sut.togglePlayback()

        // Then: Should start playing
        XCTAssertTrue(sut.isPlaying)
    }

    @MainActor
    func testSeek_UpdatesCurrentTime() {
        // When: Seeking to specific time
        sut.seek(to: 5.0)

        // Then: Current time should be updated
        XCTAssertEqual(sut.currentTime, 5.0)
    }

    @MainActor
    func testSeek_ClampsToDuration() {
        // When: Seeking beyond duration
        sut.seek(to: 100.0)

        // Then: Should clamp to duration
        XCTAssertEqual(sut.currentTime, testRecording.duration.seconds)
    }

    @MainActor
    func testSeek_ClampsToZero() {
        // When: Seeking to negative time
        sut.seek(to: -5.0)

        // Then: Should clamp to zero
        XCTAssertEqual(sut.currentTime, 0.0)
    }

    @MainActor
    func testSkipBackward_SeeksBackwardFiveSeconds() {
        // Given: Current time is 7 seconds
        sut.seek(to: 7.0)

        // When: Skipping backward
        sut.skipBackward()

        // Then: Should be at 2 seconds
        XCTAssertEqual(sut.currentTime, 2.0)
    }

    @MainActor
    func testSkipForward_SeeksForwardFiveSeconds() {
        // Given: Current time is 3 seconds
        sut.seek(to: 3.0)

        // When: Skipping forward
        sut.skipForward()

        // Then: Should be at 8 seconds
        XCTAssertEqual(sut.currentTime, 8.0)
    }

    // MARK: - Computed Properties Tests

    @MainActor
    func testDuration_ReturnsRecordingDuration() {
        // Then: Duration should match recording
        XCTAssertEqual(sut.duration, testRecording.duration.seconds)
    }

    @MainActor
    func testAnalysisResult_WhenReady_ReturnsResult() async {
        // Given: Ready state
        let expectedResult = createTestAnalysisResult()
        mockUseCase.resultToReturn = expectedResult
        await sut.startAnalysis()

        // Then: Should return result
        XCTAssertEqual(sut.analysisResult, expectedResult)
    }

    @MainActor
    func testAnalysisResult_WhenLoading_ReturnsNil() {
        // Then: Should return nil
        XCTAssertNil(sut.analysisResult)
    }

    @MainActor
    func testAnalysisResult_WhenError_ReturnsNil() async {
        // Given: Error state
        mockUseCase.shouldThrowError = true
        await sut.startAnalysis()

        // Then: Should return nil
        XCTAssertNil(sut.analysisResult)
    }

    // MARK: - Playback Completion Tests

    /// Test: After playback completes, isPlaying should become false
    /// Expected: isPlaying = false after completion
    @MainActor
    func testPlaybackCompletion_ShouldSetIsPlayingToFalse() async {
        // Given: Ready state
        mockUseCase.resultToReturn = createTestAnalysisResult()
        await sut.startAnalysis()

        // Configure quick playback (50ms total)
        mockAudioPlayer.playDurationNanoseconds = 50_000_000

        // When: Start playback
        sut.togglePlayback()
        XCTAssertTrue(sut.isPlaying, "Should be playing initially")

        // Wait for playback to complete + timer processing time
        try? await Task.sleep(nanoseconds: 150_000_000) // 150ms

        // Then: isPlaying should be false
        XCTAssertFalse(sut.isPlaying, "isPlaying should be false after playback completion")
    }

    /// Test: After playback completes, currentTime should reset to 0 (beginning)
    /// Expected: currentTime = 0.0 after completion (ready to play again from start)
    @MainActor
    func testPlaybackCompletion_ShouldResetCurrentTimeToZero() async {
        // Given: Ready state
        mockUseCase.resultToReturn = createTestAnalysisResult()
        await sut.startAnalysis()

        mockAudioPlayer.playDurationNanoseconds = 50_000_000

        // When: Start playback and wait for completion
        sut.togglePlayback()
        try? await Task.sleep(nanoseconds: 150_000_000)

        // Then: currentTime should be reset to 0 (beginning)
        XCTAssertEqual(sut.currentTime, 0.0, accuracy: 0.1,
                      "currentTime should be reset to 0.0 (beginning) after completion")
    }

}

// MARK: - Mock Objects

@MainActor
fileprivate class MockAnalyzeRecordingUseCase: AnalyzeRecordingUseCase {
    var executeCallCount = 0
    var lastRecording: Recording?
    var lastAudioURL: URL?  // Capture the audioURL passed to execute
    var shouldThrowError = false
    var resultToReturn: AnalysisResult?
    var progressCallbacks: [Double] = []

    init() {
        // Initialize with dummy dependencies (won't be used)
        let dummyFactory = DummyAnalyzerFactory()
        let dummyCache = DummyCache()
        let dummyLogger = DummyLogger()
        super.init(analyzerFactory: dummyFactory, analysisCache: dummyCache, logger: dummyLogger)
    }

    override func execute(recording: Recording, audioURL: URL? = nil, progress: @escaping @MainActor (Double) -> Void) async throws -> AnalysisResult {
        executeCallCount += 1
        lastRecording = recording
        lastAudioURL = audioURL  // Capture audioURL

        // Simulate progress updates
        await progress(0.0)
        progressCallbacks.append(0.0)

        await progress(0.5)
        progressCallbacks.append(0.5)

        await progress(1.0)
        progressCallbacks.append(1.0)

        if shouldThrowError {
            throw NSError(domain: "TestError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Test error"])
        }

        guard let result = resultToReturn else {
            throw NSError(domain: "TestError", code: -2, userInfo: [NSLocalizedDescriptionKey: "No result configured"])
        }

        return result
    }
}

@MainActor
fileprivate class DummyAnalyzerFactory: AudioFileAnalyzerFactoryProtocol {
    nonisolated func makeAnalyzer() -> AudioFileAnalyzerProtocol {
        return DummyAnalyzer()
    }
}

fileprivate class DummyAnalyzer: AudioFileAnalyzerProtocol {
    func analyze(fileURL: URL, progress: @escaping @MainActor (Double) async -> Void) async throws -> (pitchData: PitchAnalysisData, spectrogramData: SpectrogramData) {
        fatalError("Should not be called")
    }

    func analyzeSpectrogramOnly(fileURL: URL, progress: @escaping @MainActor (Double) async -> Void) async throws -> SpectrogramData {
        fatalError("Should not be called")
    }
}

fileprivate class DummyCache: AnalysisCacheProtocol {
    func get(_ id: RecordingId) -> AnalysisResult? {
        fatalError("Should not be called")
    }

    func set(_ id: RecordingId, result: AnalysisResult) {
        fatalError("Should not be called")
    }

    func clear() {
        fatalError("Should not be called")
    }
}

fileprivate class DummyLogger: LoggerProtocol {
    func debug(_ message: String, category: String) {
        fatalError("Should not be called")
    }

    func info(_ message: String, category: String) {
        fatalError("Should not be called")
    }

    func warning(_ message: String, category: String) {
        fatalError("Should not be called")
    }

    func error(_ message: String, category: String) {
        fatalError("Should not be called")
    }
}
