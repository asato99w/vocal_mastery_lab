//
//  VocalExtractionViewModelTests.swift
//  VocalMasteryLabTests
//
//  Tests for VocalExtractionViewModel including instrumental functionality
//

import XCTest
@testable import VocalMasteryLab
import VocalisDomain

@MainActor
final class VocalExtractionViewModelTests: XCTestCase {

    private var mockExtractor: VocalExtractionMockExtractor!
    private var mockRepository: VocalExtractionMockRepository!
    private var mockAudioPlayer: VocalExtractionMockPlayer!
    private var testRecording: Recording!

    override func setUp() {
        super.setUp()
        mockExtractor = VocalExtractionMockExtractor()
        mockRepository = VocalExtractionMockRepository()
        mockAudioPlayer = VocalExtractionMockPlayer()

        testRecording = Recording(
            id: RecordingId(),
            fileURL: URL(fileURLWithPath: "/tmp/test.wav"),
            createdAt: Date(),
            duration: Duration(seconds: 10.0)
        )
    }

    override func tearDown() {
        mockExtractor = nil
        mockRepository = nil
        mockAudioPlayer = nil
        testRecording = nil
        super.tearDown()
    }

    // MARK: - Initial State Tests

    func testInitialState_isIdle() {
        let viewModel = createViewModel()

        XCTAssertEqual(viewModel.state, .idle)
        XCTAssertFalse(viewModel.isSaving)
    }

    // MARK: - Extraction Tests

    func testStartExtraction_setsProcessingState() async {
        let viewModel = createViewModel()

        // Start extraction (don't wait for completion)
        Task {
            await viewModel.startExtraction()
        }

        // Give time for state to change
        try? await Task.sleep(nanoseconds: 100_000_000)

        // Should be in processing or completed state
        if case .idle = viewModel.state {
            XCTFail("State should not be idle during extraction")
        }
    }

    func testStartExtraction_completesWithResult() async {
        let viewModel = createViewModel()
        let vocalURL = URL(fileURLWithPath: "/tmp/vocal.wav")
        let instrumentalURL = URL(fileURLWithPath: "/tmp/instrumental.wav")

        mockExtractor.resultToReturn = VocalExtractionResult(
            vocalFileURL: vocalURL,
            instrumentalFileURL: instrumentalURL,
            duration: Duration(seconds: 10.0)
        )

        await viewModel.startExtraction()

        if case .completed(let result) = viewModel.state {
            XCTAssertEqual(result.vocalURL, vocalURL)
            XCTAssertEqual(result.instrumentalURL, instrumentalURL)
        } else {
            XCTFail("State should be completed after extraction")
        }
    }

    func testStartExtraction_withoutInstrumental_completesWithNilInstrumental() async {
        let viewModel = createViewModel()
        let vocalURL = URL(fileURLWithPath: "/tmp/vocal.wav")

        mockExtractor.resultToReturn = VocalExtractionResult(
            vocalFileURL: vocalURL,
            instrumentalFileURL: nil,
            duration: Duration(seconds: 10.0)
        )

        await viewModel.startExtraction()

        if case .completed(let result) = viewModel.state {
            XCTAssertEqual(result.vocalURL, vocalURL)
            XCTAssertNil(result.instrumentalURL)
        } else {
            XCTFail("State should be completed after extraction")
        }
    }

    func testStartExtraction_onError_setsErrorState() async {
        let viewModel = createViewModel()
        mockExtractor.errorToThrow = VocalExtractionError.extractionFailed("Test error")

        await viewModel.startExtraction()

        if case .error = viewModel.state {
            // Successfully set error state
        } else {
            XCTFail("State should be error after failed extraction")
        }
    }

    // MARK: - Save Tests

    func testSaveExtraction_savesVocalAndInstrumental() async {
        let viewModel = createViewModel()
        let vocalURL = URL(fileURLWithPath: "/tmp/vocal.wav")
        let instrumentalURL = URL(fileURLWithPath: "/tmp/instrumental.wav")

        mockExtractor.resultToReturn = VocalExtractionResult(
            vocalFileURL: vocalURL,
            instrumentalFileURL: instrumentalURL,
            duration: Duration(seconds: 10.0)
        )

        await viewModel.startExtraction()
        let success = await viewModel.saveExtraction()

        XCTAssertTrue(success)
        XCTAssertEqual(mockRepository.savedAudios.count, 2)

        let vocalAudio = mockRepository.savedAudios.first { $0.type == .vocal }
        let instrumentalAudio = mockRepository.savedAudios.first { $0.type == .instrumental }

        XCTAssertNotNil(vocalAudio, "Vocal audio should be saved")
        XCTAssertNotNil(instrumentalAudio, "Instrumental audio should be saved")
        XCTAssertEqual(vocalAudio?.fileURL, vocalURL)
        XCTAssertEqual(instrumentalAudio?.fileURL, instrumentalURL)
    }

    func testSaveExtraction_withoutInstrumental_savesOnlyVocal() async {
        let viewModel = createViewModel()
        let vocalURL = URL(fileURLWithPath: "/tmp/vocal.wav")

        mockExtractor.resultToReturn = VocalExtractionResult(
            vocalFileURL: vocalURL,
            instrumentalFileURL: nil,
            duration: Duration(seconds: 10.0)
        )

        await viewModel.startExtraction()
        let success = await viewModel.saveExtraction()

        XCTAssertTrue(success)
        XCTAssertEqual(mockRepository.savedAudios.count, 1)
        XCTAssertEqual(mockRepository.savedAudios.first?.type, .vocal)
    }

    func testSaveExtraction_whenNotCompleted_returnsFalse() async {
        let viewModel = createViewModel()

        let success = await viewModel.saveExtraction()

        XCTAssertFalse(success)
        XCTAssertTrue(mockRepository.savedAudios.isEmpty)
    }

    // MARK: - Playback Tests

    func testPlayVocal_playsVocalURL() async {
        let viewModel = createViewModel()
        let vocalURL = URL(fileURLWithPath: "/tmp/vocal.wav")

        mockExtractor.resultToReturn = VocalExtractionResult(
            vocalFileURL: vocalURL,
            instrumentalFileURL: nil,
            duration: Duration(seconds: 10.0)
        )

        await viewModel.startExtraction()
        await viewModel.playVocal()

        XCTAssertEqual(mockAudioPlayer.lastPlayedURL, vocalURL)
    }

    func testPlayInstrumental_playsInstrumentalURL() async {
        let viewModel = createViewModel()
        let vocalURL = URL(fileURLWithPath: "/tmp/vocal.wav")
        let instrumentalURL = URL(fileURLWithPath: "/tmp/instrumental.wav")

        mockExtractor.resultToReturn = VocalExtractionResult(
            vocalFileURL: vocalURL,
            instrumentalFileURL: instrumentalURL,
            duration: Duration(seconds: 10.0)
        )

        await viewModel.startExtraction()
        await viewModel.playInstrumental()

        XCTAssertEqual(mockAudioPlayer.lastPlayedURL, instrumentalURL)
    }

    func testPlayInstrumental_whenNoInstrumental_doesNothing() async {
        let viewModel = createViewModel()
        let vocalURL = URL(fileURLWithPath: "/tmp/vocal.wav")

        mockExtractor.resultToReturn = VocalExtractionResult(
            vocalFileURL: vocalURL,
            instrumentalFileURL: nil,
            duration: Duration(seconds: 10.0)
        )

        await viewModel.startExtraction()
        await viewModel.playInstrumental()

        XCTAssertNil(mockAudioPlayer.lastPlayedURL)
    }

    func testPlayOriginal_playsRecordingURL() async {
        let viewModel = createViewModel()

        await viewModel.playOriginal()

        XCTAssertEqual(mockAudioPlayer.lastPlayedURL, testRecording.fileURL)
    }

    // MARK: - Reset Tests

    func testReset_clearsStateToIdle() async {
        let viewModel = createViewModel()
        let vocalURL = URL(fileURLWithPath: "/tmp/vocal.wav")

        mockExtractor.resultToReturn = VocalExtractionResult(
            vocalFileURL: vocalURL,
            instrumentalFileURL: nil,
            duration: Duration(seconds: 10.0)
        )

        await viewModel.startExtraction()
        viewModel.reset()

        XCTAssertEqual(viewModel.state, .idle)
    }

    // MARK: - Recording Info Tests

    func testRecordingTitle_returnsRecordingTitleOrDefault() {
        let viewModel = createViewModel()

        // Default title when no custom title is set
        XCTAssertEqual(viewModel.recordingTitle, "録音")

        // With custom title
        var recordingWithTitle = testRecording!
        recordingWithTitle.title = "My Recording"
        let viewModelWithTitle = VocalExtractionViewModel(
            recording: recordingWithTitle,
            extractor: mockExtractor,
            extractedAudioRepository: mockRepository,
            audioPlayer: mockAudioPlayer
        )
        XCTAssertEqual(viewModelWithTitle.recordingTitle, "My Recording")
    }

    // MARK: - Helper Methods

    private func createViewModel() -> VocalExtractionViewModel {
        VocalExtractionViewModel(
            recording: testRecording,
            extractor: mockExtractor,
            extractedAudioRepository: mockRepository,
            audioPlayer: mockAudioPlayer
        )
    }
}

// MARK: - Mock Classes

private class VocalExtractionMockExtractor: VocalExtractorProtocol {
    var resultToReturn: VocalExtractionResult?
    var errorToThrow: Error?
    var progressUpdates: [(Double, String)] = []

    func extract(from sourceURL: URL, progressHandler: @escaping (Double, String) -> Void) async throws -> VocalExtractionResult {
        // Simulate progress updates
        for (progress, stage) in progressUpdates {
            progressHandler(progress, stage)
        }

        if let error = errorToThrow {
            throw error
        }

        if let result = resultToReturn {
            return result
        }

        return VocalExtractionResult(
            vocalFileURL: URL(fileURLWithPath: "/tmp/default_vocal.wav"),
            duration: Duration(seconds: 5.0)
        )
    }
}

private class VocalExtractionMockRepository: ExtractedAudioRepositoryProtocol {
    var savedAudios: [ExtractedAudio] = []
    var errorToThrow: Error?

    func save(_ audio: ExtractedAudio) async throws {
        if let error = errorToThrow {
            throw error
        }
        savedAudios.append(audio)
    }

    func findById(_ id: ExtractedAudioId) async throws -> ExtractedAudio? {
        savedAudios.first { $0.id == id }
    }

    func findByRecording(_ recordingId: RecordingId) async throws -> [ExtractedAudio] {
        savedAudios.filter { $0.sourceRecordingId == recordingId }
    }

    func findAll() async throws -> [ExtractedAudio] {
        savedAudios
    }

    func delete(_ id: ExtractedAudioId) async throws {
        savedAudios.removeAll { $0.id == id }
    }

    func deleteByRecording(_ recordingId: RecordingId) async throws {
        savedAudios.removeAll { $0.sourceRecordingId == recordingId }
    }
}

private class VocalExtractionMockPlayer: AudioPlayerProtocol {
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 10.0
    var lastPlayedURL: URL?

    func play(url: URL, withPitchDetection: Bool) async throws {
        lastPlayedURL = url
        isPlaying = true
    }

    func stop() async {
        isPlaying = false
    }

    func pause() {
        isPlaying = false
    }

    func resume() {
        isPlaying = true
    }

    func seek(to time: TimeInterval) {
        currentTime = time
    }
}
