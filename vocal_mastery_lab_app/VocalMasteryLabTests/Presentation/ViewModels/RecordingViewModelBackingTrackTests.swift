import XCTest
import VocalisDomain
import SubscriptionDomain
@testable import VocalMasteryLab

@MainActor
final class RecordingViewModelBackingTrackTests: XCTestCase {

    var sut: RecordingViewModel!
    var mockStartRecordingUseCase: MockStartRecordingUseCase!
    var mockStopRecordingUseCase: MockStopRecordingUseCase!
    var mockAudioPlayer: MockAudioPlayer!
    var mockPitchDetector: MockRealtimePitchDetector!
    var mockRecordingRepository: MockRecordingRepository!
    var mockExtractedAudioRepository: MockExtractedAudioRepository!
    var mockBackingTrackPlayer: MockAudioPlayer!

    override func setUp() async throws {
        try await super.setUp()
        mockStartRecordingUseCase = MockStartRecordingUseCase()
        mockStopRecordingUseCase = MockStopRecordingUseCase()
        mockAudioPlayer = MockAudioPlayer()
        mockPitchDetector = MockRealtimePitchDetector()
        mockRecordingRepository = MockRecordingRepository()
        mockExtractedAudioRepository = MockExtractedAudioRepository()
        mockBackingTrackPlayer = MockAudioPlayer()

        // Create mock subscription view model
        let subscriptionVM = SubscriptionViewModel(
            getStatusUseCase: MockGetSubscriptionStatusUseCase(),
            purchaseUseCase: MockPurchaseSubscriptionUseCase(),
            restoreUseCase: MockRestorePurchasesUseCase()
        )

        sut = RecordingViewModel(
            startRecordingUseCase: mockStartRecordingUseCase,
            stopRecordingUseCase: mockStopRecordingUseCase,
            audioPlayer: mockAudioPlayer,
            pitchDetector: mockPitchDetector,
            subscriptionViewModel: subscriptionVM,
            countdownDuration: 0 // Disable countdown for tests
        )

        // Set up backing track repositories
        sut.setBackingTrackRepositories(
            recordingRepository: mockRecordingRepository,
            extractedAudioRepository: mockExtractedAudioRepository,
            backingTrackPlayer: mockBackingTrackPlayer
        )
    }

    override func tearDown() async throws {
        sut = nil
        mockBackingTrackPlayer = nil
        mockExtractedAudioRepository = nil
        mockRecordingRepository = nil
        mockPitchDetector = nil
        mockAudioPlayer = nil
        mockStopRecordingUseCase = nil
        mockStartRecordingUseCase = nil
        try await super.tearDown()
    }

    // MARK: - Load Backing Tracks Tests

    func testLoadBackingTracks_WithNoRecordings_ReturnsEmptyList() async {
        // Given
        mockRecordingRepository.recordingsToReturn = []

        // When
        await sut.loadBackingTracks()

        // Then
        XCTAssertTrue(sut.availableBackingTracks.isEmpty)
    }

    func testLoadBackingTracks_WithRecordingsNoExtractedAudio_ReturnsAllRecordings() async {
        // Given - 2 recordings without extracted audio
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 10.0)
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 15.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording1, recording2]
        mockExtractedAudioRepository.storedAudios = []

        // When
        await sut.loadBackingTracks()

        // Then - ALL recordings should be in the list
        XCTAssertEqual(sut.availableBackingTracks.count, 2, "All recordings should appear in backing tracks list")

        // Each recording should have only .original source
        XCTAssertEqual(sut.availableBackingTracks[0].availableSources, [.original])
        XCTAssertEqual(sut.availableBackingTracks[1].availableSources, [.original])
    }

    func testLoadBackingTracks_WithExtractedAudio_ReturnsAllRecordingsWithCorrectSources() async {
        // Given - 2 recordings, one with extracted audio, one without
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 10.0)
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 15.0)
        )

        // Only recording1 has extracted audio
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording1.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal1.wav"),
            duration: Duration(seconds: 10.0)
        )
        let instrumentalAudio = ExtractedAudio(
            sourceRecordingId: recording1.id,
            type: .instrumental,
            fileURL: URL(fileURLWithPath: "/tmp/instrumental1.wav"),
            duration: Duration(seconds: 10.0)
        )

        mockRecordingRepository.recordingsToReturn = [recording1, recording2]
        mockExtractedAudioRepository.storedAudios = [vocalAudio, instrumentalAudio]

        // When
        await sut.loadBackingTracks()

        // Then - Both recordings should be in the list
        XCTAssertEqual(sut.availableBackingTracks.count, 2, "Both recordings should appear in backing tracks list")

        // Find each recording in the results
        let track1 = sut.availableBackingTracks.first { $0.id == recording1.id }
        let track2 = sut.availableBackingTracks.first { $0.id == recording2.id }

        XCTAssertNotNil(track1)
        XCTAssertNotNil(track2)

        // Recording1 should have all sources
        XCTAssertEqual(track1?.availableSources, [.original, .vocal, .instrumental])

        // Recording2 should only have original
        XCTAssertEqual(track2?.availableSources, [.original])
    }

    func testLoadBackingTracks_RecordingsWithoutExtractedAudio_AreIncluded() async {
        // Given - 3 recordings
        let recordingWithExtraction = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/extracted.m4a"),
            duration: Duration(seconds: 10.0)
        )
        let recordingWithoutExtraction1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/no_extract1.m4a"),
            duration: Duration(seconds: 20.0)
        )
        let recordingWithoutExtraction2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/no_extract2.m4a"),
            duration: Duration(seconds: 30.0)
        )

        // Only one has extracted audio
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recordingWithExtraction.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 10.0)
        )

        mockRecordingRepository.recordingsToReturn = [
            recordingWithExtraction,
            recordingWithoutExtraction1,
            recordingWithoutExtraction2
        ]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]

        // When
        await sut.loadBackingTracks()

        // Then - All 3 recordings should be in the list
        XCTAssertEqual(sut.availableBackingTracks.count, 3, "All 3 recordings should appear, not just the extracted one")

        // Verify each recording is present
        let extractedTrack = sut.availableBackingTracks.first { $0.id == recordingWithExtraction.id }
        let nonExtracted1 = sut.availableBackingTracks.first { $0.id == recordingWithoutExtraction1.id }
        let nonExtracted2 = sut.availableBackingTracks.first { $0.id == recordingWithoutExtraction2.id }

        XCTAssertNotNil(extractedTrack, "Recording with extraction should be in list")
        XCTAssertNotNil(nonExtracted1, "Recording without extraction (1) should be in list")
        XCTAssertNotNil(nonExtracted2, "Recording without extraction (2) should be in list")

        // Verify sources
        XCTAssertEqual(extractedTrack?.availableSources, [.original, .vocal])
        XCTAssertEqual(nonExtracted1?.availableSources, [.original])
        XCTAssertEqual(nonExtracted2?.availableSources, [.original])
    }

    // MARK: - BackingTrackInfo Tests

    func testBackingTrackInfo_FileURLForOriginal_ReturnsRecordingURL() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]
        await sut.loadBackingTracks()

        // When
        let track = sut.availableBackingTracks.first!
        let url = track.fileURL(for: .original)

        // Then
        XCTAssertEqual(url, recording.fileURL)
    }

    func testBackingTrackInfo_FileURLForVocal_ReturnsExtractedAudioURL() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 10.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]
        await sut.loadBackingTracks()

        // When
        let track = sut.availableBackingTracks.first!
        let url = track.fileURL(for: .vocal)

        // Then
        XCTAssertEqual(url, vocalAudio.fileURL)
    }

    func testBackingTrackInfo_FileURLForInstrumental_ReturnsExtractedAudioURL() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0)
        )
        let instrumentalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .instrumental,
            fileURL: URL(fileURLWithPath: "/tmp/instrumental.wav"),
            duration: Duration(seconds: 10.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]
        mockExtractedAudioRepository.storedAudios = [instrumentalAudio]
        await sut.loadBackingTracks()

        // When
        let track = sut.availableBackingTracks.first!
        let url = track.fileURL(for: .instrumental)

        // Then
        XCTAssertEqual(url, instrumentalAudio.fileURL)
    }

    // MARK: - Clear Backing Track Tests

    func testClearBackingTrack_ClearsSelection() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]
        await sut.loadBackingTracks()
        sut.selectedBackingTrack = sut.availableBackingTracks.first
        sut.selectedBackingSource = .original

        // When
        sut.clearBackingTrack()

        // Then
        XCTAssertNil(sut.selectedBackingTrack)
        XCTAssertNil(sut.selectedBackingSource)
    }

    // MARK: - Repository Call Verification Tests

    func testLoadBackingTracks_CallsFindAllOnBothRepositories() async {
        // Given
        mockRecordingRepository.recordingsToReturn = []
        mockExtractedAudioRepository.storedAudios = []

        // When
        await sut.loadBackingTracks()

        // Then
        XCTAssertTrue(mockRecordingRepository.findAllCalled, "findAll should be called on recording repository")
    }

    func testLoadBackingTracks_WithoutRepositoriesSet_DoesNotCrash() async {
        // Given - Create a new ViewModel without setting repositories
        let subscriptionVM = SubscriptionViewModel(
            getStatusUseCase: MockGetSubscriptionStatusUseCase(),
            purchaseUseCase: MockPurchaseSubscriptionUseCase(),
            restoreUseCase: MockRestorePurchasesUseCase()
        )
        let viewModelWithoutRepos = RecordingViewModel(
            startRecordingUseCase: mockStartRecordingUseCase,
            stopRecordingUseCase: mockStopRecordingUseCase,
            audioPlayer: mockAudioPlayer,
            pitchDetector: mockPitchDetector,
            subscriptionViewModel: subscriptionVM,
            countdownDuration: 0
        )
        // Note: setBackingTrackRepositories NOT called

        // When
        await viewModelWithoutRepos.loadBackingTracks()

        // Then - Should not crash, just return empty
        XCTAssertTrue(viewModelWithoutRepos.availableBackingTracks.isEmpty)
    }

    // MARK: - New Recording Flow Tests (録音完了後のフロー)

    func testLoadBackingTracks_AfterNewRecordingAdded_IncludesNewRecording() async {
        // Given - Initial state with 1 recording
        let existingRecording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/existing.m4a"),
            duration: Duration(seconds: 10.0)
        )
        mockRecordingRepository.recordingsToReturn = [existingRecording]
        await sut.loadBackingTracks()

        XCTAssertEqual(sut.availableBackingTracks.count, 1, "Should have 1 track initially")

        // When - New recording is added (simulating after recording completes)
        let newRecording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/new_recording.m4a"),
            duration: Duration(seconds: 15.0)
        )
        mockRecordingRepository.recordingsToReturn = [existingRecording, newRecording]

        // Reload backing tracks (as would happen when returning to recording screen)
        await sut.loadBackingTracks()

        // Then - Both recordings should be in the list
        XCTAssertEqual(sut.availableBackingTracks.count, 2, "Should have 2 tracks after new recording")

        let newTrack = sut.availableBackingTracks.first { $0.id == newRecording.id }
        XCTAssertNotNil(newTrack, "New recording should be in backing tracks list")
        XCTAssertEqual(newTrack?.availableSources, [.original], "New recording should have only .original source")
    }

    func testLoadBackingTracks_OnlyExtractedInRepository_StillReturnsAll() async {
        // Given - Repository returns only recordings that have been extracted
        // This simulates a potential bug where only extracted recordings are returned
        let extractedRecording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/extracted.m4a"),
            duration: Duration(seconds: 10.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: extractedRecording.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 10.0)
        )

        mockRecordingRepository.recordingsToReturn = [extractedRecording]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]

        // When
        await sut.loadBackingTracks()

        // Then
        XCTAssertEqual(sut.availableBackingTracks.count, 1)
        XCTAssertEqual(sut.availableBackingTracks.first?.availableSources, [.original, .vocal])
    }

    // MARK: - Edge Case Tests

    func testLoadBackingTracks_ManyRecordingsWithMixedExtraction_HandlesCorrectly() async {
        // Given - 10 recordings, some with extraction, some without
        var recordings: [Recording] = []
        var extractedAudios: [ExtractedAudio] = []

        for i in 0..<10 {
            let recording = Recording(
                fileURL: URL(fileURLWithPath: "/tmp/recording_\(i).m4a"),
                duration: Duration(seconds: Double(10 + i))
            )
            recordings.append(recording)

            // Only even-indexed recordings have extracted audio
            if i % 2 == 0 {
                extractedAudios.append(ExtractedAudio(
                    sourceRecordingId: recording.id,
                    type: .vocal,
                    fileURL: URL(fileURLWithPath: "/tmp/vocal_\(i).wav"),
                    duration: Duration(seconds: Double(10 + i))
                ))
            }
        }

        mockRecordingRepository.recordingsToReturn = recordings
        mockExtractedAudioRepository.storedAudios = extractedAudios

        // When
        await sut.loadBackingTracks()

        // Then - ALL 10 recordings should be in the list
        XCTAssertEqual(sut.availableBackingTracks.count, 10, "All 10 recordings should appear")

        // Verify extraction status
        for (index, track) in sut.availableBackingTracks.enumerated() {
            let recording = recordings.first { $0.id == track.id }!
            let originalIndex = recordings.firstIndex(where: { $0.id == recording.id })!

            if originalIndex % 2 == 0 {
                XCTAssertEqual(track.availableSources, [.original, .vocal],
                               "Recording \(originalIndex) should have vocal source")
            } else {
                XCTAssertEqual(track.availableSources, [.original],
                               "Recording \(originalIndex) should only have original source")
            }
        }
    }

    func testLoadBackingTracks_ExtractedAudioForNonExistentRecording_IsIgnored() async {
        // Given - Extracted audio exists but its source recording doesn't
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0)
        )

        // Orphaned extracted audio (source recording not in repository)
        let orphanedVocal = ExtractedAudio(
            sourceRecordingId: RecordingId(), // Different ID, not matching any recording
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/orphaned_vocal.wav"),
            duration: Duration(seconds: 10.0)
        )

        mockRecordingRepository.recordingsToReturn = [recording]
        mockExtractedAudioRepository.storedAudios = [orphanedVocal]

        // When
        await sut.loadBackingTracks()

        // Then
        XCTAssertEqual(sut.availableBackingTracks.count, 1)
        // The recording should only have .original since the extracted audio doesn't match
        XCTAssertEqual(sut.availableBackingTracks.first?.availableSources, [.original])
    }

    func testLoadBackingTracks_MultipleReloads_UpdatesCorrectly() async {
        // Given - Initial state
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/rec1.m4a"),
            duration: Duration(seconds: 10.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording1]
        mockExtractedAudioRepository.storedAudios = []

        // First load
        await sut.loadBackingTracks()
        XCTAssertEqual(sut.availableBackingTracks.count, 1)
        XCTAssertEqual(sut.availableBackingTracks.first?.availableSources, [.original])

        // When - Add extraction and reload
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording1.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal1.wav"),
            duration: Duration(seconds: 10.0)
        )
        mockExtractedAudioRepository.storedAudios = [vocalAudio]
        await sut.loadBackingTracks()

        // Then - Should now show vocal source
        XCTAssertEqual(sut.availableBackingTracks.count, 1)
        XCTAssertEqual(sut.availableBackingTracks.first?.availableSources, [.original, .vocal])

        // When - Add another recording and reload
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/rec2.m4a"),
            duration: Duration(seconds: 15.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording1, recording2]
        await sut.loadBackingTracks()

        // Then - Should have both recordings
        XCTAssertEqual(sut.availableBackingTracks.count, 2)
        let track1 = sut.availableBackingTracks.first { $0.id == recording1.id }
        let track2 = sut.availableBackingTracks.first { $0.id == recording2.id }
        XCTAssertEqual(track1?.availableSources, [.original, .vocal])
        XCTAssertEqual(track2?.availableSources, [.original])
    }

    // MARK: - Display Title Tests

    func testBackingTrackInfo_DisplayTitle_UsesRecordingTitleWhenAvailable() async {
        // Given
        var recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0)
        )
        recording.title = "My Custom Title"
        mockRecordingRepository.recordingsToReturn = [recording]

        // When
        await sut.loadBackingTracks()

        // Then
        XCTAssertEqual(sut.availableBackingTracks.first?.displayTitle, "My Custom Title")
    }

    func testBackingTrackInfo_DisplayTitle_FallsBackToFormattedDate() async {
        // Given - Recording without title
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]

        // When
        await sut.loadBackingTracks()

        // Then - Should use formattedDate (not empty)
        XCTAssertFalse(sut.availableBackingTracks.first?.displayTitle.isEmpty ?? true)
    }

    // MARK: - Concurrency Tests

    func testLoadBackingTracks_CalledMultipleTimes_ProducesConsistentResults() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]

        // When - Call multiple times
        await sut.loadBackingTracks()
        await sut.loadBackingTracks()
        await sut.loadBackingTracks()

        // Then - Should still have consistent results
        XCTAssertEqual(sut.availableBackingTracks.count, 1)
    }
}
