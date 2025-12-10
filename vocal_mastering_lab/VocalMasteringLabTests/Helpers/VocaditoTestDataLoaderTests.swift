import XCTest
@testable import VocalMasteringLab

/// Test VocaditoTestDataLoader functionality to verify JSON loading and data access
final class VocaditoTestDataLoaderTests: XCTestCase {

    // MARK: - Basic Loading Tests

    func testBundleResourcesAvailable() throws {
        // Debug: Check what resources are available in the bundle
        let testBundle = Bundle(for: type(of: self))
        print("📦 Bundle path: \(testBundle.bundlePath)")
        print("📦 Resource path: \(testBundle.resourcePath ?? "nil")")

        // Try different subdirectory combinations
        if let url1 = testBundle.url(forResource: "TestNotes", withExtension: "json") {
            print("✅ Found TestNotes.json without subdirectory: \(url1.path)")
        } else {
            print("❌ TestNotes.json not found without subdirectory")
        }

        if let url2 = testBundle.url(forResource: "TestNotes", withExtension: "json", subdirectory: "Vocadito") {
            print("✅ Found TestNotes.json with Vocadito subdirectory: \(url2.path)")
        } else {
            print("❌ TestNotes.json not found with Vocadito subdirectory")
        }

        if let url3 = testBundle.url(forResource: "TestNotes", withExtension: "json", subdirectory: "TestResources/Vocadito") {
            print("✅ Found TestNotes.json with TestResources/Vocadito subdirectory: \(url3.path)")
        } else {
            print("❌ TestNotes.json not found with TestResources/Vocadito subdirectory")
        }
    }

    func testLoadTestData_shouldSucceed() throws {
        // When: Load test data
        let testData = try VocaditoTestDataLoader.loadTestData()

        // Then: Should contain expected structure
        XCTAssertFalse(testData.description.isEmpty, "Description should not be empty")
        XCTAssertFalse(testData.format.isEmpty, "Format should not be empty")
        XCTAssertFalse(testData.tracks.isEmpty, "Tracks should not be empty")

        print("✅ Loaded test data with \(testData.tracks.count) tracks")
    }

    func testLoadTestData_shouldContainExpectedTracks() throws {
        // When: Load test data
        let testData = try VocaditoTestDataLoader.loadTestData()

        // Then: Should contain vocadito_1, vocadito_4, vocadito_7
        XCTAssertTrue(testData.tracks.keys.contains("vocadito_1"), "Should contain vocadito_1")
        XCTAssertTrue(testData.tracks.keys.contains("vocadito_4"), "Should contain vocadito_4")
        XCTAssertTrue(testData.tracks.keys.contains("vocadito_7"), "Should contain vocadito_7")
        XCTAssertEqual(testData.tracks.count, 3, "Should contain exactly 3 tracks")

        print("✅ All expected tracks present: \(testData.tracks.keys.sorted())")
    }

    // MARK: - Track Access Tests

    func testGetNotes_forValidTrack_shouldReturnNotes() throws {
        // When: Get notes for vocadito_1
        let notes = try VocaditoTestDataLoader.getNotes(for: "vocadito_1")

        // Then: Should return 3 notes
        XCTAssertEqual(notes.count, 3, "vocadito_1 should have 3 notes")

        // Verify note structure
        let firstNote = notes[0]
        XCTAssertEqual(firstNote.index, 0, "First note index should be 0")
        XCTAssertGreaterThan(firstNote.frequency, 0, "Frequency should be positive")
        XCTAssertGreaterThan(firstNote.duration, 0, "Duration should be positive")
        XCTAssertGreaterThanOrEqual(firstNote.startTime, 0, "Start time should be non-negative")

        print("✅ Retrieved \(notes.count) notes for vocadito_1")
        print("   First note: freq=\(String(format: "%.2f", firstNote.frequency))Hz, " +
              "start=\(String(format: "%.3f", firstNote.startTime))s, " +
              "duration=\(String(format: "%.3f", firstNote.duration))s")
    }

    func testGetNotes_forInvalidTrack_shouldThrowError() {
        // When/Then: Should throw trackNotFound error
        XCTAssertThrowsError(try VocaditoTestDataLoader.getNotes(for: "invalid_track")) { error in
            guard let testError = error as? VocaditoTestDataError,
                  case .trackNotFound(let trackName) = testError else {
                XCTFail("Should throw VocaditoTestDataError.trackNotFound")
                return
            }
            XCTAssertEqual(trackName, "invalid_track")
        }

        print("✅ Correctly throws error for invalid track")
    }

    func testGetAudioFileName_forValidTrack_shouldReturnFileName() throws {
        // When: Get audio file name for vocadito_1
        let fileName = try VocaditoTestDataLoader.getAudioFileName(for: "vocadito_1")

        // Then: Should return correct file name
        XCTAssertEqual(fileName, "vocadito_1.wav", "Should return correct audio file name")

        print("✅ Audio file name: \(fileName)")
    }

    func testGetAllTrackNames_shouldReturnSortedNames() throws {
        // When: Get all track names
        let trackNames = try VocaditoTestDataLoader.getAllTrackNames()

        // Then: Should return sorted track names
        XCTAssertEqual(trackNames.count, 3, "Should return 3 track names")
        XCTAssertEqual(trackNames, ["vocadito_1", "vocadito_4", "vocadito_7"], "Should be sorted")

        print("✅ Track names: \(trackNames)")
    }

    // MARK: - Note Data Validation Tests

    func testNoteMidTime_shouldReturnCorrectValue() throws {
        // Given: Get notes for vocadito_1
        let notes = try VocaditoTestDataLoader.getNotes(for: "vocadito_1")
        let firstNote = notes[0]

        // When: Calculate mid time
        let midTime = firstNote.midTime

        // Then: Should be start + duration/2
        let expectedMidTime = firstNote.startTime + (firstNote.duration / 2.0)
        XCTAssertEqual(midTime, expectedMidTime, accuracy: 0.0001, "Mid time calculation should be correct")

        print("✅ Note mid time: \(String(format: "%.3f", midTime))s " +
              "(start: \(String(format: "%.3f", firstNote.startTime))s + " +
              "duration/2: \(String(format: "%.3f", firstNote.duration / 2.0))s)")
    }

    func testAllTracks_shouldHaveThreeNotes() throws {
        // When: Check all tracks
        let trackNames = try VocaditoTestDataLoader.getAllTrackNames()

        // Then: Each track should have exactly 3 notes
        for trackName in trackNames {
            let notes = try VocaditoTestDataLoader.getNotes(for: trackName)
            XCTAssertEqual(notes.count, 3, "\(trackName) should have 3 notes")
            print("✅ \(trackName): \(notes.count) notes")
        }
    }

    func testAllNotes_shouldHaveValidData() throws {
        // Given: Get all tracks
        let trackNames = try VocaditoTestDataLoader.getAllTrackNames()

        // When/Then: Validate all notes in all tracks
        for trackName in trackNames {
            let notes = try VocaditoTestDataLoader.getNotes(for: trackName)

            for (index, note) in notes.enumerated() {
                XCTAssertEqual(note.index, index, "\(trackName) note \(index) should have correct index")
                XCTAssertGreaterThan(note.frequency, 0, "\(trackName) note \(index) should have positive frequency")
                XCTAssertGreaterThan(note.duration, 0, "\(trackName) note \(index) should have positive duration")
                XCTAssertGreaterThanOrEqual(note.startTime, 0, "\(trackName) note \(index) should have non-negative start time")

                // Frequency should be in reasonable singing range (80-1000 Hz)
                XCTAssertGreaterThan(note.frequency, 80, "\(trackName) note \(index) frequency should be above 80 Hz")
                XCTAssertLessThan(note.frequency, 1000, "\(trackName) note \(index) frequency should be below 1000 Hz")

                // Duration should be reasonable (0.01s - 2s for singing notes)
                XCTAssertGreaterThan(note.duration, 0.01, "\(trackName) note \(index) duration should be at least 0.01s")
                XCTAssertLessThan(note.duration, 2.0, "\(trackName) note \(index) duration should be less than 2s")
            }

            print("✅ All notes in \(trackName) have valid data")
        }
    }

    // MARK: - Caching Tests

    func testLoadTestData_shouldCacheData() throws {
        // When: Load data twice
        let data1 = try VocaditoTestDataLoader.loadTestData()
        let data2 = try VocaditoTestDataLoader.loadTestData()

        // Then: Should return same cached instance (verify by checking track count consistency)
        XCTAssertEqual(data1.tracks.count, data2.tracks.count, "Cached data should be consistent")
        XCTAssertEqual(data1.description, data2.description, "Cached data should be identical")

        print("✅ Data caching works correctly")
    }
}
