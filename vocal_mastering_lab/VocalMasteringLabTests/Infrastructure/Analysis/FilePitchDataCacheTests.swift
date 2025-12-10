import XCTest
@testable import VocalMasteringLab
@testable import VocalisDomain

final class FilePitchDataCacheTests: XCTestCase {
    var sut: FilePitchDataCache!
    var testDirectory: URL!

    override func setUp() {
        super.setUp()
        // Use temporary directory for tests
        testDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FilePitchDataCacheTests")
        sut = FilePitchDataCache(cacheDirectory: testDirectory)
    }

    override func tearDown() {
        // Clean up test directory
        try? FileManager.default.removeItem(at: testDirectory)
        sut = nil
        testDirectory = nil
        super.tearDown()
    }

    // MARK: - Test: Get returns nil for non-existent data

    func testGet_WhenDataNotExists_ReturnsNil() {
        // Given
        let recordingId = RecordingId()

        // When
        let result = sut.get(recordingId)

        // Then
        XCTAssertNil(result)
    }

    // MARK: - Test: Set and Get roundtrip

    func testSetAndGet_WhenDataSaved_ReturnsCorrectData() throws {
        // Given
        let recordingId = RecordingId()
        let pitchData = createSamplePitchData()

        // When
        sut.set(recordingId, pitchData: pitchData)
        let result = sut.get(recordingId)

        // Then
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.timeStamps, pitchData.timeStamps)
        XCTAssertEqual(result?.frequencies, pitchData.frequencies)
        XCTAssertEqual(result?.confidences, pitchData.confidences)
    }

    // MARK: - Test: Data persists across instances

    func testPersistence_WhenNewCacheInstance_DataStillExists() throws {
        // Given
        let recordingId = RecordingId()
        let pitchData = createSamplePitchData()
        sut.set(recordingId, pitchData: pitchData)

        // When: Create new cache instance pointing to same directory
        let newCache = FilePitchDataCache(cacheDirectory: testDirectory)
        let result = newCache.get(recordingId)

        // Then
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.timeStamps, pitchData.timeStamps)
    }

    // MARK: - Test: Delete removes data

    func testDelete_WhenCalled_RemovesData() throws {
        // Given
        let recordingId = RecordingId()
        let pitchData = createSamplePitchData()
        sut.set(recordingId, pitchData: pitchData)

        // Verify data exists
        XCTAssertNotNil(sut.get(recordingId))

        // When
        sut.delete(recordingId)

        // Then
        XCTAssertNil(sut.get(recordingId))
    }

    // MARK: - Test: ClearAll removes all data

    func testClearAll_WhenCalled_RemovesAllData() throws {
        // Given
        let id1 = RecordingId()
        let id2 = RecordingId()
        let pitchData = createSamplePitchData()

        sut.set(id1, pitchData: pitchData)
        sut.set(id2, pitchData: pitchData)

        // When
        sut.clearAll()

        // Then
        XCTAssertNil(sut.get(id1))
        XCTAssertNil(sut.get(id2))
    }

    // MARK: - Test: Multiple recordings stored separately

    func testMultipleRecordings_WhenStored_EachHasCorrectData() throws {
        // Given
        let id1 = RecordingId()
        let id2 = RecordingId()

        let pitchData1 = PitchAnalysisData(
            timeStamps: [0.0, 0.1],
            frequencies: [440.0, 441.0],
            confidences: [0.9, 0.9],
            targetNotes: [nil, nil]
        )

        let pitchData2 = PitchAnalysisData(
            timeStamps: [0.0, 0.1, 0.2],
            frequencies: [220.0, 221.0, 222.0],
            confidences: [0.8, 0.8, 0.8],
            targetNotes: [nil, nil, nil]
        )

        // When
        sut.set(id1, pitchData: pitchData1)
        sut.set(id2, pitchData: pitchData2)

        // Then
        let result1 = sut.get(id1)
        let result2 = sut.get(id2)

        XCTAssertEqual(result1?.frequencies, [440.0, 441.0])
        XCTAssertEqual(result2?.frequencies, [220.0, 221.0, 222.0])
    }

    // MARK: - Test: Exists returns correct status

    func testExists_WhenDataExists_ReturnsTrue() throws {
        // Given
        let recordingId = RecordingId()
        let pitchData = createSamplePitchData()
        sut.set(recordingId, pitchData: pitchData)

        // When
        let exists = sut.exists(recordingId)

        // Then
        XCTAssertTrue(exists)
    }

    func testExists_WhenDataNotExists_ReturnsFalse() {
        // Given
        let recordingId = RecordingId()

        // When
        let exists = sut.exists(recordingId)

        // Then
        XCTAssertFalse(exists)
    }

    // MARK: - Test: Overwrite existing data

    func testSet_WhenDataAlreadyExists_OverwritesData() throws {
        // Given
        let recordingId = RecordingId()

        let originalData = PitchAnalysisData(
            timeStamps: [0.0],
            frequencies: [440.0],
            confidences: [0.9],
            targetNotes: [nil]
        )

        let newData = PitchAnalysisData(
            timeStamps: [0.0, 0.1],
            frequencies: [880.0, 881.0],
            confidences: [0.95, 0.95],
            targetNotes: [nil, nil]
        )

        sut.set(recordingId, pitchData: originalData)

        // When
        sut.set(recordingId, pitchData: newData)

        // Then
        let result = sut.get(recordingId)
        XCTAssertEqual(result?.frequencies, [880.0, 881.0])
    }

    // MARK: - Helper Methods

    private func createSamplePitchData() -> PitchAnalysisData {
        return PitchAnalysisData(
            timeStamps: [0.0, 0.05, 0.10, 0.15],
            frequencies: [261.6, 262.3, 261.9, 263.0],
            confidences: [0.85, 0.92, 0.88, 0.90],
            targetNotes: [nil, nil, nil, nil]
        )
    }
}
