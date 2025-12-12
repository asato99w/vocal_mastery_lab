import XCTest
import VocalisDomain
@testable import VocalMasteryLab

final class AnalysisCacheTests: XCTestCase {
    var sut: AnalysisCache!
    var testRecordingId: RecordingId!
    var testAnalysisResult: AnalysisResult!

    override func setUp() {
        super.setUp()
        sut = AnalysisCache(maxCacheSize: 3)
        testRecordingId = RecordingId()
        testAnalysisResult = createTestAnalysisResult()
    }

    override func tearDown() {
        sut = nil
        testRecordingId = nil
        testAnalysisResult = nil
        super.tearDown()
    }

    // MARK: - Test Helpers

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

    // MARK: - Basic Operations Tests

    func testGetFromEmptyCache_ReturnsNil() {
        // When: Getting from empty cache
        let result = sut.get(testRecordingId)

        // Then: Should return nil
        XCTAssertNil(result)
    }

    func testSetAndGet_StoresAndRetrievesResult() {
        // When: Setting and getting result
        sut.set(testRecordingId, result: testAnalysisResult)
        let result = sut.get(testRecordingId)

        // Then: Should retrieve the same result
        XCTAssertNotNil(result)
        XCTAssertEqual(result, testAnalysisResult)
    }

    func testClear_RemovesAllEntries() {
        // Given: Cache with data
        sut.set(testRecordingId, result: testAnalysisResult)
        XCTAssertEqual(sut.count, 1)

        // When: Clearing cache
        sut.clear()

        // Then: Cache should be empty
        XCTAssertEqual(sut.count, 0)
        XCTAssertNil(sut.get(testRecordingId))
    }

    // MARK: - LRU Eviction Tests

    func testLRUEviction_RemovesOldestEntry() {
        // Given: Cache with max size 3
        let id1 = RecordingId()
        let id2 = RecordingId()
        let id3 = RecordingId()
        let id4 = RecordingId()

        // When: Adding 4 entries (exceeds max size)
        sut.set(id1, result: testAnalysisResult)
        sut.set(id2, result: testAnalysisResult)
        sut.set(id3, result: testAnalysisResult)
        sut.set(id4, result: testAnalysisResult)

        // Then: First entry should be evicted
        XCTAssertEqual(sut.count, 3)
        XCTAssertNil(sut.get(id1))
        XCTAssertNotNil(sut.get(id2))
        XCTAssertNotNil(sut.get(id3))
        XCTAssertNotNil(sut.get(id4))
    }

    func testLRUEviction_AccessUpdatesOrder() {
        // Given: Cache with 3 entries
        let id1 = RecordingId()
        let id2 = RecordingId()
        let id3 = RecordingId()

        sut.set(id1, result: testAnalysisResult)
        sut.set(id2, result: testAnalysisResult)
        sut.set(id3, result: testAnalysisResult)

        // When: Accessing id1 (making it most recently used)
        _ = sut.get(id1)

        // And: Adding a new entry
        let id4 = RecordingId()
        sut.set(id4, result: testAnalysisResult)

        // Then: id2 should be evicted (oldest unused), id1 should remain
        XCTAssertEqual(sut.count, 3)
        XCTAssertNotNil(sut.get(id1))
        XCTAssertNil(sut.get(id2))
        XCTAssertNotNil(sut.get(id3))
        XCTAssertNotNil(sut.get(id4))
    }

    func testUpdateExistingEntry_DoesNotIncreaseCount() {
        // Given: Cache with one entry
        sut.set(testRecordingId, result: testAnalysisResult)
        XCTAssertEqual(sut.count, 1)

        // When: Updating the same entry
        let newResult = createTestAnalysisResult()
        sut.set(testRecordingId, result: newResult)

        // Then: Count should remain 1
        XCTAssertEqual(sut.count, 1)
        XCTAssertEqual(sut.get(testRecordingId), newResult)
    }

    // MARK: - Thread Safety Tests

    func testConcurrentAccess_ThreadSafe() {
        let expectation = XCTestExpectation(description: "Concurrent operations complete")
        expectation.expectedFulfillmentCount = 100

        // When: Performing concurrent operations
        DispatchQueue.concurrentPerform(iterations: 100) { index in
            let id = RecordingId()
            sut.set(id, result: testAnalysisResult)
            _ = sut.get(id)
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 5.0)

        // Then: Should not crash and maintain consistency
        XCTAssertTrue(sut.count <= 3) // Max cache size
    }
}
