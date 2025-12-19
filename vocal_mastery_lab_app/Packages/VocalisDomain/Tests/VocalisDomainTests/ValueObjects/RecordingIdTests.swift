import XCTest
@testable import VocalisDomain

final class RecordingIdTests: XCTestCase {

    // MARK: - Initialization Tests

    func testInit_DefaultInitializer_GeneratesUUID() {
        let id = RecordingId()

        XCTAssertNotNil(id.value)
        XCTAssertFalse(id.value.uuidString.isEmpty)
    }

    func testInit_WithUUID_StoresValue() {
        let uuid = UUID()
        let id = RecordingId(value: uuid)

        XCTAssertEqual(id.value, uuid)
    }

    func testInit_MultipleCalls_GeneratesDifferentUUIDs() {
        let id1 = RecordingId()
        let id2 = RecordingId()

        XCTAssertNotEqual(id1.value, id2.value)
    }

    // MARK: - Equatable Tests

    func testEquatable_SameUUID_Equal() {
        let uuid = UUID()
        let id1 = RecordingId(value: uuid)
        let id2 = RecordingId(value: uuid)

        XCTAssertEqual(id1, id2)
    }

    func testEquatable_DifferentUUID_NotEqual() {
        let id1 = RecordingId()
        let id2 = RecordingId()

        XCTAssertNotEqual(id1, id2)
    }

    // MARK: - Hashable Tests

    func testHashable_SameUUID_SameHash() {
        let uuid = UUID()
        let id1 = RecordingId(value: uuid)
        let id2 = RecordingId(value: uuid)

        XCTAssertEqual(id1.hashValue, id2.hashValue)
    }

    func testHashable_CanBeUsedInSet() {
        let id1 = RecordingId()
        let id2 = RecordingId()
        let id3 = RecordingId(value: id1.value)

        let set: Set<RecordingId> = [id1, id2, id3]

        XCTAssertEqual(set.count, 2) // id1 and id3 are equal, so only 2 unique values
        XCTAssertTrue(set.contains(id1))
        XCTAssertTrue(set.contains(id2))
    }

    // MARK: - Identifiable Tests

    func testIdentifiable_IDProperty_ReturnsValue() {
        let uuid = UUID()
        let recordingId = RecordingId(value: uuid)

        XCTAssertEqual(recordingId.id, uuid)
    }

    // MARK: - Codable Tests

    func testCodable_Encode_Success() throws {
        let uuid = UUID()
        let id = RecordingId(value: uuid)

        let encoder = JSONEncoder()
        let data = try encoder.encode(id)

        XCTAssertFalse(data.isEmpty)
    }

    func testCodable_Decode_Success() throws {
        let uuid = UUID()
        let originalId = RecordingId(value: uuid)

        let encoder = JSONEncoder()
        let data = try encoder.encode(originalId)

        let decoder = JSONDecoder()
        let decodedId = try decoder.decode(RecordingId.self, from: data)

        XCTAssertEqual(decodedId, originalId)
        XCTAssertEqual(decodedId.value, uuid)
    }

    func testCodable_RoundTrip_PreservesValue() throws {
        let originalId = RecordingId()

        let encoder = JSONEncoder()
        let data = try encoder.encode(originalId)

        let decoder = JSONDecoder()
        let decodedId = try decoder.decode(RecordingId.self, from: data)

        XCTAssertEqual(decodedId, originalId)
    }
}
