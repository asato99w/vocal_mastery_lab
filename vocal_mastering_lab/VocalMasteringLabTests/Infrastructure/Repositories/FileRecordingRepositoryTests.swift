import XCTest
import VocalisDomain
@testable import VocalMasteringLab

final class FileRecordingRepositoryTests: XCTestCase {

    var sut: FileRecordingRepository!
    var testUserDefaults: UserDefaults!

    override func setUp() {
        super.setUp()
        // Use a unique suite name for test isolation
        testUserDefaults = UserDefaults(suiteName: "FileRecordingRepositoryTests")
        testUserDefaults.removePersistentDomain(forName: "FileRecordingRepositoryTests")
        sut = FileRecordingRepository(userDefaults: testUserDefaults)
    }

    override func tearDown() {
        testUserDefaults.removePersistentDomain(forName: "FileRecordingRepositoryTests")
        testUserDefaults = nil
        sut = nil
        super.tearDown()
    }

    // MARK: - Update Tests

    func testUpdate_ExistingRecording_UpdatesTitle() async throws {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0),
            scaleSettings: ScaleSettings.mvpDefault
        )
        try await sut.save(recording)

        // When
        var updatedRecording = recording
        updatedRecording.title = "新しいタイトル"
        try await sut.update(updatedRecording)

        // Then
        let result = try await sut.findById(recording.id)
        XCTAssertEqual(result?.title, "新しいタイトル")
    }

    func testUpdate_NonExistentRecording_ThrowsError() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0),
            scaleSettings: ScaleSettings.mvpDefault
        )

        // When/Then
        do {
            try await sut.update(recording)
            XCTFail("Expected error to be thrown")
        } catch {
            XCTAssertTrue(error is RecordingRepositoryError)
        }
    }

    func testUpdate_PreservesOtherRecordings() async throws {
        // Given - create actual files for findAll to not filter them out
        let url1 = FileManager.default.temporaryDirectory.appendingPathComponent("test1_\(UUID().uuidString).m4a")
        let url2 = FileManager.default.temporaryDirectory.appendingPathComponent("test2_\(UUID().uuidString).m4a")
        FileManager.default.createFile(atPath: url1.path, contents: Data())
        FileManager.default.createFile(atPath: url2.path, contents: Data())
        defer {
            try? FileManager.default.removeItem(at: url1)
            try? FileManager.default.removeItem(at: url2)
        }

        let recording1 = Recording(
            fileURL: url1,
            duration: Duration(seconds: 10.0),
            scaleSettings: ScaleSettings.mvpDefault
        )
        let recording2 = Recording(
            fileURL: url2,
            duration: Duration(seconds: 15.0),
            scaleSettings: ScaleSettings.mvpDefault
        )
        try await sut.save(recording1)
        try await sut.save(recording2)

        // When - update only recording1
        var updatedRecording1 = recording1
        updatedRecording1.title = "更新済み"
        try await sut.update(updatedRecording1)

        // Then - recording2 should be unchanged
        let allRecordings = try await sut.findAll()
        XCTAssertEqual(allRecordings.count, 2)

        let result2 = try await sut.findById(recording2.id)
        XCTAssertNil(result2?.title) // recording2 should still have no title
    }

    func testUpdate_ClearsTitle_WhenSetToNil() async throws {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0),
            scaleSettings: ScaleSettings.mvpDefault,
            title: "初期タイトル"
        )
        try await sut.save(recording)

        // When
        var updatedRecording = recording
        updatedRecording.title = nil
        try await sut.update(updatedRecording)

        // Then
        let result = try await sut.findById(recording.id)
        XCTAssertNil(result?.title)
    }

    // MARK: - Save and FindAll Tests (existing functionality verification)

    func testSave_NewRecording_PersistsToUserDefaults() async throws {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0),
            scaleSettings: ScaleSettings.mvpDefault,
            title: "テスト録音"
        )

        // When
        try await sut.save(recording)

        // Then
        let result = try await sut.findById(recording.id)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.title, "テスト録音")
    }

    func testFindAll_ReturnsAllRecordings() async throws {
        // Given - create actual files for findAll to not filter them out
        let url1 = FileManager.default.temporaryDirectory.appendingPathComponent("test1_\(UUID().uuidString).m4a")
        let url2 = FileManager.default.temporaryDirectory.appendingPathComponent("test2_\(UUID().uuidString).m4a")
        FileManager.default.createFile(atPath: url1.path, contents: Data())
        FileManager.default.createFile(atPath: url2.path, contents: Data())
        defer {
            try? FileManager.default.removeItem(at: url1)
            try? FileManager.default.removeItem(at: url2)
        }

        let recording1 = Recording(
            fileURL: url1,
            duration: Duration(seconds: 10.0),
            scaleSettings: ScaleSettings.mvpDefault,
            title: "録音1"
        )
        let recording2 = Recording(
            fileURL: url2,
            duration: Duration(seconds: 15.0),
            scaleSettings: ScaleSettings.mvpDefault,
            title: "録音2"
        )
        try await sut.save(recording1)
        try await sut.save(recording2)

        // When
        let results = try await sut.findAll()

        // Then
        XCTAssertEqual(results.count, 2)
    }
}
