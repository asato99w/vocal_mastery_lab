import XCTest
@testable import VocalisDomain

final class KeyProgressionPatternTests: XCTestCase {

    // MARK: - Display Name Tests

    func testAscendingOnly_DisplayName() {
        // Given
        let pattern = KeyProgressionPattern.ascendingOnly

        // When
        let displayName = pattern.displayName

        // Then
        XCTAssertEqual(displayName, "上昇のみ")
    }

    func testDescendingOnly_DisplayName() {
        // Given
        let pattern = KeyProgressionPattern.descendingOnly

        // When
        let displayName = pattern.displayName

        // Then
        XCTAssertEqual(displayName, "下降のみ")
    }

    func testAscendingThenDescending_DisplayName() {
        // Given
        let pattern = KeyProgressionPattern.ascendingThenDescending

        // When
        let displayName = pattern.displayName

        // Then
        XCTAssertEqual(displayName, "上昇→下降")
    }

    func testDescendingThenAscending_DisplayName() {
        // Given
        let pattern = KeyProgressionPattern.descendingThenAscending

        // When
        let displayName = pattern.displayName

        // Then
        XCTAssertEqual(displayName, "下降→上昇")
    }

    // MARK: - Visibility Tests

    func testAscendingOnly_ShowsAscendingCount() {
        let pattern = KeyProgressionPattern.ascendingOnly
        XCTAssertTrue(pattern.showsAscendingCount)
        XCTAssertFalse(pattern.showsDescendingCount)
    }

    func testDescendingOnly_ShowsDescendingCount() {
        let pattern = KeyProgressionPattern.descendingOnly
        XCTAssertFalse(pattern.showsAscendingCount)
        XCTAssertTrue(pattern.showsDescendingCount)
    }

    func testAscendingThenDescending_ShowsBothCounts() {
        let pattern = KeyProgressionPattern.ascendingThenDescending
        XCTAssertTrue(pattern.showsAscendingCount)
        XCTAssertTrue(pattern.showsDescendingCount)
    }

    func testDescendingThenAscending_ShowsBothCounts() {
        let pattern = KeyProgressionPattern.descendingThenAscending
        XCTAssertTrue(pattern.showsAscendingCount)
        XCTAssertTrue(pattern.showsDescendingCount)
    }

    // MARK: - Codable Tests

    func testCodable_RoundTrip() throws {
        // Given
        let patterns: [KeyProgressionPattern] = [
            .ascendingOnly,
            .descendingOnly,
            .ascendingThenDescending,
            .descendingThenAscending
        ]

        for pattern in patterns {
            // When
            let encoded = try JSONEncoder().encode(pattern)
            let decoded = try JSONDecoder().decode(KeyProgressionPattern.self, from: encoded)

            // Then
            XCTAssertEqual(pattern, decoded)
        }
    }
}
