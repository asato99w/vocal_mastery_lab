import XCTest
@testable import VocalisDomain

final class DurationTests: XCTestCase {
    func testInit_PositiveValue() {
        // Given & When
        let duration = Duration(seconds: 120)
        
        // Then
        XCTAssertEqual(duration.seconds, 120)
    }
    
    func testInit_NegativeValue_ClampsToZero() {
        // Given & When
        let duration = Duration(seconds: -10)
        
        // Then
        XCTAssertEqual(duration.seconds, 0)
    }
    
    func testInit_FromDates() {
        // Given
        let start = Date()
        let end = start.addingTimeInterval(60)
        
        // When
        let duration = Duration(from: start, to: end)
        
        // Then
        XCTAssertEqual(duration.seconds, 60, accuracy: 0.001)
    }
    
    func testFormatted() {
        // Given
        let duration = Duration(seconds: 125) // 2:05
        
        // When
        let formatted = duration.formatted
        
        // Then
        XCTAssertEqual(formatted, "02:05")
    }
    
    func testFormatted_ZeroSeconds() {
        // Given
        let duration = Duration(seconds: 0)
        
        // When
        let formatted = duration.formatted
        
        // Then
        XCTAssertEqual(formatted, "00:00")
    }
    
    func testComparable() {
        // Given
        let short = Duration(seconds: 10)
        let long = Duration(seconds: 20)
        
        // When & Then
        XCTAssertTrue(short < long)
        XCTAssertFalse(long < short)
    }
}
