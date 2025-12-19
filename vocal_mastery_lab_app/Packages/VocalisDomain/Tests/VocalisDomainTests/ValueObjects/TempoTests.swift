import XCTest
@testable import VocalisDomain

final class TempoTests: XCTestCase {
    func testInit_ValidValue_Success() throws {
        // Given & When
        let tempo = try Tempo(secondsPerNote: 1.0)
        
        // Then
        XCTAssertEqual(tempo.secondsPerNote, 1.0)
    }
    
    func testInit_ZeroValue_ThrowsError() {
        // Given & When & Then
        XCTAssertThrowsError(try Tempo(secondsPerNote: 0.0)) { error in
            XCTAssertTrue(error is TempoError)
        }
    }
    
    func testInit_NegativeValue_ThrowsError() {
        // Given & When & Then
        XCTAssertThrowsError(try Tempo(secondsPerNote: -1.0)) { error in
            XCTAssertTrue(error is TempoError)
        }
    }
    
    func testStandardTempo() {
        // Given & When
        let standard = Tempo.standard
        
        // Then
        XCTAssertEqual(standard.secondsPerNote, 1.0)
    }
}
