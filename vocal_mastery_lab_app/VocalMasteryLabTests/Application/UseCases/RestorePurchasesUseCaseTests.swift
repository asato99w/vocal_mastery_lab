//
//  RestorePurchasesUseCaseTests.swift
//  VocalMasteryLabTests
//
//  Tests for RestorePurchasesUseCase
//

import XCTest
@testable import VocalMasteryLab

final class RestorePurchasesUseCaseTests: XCTestCase {

    var useCase: RestorePurchasesUseCase!
    var mockRepository: MockSubscriptionRepository!

    override func setUp() {
        super.setUp()
        mockRepository = MockSubscriptionRepository()
        useCase = RestorePurchasesUseCase(repository: mockRepository)
    }

    override func tearDown() {
        useCase = nil
        mockRepository = nil
        super.tearDown()
    }

    // MARK: - Basic Restore Tests

    func testExecuteRestoresExistingPurchases() async throws {
        // Given: User has previous purchases
        mockRepository.mockRestoreResult = .success(())

        // When: Execute restore
        try await useCase.execute()

        // Then: Should call repository restore
        XCTAssertTrue(mockRepository.didCallRestorePurchases)
    }

    func testExecuteSucceedsWhenPurchasesFound() async throws {
        // Given: Purchases exist
        mockRepository.mockRestoreResult = .success(())

        // When: Execute restore
        // Then: Should succeed without error
        try await useCase.execute()
    }

    // MARK: - Error Handling Tests

    func testExecuteThrowsErrorWhenNoPurchases() async {
        // Given: No previous purchases
        mockRepository.mockRestoreResult = .failure(SubscriptionError.noPurchasesToRestore)

        // When/Then: Should throw error
        do {
            try await useCase.execute()
            XCTFail("Should have thrown error")
        } catch let error as SubscriptionError {
            XCTAssertEqual(error, SubscriptionError.noPurchasesToRestore)
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    func testExecuteThrowsErrorWhenNetworkFails() async {
        // Given: Network error occurs
        mockRepository.mockRestoreResult = .failure(SubscriptionError.networkError)

        // When/Then: Should throw error
        do {
            try await useCase.execute()
            XCTFail("Should have thrown error")
        } catch let error as SubscriptionError {
            XCTAssertEqual(error, SubscriptionError.networkError)
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    func testExecuteThrowsErrorWhenRestoreFails() async {
        // Given: Restore fails for some reason
        mockRepository.mockRestoreResult = .failure(SubscriptionError.unknown)

        // When/Then: Should propagate error
        do {
            try await useCase.execute()
            XCTFail("Should have thrown error")
        } catch let error as SubscriptionError {
            XCTAssertEqual(error, SubscriptionError.unknown)
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    // MARK: - Integration Tests

    func testExecuteCanBeCalledMultipleTimes() async throws {
        // Given: Successful restore
        mockRepository.mockRestoreResult = .success(())

        // When: Execute restore multiple times
        try await useCase.execute()
        try await useCase.execute()

        // Then: Should succeed both times
        XCTAssertTrue(mockRepository.didCallRestorePurchases)
    }
}
