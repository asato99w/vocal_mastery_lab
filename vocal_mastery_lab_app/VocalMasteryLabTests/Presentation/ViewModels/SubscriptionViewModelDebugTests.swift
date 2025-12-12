import XCTest
import Combine
import VocalisDomain
import SubscriptionDomain
@testable import VocalMasteryLab

@MainActor
final class SubscriptionViewModelDebugTests: XCTestCase {

    var sut: SubscriptionViewModel!
    var mockGetStatusUseCase: MockGetSubscriptionStatusUseCase!
    var mockPurchaseUseCase: MockPurchaseSubscriptionUseCase!
    var mockRestoreUseCase: MockRestorePurchasesUseCase!
    var cancellables: Set<AnyCancellable>!

    override func setUp() async throws {
        try await super.setUp()
        mockGetStatusUseCase = MockGetSubscriptionStatusUseCase()
        mockPurchaseUseCase = MockPurchaseSubscriptionUseCase()
        mockRestoreUseCase = MockRestorePurchasesUseCase()
        sut = SubscriptionViewModel(
            getStatusUseCase: mockGetStatusUseCase,
            purchaseUseCase: mockPurchaseUseCase,
            restoreUseCase: mockRestoreUseCase
        )
        cancellables = []
    }

    override func tearDown() async throws {
        cancellables = nil
        sut = nil
        mockRestoreUseCase = nil
        mockPurchaseUseCase = nil
        mockGetStatusUseCase = nil
        try await super.tearDown()
    }

    // MARK: - Debug Tier Tests

    func testSetDebugTier_Free_SetsCorrectStatus() {
        // When
        sut.setDebugTier(.free)

        // Then
        XCTAssertNotNil(sut.currentStatus)
        XCTAssertEqual(sut.currentStatus?.tier, .free)
        XCTAssertFalse(sut.currentStatus?.isActive ?? true)
        XCTAssertNil(sut.currentStatus?.expirationDate)
        XCTAssertNil(sut.currentStatus?.purchaseDate)
        XCTAssertFalse(sut.currentStatus?.willAutoRenew ?? true)
    }

    func testSetDebugTier_Premium_SetsCorrectStatus() {
        // When
        sut.setDebugTier(.premium)

        // Then
        XCTAssertNotNil(sut.currentStatus)
        XCTAssertEqual(sut.currentStatus?.tier, .premium)
        XCTAssertTrue(sut.currentStatus?.isActive ?? false)
        XCTAssertNotNil(sut.currentStatus?.expirationDate)
        XCTAssertNotNil(sut.currentStatus?.purchaseDate)
        XCTAssertFalse(sut.currentStatus?.willAutoRenew ?? true)
    }

    func testSetDebugTier_PremiumPlus_SetsCorrectStatus() {
        // When
        sut.setDebugTier(.premiumPlus)

        // Then
        XCTAssertNotNil(sut.currentStatus)
        XCTAssertEqual(sut.currentStatus?.tier, .premiumPlus)
        XCTAssertTrue(sut.currentStatus?.isActive ?? false)
        XCTAssertNotNil(sut.currentStatus?.expirationDate)
        XCTAssertNotNil(sut.currentStatus?.purchaseDate)
        XCTAssertFalse(sut.currentStatus?.willAutoRenew ?? true)
    }

    func testSetDebugTier_MultipleTimes_RetainsLatestValue() {
        // When
        sut.setDebugTier(.free)
        XCTAssertEqual(sut.currentStatus?.tier, .free)

        sut.setDebugTier(.premium)
        XCTAssertEqual(sut.currentStatus?.tier, .premium)

        sut.setDebugTier(.premiumPlus)
        XCTAssertEqual(sut.currentStatus?.tier, .premiumPlus)

        sut.setDebugTier(.free)
        XCTAssertEqual(sut.currentStatus?.tier, .free)
    }

    func testSetDebugTier_PublishesChanges() {
        // Given
        var receivedStatuses: [SubscriptionStatus?] = []
        sut.$currentStatus
            .sink { status in
                receivedStatuses.append(status)
            }
            .store(in: &cancellables)

        // When
        sut.setDebugTier(.premium)

        // Then
        XCTAssertEqual(receivedStatuses.count, 2) // Initial nil + premium
        XCTAssertNil(receivedStatuses[0])
        XCTAssertEqual(receivedStatuses[1]?.tier, .premium)
    }

    func testLoadStatus_OverridesDebugTier() async {
        // When - Set debug tier first
        sut.setDebugTier(.premium)
        XCTAssertEqual(sut.currentStatus?.tier, .premium)

        // Then - clearDebugTier should load actual status and override debug tier
        await sut.clearDebugTier()
        XCTAssertEqual(sut.currentStatus?.tier, .free, "clearDebugTier() should load actual status from use case")
    }

    // MARK: - Singleton Sharing Test

    func testDependencyContainer_ReturnsSameInstance() {
        // Given
        let instance1 = DependencyContainer.shared.subscriptionViewModel
        let instance2 = DependencyContainer.shared.subscriptionViewModel

        // Then
        XCTAssertTrue(instance1 === instance2, "DependencyContainer should return the same singleton instance")
    }

    func testDependencyContainer_DebugTierPersistsAcrossAccess() {
        // Given
        let viewModel = DependencyContainer.shared.subscriptionViewModel

        // When
        viewModel.setDebugTier(.premium)

        // Then - Access again and verify
        let sameViewModel = DependencyContainer.shared.subscriptionViewModel
        XCTAssertEqual(sameViewModel.currentStatus?.tier, .premium, "Debug tier should persist across DependencyContainer accesses")
        XCTAssertTrue(viewModel === sameViewModel, "Should be the exact same instance")
    }

    // MARK: - Print Debug Info

    func testPrintDebugInfo_ViewModelState() {
        // Given
        sut.setDebugTier(.premium)

        // Print
        print("=== SubscriptionViewModel Debug Info ===")
        print("ViewModel instance: \(ObjectIdentifier(sut))")
        print("Current tier: \(sut.currentStatus?.tier.displayName ?? "nil")")
        print("Is active: \(sut.currentStatus?.isActive ?? false)")
        print("=========================================")
    }

    func testPrintDebugInfo_DependencyContainerSingleton() {
        // Given
        let vm1 = DependencyContainer.shared.subscriptionViewModel
        let vm2 = DependencyContainer.shared.subscriptionViewModel

        // Set tier on first access
        vm1.setDebugTier(.premiumPlus)

        // Print
        print("=== DependencyContainer Singleton Debug Info ===")
        print("First access instance: \(ObjectIdentifier(vm1))")
        print("Second access instance: \(ObjectIdentifier(vm2))")
        print("Are same instance: \(vm1 === vm2)")
        print("First access tier: \(vm1.currentStatus?.tier.displayName ?? "nil")")
        print("Second access tier: \(vm2.currentStatus?.tier.displayName ?? "nil")")
        print("==============================================")
    }
}
