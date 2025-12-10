//
//  SubscriptionViewModel.swift
//  VocalMasteringLab
//
//  ViewModel for subscription management
//

import Foundation
import SubscriptionDomain
import Combine

@MainActor
public final class SubscriptionViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published public private(set) var currentStatus: SubscriptionDomain.SubscriptionStatus?
    @Published public private(set) var isLoading = false
    @Published public var errorMessage: String?

    // MARK: - Dependencies

    private let getStatusUseCase: GetSubscriptionStatusUseCaseProtocol
    private let purchaseUseCase: PurchaseSubscriptionUseCaseProtocol
    private let restoreUseCase: RestorePurchasesUseCaseProtocol

    #if DEBUG
    private var isDebugTierSet = false
    #endif

    // MARK: - Initialization

    public init(
        getStatusUseCase: GetSubscriptionStatusUseCaseProtocol,
        purchaseUseCase: PurchaseSubscriptionUseCaseProtocol,
        restoreUseCase: RestorePurchasesUseCaseProtocol
    ) {
        self.getStatusUseCase = getStatusUseCase
        self.purchaseUseCase = purchaseUseCase
        self.restoreUseCase = restoreUseCase
    }

    // MARK: - Public Methods

    /// Load current subscription status
    /// - Parameter force: If true, clears debug tier and forces reload from StoreKit (for Transaction.updates)
    public func loadStatus(force: Bool = false) async {
        #if DEBUG
        // Skip loading from StoreKit if debug tier is manually set
        // Unless force=true (e.g., when Transaction.updates fires)
        if isDebugTierSet && !force {
            return
        }
        // Clear debug tier flag when force loading
        if force {
            isDebugTierSet = false
        }
        #endif

        isLoading = true
        errorMessage = nil

        do {
            currentStatus = try await getStatusUseCase.execute()
        } catch {
            errorMessage = error.localizedDescription
            currentStatus = nil
        }

        isLoading = false
    }

    /// Purchase a subscription tier
    public func purchase(tier: SubscriptionTier) async {
        #if DEBUG
        // Clear debug tier flag when making real purchase
        isDebugTierSet = false
        #endif

        isLoading = true
        errorMessage = nil

        do {
            try await purchaseUseCase.execute(tier: tier)
            // Refresh status after successful purchase
            await loadStatus()
        } catch {
            errorMessage = error.localizedDescription
            isLoading = false
        }
    }

    /// Restore previous purchases
    public func restorePurchases() async {
        #if DEBUG
        // Clear debug tier flag when restoring purchases
        isDebugTierSet = false
        #endif

        isLoading = true
        errorMessage = nil

        do {
            try await restoreUseCase.execute()
            // Refresh status after successful restore
            await loadStatus()
        } catch {
            errorMessage = error.localizedDescription
            isLoading = false
        }
    }

    /// Check if user has access to a feature
    public func hasAccessTo(_ feature: Feature) -> Bool {
        guard let status = currentStatus else {
            return false
        }
        return status.hasAccessTo(feature)
    }

    /// Clear error message
    public func clearError() {
        errorMessage = nil
    }

    // MARK: - Debug Methods

    #if DEBUG
    /// Set debug tier for testing (overrides StoreKit state)
    public func setDebugTier(_ tier: SubscriptionTier) {
        isDebugTierSet = true
        currentStatus = SubscriptionDomain.SubscriptionStatus(
            tier: tier,
            cohort: .v2_0,
            isActive: tier != .free,
            expirationDate: tier != .free ? Date().addingTimeInterval(86400 * 30) : nil,
            purchaseDate: tier != .free ? Date() : nil,
            willAutoRenew: false
        )
    }

    /// Clear debug tier and reload from StoreKit
    public func clearDebugTier() async {
        isDebugTierSet = false
        await loadStatus()
    }
    #endif

}
