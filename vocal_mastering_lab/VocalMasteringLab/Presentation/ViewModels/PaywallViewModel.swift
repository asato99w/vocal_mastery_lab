//
//  PaywallViewModel.swift
//  VocalMasteringLab
//
//  ViewModel for paywall display
//

import Foundation
import SubscriptionDomain
import Combine
import SubscriptionDomain

@MainActor
public final class PaywallViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published public private(set) var currentStatus: SubscriptionStatus?
    @Published public private(set) var selectedTier: SubscriptionTier = .premium
    @Published public private(set) var isLoading = false
    @Published public var errorMessage: String?
    @Published public private(set) var isPurchaseSuccessful = false
    @Published public private(set) var availableProducts: [SubscriptionProduct] = []

    // MARK: - Dependencies

    private let getStatusUseCase: GetSubscriptionStatusUseCaseProtocol
    private let getProductsUseCase: GetAvailableProductsUseCaseProtocol
    private let purchaseUseCase: PurchaseSubscriptionUseCaseProtocol
    private let restoreUseCase: RestorePurchasesUseCaseProtocol

    // MARK: - Initialization

    public init(
        getStatusUseCase: GetSubscriptionStatusUseCaseProtocol,
        purchaseUseCase: PurchaseSubscriptionUseCaseProtocol,
        restoreUseCase: RestorePurchasesUseCaseProtocol,
        getProductsUseCase: GetAvailableProductsUseCaseProtocol
    ) {
        self.getStatusUseCase = getStatusUseCase
        self.purchaseUseCase = purchaseUseCase
        self.restoreUseCase = restoreUseCase
        self.getProductsUseCase = getProductsUseCase
    }

    // MARK: - Public Methods

    /// Load current subscription status and available products
    public func loadStatus() async {
        isLoading = true
        errorMessage = nil

        FileLogger.shared.log(level: "INFO", category: "paywall", message: "[loadStatus] Starting to load status and products")

        do {
            FileLogger.shared.log(level: "INFO", category: "paywall", message: "[loadStatus] Fetching subscription status...")
            currentStatus = try await getStatusUseCase.execute()
            FileLogger.shared.log(level: "INFO", category: "paywall", message: "[loadStatus] Status loaded: \(String(describing: currentStatus?.tier))")

            FileLogger.shared.log(level: "INFO", category: "paywall", message: "[loadStatus] Fetching available products...")
            availableProducts = try await getProductsUseCase.execute()
            FileLogger.shared.log(level: "INFO", category: "paywall", message: "[loadStatus] Products loaded: \(availableProducts.count) products")

            for product in availableProducts {
                FileLogger.shared.log(level: "INFO", category: "paywall", message: "[loadStatus] Product: \(product.productId) - \(product.displayPrice)")
            }
        } catch {
            FileLogger.shared.log(level: "ERROR", category: "paywall", message: "[loadStatus] Error: \(error.localizedDescription)")
            errorMessage = error.localizedDescription
            currentStatus = nil
        }

        isLoading = false
        FileLogger.shared.log(level: "INFO", category: "paywall", message: "[loadStatus] Completed. isLoading=false, products=\(availableProducts.count)")
    }

    /// Get product for a specific tier
    public func product(for tier: SubscriptionTier) -> SubscriptionProduct? {
        return availableProducts.first { $0.tier == tier }
    }

    /// Select a subscription tier
    public func selectTier(_ tier: SubscriptionTier) {
        guard tier != .free else { return }
        selectedTier = tier
    }

    /// Purchase the selected tier
    public func purchaseSelectedTier() async {
        FileLogger.shared.log(level: "INFO", category: "purchase", message: "[paywall] 🛒 Purchase started for tier: \(selectedTier)")
        isLoading = true
        errorMessage = nil
        isPurchaseSuccessful = false

        do {
            FileLogger.shared.log(level: "INFO", category: "purchase", message: "[paywall] 🛒 Calling purchaseUseCase.execute()")
            try await purchaseUseCase.execute(tier: selectedTier)
            FileLogger.shared.log(level: "INFO", category: "purchase", message: "[paywall] ✅ Purchase completed successfully")

            // Refresh status after successful purchase
            FileLogger.shared.log(level: "INFO", category: "purchase", message: "[paywall] 🔄 Loading status after purchase")
            await loadStatus()
            FileLogger.shared.log(level: "INFO", category: "purchase", message: "[paywall] ✅ Status loaded, setting isPurchaseSuccessful=true")

            isPurchaseSuccessful = true
            isLoading = false
        } catch SubscriptionError.userCancelled {
            // User cancelled - not an error, just reset loading state silently
            FileLogger.shared.log(level: "INFO", category: "purchase", message: "[paywall] 🚫 User cancelled purchase - no error shown")
            isPurchaseSuccessful = false
            isLoading = false
        } catch {
            FileLogger.shared.log(level: "ERROR", category: "purchase", message: "[paywall] ❌ Purchase failed: \(error.localizedDescription)")
            errorMessage = error.localizedDescription
            isPurchaseSuccessful = false
            isLoading = false
        }
    }

    /// Restore previous purchases
    public func restorePurchases() async {
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

    /// Clear error message
    public func clearError() {
        errorMessage = nil
    }

    /// Reset purchase success state
    public func resetPurchaseSuccess() {
        isPurchaseSuccessful = false
    }

    // MARK: - Computed Properties

    /// Available tiers for purchase
    public var availableTiers: [SubscriptionTier] {
        return [.premium, .premiumPlus]
    }

    /// Features for the selected tier
    public var selectedTierFeatures: [Feature] {
        return Feature.allCases.filter { feature in
            feature.minimumTier == selectedTier ||
            (selectedTier == .premiumPlus && feature.minimumTier == .premium)
        }
    }

    /// Check if a tier is recommended
    public func isRecommended(_ tier: SubscriptionTier) -> Bool {
        // Premium Plus is recommended for most users
        return tier == .premiumPlus
    }

    /// Get features for a specific tier
    public func features(for tier: SubscriptionTier) -> [Feature] {
        return Feature.allCases.filter { feature in
            feature.minimumTier == tier ||
            (tier == .premiumPlus && feature.minimumTier == .premium) ||
            (tier == .premiumPlus && feature.minimumTier == .free) ||
            (tier == .premium && feature.minimumTier == .free)
        }
    }

    /// Check if user already has a subscription
    public var hasActiveSubscription: Bool {
        guard let status = currentStatus else { return false }
        return status.tier != .free && status.isActive
    }
}
