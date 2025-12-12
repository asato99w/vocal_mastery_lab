//
//  StoreKitSubscriptionRepository.swift
//  VocalMasteryLab
//
//  StoreKit-based implementation of SubscriptionRepositoryProtocol
//

import Foundation
import SubscriptionDomain
import StoreKit

/// StoreKit implementation of subscription repository
@available(iOS 15.0, *)
public final class StoreKitSubscriptionRepository: SubscriptionRepositoryProtocol {
    private let productService: StoreKitProductServiceProtocol
    private let purchaseService: StoreKitPurchaseServiceProtocol
    private let cohortStore: UserCohortStoreProtocol

    public init(
        productService: StoreKitProductServiceProtocol,
        purchaseService: StoreKitPurchaseServiceProtocol,
        cohortStore: UserCohortStoreProtocol
    ) {
        self.productService = productService
        self.purchaseService = purchaseService
        self.cohortStore = cohortStore
    }

    // MARK: - SubscriptionRepositoryProtocol

    public func getCurrentStatus() async throws -> SubscriptionDomain.SubscriptionStatus {
        #if DEBUG
        // UIテスト用: 環境変数でティアをオーバーライド
        if let tierString = ProcessInfo.processInfo.environment["SUBSCRIPTION_TIER"],
           let tier = SubscriptionTier(rawValue: tierString) {
            let cohort = (try? await getUserCohort()) ?? .v2_0
            return SubscriptionStatus(
                tier: tier,
                cohort: cohort,
                isActive: true,
                expirationDate: nil,
                purchaseDate: nil,
                willAutoRenew: false
            )
        }
        #endif

        // Get or determine user cohort
        let cohort = try await getUserCohort()

        // Get current entitlements
        let entitlements = await purchaseService.currentEntitlements()

        // Find active subscription if any
        guard let activeTransaction = entitlements.first else {
            // No active subscription - return free tier
            return SubscriptionStatus.defaultFree(cohort: cohort)
        }

        // Determine tier from product ID
        let tier = determineTier(from: activeTransaction.productID)

        // Check if subscription is active
        let expirationDate = activeTransaction.expirationDate
        let isActive = expirationDate.map { $0 > Date() } ?? true

        return SubscriptionStatus(
            tier: tier,
            cohort: cohort,
            isActive: isActive,
            expirationDate: expirationDate,
            purchaseDate: activeTransaction.purchaseDate,
            willAutoRenew: activeTransaction.isUpgraded == false
        )
    }

    public func purchaseProduct(productId: String) async throws {
        FileLogger.shared.log(level: "INFO", category: "purchase", message: "[repo] 🛒 purchaseProduct called with productId: \(productId)")

        // Fetch product
        FileLogger.shared.log(level: "INFO", category: "purchase", message: "[repo] 📦 Fetching products from StoreKit")
        let products = try await productService.fetchProducts(productIds: [productId])
        FileLogger.shared.log(level: "INFO", category: "purchase", message: "[repo] 📦 Fetched \(products.count) products")

        guard let product = products.first else {
            FileLogger.shared.log(level: "ERROR", category: "purchase", message: "[repo] ❌ Product not found for id: \(productId)")
            throw SubscriptionError.productNotFound
        }

        FileLogger.shared.log(level: "INFO", category: "purchase", message: "[repo] 📦 Product found: \(product.displayName), price: \(product.displayPrice)")

        // Purchase product
        FileLogger.shared.log(level: "INFO", category: "purchase", message: "[repo] 🛒 Calling purchaseService.purchase()")
        let result = try await purchaseService.purchase(product)
        FileLogger.shared.log(level: "INFO", category: "purchase", message: "[repo] ✅ purchase() returned with result")

        switch result {
        case .success(let verification):
            FileLogger.shared.log(level: "INFO", category: "purchase", message: "[repo] ✅ Purchase result: success")
            // Verify transaction
            guard case .verified = verification else {
                FileLogger.shared.log(level: "ERROR", category: "purchase", message: "[repo] ❌ Transaction verification failed")
                throw SubscriptionError.purchaseFailed("Transaction verification failed")
            }
            FileLogger.shared.log(level: "INFO", category: "purchase", message: "[repo] ✅ Transaction verified successfully")
            // Purchase successful

        case .pending:
            FileLogger.shared.log(level: "INFO", category: "purchase", message: "[repo] ⏳ Purchase result: pending")
            throw SubscriptionError.purchaseFailed("Purchase is pending")

        case .userCancelled:
            FileLogger.shared.log(level: "INFO", category: "purchase", message: "[repo] 🚫 Purchase result: user cancelled")
            throw SubscriptionError.userCancelled

        @unknown default:
            FileLogger.shared.log(level: "ERROR", category: "purchase", message: "[repo] ❌ Purchase result: unknown")
            throw SubscriptionError.unknown
        }
    }

    public func getAvailableProducts() async throws -> [SubscriptionProduct] {
        // Get all subscription product IDs
        let productIds: Set<String> = [
            SubscriptionTier.premium.productId,
            SubscriptionTier.premiumPlus.productId
        ]

        FileLogger.shared.log(level: "INFO", category: "products", message: "[repo] 📦 Fetching products with IDs: \(productIds)")

        // Fetch products from StoreKit
        do {
            let products = try await productService.fetchProducts(productIds: productIds)
            FileLogger.shared.log(level: "INFO", category: "products", message: "[repo] 📦 Fetched \(products.count) products: \(products.map { $0.id })")

            if products.isEmpty {
                FileLogger.shared.log(level: "WARN", category: "products", message: "[repo] ⚠️ No products returned from StoreKit. Check App Store Connect configuration.")
            }

            // Convert to domain model
            return products.map { product in
                let tier = determineTier(from: product.id)
                let period = convertSubscriptionPeriod(from: product)
                return SubscriptionProduct(
                    productId: product.id,
                    displayName: product.displayName,
                    description: product.description,
                    displayPrice: product.displayPrice,
                    subscriptionPeriod: period,
                    tier: tier
                )
            }
        } catch {
            FileLogger.shared.log(level: "ERROR", category: "products", message: "[repo] ❌ Failed to fetch products: \(error.localizedDescription)")
            throw error
        }
    }

    public func restorePurchases() async throws {
        // Sync with App Store
        try await purchaseService.restore()

        // Check if we have any entitlements after restore
        let entitlements = await purchaseService.currentEntitlements()

        guard !entitlements.isEmpty else {
            throw SubscriptionError.noPurchasesToRestore
        }
    }

    public func getUserCohort() async throws -> UserCohort {
        // Check if cohort is already stored
        if let storedCohort = cohortStore.load() {
            return storedCohort
        }

        // Determine cohort based on current date (first installation)
        let cohort = UserCohort.determine(from: Date())

        // Save cohort for future use
        cohortStore.save(cohort: cohort)

        return cohort
    }

    // MARK: - Private Helpers

    private func determineTier(from productId: String) -> SubscriptionTier {
        switch productId {
        case SubscriptionTier.premium.productId:
            return .premium
        case SubscriptionTier.premiumPlus.productId:
            return .premiumPlus
        default:
            return .free
        }
    }

    private func convertSubscriptionPeriod(from product: Product) -> SubscriptionDomain.SubscriptionPeriod? {
        guard let subscription = product.subscription else {
            return nil
        }

        let period = subscription.subscriptionPeriod
        let unit: SubscriptionDomain.SubscriptionPeriodUnit

        switch period.unit {
        case .day:
            unit = .day
        case .week:
            unit = .week
        case .month:
            unit = .month
        case .year:
            unit = .year
        @unknown default:
            unit = .month
        }

        return SubscriptionDomain.SubscriptionPeriod(value: period.value, unit: unit)
    }
}
