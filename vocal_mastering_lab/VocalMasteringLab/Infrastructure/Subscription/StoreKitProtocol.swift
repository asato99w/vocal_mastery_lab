//
//  StoreKitProtocol.swift
//  VocalMasteringLab
//
//  Protocol abstraction for StoreKit to enable testing
//

import Foundation
import StoreKit

/// Protocol for StoreKit product operations
public protocol StoreKitProductServiceProtocol {
    /// Fetch products from App Store
    func fetchProducts(productIds: Set<String>) async throws -> [Product]
}

/// Protocol for StoreKit purchase operations
public protocol StoreKitPurchaseServiceProtocol {
    /// Purchase a product
    func purchase(_ product: Product) async throws -> Product.PurchaseResult

    /// Get current entitlements
    func currentEntitlements() async -> [Transaction]

    /// Restore purchases
    func restore() async throws
}

/// Real StoreKit implementation
@available(iOS 15.0, *)
public final class StoreKitProductService: StoreKitProductServiceProtocol {
    public init() {}

    public func fetchProducts(productIds: Set<String>) async throws -> [Product] {
        return try await Product.products(for: productIds)
    }
}

/// Real StoreKit purchase implementation
@available(iOS 15.0, *)
public final class StoreKitPurchaseService: StoreKitPurchaseServiceProtocol {
    public init() {}

    public func purchase(_ product: Product) async throws -> Product.PurchaseResult {
        FileLogger.shared.log(level: "INFO", category: "purchase", message: "[storekit] 🛒 Starting product.purchase() for: \(product.id)")
        let result = try await product.purchase()
        FileLogger.shared.log(level: "INFO", category: "purchase", message: "[storekit] ✅ product.purchase() completed")
        return result
    }

    public func currentEntitlements() async -> [Transaction] {
        var transactions: [Transaction] = []

        for await result in Transaction.currentEntitlements {
            if case .verified(let transaction) = result {
                transactions.append(transaction)
            }
        }

        return transactions
    }

    public func restore() async throws {
        try await AppStore.sync()
    }
}
