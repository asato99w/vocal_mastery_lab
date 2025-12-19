//
//  SubscriptionRepositoryProtocol.swift
//  VocalisStudio
//
//  Repository protocol for subscription management
//

import Foundation

/// Repository interface for subscription operations
public protocol SubscriptionRepositoryProtocol {
    /// Get current subscription status for the user
    /// - Returns: Current subscription status
    /// - Throws: SubscriptionError if unable to retrieve status
    func getCurrentStatus() async throws -> SubscriptionStatus

    /// Get available subscription products with pricing
    /// - Returns: Array of available subscription products
    /// - Throws: SubscriptionError if unable to fetch products
    func getAvailableProducts() async throws -> [SubscriptionProduct]

    /// Purchase a subscription product
    /// - Parameter productId: Product identifier from App Store Connect
    /// - Throws: SubscriptionError if purchase fails
    func purchaseProduct(productId: String) async throws

    /// Restore previous purchases
    /// - Throws: SubscriptionError if restore fails or no purchases found
    func restorePurchases() async throws

    /// Get user's cohort (for Grandfather clause)
    /// - Returns: User cohort based on first installation date
    /// - Throws: SubscriptionError if unable to determine cohort
    func getUserCohort() async throws -> UserCohort
}
