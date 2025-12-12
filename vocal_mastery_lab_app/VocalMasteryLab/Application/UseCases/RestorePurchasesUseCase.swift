//
//  RestorePurchasesUseCase.swift
//  VocalMasteryLab
//
//  Use case for restoring previous purchases
//

import Foundation
import SubscriptionDomain
import StoreKit

/// Use case that restores previous subscription purchases
public final class RestorePurchasesUseCase {
    private let repository: SubscriptionRepositoryProtocol

    public init(repository: SubscriptionRepositoryProtocol) {
        self.repository = repository
    }

    /// Execute the use case to restore previous purchases
    /// - Throws: SubscriptionError if restore fails or no purchases found
    /// - Note: User cancellation errors are silently ignored (not thrown)
    public func execute() async throws {
        do {
            try await repository.restorePurchases()
        } catch is CancellationError {
            // User cancelled - silently ignore
            return
        } catch SubscriptionError.userCancelled {
            // User cancelled - silently ignore
            return
        } catch StoreKitError.userCancelled {
            // User cancelled StoreKit dialog - silently ignore
            return
        }
    }
}
