//
//  GetAvailableProductsUseCase.swift
//  VocalisStudio
//
//  Use case for fetching available subscription products
//

import Foundation

/// Protocol for getting available subscription products
public protocol GetAvailableProductsUseCaseProtocol {
    /// Execute the use case
    /// - Returns: Array of available subscription products with pricing
    /// - Throws: SubscriptionError if unable to fetch products
    func execute() async throws -> [SubscriptionProduct]
}

/// Use case for fetching available subscription products from StoreKit
public final class GetAvailableProductsUseCase: GetAvailableProductsUseCaseProtocol {
    private let repository: SubscriptionRepositoryProtocol

    public init(repository: SubscriptionRepositoryProtocol) {
        self.repository = repository
    }

    public func execute() async throws -> [SubscriptionProduct] {
        return try await repository.getAvailableProducts()
    }
}
