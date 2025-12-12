//
//  GetSubscriptionStatusUseCase.swift
//  VocalMasteryLab
//
//  Use case for retrieving current subscription status
//

import Foundation
import SubscriptionDomain

/// Use case that retrieves current subscription status for the user
public final class GetSubscriptionStatusUseCase {
    private let repository: SubscriptionRepositoryProtocol

    public init(repository: SubscriptionRepositoryProtocol) {
        self.repository = repository
    }

    /// Execute the use case to get current subscription status
    /// - Returns: Current subscription status
    /// - Throws: Error if unable to retrieve status
    public func execute() async throws -> SubscriptionStatus {
        return try await repository.getCurrentStatus()
    }
}
