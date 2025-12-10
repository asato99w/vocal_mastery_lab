//
//  GetSubscriptionStatusUseCaseProtocol.swift
//  VocalMasteringLab
//
//  Protocol for getting subscription status use case
//

import Foundation
import SubscriptionDomain

public protocol GetSubscriptionStatusUseCaseProtocol {
    func execute() async throws -> SubscriptionStatus
}

extension GetSubscriptionStatusUseCase: GetSubscriptionStatusUseCaseProtocol {}
