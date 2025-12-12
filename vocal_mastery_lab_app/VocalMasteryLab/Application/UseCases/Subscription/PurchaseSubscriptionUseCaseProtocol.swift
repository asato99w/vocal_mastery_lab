//
//  PurchaseSubscriptionUseCaseProtocol.swift
//  VocalMasteryLab
//
//  Protocol for purchase subscription use case
//

import Foundation
import SubscriptionDomain

public protocol PurchaseSubscriptionUseCaseProtocol {
    func execute(tier: SubscriptionTier) async throws
}

extension PurchaseSubscriptionUseCase: PurchaseSubscriptionUseCaseProtocol {}
