//
//  RestorePurchasesUseCaseProtocol.swift
//  VocalMasteryLab
//
//  Protocol for restore purchases use case
//

import Foundation

public protocol RestorePurchasesUseCaseProtocol {
    func execute() async throws
}

extension RestorePurchasesUseCase: RestorePurchasesUseCaseProtocol {}
