//
//  UserDefaultsUserCohortStore.swift
//  VocalMasteringLab
//
//  UserDefaults-based storage for user cohort
//

import Foundation
import SubscriptionDomain

/// Protocol for user cohort storage
public protocol UserCohortStoreProtocol {
    /// Load saved cohort
    func load() -> UserCohort?

    /// Save cohort
    func save(cohort: UserCohort)

    /// Clear saved cohort
    func clear()
}

/// UserDefaults-based implementation of user cohort storage
public final class UserDefaultsUserCohortStore: UserCohortStoreProtocol {
    private let userDefaults: UserDefaults
    private let cohortKey = "vocalisstudio.subscription.userCohort"

    public init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
    }

    public func load() -> UserCohort? {
        guard let rawValue = userDefaults.string(forKey: cohortKey) else {
            return nil
        }
        return UserCohort(rawValue: rawValue)
    }

    public func save(cohort: UserCohort) {
        userDefaults.set(cohort.rawValue, forKey: cohortKey)
    }

    public func clear() {
        userDefaults.removeObject(forKey: cohortKey)
    }
}
