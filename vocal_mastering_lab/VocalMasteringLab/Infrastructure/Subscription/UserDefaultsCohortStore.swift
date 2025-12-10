//
//  UserDefaultsCohortStore.swift
//  VocalMasteringLab
//
//  UserDefaults-based implementation of UserCohortStoreProtocol
//

import Foundation
import SubscriptionDomain

public final class UserDefaultsCohortStore: UserCohortStoreProtocol {

    private let userDefaults: UserDefaults
    private let cohortKey = "user_cohort"

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
