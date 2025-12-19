//
//  UserCohort.swift
//  VocalisStudio
//
//  User cohort value object for Grandfather clause implementation
//

import Foundation

/// User cohort based on first app installation version
/// Used to implement Grandfather clause for existing users
public enum UserCohort: String, Codable {
    /// Users who installed v1.0 (all features free forever)
    case v1_0 = "v1_0"

    /// Users who installed v2.0 or later (freemium model applies)
    case v2_0 = "v2_0"

    /// Users who installed v2.5 or later (Premium Plus available)
    case v2_5 = "v2_5"

    /// Users who installed v3.0 or later (ads introduced)
    case v3_0 = "v3_0"

    /// Localized display name
    public var displayName: String {
        switch self {
        case .v1_0:
            return "初期ユーザー（全機能無料）"
        case .v2_0:
            return "v2.0ユーザー"
        case .v2_5:
            return "v2.5ユーザー"
        case .v3_0:
            return "v3.0ユーザー"
        }
    }

    /// Whether this cohort has Grandfather privileges (all features free)
    public var hasGrandfatherPrivileges: Bool {
        return self == .v1_0
    }

    /// Release date for each version (for cohort determination)
    public static var releaseDates: [UserCohort: Date] {
        let calendar = Calendar.current

        // v1.0: 2025-04-01
        let v1_0_date = calendar.date(from: DateComponents(year: 2025, month: 4, day: 1))!

        // v2.0: 2025-10-01
        let v2_0_date = calendar.date(from: DateComponents(year: 2025, month: 10, day: 1))!

        // v2.5: 2026-04-01
        let v2_5_date = calendar.date(from: DateComponents(year: 2026, month: 4, day: 1))!

        // v3.0: 2026-10-01
        let v3_0_date = calendar.date(from: DateComponents(year: 2026, month: 10, day: 1))!

        return [
            .v1_0: v1_0_date,
            .v2_0: v2_0_date,
            .v2_5: v2_5_date,
            .v3_0: v3_0_date
        ]
    }

    /// Determine cohort based on installation date
    public static func determine(from installationDate: Date) -> UserCohort {
        let dates = releaseDates

        if installationDate < dates[.v2_0]! {
            return .v1_0
        } else if installationDate < dates[.v2_5]! {
            return .v2_0
        } else if installationDate < dates[.v3_0]! {
            return .v2_5
        } else {
            return .v3_0
        }
    }
}
