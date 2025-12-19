//
//  SubscriptionStatus.swift
//  VocalisStudio
//
//  Subscription status entity representing user's current subscription state
//

import Foundation

/// User's subscription status including tier, cohort, and feature access
public struct SubscriptionStatus: Equatable, Codable {
    /// Current subscription tier
    public let tier: SubscriptionTier

    /// User cohort (for Grandfather clause)
    public let cohort: UserCohort

    /// Whether subscription is currently active
    public let isActive: Bool

    /// Subscription expiration date (nil if free tier)
    public let expirationDate: Date?

    /// Purchase date of current subscription (nil if free tier)
    public let purchaseDate: Date?

    /// Whether subscription will auto-renew
    public let willAutoRenew: Bool

    public init(
        tier: SubscriptionTier,
        cohort: UserCohort,
        isActive: Bool = true,
        expirationDate: Date? = nil,
        purchaseDate: Date? = nil,
        willAutoRenew: Bool = false
    ) {
        self.tier = tier
        self.cohort = cohort
        self.isActive = isActive
        self.expirationDate = expirationDate
        self.purchaseDate = purchaseDate
        self.willAutoRenew = willAutoRenew
    }

    // MARK: - Feature Access

    /// Check if user has access to a specific feature
    public func hasAccessTo(_ feature: Feature) -> Bool {
        // v1.0 users have access to all features (Grandfather clause)
        if cohort.hasGrandfatherPrivileges {
            return true
        }

        // Free tier features are available to everyone
        if feature.minimumTier == .free {
            return true
        }

        // Premium and Premium Plus features require active subscription
        guard isActive else {
            return false
        }

        // Check if user's tier meets the feature's minimum required tier
        return tier >= feature.minimumTier
    }

    /// Get list of features user has access to
    public var accessibleFeatures: [Feature] {
        return Feature.allCases.filter { hasAccessTo($0) }
    }

    /// Get list of features user does NOT have access to
    public var lockedFeatures: [Feature] {
        return Feature.allCases.filter { !hasAccessTo($0) }
    }

    // MARK: - Ad Policy

    /// Get ad policy for this user
    public func getAdPolicy() -> AdPolicy {
        // Premium and above: no ads
        if tier >= .premium && isActive {
            return .noAds
        }

        // v1.0 users: gradual ad introduction (Phase 1-3)
        if cohort == .v1_0 {
            return getV1_0AdPolicy()
        }

        // v2.0+ free users: standard ads
        return .standardAdsWithInterstitial
    }

    /// Ad policy for v1.0 users (gradual introduction)
    private func getV1_0AdPolicy() -> AdPolicy {
        // v3.0 release date (when ads were introduced)
        let v3_0_releaseDate = UserCohort.releaseDates[.v3_0]!

        // Calculate months since ad introduction
        let monthsSinceAdIntroduction = Date().monthsSince(v3_0_releaseDate)

        // Return current date if before v3.0 release (no ads)
        guard Date() >= v3_0_releaseDate else {
            return .noAds
        }

        // Phase 1 (0-2 months): Light ads (banner only)
        if monthsSinceAdIntroduction < 2 {
            return .lightAds
        }

        // Phase 2 (2-4 months): Standard ads (banner + rewarded)
        if monthsSinceAdIntroduction < 4 {
            return .standardAds
        }

        // Phase 3 (4+ months): Standard ads with interstitial
        return .standardAdsWithInterstitial
    }

    // MARK: - Subscription State

    /// Whether subscription is expired
    public var isExpired: Bool {
        guard let expirationDate = expirationDate else {
            return false // Free tier never expires
        }
        return Date() > expirationDate
    }

    /// Days until expiration (nil if free tier or expired)
    public var daysUntilExpiration: Int? {
        guard let expirationDate = expirationDate, !isExpired else {
            return nil
        }

        let calendar = Calendar.current
        let components = calendar.dateComponents([.day], from: Date(), to: expirationDate)
        return components.day
    }

    /// Whether subscription is in grace period (expired but still active)
    public var isInGracePeriod: Bool {
        return isExpired && isActive
    }

    // MARK: - Premium Status

    /// Whether user is on free tier
    public var isFree: Bool {
        return tier == .free
    }

    /// Whether user has active Premium subscription
    public var hasPremium: Bool {
        return tier == .premium && isActive
    }

    /// Whether user has active Premium Plus subscription
    public var hasPremiumPlus: Bool {
        return tier == .premiumPlus && isActive
    }

    /// Whether user has any paid subscription
    public var hasPaidSubscription: Bool {
        return tier > .free && isActive
    }

    // MARK: - Default Status

    /// Default free tier status for new v2.0+ users
    public static func defaultFree(cohort: UserCohort = .v2_0) -> SubscriptionStatus {
        return SubscriptionStatus(
            tier: .free,
            cohort: cohort,
            isActive: true,
            expirationDate: nil,
            purchaseDate: nil,
            willAutoRenew: false
        )
    }

    /// Free status for v1.0 users (Grandfather)
    public static let grandfatherFree = SubscriptionStatus(
        tier: .free,
        cohort: .v1_0,
        isActive: true,
        expirationDate: nil,
        purchaseDate: nil,
        willAutoRenew: false
    )
}

// MARK: - Date Extension

private extension Date {
    /// Calculate months between two dates
    func monthsSince(_ date: Date) -> Int {
        let calendar = Calendar.current
        let components = calendar.dateComponents([.month], from: date, to: self)
        return max(0, components.month ?? 0)
    }
}
