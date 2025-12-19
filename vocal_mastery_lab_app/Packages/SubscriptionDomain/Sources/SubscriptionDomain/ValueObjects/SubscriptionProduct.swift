//
//  SubscriptionProduct.swift
//  VocalisStudio
//
//  Value object representing a subscription product with pricing
//

import Foundation

/// Subscription period unit
public enum SubscriptionPeriodUnit: String, Equatable, Sendable {
    case day
    case week
    case month
    case year

    /// Localized display string for Apple requirements
    public var localizedKey: String {
        switch self {
        case .day: return "subscription.period.day"
        case .week: return "subscription.period.week"
        case .month: return "subscription.period.month"
        case .year: return "subscription.period.year"
        }
    }
}

/// Subscription period (value + unit)
public struct SubscriptionPeriod: Equatable, Sendable {
    public let value: Int
    public let unit: SubscriptionPeriodUnit

    public init(value: Int, unit: SubscriptionPeriodUnit) {
        self.value = value
        self.unit = unit
    }

    /// Localization key for auto-renewal description (Apple requirement)
    public var autoRenewalDescriptionKey: String {
        switch unit {
        case .day:
            return value == 1 ? "subscription.auto_renew.daily" : "subscription.auto_renew.days"
        case .week:
            return value == 1 ? "subscription.auto_renew.weekly" : "subscription.auto_renew.weeks"
        case .month:
            return value == 1 ? "subscription.auto_renew.monthly" : "subscription.auto_renew.months"
        case .year:
            return value == 1 ? "subscription.auto_renew.yearly" : "subscription.auto_renew.years"
        }
    }
}

/// Represents a subscription product with its metadata
public struct SubscriptionProduct: Equatable, Sendable {
    /// Product identifier
    public let productId: String

    /// Display name (from StoreKit)
    public let displayName: String

    /// Product description (from StoreKit)
    public let description: String

    /// Localized price (e.g., "¥500", "$4.99")
    public let displayPrice: String

    /// Subscription period (e.g., 1 month, 1 year)
    public let subscriptionPeriod: SubscriptionPeriod?

    /// Subscription tier
    public let tier: SubscriptionTier

    public init(
        productId: String,
        displayName: String,
        description: String = "",
        displayPrice: String,
        subscriptionPeriod: SubscriptionPeriod? = nil,
        tier: SubscriptionTier
    ) {
        self.productId = productId
        self.displayName = displayName
        self.description = description
        self.displayPrice = displayPrice
        self.subscriptionPeriod = subscriptionPeriod
        self.tier = tier
    }
}
