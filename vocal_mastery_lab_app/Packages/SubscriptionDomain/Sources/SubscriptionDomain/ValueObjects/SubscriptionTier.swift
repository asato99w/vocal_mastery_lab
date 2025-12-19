//
//  SubscriptionTier.swift
//  VocalisStudio
//
//  Subscription tier value object representing user's subscription level
//

import Foundation

/// Subscription tier levels
public enum SubscriptionTier: String, Codable, Comparable, CaseIterable {
    case free = "free"
    case premium = "premium"
    case premiumPlus = "premium_plus"

    /// Product ID for App Store Connect
    public var productId: String {
        switch self {
        case .free:
            return ""
        case .premium:
            return "com.vocalisstudio.premium.monthly"
        case .premiumPlus:
            return "com.vocalisstudio.premiumplus.monthly"
        }
    }

    /// Localization key for display name
    public var displayNameKey: String {
        switch self {
        case .free:
            return "paywall.price_free"
        case .premium:
            return "paywall.premium"
        case .premiumPlus:
            return "paywall.premium_plus"
        }
    }

    /// Localized display name (uses NSLocalizedString for proper localization)
    public var displayName: String {
        return NSLocalizedString(displayNameKey, comment: "")
    }

    /// Monthly price in yen (for display purposes)
    public var monthlyPrice: Int {
        switch self {
        case .free:
            return 0
        case .premium:
            return 480
        case .premiumPlus:
            return 980
        }
    }

    /// Yearly price in yen (for display purposes)
    public var yearlyPrice: Int {
        switch self {
        case .free:
            return 0
        case .premium:
            return 4800
        case .premiumPlus:
            return 9800
        }
    }

    /// Yearly product ID for App Store Connect
    public var yearlyProductId: String {
        switch self {
        case .free:
            return ""
        case .premium:
            return "com.vocalisstudio.premium.yearly"
        case .premiumPlus:
            return "com.vocalisstudio.premiumplus.yearly"
        }
    }

    // MARK: - Comparable

    public static func < (lhs: SubscriptionTier, rhs: SubscriptionTier) -> Bool {
        return lhs.order < rhs.order
    }

    private var order: Int {
        switch self {
        case .free:
            return 0
        case .premium:
            return 1
        case .premiumPlus:
            return 2
        }
    }
}
