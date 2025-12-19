//
//  AdPolicy.swift
//  VocalisStudio
//
//  Ad policy value object representing advertising rules for users
//

import Foundation

/// Ad frequency levels
public enum AdFrequency: String, Codable {
    /// No ads
    case none

    /// Light ads (1-2 times per session, banner only)
    case light

    /// Standard ads (3-5 times per session, banner + interstitial)
    case standard

    /// Heavy ads (aggressive display, all types)
    case heavy

    /// Localized display name
    public var displayName: String {
        switch self {
        case .none:
            return "広告なし"
        case .light:
            return "軽度"
        case .standard:
            return "標準"
        case .heavy:
            return "高頻度"
        }
    }
}

/// Ad types
public enum AdType: String, Codable {
    /// Banner ad (always visible at bottom)
    case banner

    /// Interstitial ad (full-screen, skippable after 5 seconds)
    case interstitial

    /// Rewarded ad (user choice, provides benefits)
    case rewarded

    /// Localized display name
    public var displayName: String {
        switch self {
        case .banner:
            return "バナー広告"
        case .interstitial:
            return "インタースティシャル広告"
        case .rewarded:
            return "リワード広告"
        }
    }
}

/// Ad policy for a user
public struct AdPolicy: Equatable, Codable {
    /// Ad frequency level
    public let frequency: AdFrequency

    /// Allowed ad types
    public let allowedTypes: Set<AdType>

    public init(frequency: AdFrequency, allowedTypes: Set<AdType>) {
        self.frequency = frequency
        self.allowedTypes = allowedTypes
    }

    /// Whether ads should be shown
    public var shouldShowAds: Bool {
        return frequency != .none && !allowedTypes.isEmpty
    }

    /// Whether banner ads are allowed
    public var allowsBanner: Bool {
        return allowedTypes.contains(.banner)
    }

    /// Whether interstitial ads are allowed
    public var allowsInterstitial: Bool {
        return allowedTypes.contains(.interstitial)
    }

    /// Whether rewarded ads are allowed
    public var allowsRewarded: Bool {
        return allowedTypes.contains(.rewarded)
    }

    // MARK: - Predefined Policies

    /// No ads policy (for Premium and above)
    public static let noAds = AdPolicy(frequency: .none, allowedTypes: [])

    /// Light ads policy (banner only)
    public static let lightAds = AdPolicy(frequency: .light, allowedTypes: [.banner])

    /// Standard ads policy (banner + rewarded)
    public static let standardAds = AdPolicy(frequency: .standard, allowedTypes: [.banner, .rewarded])

    /// Standard ads with interstitial (banner + interstitial + rewarded)
    public static let standardAdsWithInterstitial = AdPolicy(
        frequency: .standard,
        allowedTypes: [.banner, .interstitial, .rewarded]
    )

    /// Heavy ads policy (all types)
    public static let heavyAds = AdPolicy(
        frequency: .heavy,
        allowedTypes: [.banner, .interstitial, .rewarded]
    )
}

// MARK: - AdType Codable Conformance for Set

extension AdType: Hashable {}
