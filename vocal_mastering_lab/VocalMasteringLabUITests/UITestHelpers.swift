//
//  UITestHelpers.swift
//  VocalMasteringLabUITests
//
//  Common helpers for UI tests
//

import XCTest

// MARK: - Localization Helper

/// Helper class to load localized strings from the app's Localizable.strings
/// This allows UI tests to work correctly regardless of the test language setting
final class LocalizationHelper {
    static let shared = LocalizationHelper()

    private var bundle: Bundle?
    private var locale: String = "en"

    private init() {
        // Auto-configure with English locale by default
        // This ensures fallback strings work even if configure() is not called
        configure(locale: "en")
    }

    /// Configure the localization helper with the specified locale
    /// - Parameter locale: The locale identifier (e.g., "en", "ja")
    func configure(locale: String) {
        self.locale = locale
        // Find the app's Resources directory to load Localizable.strings
        // UI tests run in a separate process, so we need to find the app bundle path
        let resourcesPath = findResourcesPath(for: locale)
        if let path = resourcesPath {
            bundle = Bundle(path: path)
        }
    }

    private func findResourcesPath(for locale: String) -> String? {
        // Try to find the Localizable.strings in the source directory
        // This works because UI tests have access to the file system
        let fileManager = FileManager.default
        let currentPath = fileManager.currentDirectoryPath

        // Look for the lproj directory relative to the project
        let possiblePaths = [
            "\(currentPath)/VocalMasteringLab/Resources/\(locale).lproj",
            "\(currentPath)/../VocalMasteringLab/Resources/\(locale).lproj",
            // For Xcode builds, check DerivedData paths
        ]

        for path in possiblePaths {
            if fileManager.fileExists(atPath: "\(path)/Localizable.strings") {
                return path
            }
        }

        // Fallback: try to find via environment or process info
        if let srcRoot = ProcessInfo.processInfo.environment["SRCROOT"] {
            let srcPath = "\(srcRoot)/VocalMasteringLab/Resources/\(locale).lproj"
            if fileManager.fileExists(atPath: "\(srcPath)/Localizable.strings") {
                return srcPath
            }
        }

        return nil
    }

    /// Get a localized string for the given key
    /// - Parameter key: The localization key
    /// - Returns: The localized string, or the key if not found
    func string(for key: String) -> String {
        if let bundle = bundle {
            let value = bundle.localizedString(forKey: key, value: nil, table: nil)
            if value != key {
                return value
            }
        }
        // Fallback to hardcoded values if bundle not available
        return fallbackStrings[key] ?? key
    }

    // Fallback strings for cases where bundle loading fails
    // These should match the English Localizable.strings
    private let fallbackStrings: [String: String] = [
        // Common
        "ok": "OK",
        "cancel": "Cancel",
        "save": "Save",
        "delete": "Delete",
        "error": "Error",
        // Recording
        "recording.show_settings": "Show Settings",
        "recording.hide_settings": "Hide Settings",
        "recording.scale_five_tone": "5-Tone",
        "recording.scale_five_tone_down": "5-Tone Down",
        "recording.scale_octave_repeat": "Octave Repeat",
        "recording.scale_broken": "Broken Scale",
        "recording.scale_broken_double": "Broken Scale (x2)",
        "recording.scale_rossini": "1.5 Octave",
        "recording.scale_off": "Off",
        "recording.countdown_message": "Counting down...",
        "recording.rename": "Rename",
        "recording.limit_reached": "You have reached today's recording limit",
        // List
        "list.title": "Recording List",
        // Analysis
        "analysis.info_title": "Recording Info",
        "analysis.analyzing": "Analyzing...",
        // Settings
        "settings.title": "Settings",
        "settings.manage_subscription": "Manage Subscription",
        "settings.terms": "Terms of Use",
        "settings.privacy": "Privacy Policy",
        // Paywall
        "paywall.purchase": "Purchase",
        "paywall.restore": "Restore Purchase",
        "paywall.free_limit": "5 times/day / 30 sec max",
        "paywall.premium_limit": "Unlimited / 5 min max",
        "paywall.price_free": "Free",
        "paywall.unlock_unlimited": "Unlock Unlimited Recording",
        "paywall.unlimited_description": "Record as many times as you want every day with Premium",
        "paywall.terms_agreement": "By purchasing, you agree to the Terms of Use and Privacy Policy",
        "paywall.terms": "Terms of Use",
        "paywall.privacy": "Privacy Policy",
        // Subscription
        "subscription.title": "Subscription Management",
        "subscription.cancel": "Cancel Subscription",
        "subscription.active": "Active",
        "subscription.inactive": "Inactive",
        "subscription.upgrade": "Upgrade",
        // Home
        "home.unlock_unlimited": "Unlock Unlimited Recording",
        // Debug
        "debug.paywall": "Premium Plan (Paywall)",
        "debug.current_tier": "Current",
        // Note: Scale names now use recording.scale_* keys for consistency
    ]
}

// MARK: - Common Localized Strings for UI Tests

/// Namespace for commonly used localized strings in UI tests
/// Strings are loaded dynamically based on the configured test locale
enum L10n {
    private static var helper: LocalizationHelper { LocalizationHelper.shared }

    // MARK: - Common
    static var ok: String { helper.string(for: "ok") }
    static var cancel: String { helper.string(for: "cancel") }
    static var save: String { helper.string(for: "save") }
    static var delete: String { helper.string(for: "delete") }
    static var error: String { helper.string(for: "error") }

    // MARK: - Recording
    static var showSettings: String { helper.string(for: "recording.show_settings") }
    static var hideSettings: String { helper.string(for: "recording.hide_settings") }
    static var scaleFiveTone: String { helper.string(for: "recording.scale_five_tone") }
    static var scaleFiveToneDown: String { helper.string(for: "recording.scale_five_tone_down") }
    static var scaleOctaveRepeat: String { helper.string(for: "recording.scale_octave_repeat") }
    static var scaleBroken: String { helper.string(for: "recording.scale_broken") }
    static var scaleBrokenDouble: String { helper.string(for: "recording.scale_broken_double") }
    static var scaleRossini: String { helper.string(for: "recording.scale_rossini") }
    static var scaleArpeggioDownTriple: String { helper.string(for: "recording.scale_arpeggio_down_triple") }
    static var scaleOff: String { helper.string(for: "recording.scale_off") }
    // Note: DisplayName aliases removed - now using recording.scale_* keys consistently
    static var countdownMessage: String { helper.string(for: "recording.countdown_message") }
    static var rename: String { helper.string(for: "recording.rename") }
    static var recordingLimitReached: String { helper.string(for: "recording.limit_reached") }

    // MARK: - List
    static var listTitle: String { helper.string(for: "list.title") }

    // MARK: - Analysis
    static var analysisInfoTitle: String { helper.string(for: "analysis.info_title") }
    static var analyzing: String { helper.string(for: "analysis.analyzing") }

    // MARK: - Settings
    static var settingsTitle: String { helper.string(for: "settings.title") }
    static var manageSubscription: String { helper.string(for: "settings.manage_subscription") }
    static var terms: String { helper.string(for: "settings.terms") }
    static var privacy: String { helper.string(for: "settings.privacy") }

    // MARK: - Paywall
    static var purchase: String { helper.string(for: "paywall.purchase") }
    static var restore: String { helper.string(for: "paywall.restore") }
    static var freeLimit: String { helper.string(for: "paywall.free_limit") }
    static var premiumLimit: String { helper.string(for: "paywall.premium_limit") }
    static var priceFree: String { helper.string(for: "paywall.price_free") }
    static var unlockUnlimited: String { helper.string(for: "paywall.unlock_unlimited") }
    static var unlimitedDescription: String { helper.string(for: "paywall.unlimited_description") }
    static var termsAgreement: String { helper.string(for: "paywall.terms_agreement") }
    static var paywallTerms: String { helper.string(for: "paywall.terms") }
    static var paywallPrivacy: String { helper.string(for: "paywall.privacy") }

    // MARK: - Subscription Management
    static var subscriptionTitle: String { helper.string(for: "subscription.title") }
    static var subscriptionCancel: String { helper.string(for: "subscription.cancel") }
    static var subscriptionUpgrade: String { helper.string(for: "subscription.upgrade") }
    static var subscriptionActive: String { helper.string(for: "subscription.active") }
    static var subscriptionInactive: String { helper.string(for: "subscription.inactive") }

    // MARK: - Home
    static var homeUnlockUnlimited: String { helper.string(for: "home.unlock_unlimited") }

    // MARK: - Debug
    static var debugPaywall: String { helper.string(for: "debug.paywall") }
    static var debugCurrentTier: String { helper.string(for: "debug.current_tier") }

    // MARK: - Dynamic strings with format
    static func currentTier(_ tier: String) -> String {
        return "\(debugCurrentTier): \(tier)"
    }
}

extension XCTestCase {
    /// Launch app with recording count reset and animations disabled for UI tests
    /// - Parameters:
    ///   - language: The language code to launch the app with (default: "en")
    ///   - locale: The locale identifier to launch the app with (default: "en")
    ///   - premium: If true, launch app with premium subscription tier (default: false)
    /// - Returns: The launched XCUIApplication
    func launchAppWithResetRecordingCount(
        language: String = "en",
        locale: String = "en",
        premium: Bool = false
    ) -> XCUIApplication {
        // Configure the localization helper to match the app's language
        LocalizationHelper.shared.configure(locale: language)

        let app = XCUIApplication()
        app.launchArguments = [
            "-UITestResetRecordingCount",
            "-UITestDisableAnimations",
            "-AppleLanguages", "(\(language))",
            "-AppleLocale", locale
        ]

        // Set subscription tier via environment variable
        if premium {
            app.launchEnvironment["SUBSCRIPTION_TIER"] = "premium"
        }

        app.launch()
        return app
    }
}
