import Foundation
import SwiftUI
import OSLog

/// Manager for handling app localization
@MainActor
class LocalizationManager: ObservableObject {
    private static let logger = Logger(
        subsystem: "com.kazuasato.VocalMasteryLab",
        category: "localization"
    )
    static let shared = LocalizationManager()
    
    @Published var currentLanguage: String {
        didSet {
            UserDefaults.standard.set(currentLanguage, forKey: "appLanguage")
            // Force bundle refresh
            Bundle.setLanguage(currentLanguage)
        }
    }
    
    /// Supported languages with their locale codes
    static let supportedLanguages: [String] = [
        "ja",      // Japanese
        "en",      // English
        "zh-Hans", // Simplified Chinese
        "zh-Hant", // Traditional Chinese
        "ko",      // Korean
        "es",      // Spanish
        "fr",      // French
        "de"       // German
    ]

    private init() {
        // Priority order:
        // 1. Saved user preference (user explicitly selected in app settings)
        // 2. Preferred Languages (system settings, includes -AppleLanguages for testing)
        // 3. System default (fallback to English)

        let finalLanguage: String
        let savedLanguage = UserDefaults.standard.string(forKey: "appLanguage")
        let preferredLanguages = Locale.preferredLanguages
        let firstPreferred = preferredLanguages.first ?? "en"

        // Log language determination process (both OSLog and FileLogger for reliable capture)
        Self.logger.error("🔴 LANG_INIT: savedLanguage=\(savedLanguage ?? "nil", privacy: .public)")
        Self.logger.error("🔴 LANG_INIT: preferredLanguages=\(preferredLanguages, privacy: .public)")
        Self.logger.error("🔴 LANG_INIT: firstPreferred=\(firstPreferred, privacy: .public)")

        FileLogger.shared.log(level: "INFO", category: "localization", message: "🔴 LANG_INIT: savedLanguage=\(savedLanguage ?? "nil")")
        FileLogger.shared.log(level: "INFO", category: "localization", message: "🔴 LANG_INIT: preferredLanguages=\(preferredLanguages)")
        FileLogger.shared.log(level: "INFO", category: "localization", message: "🔴 LANG_INIT: firstPreferred=\(firstPreferred)")

        if let savedLanguage = savedLanguage {
            // User explicitly chose this language in app settings
            finalLanguage = savedLanguage
            Self.logger.error("🔴 LANG_INIT: Using saved language: \(finalLanguage, privacy: .public)")
            FileLogger.shared.log(level: "INFO", category: "localization", message: "🔴 LANG_INIT: Using saved language: \(finalLanguage)")
        } else {
            // Use preferred languages (respects -AppleLanguages launch argument)
            finalLanguage = Self.mapToSupportedLanguage(firstPreferred)
            Self.logger.error("🔴 LANG_INIT: Using mapped preferred language: \(firstPreferred, privacy: .public) -> \(finalLanguage, privacy: .public)")
            FileLogger.shared.log(level: "INFO", category: "localization", message: "🔴 LANG_INIT: Using mapped preferred language: \(firstPreferred) -> \(finalLanguage)")
        }

        self.currentLanguage = finalLanguage
        Bundle.setLanguage(finalLanguage)
        Self.logger.error("🔴 LANG_INIT: Final language set to: \(finalLanguage, privacy: .public)")
        FileLogger.shared.log(level: "INFO", category: "localization", message: "🔴 LANG_INIT: Final language set to: \(finalLanguage)")
    }

    /// Maps a system locale to a supported app language
    private static func mapToSupportedLanguage(_ locale: String) -> String {
        // Check for exact match first
        if supportedLanguages.contains(locale) {
            return locale
        }

        // Check for language prefix matches
        if locale.hasPrefix("ja") { return "ja" }
        if locale.hasPrefix("zh-Hans") || locale.hasPrefix("zh_CN") { return "zh-Hans" }
        if locale.hasPrefix("zh-Hant") || locale.hasPrefix("zh_TW") || locale.hasPrefix("zh_HK") { return "zh-Hant" }
        if locale.hasPrefix("zh") { return "zh-Hans" } // Default Chinese to Simplified
        if locale.hasPrefix("ko") { return "ko" }
        if locale.hasPrefix("es") { return "es" }
        if locale.hasPrefix("fr") { return "fr" }
        if locale.hasPrefix("de") { return "de" }

        // Default to English
        return "en"
    }
    
    func changeLanguage(_ language: String) {
        currentLanguage = language
    }
}

// MARK: - Bundle Extension for Language Switching

private var bundleKey: UInt8 = 0

extension Bundle {
    static func setLanguage(_ language: String) {
        defer {
            // Force SwiftUI to refresh
            object_setClass(Bundle.main, PrivateBundle.self)
        }
        objc_setAssociatedObject(Bundle.main, &bundleKey, Bundle(path: Bundle.main.path(forResource: language, ofType: "lproj") ?? ""), .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    }
    
    class PrivateBundle: Bundle {
        override func localizedString(forKey key: String, value: String?, table tableName: String?) -> String {
            guard let bundle = objc_getAssociatedObject(self, &bundleKey) as? Bundle else {
                return super.localizedString(forKey: key, value: value, table: tableName)
            }
            return bundle.localizedString(forKey: key, value: value, table: tableName)
        }
    }
}

// MARK: - SwiftUI Localization Helpers

extension String {
    var localized: String {
        return NSLocalizedString(self, comment: "")
    }
    
    func localized(with arguments: CVarArg...) -> String {
        return String(format: NSLocalizedString(self, comment: ""), arguments: arguments)
    }
}
