//
//  PaywallUITests.swift
//  VocalMasteryLabUITests
//
//  UI tests for subscription paywall flow
//  Note: Most tests are skipped while all features are free.
//  Structure preserved for future paid plan restoration.
//

import XCTest
import StoreKitTest

final class PaywallUITests: XCTestCase {

    var app: XCUIApplication!
    var session: SKTestSession!

    override func setUpWithError() throws {
        continueAfterFailure = false

        // Initialize StoreKit Test session with Configuration.storekit using relative path
        let testBundle = Bundle(for: type(of: self))
        guard let configURL = testBundle.url(forResource: "Configuration", withExtension: "storekit") else {
            XCTFail("Failed to find Configuration.storekit in test bundle")
            return
        }
        session = try SKTestSession(contentsOf: configURL)
        session.disableDialogs = true  // Disable dialogs for automated testing
        session.clearTransactions()

        // Configure L10n to use English locale (must match app launch language)
        LocalizationHelper.shared.configure(locale: "en")

        app = XCUIApplication()
        app.launchArguments = [
            "UI-Testing",
            "-AppleLanguages", "(en)",
            "-AppleLocale", "en"
        ]
        app.launch()
    }

    override func tearDownWithError() throws {
        session?.clearTransactions()
        session = nil
        app = nil
    }

    // MARK: - Paywall Display Tests (SKIPPED - UI removed for "all free" policy)

    func testPaywallDisplay_showsCorrectPricing() throws {
        throw XCTSkip("Skipped: Paywall UI removed while all features are free. Preserved for future paid plan.")
    }

    func testPaywallDisplay_showsTermsAndPrivacy() throws {
        // Navigate to paywall via Debug Menu
        navigateToPaywall()

        // Verify terms and privacy links exist
        // Note: SwiftUI Link may appear as other elements (buttons, staticTexts) in XCUITest
        let termsElement = app.descendants(matching: .any).containing(NSPredicate(format: "label CONTAINS %@", L10n.paywallTerms))
        XCTAssertTrue(termsElement.firstMatch.exists, "Should have terms link")

        let privacyElement = app.descendants(matching: .any).containing(NSPredicate(format: "label CONTAINS %@", L10n.paywallPrivacy))
        XCTAssertTrue(privacyElement.firstMatch.exists, "Should have privacy policy link")

        // Verify disclaimer text
        XCTAssertTrue(app.staticTexts[L10n.termsAgreement].exists,
                     "Should show purchase agreement text")
    }

    // MARK: - Recording Limit → Paywall Flow (SKIPPED - No recording limits)

    func SKIP_testRecordingLimitReached_showsPaywall() throws {
        throw XCTSkip("Feature not implemented and skipped: Recording limits removed for 'all free' policy")
    }

    // MARK: - Purchase Flow Tests (SKIPPED - UI removed)

    func testPurchaseButton_isAccessible() throws {
        throw XCTSkip("Skipped: Paywall UI removed while all features are free. Preserved for future paid plan.")
    }

    func testRestoreButton_isAccessible() throws {
        throw XCTSkip("Skipped: Subscription management UI removed while all features are free. Preserved for future paid plan.")
    }

    // MARK: - Settings Navigation Tests (SKIPPED - Subscription section removed)

    func testSettings_hasSubscriptionLink() throws {
        throw XCTSkip("Skipped: Subscription section removed from Settings while all features are free. Preserved for future paid plan.")
    }

    func testSettings_hasTermsAndPrivacyLinks() throws {
        // Navigate to settings from Home
        let homeSettingsButton = app.buttons["HomeSettingsButton"]
        XCTAssertTrue(homeSettingsButton.waitForExistence(timeout: 5), "Home settings button should exist")
        homeSettingsButton.tap()

        // Wait for settings view to appear by checking terms link
        let termsLink = app.staticTexts[L10n.terms]
        XCTAssertTrue(termsLink.waitForExistence(timeout: 5), "Should have terms link in settings")

        let privacyLink = app.staticTexts[L10n.privacy]
        XCTAssertTrue(privacyLink.exists, "Should have privacy link in settings")
    }

    // MARK: - Subscription Status Display Tests (SKIPPED - UI removed)

    func testSubscriptionManagement_showsCurrentPlan() throws {
        throw XCTSkip("Skipped: Subscription management UI removed while all features are free. Preserved for future paid plan.")
    }

    func testSubscriptionManagement_hasCancelLink() throws {
        throw XCTSkip("Skipped: Subscription management UI removed while all features are free. Preserved for future paid plan.")
    }

    // MARK: - Loading States Tests

    func testPurchaseButton_showsLoadingState() throws {
        navigateToPaywall()

        let purchaseButton = app.buttons[L10n.purchase]
        XCTAssertTrue(purchaseButton.exists)

        // Tap purchase button
        purchaseButton.tap()

        // Note: In real StoreKit environment, loading indicator would appear
        // In test environment, this depends on how StoreKit test is configured
    }

    // MARK: - Helper Methods

    private func navigateToSubscriptionManagement() {
        // Note: Subscription section removed from Settings while all features are free
        // This helper preserved for future restoration
        let homeSettingsButton = app.buttons["HomeSettingsButton"]
        if homeSettingsButton.waitForExistence(timeout: 5) {
            homeSettingsButton.tap()

            let subscriptionLink = app.buttons.containing(NSPredicate(format: "label CONTAINS[cd] %@", L10n.manageSubscription))
            if subscriptionLink.firstMatch.waitForExistence(timeout: 5) {
                subscriptionLink.firstMatch.tap()
                _ = app.navigationBars[L10n.subscriptionTitle].waitForExistence(timeout: 5)
            }
        }
    }

    private func navigateToPaywall() {
        // Option 1: Use Upgrade Banner on home screen (REMOVED for "all free" policy)
        // Note: Upgrade banner removed while all features are free

        // Option 2: Use Debug Menu on home screen (debug builds only)
        #if DEBUG
        let debugButton = app.staticTexts["Debug"]
        if debugButton.waitForExistence(timeout: 2) {
            debugButton.tap()

            // Wait for debug menu and find paywall link
            let paywallLink = app.buttons.containing(NSPredicate(format: "label CONTAINS[cd] %@", L10n.debugPaywall))
            if paywallLink.firstMatch.waitForExistence(timeout: 5) {
                paywallLink.firstMatch.tap()

                // Wait for paywall view to appear
                _ = app.buttons[L10n.purchase].waitForExistence(timeout: 5)
            }
        }
        #endif
    }

    // MARK: - Accessibility Tests

    func testPaywall_isAccessible() throws {
        navigateToPaywall()

        // Verify all important elements have accessibility identifiers or labels
        let purchaseButton = app.buttons[L10n.purchase]
        XCTAssertTrue(purchaseButton.exists)

        let restoreButton = app.buttons.containing(NSPredicate(format: "label CONTAINS[cd] %@", L10n.restore))
        XCTAssertTrue(restoreButton.firstMatch.exists)

        // Verify text is readable
        XCTAssertTrue(app.staticTexts[L10n.unlockUnlimited].exists)
        XCTAssertTrue(app.staticTexts[L10n.unlimitedDescription].exists)
    }

    // MARK: - Purchase Status Update Tests (SKIPPED - UI removed)

    func testPurchase_shouldUpdateToPremiumStatus() throws {
        throw XCTSkip("Skipped: Subscription management UI removed while all features are free. Preserved for future paid plan.")
    }

    func testDebugMenu_tierSwitch_shouldPersistAcrossScreens() throws {
        throw XCTSkip("Skipped: Subscription management UI removed while all features are free. Preserved for future paid plan.")
    }

    // MARK: - Subscription Management Button Visibility Tests (SKIPPED - UI removed)

    func testSubscriptionManagement_freeTier_buttonVisibility() throws {
        throw XCTSkip("Skipped: Subscription management UI removed while all features are free. Preserved for future paid plan.")
    }

    func testSubscriptionManagement_premiumTier_buttonVisibility() throws {
        throw XCTSkip("Skipped: Subscription management UI removed while all features are free. Preserved for future paid plan.")
    }

    func testDebugTier_shouldBeClearedAfterRestorePurchase() throws {
        throw XCTSkip("Skipped: Subscription management UI removed while all features are free. Preserved for future paid plan.")
    }
}
