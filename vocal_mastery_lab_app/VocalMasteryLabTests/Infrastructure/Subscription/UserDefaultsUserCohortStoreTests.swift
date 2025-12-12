//
//  UserDefaultsUserCohortStoreTests.swift
//  VocalMasteryLabTests
//
//  Tests for UserDefaultsUserCohortStore
//

import XCTest
import SubscriptionDomain
@testable import VocalMasteryLab

final class UserDefaultsUserCohortStoreTests: XCTestCase {

    var store: UserDefaultsUserCohortStore!
    var userDefaults: UserDefaults!

    override func setUp() {
        super.setUp()
        // Use a test suite to avoid affecting real user defaults
        userDefaults = UserDefaults(suiteName: "test.vocalisstudio.subscription")!
        userDefaults.removePersistentDomain(forName: "test.vocalisstudio.subscription")
        store = UserDefaultsUserCohortStore(userDefaults: userDefaults)
    }

    override func tearDown() {
        userDefaults.removePersistentDomain(forName: "test.vocalisstudio.subscription")
        userDefaults = nil
        store = nil
        super.tearDown()
    }

    // MARK: - Save and Load Tests

    func testSavesAndLoadsV1_0Cohort() {
        // Given: v1.0 cohort
        let cohort = UserCohort.v1_0

        // When: Save cohort
        store.save(cohort: cohort)

        // Then: Should load same cohort
        let loaded = store.load()
        XCTAssertEqual(loaded, cohort)
    }

    func testSavesAndLoadsV2_0Cohort() {
        // Given: v2.0 cohort
        let cohort = UserCohort.v2_0

        // When: Save cohort
        store.save(cohort: cohort)

        // Then: Should load same cohort
        let loaded = store.load()
        XCTAssertEqual(loaded, cohort)
    }

    func testSavesAndLoadsV2_5Cohort() {
        // Given: v2.5 cohort
        let cohort = UserCohort.v2_5

        // When: Save cohort
        store.save(cohort: cohort)

        // Then: Should load same cohort
        let loaded = store.load()
        XCTAssertEqual(loaded, cohort)
    }

    func testSavesAndLoadsV3_0Cohort() {
        // Given: v3.0 cohort
        let cohort = UserCohort.v3_0

        // When: Save cohort
        store.save(cohort: cohort)

        // Then: Should load same cohort
        let loaded = store.load()
        XCTAssertEqual(loaded, cohort)
    }

    // MARK: - First Launch Tests

    func testReturnsNilWhenNoCohortSaved() {
        // Given: No cohort saved (first launch)

        // When: Load cohort
        let loaded = store.load()

        // Then: Should return nil
        XCTAssertNil(loaded)
    }

    func testReturnsNilAfterClear() {
        // Given: Cohort saved then cleared
        store.save(cohort: .v1_0)
        store.clear()

        // When: Load cohort
        let loaded = store.load()

        // Then: Should return nil
        XCTAssertNil(loaded)
    }

    // MARK: - Overwrite Tests

    func testOverwritesExistingCohort() {
        // Given: v1.0 cohort saved
        store.save(cohort: .v1_0)

        // When: Save v2.0 cohort
        store.save(cohort: .v2_0)

        // Then: Should load v2.0 (overwritten)
        let loaded = store.load()
        XCTAssertEqual(loaded, .v2_0)
    }

    // MARK: - Persistence Tests

    func testPersistsAcrossInstances() {
        // Given: Save cohort in first instance
        store.save(cohort: .v1_0)

        // When: Create new instance with same UserDefaults
        let newStore = UserDefaultsUserCohortStore(userDefaults: userDefaults)

        // Then: Should load same cohort
        let loaded = newStore.load()
        XCTAssertEqual(loaded, .v1_0)
    }

    // MARK: - Grandfather Clause Tests

    func testV1_0CohortHasGrandfatherPrivileges() {
        // Given: v1.0 cohort saved
        store.save(cohort: .v1_0)

        // When: Load cohort
        let loaded = store.load()

        // Then: Should have grandfather privileges
        XCTAssertTrue(loaded?.hasGrandfatherPrivileges ?? false)
    }

    func testV2_0CohortDoesNotHaveGrandfatherPrivileges() {
        // Given: v2.0 cohort saved
        store.save(cohort: .v2_0)

        // When: Load cohort
        let loaded = store.load()

        // Then: Should NOT have grandfather privileges
        XCTAssertFalse(loaded?.hasGrandfatherPrivileges ?? true)
    }
}
