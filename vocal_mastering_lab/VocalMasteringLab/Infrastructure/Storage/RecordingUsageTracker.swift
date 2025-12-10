//
//  RecordingUsageTracker.swift
//  VocalMasteringLab
//
//  Tracks daily recording usage for subscription limits
//

import Foundation

/// Tracks recording usage for subscription limit enforcement
public final class RecordingUsageTracker {

    private let userDefaults: UserDefaults
    private let recordingCountKey = "daily_recording_count"
    private let lastResetDateKey = "last_reset_date"

    public init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
    }

    /// Get today's recording count
    public func getTodayCount() -> Int {
        #if DEBUG
        // UIテスト用: 環境変数でカウントをオーバーライド
        if let testCount = ProcessInfo.processInfo.environment["DAILY_RECORDING_COUNT"],
           let count = Int(testCount) {
            return count
        }
        #endif

        resetIfNeeded()
        return userDefaults.integer(forKey: recordingCountKey)
    }

    /// Increment recording count for today
    public func incrementCount() {
        resetIfNeeded()
        let current = userDefaults.integer(forKey: recordingCountKey)
        userDefaults.set(current + 1, forKey: recordingCountKey)
    }

    /// Reset count if it's a new day
    private func resetIfNeeded() {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())

        if let lastResetDate = userDefaults.object(forKey: lastResetDateKey) as? Date {
            let lastResetDay = calendar.startOfDay(for: lastResetDate)

            // If it's a new day, reset the count
            if today > lastResetDay {
                userDefaults.set(0, forKey: recordingCountKey)
                userDefaults.set(today, forKey: lastResetDateKey)
            }
        } else {
            // First time - initialize
            userDefaults.set(0, forKey: recordingCountKey)
            userDefaults.set(today, forKey: lastResetDateKey)
        }
    }

    #if DEBUG
    /// Reset count for testing
    public func resetForTesting() {
        userDefaults.set(0, forKey: recordingCountKey)
        userDefaults.set(Date(), forKey: lastResetDateKey)
    }
    #endif
}
