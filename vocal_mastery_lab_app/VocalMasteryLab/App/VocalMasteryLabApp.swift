import SwiftUI
import VocalisDomain
import OSLog
import StoreKit
import AVFoundation

@available(iOS 15.0, macOS 11.0, *)
@main
public struct VocalMasteryLabApp: App {
    private static let boot = Logger(
        subsystem: "com.kazuasato.VocalMasteryLab",
        category: "boot"
    )

    /// Track app lifecycle for background audio debugging
    @Environment(\.scenePhase) private var scenePhase

    #if DEBUG
    private let animationsDisabled = CommandLine.arguments.contains("-UITestDisableAnimations")
    #else
    private let animationsDisabled = false
    #endif

    public init() {
        // Initialize file logging system (DEBUG builds only)
        #if DEBUG
        // Add error-level markers for reliable log capture (log_capture_guide_v2.md)
        Self.boot.error("UI_TEST_MARK: APP_INIT")
        FileLogger.shared.log(level: "INFO", category: "boot", message: "APP_INIT_FILE")

        let logPath = FileLogger.shared.currentLogPath
        Logger.viewModel.info("File logging enabled")
        Logger.viewModel.info("Log file: \(logPath)")
        FileLogger.shared.log(level: "INFO", category: "system", message: "VocalMasteryLab started")

        // Reset recording count and delete all recordings for UI tests
        if CommandLine.arguments.contains("-UITestResetRecordingCount") {
            Logger.viewModel.info("UI Test mode detected: Resetting recording count and deleting all recordings")
            RecordingUsageTracker().resetForTesting()

            // Delete all existing recordings
            Task {
                do {
                    let allRecordings = try await DependencyContainer.shared.recordingRepository.findAll()
                    Logger.viewModel.info("Found \(allRecordings.count) recordings to delete")
                    for recording in allRecordings {
                        try await DependencyContainer.shared.recordingRepository.delete(recording.id)
                    }
                    Logger.viewModel.info("All recordings deleted successfully")
                } catch {
                    Logger.viewModel.error("Failed to delete recordings: \(error)")
                }
            }

            Logger.viewModel.info("Recording count reset complete")
        }

        // Log animation disabling for UI tests
        if animationsDisabled {
            Logger.viewModel.info("UI Test mode: Animations disabled")
        }
        #endif
    }

    public var body: some Scene {
        WindowGroup {
            HomeView()
                .environmentObject(DependencyContainer.shared.subscriptionViewModel)
                .environment(\.uiTestAnimationsDisabled, animationsDisabled)
                .task {
                    await observeTransactionUpdates()
                }
        }
        .onChange(of: scenePhase) { oldPhase, newPhase in
            logScenePhaseChange(from: oldPhase, to: newPhase)
        }
    }

    // MARK: - App Lifecycle Logging

    /// Log scene phase changes for background audio debugging
    private func logScenePhaseChange(from oldPhase: ScenePhase, to newPhase: ScenePhase) {
        let oldPhaseString = phaseDescription(oldPhase)
        let newPhaseString = phaseDescription(newPhase)

        let message = "[DEBUG-lifecycle] ScenePhase changed: \(oldPhaseString) → \(newPhaseString)"
        Self.boot.info("\(message)")
        FileLogger.shared.log(level: "DEBUG", category: "lifecycle", message: message)

        // Log audio session state on each lifecycle change
        let audioSession = AVFoundation.AVAudioSession.sharedInstance()
        let category = audioSession.category.rawValue
        let isActive = audioSession.isOtherAudioPlaying
        let stateMessage = "[DEBUG-lifecycle] AudioSession: category=\(category), isOtherAudioPlaying=\(isActive)"
        Self.boot.info("\(stateMessage)")
        FileLogger.shared.log(level: "DEBUG", category: "lifecycle", message: stateMessage)
    }

    private func phaseDescription(_ phase: ScenePhase) -> String {
        switch phase {
        case .active: return "active"
        case .inactive: return "inactive"
        case .background: return "background"
        @unknown default: return "unknown"
        }
    }

    // MARK: - StoreKit Transaction Observation

    /// Observe StoreKit transaction updates
    /// This is required for StoreKit 2 to properly handle purchase completions
    @MainActor
    private func observeTransactionUpdates() async {
        FileLogger.shared.log(level: "INFO", category: "storekit", message: "🔄 Starting Transaction.updates observation")

        for await result in Transaction.updates {
            FileLogger.shared.log(level: "INFO", category: "storekit", message: "🔔 Transaction update received")

            switch result {
            case .verified(let transaction):
                FileLogger.shared.log(level: "INFO", category: "storekit", message: "✅ Transaction verified: \(transaction.productID)")

                // Finish the transaction
                await transaction.finish()
                FileLogger.shared.log(level: "INFO", category: "storekit", message: "✅ Transaction finished: \(transaction.productID)")

                // Refresh subscription status
                // Use force=true to clear any debug tier override when Transaction.updates fires
                Task {
                    await DependencyContainer.shared.subscriptionViewModel.loadStatus(force: true)
                    FileLogger.shared.log(level: "INFO", category: "storekit", message: "🔄 Subscription status refreshed (debug tier cleared)")
                }

            case .unverified(let transaction, let error):
                FileLogger.shared.log(level: "ERROR", category: "storekit", message: "❌ Transaction verification failed: \(transaction.productID), error: \(error)")
            }
        }
    }
}