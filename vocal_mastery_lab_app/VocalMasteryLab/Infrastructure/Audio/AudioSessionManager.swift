import Foundation
import AVFoundation
import OSLog

/// Centralized audio session manager for the entire application
/// Manages AVAudioSession configuration, activation/deactivation, interruptions, and route changes
public class AudioSessionManager {

    // MARK: - Singleton

    public static let shared = AudioSessionManager()

    // MARK: - Session State

    /// Cached audio session mode for the current recording session
    /// This prevents mode changes during recording (which would cause error -10868)
    private var sessionMode: AVAudioSession.Mode?

    /// Reference count for active audio components
    /// Tracks how many components are currently using the audio session
    /// Only deactivate when count reaches 0
    private var activeComponentCount: Int = 0

    /// Lock for thread-safe access to activeComponentCount
    private let countLock = NSLock()

    private init() {
        setupNotificationObservers()
        Logger.audio.info("AudioSessionManager initialized")
        FileLogger.shared.log(level: "INFO", category: "audio", message: "AudioSessionManager initialized")

        // Log initial audio session state for debugging
        logAudioSessionState("initialization")
    }

    // MARK: - Debug Logging

    /// Log current audio session state for debugging background audio issues
    private func logAudioSessionState(_ context: String) {
        let audioSession = AVAudioSession.sharedInstance()
        let category = audioSession.category.rawValue
        let mode = audioSession.mode.rawValue
        let options = describeOptions(audioSession.categoryOptions)
        let isOtherAudioPlaying = audioSession.isOtherAudioPlaying
        let secondaryAudioHint = audioSession.secondaryAudioShouldBeSilencedHint

        let message = """
        [DEBUG-\(context)] AudioSession State:
          - category: \(category)
          - mode: \(mode)
          - options: \(options)
          - isOtherAudioPlaying: \(isOtherAudioPlaying)
          - secondaryAudioShouldBeSilencedHint: \(secondaryAudioHint)
        """
        Logger.audio.info("\(message)")
        FileLogger.shared.log(level: "DEBUG", category: "audio", message: message)
    }

    /// Describe category options as readable string
    private func describeOptions(_ options: AVAudioSession.CategoryOptions) -> String {
        var descriptions: [String] = []
        if options.contains(.mixWithOthers) { descriptions.append("mixWithOthers") }
        if options.contains(.duckOthers) { descriptions.append("duckOthers") }
        if options.contains(.allowBluetooth) { descriptions.append("allowBluetooth") }
        if options.contains(.defaultToSpeaker) { descriptions.append("defaultToSpeaker") }
        if options.contains(.interruptSpokenAudioAndMixWithOthers) { descriptions.append("interruptSpokenAudioAndMixWithOthers") }
        if options.contains(.allowBluetoothA2DP) { descriptions.append("allowBluetoothA2DP") }
        if options.contains(.allowAirPlay) { descriptions.append("allowAirPlay") }
        if options.contains(.overrideMutedMicrophoneInterruption) { descriptions.append("overrideMutedMicrophoneInterruption") }
        return descriptions.isEmpty ? "(none)" : descriptions.joined(separator: ", ")
    }

    // MARK: - Audio Session Configuration

    /// Configure audio session for recording (with simultaneous playback support)
    public func configureForRecording() throws {
        let audioSession = AVAudioSession.sharedInstance()

        // Log initial state before configuration for debugging audio mixing issues
        let wasOtherAudioPlaying = audioSession.isOtherAudioPlaying
        if wasOtherAudioPlaying {
            Logger.audio.info("[DEBUG] Other audio is playing before configureForRecording")
            FileLogger.shared.log(level: "DEBUG", category: "audio", message: "Other audio is playing before configureForRecording")
        }
        logAudioSessionState("before-configureForRecording")

        do {
            // Select mode and cache it for the session
            // This prevents mode changes during recording (which would cause error -10868)
            let mode = selectOptimalMode(for: audioSession)
            sessionMode = mode

            // .playAndRecord: allows recording and playback simultaneously
            // .measurement mode: highest precision for pitch detection and audio quality
            // .defaultToSpeaker: plays audio through speaker even when recording
            // .allowBluetooth: supports bluetooth headsets for calls
            // .allowBluetoothA2DP: enables Bluetooth recording (required for Bluetooth microphone)
            // .mixWithOthers: allows recording to continue when other apps play audio (e.g., YouTube)
            //
            // NOTE: The .playAndRecord category with .measurement mode may cause iOS to reduce
            // other apps' audio volumes (ducking behavior) to ensure recording quality.
            // This is intentional iOS behavior and not caused by .duckOthers option.
            try audioSession.setCategory(
                .playAndRecord,
                mode: mode,
                options: [.defaultToSpeaker, .allowBluetooth, .allowBluetoothA2DP, .mixWithOthers]
            )

            // Set preferred sample rate (44.1 kHz for high quality)
            try audioSession.setPreferredSampleRate(44100.0)

            // Always set input gain to maximum when possible
            // .measurement mode has no auto-gain, so we set it explicitly
            // Setting gain explicitly ensures consistent recording levels across all environments
            if audioSession.isInputGainSettable {
                try audioSession.setInputGain(1.0)  // 1.0 = maximum gain
                Logger.audio.info("Input gain set to maximum (1.0) for mode: \(String(describing: mode))")
                FileLogger.shared.log(level: "INFO", category: "audio", message: "Input gain set to 1.0 for mode: \(String(describing: mode))")
            } else {
                Logger.audio.info("Input gain not settable for current audio route")
                FileLogger.shared.log(level: "INFO", category: "audio", message: "Input gain not settable for current audio route")
            }

            Logger.audio.info("Audio session configured for recording: category=playAndRecord, mode=\(String(describing: mode)), sampleRate=44100Hz")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio session configured for recording: mode=\(String(describing: mode)), sampleRate=44100Hz")

            // Log state after configuration for debugging
            logAudioSessionState("after-configureForRecording")
        } catch {
            // Log detailed error information for debugging
            let nsError = error as NSError
            let isOtherAudioPlaying = audioSession.isOtherAudioPlaying
            Logger.audio.error("Failed to configure audio session for recording - domain: \(nsError.domain), code: \(nsError.code), otherAudioPlaying: \(isOtherAudioPlaying)")
            FileLogger.shared.log(
                level: "ERROR",
                category: "audio",
                message: "Failed to configure audio session for recording: \(error.localizedDescription), domain: \(nsError.domain), code: \(nsError.code), otherAudioPlaying: \(isOtherAudioPlaying)"
            )
            throw error
        }
    }

    /// Configure audio session for playback only
    public func configureForPlayback() throws {
        let audioSession = AVAudioSession.sharedInstance()

        do {
            // .playback: optimized for audio playback
            // .default mode: general-purpose mode
            // .mixWithOthers: allows playback to mix with other apps' audio
            try audioSession.setCategory(
                .playback,
                mode: .default,
                options: [.mixWithOthers]
            )

            Logger.audio.info("Audio session configured for playback: category=playback")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio session configured for playback: category=playback")
        } catch {
            Logger.audio.logError(error)
            FileLogger.shared.log(level: "ERROR", category: "audio", message: "Failed to configure audio session for playback: \(error.localizedDescription)")
            throw error
        }
    }

    /// Configure audio session for recording with playback (used during playback with pitch detection)
    public func configureForRecordingAndPlayback() throws {
        let audioSession = AVAudioSession.sharedInstance()

        do {
            // Use cached mode if available (prevents mode change during recording)
            // Otherwise, select mode based on current audio route
            let mode = sessionMode ?? selectOptimalMode(for: audioSession)

            // .playAndRecord: recording + playback
            // .measurement mode: highest precision for pitch detection and audio quality
            // Note: .measurement mode is incompatible with .defaultToSpeaker option
            // so we conditionally include .defaultToSpeaker based on mode
            // .mixWithOthers: allows recording to continue when other apps play audio
            var options: AVAudioSession.CategoryOptions = [.allowBluetooth, .allowBluetoothA2DP, .mixWithOthers]
            if mode != .measurement {
                options.insert(.defaultToSpeaker)
            }

            try audioSession.setCategory(
                .playAndRecord,
                mode: mode,
                options: options
            )

            // Always set input gain to maximum when possible
            // .measurement mode has no auto-gain, so we set it explicitly
            // Setting gain explicitly ensures consistent recording levels across all environments
            if audioSession.isInputGainSettable {
                try audioSession.setInputGain(1.0)  // 1.0 = maximum gain
                Logger.audio.info("Input gain set to maximum (1.0) for mode: \(String(describing: mode))")
                FileLogger.shared.log(level: "INFO", category: "audio", message: "Input gain set to 1.0 for mode: \(String(describing: mode))")
            } else {
                Logger.audio.info("Input gain not settable for current audio route")
                FileLogger.shared.log(level: "INFO", category: "audio", message: "Input gain not settable for current audio route")
            }

            Logger.audio.info("Audio session configured for recording and playback: category=playAndRecord, mode=\(String(describing: mode)) with full Bluetooth support")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio session configured for recording and playback: mode=\(String(describing: mode))")
        } catch {
            // Log detailed error information for debugging
            let nsError = error as NSError
            Logger.audio.logError(error)
            FileLogger.shared.log(level: "ERROR", category: "audio", message: "Failed to configure audio session for recording and playback: \(error.localizedDescription), domain: \(nsError.domain), code: \(nsError.code)")
            throw error
        }
    }

    // MARK: - Reference Counting

    /// Register an audio component as active
    /// Call this when a component starts using the audio session
    /// Components: AVAudioPlayer, AVAudioEngine (pitch detection), ScalePlayer
    public func registerActiveComponent(_ componentName: String) {
        countLock.lock()
        defer { countLock.unlock() }

        activeComponentCount += 1
        Logger.audio.info("Audio component registered: \(componentName), active count: \(self.activeComponentCount)")
        FileLogger.shared.log(level: "INFO", category: "audio", message: "Component registered: \(componentName), count: \(self.activeComponentCount)")
    }

    /// Unregister an audio component
    /// Call this when a component stops using the audio session
    /// Returns true if this was the last component (safe to deactivate)
    @discardableResult
    public func unregisterActiveComponent(_ componentName: String) -> Bool {
        countLock.lock()
        defer { countLock.unlock() }

        activeComponentCount = max(0, activeComponentCount - 1)
        Logger.audio.info("Audio component unregistered: \(componentName), active count: \(self.activeComponentCount)")
        FileLogger.shared.log(level: "INFO", category: "audio", message: "Component unregistered: \(componentName), count: \(self.activeComponentCount)")

        return activeComponentCount == 0
    }

    /// Check if any audio components are currently active
    public var hasActiveComponents: Bool {
        countLock.lock()
        defer { countLock.unlock() }
        return activeComponentCount > 0
    }

    // MARK: - Session Activation

    /// Activate the audio session
    /// Note: Multiple activations are safe - AVAudioSession handles this gracefully
    /// - Parameter allowMixing: If true, activates with options to allow mixing with other audio.
    ///                         This helps prevent errors when other apps are playing audio.
    public func activate(allowMixing: Bool = true) throws {
        let audioSession = AVAudioSession.sharedInstance()

        // Log if other audio is playing for debugging
        if audioSession.isOtherAudioPlaying {
            Logger.audio.info("Other audio is playing - will attempt to mix")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Other audio is playing - attempting mixed activation")
        }

        do {
            // setActive(true) can be called multiple times safely
            // AVAudioSession maintains an internal activation count
            // We don't use .notifyOthersOnDeactivation here as it's only for deactivation
            try audioSession.setActive(true)
            Logger.audio.info("Audio session activated")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio session activated")

            // Log state after activation for debugging
            logAudioSessionState("after-activate")
        } catch {
            // Handle specific errors that are acceptable
            let nsError = error as NSError
            // Error code -50 (kAudioSessionInvalidPropertySizeError) can occur on simulator
            // Error 560030580 (AVAudioSessionErrorCodeBadParam) can occur in some edge cases
            // Error 561017449 ('!act') can occur when trying to activate while other audio is playing
            //   with incompatible category - we handle this by using mixWithOthers option
            let ignorableErrorCodes: [Int] = [-50, 560030580]
            if nsError.domain == NSOSStatusErrorDomain && ignorableErrorCodes.contains(nsError.code) {
                Logger.audio.warning("Audio session activation warning (ignorable): \(error.localizedDescription)")
                FileLogger.shared.log(level: "WARNING", category: "audio", message: "Audio session activation warning: \(error.localizedDescription)")
                return
            }

            // Log detailed error for debugging
            Logger.audio.error("Audio session activation failed - domain: \(nsError.domain), code: \(nsError.code), description: \(error.localizedDescription)")
            FileLogger.shared.log(level: "ERROR", category: "audio", message: "Failed to activate audio session: domain=\(nsError.domain), code=\(nsError.code), description=\(error.localizedDescription)")
            throw error
        }
    }

    /// Activate the audio session only if no other audio is playing
    public func activateIfNeeded() throws {
        let audioSession = AVAudioSession.sharedInstance()

        guard !audioSession.isOtherAudioPlaying else {
            Logger.audio.info("Audio session activation skipped: other audio is playing")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio session activation skipped: other audio is playing")
            return
        }

        try activate()
    }

    /// Deactivate the audio session
    /// Note: Only deactivates if no active components are using the session
    public func deactivate() throws {
        // Check if other components are still using the audio session
        if hasActiveComponents {
            Logger.audio.info("Audio session deactivation skipped: \(self.activeComponentCount) component(s) still active")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Deactivation skipped: \(self.activeComponentCount) component(s) active")
            return
        }

        let audioSession = AVAudioSession.sharedInstance()

        do {
            try audioSession.setActive(false, options: .notifyOthersOnDeactivation)
            Logger.audio.info("Audio session deactivated")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio session deactivated")
        } catch {
            Logger.audio.logError(error)
            FileLogger.shared.log(level: "ERROR", category: "audio", message: "Failed to deactivate audio session: \(error.localizedDescription)")
            throw error
        }
    }

    /// Force deactivate the audio session regardless of active components
    /// Use only when shutting down or in error recovery
    public func forceDeactivate() throws {
        let audioSession = AVAudioSession.sharedInstance()

        // Reset component count
        countLock.lock()
        activeComponentCount = 0
        countLock.unlock()

        do {
            try audioSession.setActive(false, options: .notifyOthersOnDeactivation)
            Logger.audio.info("Audio session force deactivated")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio session force deactivated")
        } catch {
            Logger.audio.logError(error)
            FileLogger.shared.log(level: "ERROR", category: "audio", message: "Failed to force deactivate audio session: \(error.localizedDescription)")
            throw error
        }
    }

    /// Reset the cached session mode (call after recording ends)
    public func resetSessionMode() {
        sessionMode = nil
        Logger.audio.info("Audio session mode cache reset")
        FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio session mode cache reset")
    }

    // MARK: - Notification Observers

    private func setupNotificationObservers() {
        // Listen for audio session interruptions (e.g., phone calls)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleInterruption),
            name: AVAudioSession.interruptionNotification,
            object: nil
        )

        // Listen for route changes (e.g., headphone plug/unplug)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleRouteChange),
            name: AVAudioSession.routeChangeNotification,
            object: nil
        )

        // Listen for secondary audio hint changes (other apps playing audio)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleSilenceSecondaryAudioHint),
            name: AVAudioSession.silenceSecondaryAudioHintNotification,
            object: nil
        )
    }

    @objc private func handleSilenceSecondaryAudioHint(notification: Notification) {
        guard let userInfo = notification.userInfo,
              let typeValue = userInfo[AVAudioSessionSilenceSecondaryAudioHintTypeKey] as? UInt,
              let type = AVAudioSession.SilenceSecondaryAudioHintType(rawValue: typeValue) else {
            return
        }

        switch type {
        case .begin:
            Logger.audio.info("[DEBUG] Secondary audio hint: BEGIN - another app started playing audio")
            FileLogger.shared.log(level: "DEBUG", category: "audio", message: "Secondary audio hint: BEGIN - another app started playing audio")
            logAudioSessionState("secondary-audio-begin")

        case .end:
            Logger.audio.info("[DEBUG] Secondary audio hint: END - another app stopped playing audio")
            FileLogger.shared.log(level: "DEBUG", category: "audio", message: "Secondary audio hint: END - another app stopped playing audio")
            logAudioSessionState("secondary-audio-end")

        @unknown default:
            Logger.audio.warning("[DEBUG] Unknown secondary audio hint type: \(typeValue)")
            FileLogger.shared.log(level: "WARNING", category: "audio", message: "Unknown secondary audio hint type: \(typeValue)")
        }
    }

    @objc private func handleInterruption(notification: Notification) {
        guard let userInfo = notification.userInfo,
              let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else {
            return
        }

        switch type {
        case .began:
            Logger.audio.info("Audio session interrupted (e.g., phone call started)")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio session interrupted")

        case .ended:
            Logger.audio.info("Audio session interruption ended")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio session interruption ended")

            // Check if we should resume audio
            if let optionsValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt {
                let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue)
                if options.contains(.shouldResume) {
                    Logger.audio.info("Should resume audio after interruption")
                    FileLogger.shared.log(level: "INFO", category: "audio", message: "Should resume audio after interruption")
                    // Note: Actual resume logic should be handled by the audio components themselves
                }
            }

        @unknown default:
            Logger.audio.warning("Unknown audio session interruption type: \(typeValue)")
            FileLogger.shared.log(level: "WARNING", category: "audio", message: "Unknown interruption type: \(typeValue)")
        }
    }

    @objc private func handleRouteChange(notification: Notification) {
        guard let userInfo = notification.userInfo,
              let reasonValue = userInfo[AVAudioSessionRouteChangeReasonKey] as? UInt,
              let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue) else {
            return
        }

        switch reason {
        case .newDeviceAvailable:
            Logger.audio.info("New audio device available (e.g., headphones plugged in)")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "New audio device available")

        case .oldDeviceUnavailable:
            Logger.audio.info("Audio device removed (e.g., headphones unplugged)")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio device removed")
            // Note: Components should handle pausing playback/recording

        case .categoryChange:
            Logger.audio.info("Audio session category changed")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio session category changed")

        case .override:
            Logger.audio.info("Audio route override")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio route override")

        case .wakeFromSleep:
            Logger.audio.info("Audio session woke from sleep")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio session woke from sleep")

        case .noSuitableRouteForCategory:
            Logger.audio.warning("No suitable audio route for current category")
            FileLogger.shared.log(level: "WARNING", category: "audio", message: "No suitable audio route for category")

        case .routeConfigurationChange:
            Logger.audio.info("Audio route configuration changed")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio route configuration changed")

        @unknown default:
            Logger.audio.warning("Unknown audio route change reason: \(reasonValue)")
            FileLogger.shared.log(level: "WARNING", category: "audio", message: "Unknown route change reason: \(reasonValue)")
        }
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - Private Helpers

    /// Select optimal audio session mode based on current audio route
    /// - Returns: .measurement for high precision recording (all audio routes)
    private func selectOptimalMode(for audioSession: AVAudioSession) -> AVAudioSession.Mode {
        // Always use .measurement mode for maximum audio precision
        // This provides the most accurate pitch detection and highest quality recording
        // without any audio processing (no AGC, no noise reduction, no echo cancellation)
        let selectedMode: AVAudioSession.Mode = .measurement

        // Log detected outputs for debugging
        let currentRoute = audioSession.currentRoute
        let outputTypes = currentRoute.outputs.map { $0.portType.rawValue }.joined(separator: ", ")
        Logger.audio.info("Audio route detection: mode=\(String(describing: selectedMode)), outputs=\(outputTypes)")
        FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio route: mode=\(String(describing: selectedMode)), outputs=\(outputTypes)")

        return selectedMode
    }
}
