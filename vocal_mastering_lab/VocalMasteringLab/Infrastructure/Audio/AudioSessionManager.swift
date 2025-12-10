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
    }

    // MARK: - Audio Session Configuration

    /// Configure audio session for recording (with simultaneous playback support)
    public func configureForRecording() throws {
        let audioSession = AVAudioSession.sharedInstance()

        do {
            // Select mode based on audio route and cache it for the session
            // This prevents mode changes during recording (which would cause error -10868)
            let mode = selectOptimalMode(for: audioSession)
            sessionMode = mode

            // .playAndRecord: allows recording and playback simultaneously
            // Mode selection:
            //   - Headphones connected (Bluetooth or wired): .measurement (precision priority)
            //   - No headphones (built-in speaker/mic): .videoRecording (volume priority)
            // .defaultToSpeaker: plays audio through speaker even when recording
            // .allowBluetooth: supports bluetooth headsets for calls
            // .allowBluetoothA2DP: enables Bluetooth recording (required for Bluetooth microphone)
            try audioSession.setCategory(
                .playAndRecord,
                mode: mode,
                options: [.defaultToSpeaker, .allowBluetooth, .allowBluetoothA2DP]
            )

            // Set preferred sample rate (44.1 kHz for high quality)
            try audioSession.setPreferredSampleRate(44100.0)

            // Always set input gain to maximum when possible
            // Note: .videoRecording has auto-gain but iOS Simulator may not apply it correctly
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
        } catch {
            // Log detailed error information for debugging
            let nsError = error as NSError
            Logger.audio.logError(error)
            FileLogger.shared.log(level: "ERROR", category: "audio", message: "Failed to configure audio session for recording: \(error.localizedDescription), domain: \(nsError.domain), code: \(nsError.code)")
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
            // Mode selection:
            //   - Headphones (wired or Bluetooth): .measurement (precision priority)
            //   - Built-in: .videoRecording (volume priority with auto-gain)
            // Note: .measurement mode is incompatible with .defaultToSpeaker option
            // so we conditionally include .defaultToSpeaker based on mode
            var options: AVAudioSession.CategoryOptions = [.allowBluetooth, .allowBluetoothA2DP]
            if mode != .measurement {
                options.insert(.defaultToSpeaker)
            }

            try audioSession.setCategory(
                .playAndRecord,
                mode: mode,
                options: options
            )

            // Always set input gain to maximum when possible
            // Note: .videoRecording has auto-gain but iOS Simulator may not apply it correctly
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
    public func activate() throws {
        let audioSession = AVAudioSession.sharedInstance()

        do {
            // setActive(true) can be called multiple times safely
            // AVAudioSession maintains an internal activation count
            try audioSession.setActive(true)
            Logger.audio.info("Audio session activated")
            FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio session activated")
        } catch {
            // Handle specific errors that are acceptable
            let nsError = error as NSError
            // Error code -50 (kAudioSessionInvalidPropertySizeError) can occur on simulator
            // Error 560030580 (AVAudioSessionErrorCodeBadParam) can occur in some edge cases
            if nsError.domain == NSOSStatusErrorDomain && (nsError.code == -50 || nsError.code == 560030580) {
                Logger.audio.warning("Audio session activation warning (ignorable): \(error.localizedDescription)")
                FileLogger.shared.log(level: "WARNING", category: "audio", message: "Audio session activation warning: \(error.localizedDescription)")
                return
            }

            Logger.audio.logError(error)
            FileLogger.shared.log(level: "ERROR", category: "audio", message: "Failed to activate audio session: \(error.localizedDescription)")
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
    /// - Returns: .measurement for headphones (wired/Bluetooth), .videoRecording for built-in
    private func selectOptimalMode(for audioSession: AVAudioSession) -> AVAudioSession.Mode {
        let currentRoute = audioSession.currentRoute

        // Check if headphones (Bluetooth or wired) are connected
        let hasHeadphones = currentRoute.outputs.contains { output in
            switch output.portType {
            case .headphones:           // Wired headphones
                return true
            case .bluetoothHFP:         // Bluetooth Hands-Free Profile
                return true
            case .bluetoothA2DP:        // Bluetooth Advanced Audio Distribution Profile
                return true
            case .bluetoothLE:          // Bluetooth Low Energy
                return true
            default:
                return false
            }
        }

        // Select mode based on audio route:
        // - Headphones (wired or Bluetooth): .measurement (precision priority for pitch detection)
        // - Built-in speaker/mic: .videoRecording (volume priority with auto-gain)
        let selectedMode: AVAudioSession.Mode = hasHeadphones ? .measurement : .videoRecording

        // Log detected outputs for debugging
        let outputTypes = currentRoute.outputs.map { $0.portType.rawValue }.joined(separator: ", ")
        Logger.audio.info("Audio route detection: hasHeadphones=\(hasHeadphones), mode=\(String(describing: selectedMode)), outputs=\(outputTypes)")
        FileLogger.shared.log(level: "INFO", category: "audio", message: "Audio route: hasHeadphones=\(hasHeadphones), mode=\(String(describing: selectedMode)), outputs=\(outputTypes)")

        return selectedMode
    }
}
