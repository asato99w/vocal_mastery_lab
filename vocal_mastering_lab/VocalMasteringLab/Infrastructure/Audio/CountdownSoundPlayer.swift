import Foundation
import AVFoundation
import OSLog

/// Plays countdown click sounds using SF2 sound bank
/// Uses AVAudioEngine + AVAudioUnitSampler to play sounds even in silent mode
/// (because AudioSession is configured for .playAndRecord)
public class CountdownSoundPlayer {

    // MARK: - Singleton

    public static let shared = CountdownSoundPlayer()

    // MARK: - Properties

    private var engine: AVAudioEngine?
    private var sampler: AVAudioUnitSampler?
    private var isLoaded = false

    // Woodblock sound: MIDI note 76 (High Wood Block) on percussion channel
    // GM Percussion channel is 10 (0-indexed: 9)
    private let percussionChannel: UInt8 = 9
    private let woodblockNote: UInt8 = 76  // High Wood Block
    private let velocity: UInt8 = 100

    // MARK: - Initialization

    private init() {
        setupEngine()
    }

    // MARK: - Setup

    private func setupEngine() {
        engine = AVAudioEngine()
        sampler = AVAudioUnitSampler()

        guard let engine = engine, let sampler = sampler else { return }

        engine.attach(sampler)
        engine.connect(sampler, to: engine.mainMixerNode, format: nil)
        engine.mainMixerNode.outputVolume = 1.0

        Logger.audio.info("CountdownSoundPlayer: Engine setup complete")
    }

    /// Load SF2 sound bank for percussion sounds
    public func loadSoundBank() async throws {
        guard !isLoaded else { return }

        guard let sampler = sampler else {
            throw CountdownSoundError.engineNotInitialized
        }

        guard let sf2URL = Bundle.main.url(forResource: "GeneralUserGS", withExtension: "sf2") else {
            Logger.audio.error("CountdownSoundPlayer: SF2 file not found")
            throw CountdownSoundError.soundBankNotFound
        }

        do {
            // Load percussion bank (bank 128 for drums in GM)
            try sampler.loadSoundBankInstrument(
                at: sf2URL,
                program: 0,
                bankMSB: UInt8(kAUSampler_DefaultPercussionBankMSB),
                bankLSB: UInt8(kAUSampler_DefaultBankLSB)
            )
            isLoaded = true
            Logger.audio.info("CountdownSoundPlayer: SF2 percussion bank loaded")
        } catch {
            Logger.audio.error("CountdownSoundPlayer: Failed to load SF2: \(error.localizedDescription)")
            throw CountdownSoundError.loadFailed(error.localizedDescription)
        }
    }

    /// Play countdown click sound
    /// This works even in silent mode because AudioSession is set to .playAndRecord
    public func playClick() async {
        guard let engine = engine, let sampler = sampler else {
            Logger.audio.warning("CountdownSoundPlayer: Engine not initialized")
            return
        }

        // Load sound bank if not loaded
        if !isLoaded {
            do {
                try await loadSoundBank()
            } catch {
                Logger.audio.error("CountdownSoundPlayer: Failed to load sound bank: \(error.localizedDescription)")
                return
            }
        }

        do {
            // Start engine if not running
            if !engine.isRunning {
                try engine.start()
                Logger.audio.info("CountdownSoundPlayer: Engine started")
            }

            // Play woodblock sound
            sampler.startNote(woodblockNote, withVelocity: velocity, onChannel: percussionChannel)

            // Note off after short duration (50ms)
            try? await Task.sleep(nanoseconds: 50_000_000)
            sampler.stopNote(woodblockNote, onChannel: percussionChannel)

            Logger.audio.info("CountdownSoundPlayer: Click played")

        } catch {
            Logger.audio.error("CountdownSoundPlayer: Failed to play click: \(error.localizedDescription)")
        }
    }

    /// Stop engine to release resources
    public func stop() {
        engine?.stop()
        Logger.audio.info("CountdownSoundPlayer: Engine stopped")
    }
}

// MARK: - Errors

public enum CountdownSoundError: Error, LocalizedError {
    case engineNotInitialized
    case soundBankNotFound
    case loadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .engineNotInitialized:
            return "Audio engine not initialized"
        case .soundBankNotFound:
            return "SF2 sound bank not found"
        case .loadFailed(let message):
            return "Failed to load sound bank: \(message)"
        }
    }
}
