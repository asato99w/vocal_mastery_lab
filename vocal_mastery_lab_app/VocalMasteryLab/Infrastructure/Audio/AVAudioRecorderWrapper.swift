import Foundation
import VocalisDomain
import AVFoundation
import OSLog

/// Wrapper for AVAudioRecorder that implements AudioRecorderProtocol
public class AVAudioRecorderWrapper: NSObject, AudioRecorderProtocol {

    private var audioRecorder: AVAudioRecorder?
    private var recordingURL: URL?
    private var startTime: Date?
    private var meteringTimer: DispatchSourceTimer?
    private var peakLevels: [Float] = []
    private var averageLevels: [Float] = []
    private let meteringQueue = DispatchQueue(label: "com.vocalis.metering")

    public var isRecording: Bool {
        return audioRecorder?.isRecording ?? false
    }

    public override init() {
        super.init()
    }

    // MARK: - AudioRecorderProtocol

    public func prepareRecording() async throws -> URL {
        Logger.recording.info("Preparing recording")

        // Configure audio session for recording using centralized manager
        do {
            try AudioSessionManager.shared.configureForRecording()
            try AudioSessionManager.shared.activate()
        } catch {
            Logger.recording.logError(error)
            throw AudioRecorderError.recordingFailed("Failed to configure audio session: \(error.localizedDescription)")
        }

        // Generate unique recording file URL
        let url = generateRecordingURL()
        recordingURL = url
        Logger.recording.debug("Recording URL: \(url.lastPathComponent)")

        // Configure audio settings - High quality Linear PCM (WAV)
        // 44.1kHz, 32-bit float, stereo for maximum quality
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 44100.0,
            AVNumberOfChannelsKey: 2,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false
        ]

        // Create AVAudioRecorder
        do {
            audioRecorder = try AVAudioRecorder(url: url, settings: settings)
            audioRecorder?.delegate = self
            audioRecorder?.isMeteringEnabled = true  // Enable metering for level monitoring
            audioRecorder?.prepareToRecord()

            Logger.recording.info("Recording prepared successfully")
            return url
        } catch {
            Logger.recording.logError(error)
            throw AudioRecorderError.recordingFailed("Failed to prepare recording: \(error.localizedDescription)")
        }
    }

    public func startRecording() async throws {
        guard let recorder = audioRecorder else {
            Logger.recording.error("Start recording failed: not prepared")
            throw AudioRecorderError.notPrepared
        }

        guard !recorder.isRecording else {
            Logger.recording.warning("Start recording ignored: already recording")
            throw AudioRecorderError.recordingFailed("Already recording")
        }

        // Start recording
        let success = recorder.record()
        if success {
            startTime = Date()
            peakLevels = []
            averageLevels = []
            startMeteringTimer()
            Logger.recording.info("Recording started")
            FileLogger.shared.log(level: "INFO", category: "recording", message: "Recording started with metering enabled")
        } else {
            Logger.recording.error("Failed to start AVAudioRecorder")
            throw AudioRecorderError.recordingFailed("Failed to start recording")
        }
    }

    public func stopRecording() async throws -> TimeInterval {
        guard let recorder = audioRecorder else {
            Logger.recording.error("Stop recording failed: not initialized")
            throw AudioRecorderError.notRecording
        }

        guard recorder.isRecording else {
            Logger.recording.warning("Stop recording ignored: not recording")
            throw AudioRecorderError.notRecording
        }

        // Stop metering and log results
        stopMeteringTimer()
        logMeteringResults()

        // Stop recording
        recorder.stop()
        Logger.recording.info("Recording stopped")

        // Calculate duration
        guard let startTime = startTime else {
            Logger.recording.warning("Recording duration unknown: startTime was nil")
            return 0
        }

        let duration = Date().timeIntervalSince(startTime)
        Logger.recording.info("Recording duration: \(String(format: "%.2f", duration))s")

        // Reset state
        self.startTime = nil

        return duration
    }

    // MARK: - Private Methods

    // MARK: - Metering (for input level investigation)

    private func startMeteringTimer() {
        let timer = DispatchSource.makeTimerSource(queue: meteringQueue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(100))
        timer.setEventHandler { [weak self] in
            self?.updateMeters()
        }
        meteringTimer = timer
        timer.resume()
        FileLogger.shared.log(level: "DEBUG", category: "recording_level", message: "Metering timer started (DispatchSource)")
    }

    private func stopMeteringTimer() {
        meteringTimer?.cancel()
        meteringTimer = nil
        FileLogger.shared.log(level: "DEBUG", category: "recording_level", message: "Metering timer stopped, samples collected: \(peakLevels.count)")
    }

    private func updateMeters() {
        guard let recorder = audioRecorder, recorder.isRecording else { return }
        recorder.updateMeters()

        let peak = recorder.peakPower(forChannel: 0)
        let average = recorder.averagePower(forChannel: 0)

        peakLevels.append(peak)
        averageLevels.append(average)
    }

    private func logMeteringResults() {
        guard !peakLevels.isEmpty else {
            FileLogger.shared.log(level: "WARNING", category: "recording_level", message: "No metering data collected")
            return
        }

        let maxPeak = peakLevels.max() ?? -160
        let avgOfPeaks = peakLevels.reduce(0, +) / Float(peakLevels.count)
        let avgOfAvg = averageLevels.reduce(0, +) / Float(averageLevels.count)

        // dBFS scale: 0 = full scale, -160 = silence
        // Typical healthy recording levels: peak -6 to -3 dBFS, average -20 to -12 dBFS
        let levelAssessment: String
        if maxPeak > -3 {
            levelAssessment = "⚠️ CLIPPING (too loud)"
        } else if maxPeak > -12 {
            levelAssessment = "✅ GOOD"
        } else if maxPeak > -24 {
            levelAssessment = "⚠️ LOW (may need gain boost)"
        } else {
            levelAssessment = "❌ VERY LOW (likely inaudible)"
        }

        let message = """
        📊 RECORDING INPUT LEVELS:
        - Max Peak: \(String(format: "%.1f", maxPeak)) dBFS
        - Avg Peak: \(String(format: "%.1f", avgOfPeaks)) dBFS
        - Avg Level: \(String(format: "%.1f", avgOfAvg)) dBFS
        - Samples: \(peakLevels.count)
        - Assessment: \(levelAssessment)
        """

        FileLogger.shared.log(level: "INFO", category: "recording_level", message: message)
        Logger.recording.info("Recording levels - MaxPeak: \(maxPeak) dBFS, Assessment: \(levelAssessment)")
    }

    private func generateRecordingURL() -> URL {
        // Use Documents directory for persistent storage
        let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

        // Generate filename with timestamp + milliseconds for uniqueness
        // Format: recording_yyyyMMdd_HHmmss_SSS.wav
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = dateFormatter.string(from: Date())

        // Add milliseconds for uniqueness when called in rapid succession
        let milliseconds = Int(Date().timeIntervalSince1970 * 1000) % 1000
        let fileName = "recording_\(timestamp)_\(String(format: "%03d", milliseconds)).wav"

        let url = documentsDir.appendingPathComponent(fileName)
        return url
    }
}

// MARK: - AVAudioRecorderDelegate

extension AVAudioRecorderWrapper: AVAudioRecorderDelegate {

    public func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        if flag {
            Logger.recording.info("Recording finished successfully")
        } else {
            Logger.recording.error("Recording finished with failure")
        }
    }

    public func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
        if let error = error {
            Logger.recording.error("Encoding error: \(error.localizedDescription)")
        }
    }
}
