import Foundation
import VocalisDomain
import Combine

/// ViewModel for recording list screen
@MainActor
public class RecordingListViewModel: ObservableObject {
    @Published public private(set) var recordings: [Recording] = []
    @Published public private(set) var isLoading: Bool = false
    @Published public private(set) var errorMessage: String?
    @Published public private(set) var playingRecordingId: RecordingId?
    @Published public private(set) var currentTime: Double = 0.0
    @Published public private(set) var currentPlaybackPosition: [RecordingId: TimeInterval] = [:]
    @Published public private(set) var selectedRecording: Recording?
    @Published public private(set) var extractedRecordingIds: Set<RecordingId> = []

    private let recordingRepository: RecordingRepositoryProtocol
    private let extractedAudioRepository: ExtractedAudioRepositoryProtocol
    private let audioPlayer: AudioPlayerProtocol
    private var positionTrackingTask: Task<Void, Never>?
    private var playbackFinishObserver: NSObjectProtocol?

    public init(
        recordingRepository: RecordingRepositoryProtocol,
        extractedAudioRepository: ExtractedAudioRepositoryProtocol,
        audioPlayer: AudioPlayerProtocol
    ) {
        self.recordingRepository = recordingRepository
        self.extractedAudioRepository = extractedAudioRepository
        self.audioPlayer = audioPlayer

        // Observe playback finish notification
        playbackFinishObserver = NotificationCenter.default.addObserver(
            forName: .audioPlaybackDidFinish,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.handlePlaybackFinished()
            }
        }
    }

    deinit {
        if let observer = playbackFinishObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    /// Handle playback finished notification
    private func handlePlaybackFinished() {
        guard let recordingId = playingRecordingId else { return }
        playingRecordingId = nil
        stopPositionTracking()
        currentPlaybackPosition[recordingId] = 0.0
        currentTime = 0.0
    }

    /// Load all recordings
    public func loadRecordings() async {
        isLoading = true
        errorMessage = nil

        do {
            recordings = try await recordingRepository.findAll()
            await loadExtractionStatus()
        } catch {
            errorMessage = error.localizedDescription
        }

        isLoading = false
    }

    /// Load extraction status for all recordings
    private func loadExtractionStatus() async {
        do {
            let allExtracted = try await extractedAudioRepository.findAll()
            var extractedIds = Set<RecordingId>()
            for extracted in allExtracted {
                extractedIds.insert(extracted.sourceRecordingId)
            }
            extractedRecordingIds = extractedIds
        } catch {
            // Silently ignore extraction status errors
        }
    }

    /// Check if a recording has extracted audio
    public func hasExtractedAudio(_ recording: Recording) -> Bool {
        extractedRecordingIds.contains(recording.id)
    }

    /// Play a recording (assumes any previous playback is already stopped)
    public func playRecording(_ recording: Recording) async {
        playingRecordingId = recording.id

        // Start playback without waiting for completion
        Task {
            do {
                try await audioPlayer.play(url: recording.fileURL)
                // Playback finished naturally
                await MainActor.run {
                    if playingRecordingId == recording.id {
                        playingRecordingId = nil
                        stopPositionTracking()
                        // Reset position to beginning
                        currentPlaybackPosition[recording.id] = 0.0
                        currentTime = 0.0
                    }
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    if playingRecordingId == recording.id {
                        playingRecordingId = nil
                        stopPositionTracking()
                        // Reset position to beginning on error too
                        currentPlaybackPosition[recording.id] = 0.0
                        currentTime = 0.0
                    }
                }
            }
        }
    }

    /// Pause playback (keeps position)
    public func pausePlayback() {
        audioPlayer.pause()
        playingRecordingId = nil
    }

    /// Resume playback from current position
    public func resumePlayback() {
        guard let recording = selectedRecording else { return }
        audioPlayer.resume()
        playingRecordingId = recording.id
    }

    /// Stop playback completely (resets position and selection)
    public func stopPlayback() async {
        await audioPlayer.stop()
        stopPositionTracking()
        playingRecordingId = nil
        if let recording = selectedRecording {
            currentPlaybackPosition[recording.id] = 0.0
        }
        currentTime = 0.0
        selectedRecording = nil  // Reset selection to allow re-selecting same recording
    }

    /// Delete a recording
    public func deleteRecording(_ recording: Recording) async {
        do {
            // Stop playback if this recording is playing
            if playingRecordingId == recording.id {
                await stopPlayback()
            }

            // Clear selection if this recording is selected
            if selectedRecording?.id == recording.id {
                selectedRecording = nil
            }

            // Delete associated extracted audio
            try await extractedAudioRepository.deleteByRecording(recording.id)

            try await recordingRepository.delete(recording.id)

            // Reload recordings
            await loadRecordings()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Rename a recording
    public func renameRecording(_ recording: Recording, newTitle: String?) async {
        do {
            var updatedRecording = recording
            // Trim whitespace and set nil if empty
            let trimmedTitle = newTitle?.trimmingCharacters(in: .whitespacesAndNewlines)
            updatedRecording.title = trimmedTitle?.isEmpty == true ? nil : trimmedTitle

            try await recordingRepository.update(updatedRecording)

            // Update local state
            if let index = recordings.firstIndex(where: { $0.id == recording.id }) {
                recordings[index] = updatedRecording
            }

            // Update selected recording if it was renamed
            if selectedRecording?.id == recording.id {
                selectedRecording = updatedRecording
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Start position tracking
    public func startPositionTracking() async {
        stopPositionTracking()

        positionTrackingTask = Task { @MainActor in
            while !Task.isCancelled {
                if let recordingId = playingRecordingId {
                    let position = audioPlayer.currentTime
                    currentTime = position
                    currentPlaybackPosition[recordingId] = position
                }
                try? await Task.sleep(nanoseconds: 100_000_000) // 100ms update interval
            }
        }
    }

    /// Stop position tracking
    public func stopPositionTracking() {
        positionTrackingTask?.cancel()
        positionTrackingTask = nil
    }

    /// Seek to a specific position
    public func seekToPosition(_ time: Double) async {
        audioPlayer.seek(to: time)
        currentTime = time
    }

    /// Seek to a specific position for a specific recording
    public func seek(to position: TimeInterval, for recordingId: RecordingId) async {
        guard playingRecordingId == recordingId else { return }
        audioPlayer.seek(to: position)
        currentPlaybackPosition[recordingId] = position
        currentTime = position
    }

    // MARK: - Selection and Playback Control

    /// Select a recording and start playback
    public func selectAndPlay(_ recording: Recording) async {
        // If same recording is already selected, do nothing
        // (pause/resume is handled by the panel's play button)
        if selectedRecording?.id == recording.id {
            return
        }

        // Different recording selected - stop current playback immediately
        if playingRecordingId != nil {
            // Use synchronous stop to avoid blocking UI
            audioPlayer.pause()
            playingRecordingId = nil
            stopPositionTracking()
        }

        // Select recording and update UI immediately
        selectedRecording = recording
        playingRecordingId = recording.id  // Update UI before async operation

        // Pre-configure audio session to reduce latency
        try? AudioSessionManager.shared.configureForPlayback()
        try? AudioSessionManager.shared.activate()

        await playRecording(recording)
        await startPositionTracking()
    }

    /// Toggle playback for selected recording (pause/resume)
    public func togglePlayback() async {
        guard selectedRecording != nil else { return }

        if audioPlayer.isPlaying {
            // Currently playing -> pause
            pausePlayback()
        } else {
            // Currently paused -> resume
            resumePlayback()
            await startPositionTracking()
        }
    }

    /// Play previous recording in list
    public func playPrevious() async {
        guard let current = selectedRecording,
              let currentIndex = recordings.firstIndex(where: { $0.id == current.id }),
              currentIndex > 0 else {
            return
        }

        let previousRecording = recordings[currentIndex - 1]
        selectedRecording = previousRecording
        await playRecording(previousRecording)
        await startPositionTracking()
    }

    /// Play next recording in list
    public func playNext() async {
        guard let current = selectedRecording,
              let currentIndex = recordings.firstIndex(where: { $0.id == current.id }),
              currentIndex < recordings.count - 1 else {
            return
        }

        let nextRecording = recordings[currentIndex + 1]
        selectedRecording = nextRecording
        await playRecording(nextRecording)
        await startPositionTracking()
    }

    /// Check if can play previous recording
    public var canPlayPrevious: Bool {
        guard let current = selectedRecording,
              let currentIndex = recordings.firstIndex(where: { $0.id == current.id }) else {
            return false
        }
        return currentIndex > 0
    }

    /// Check if can play next recording
    public var canPlayNext: Bool {
        guard let current = selectedRecording,
              let currentIndex = recordings.firstIndex(where: { $0.id == current.id }) else {
            return false
        }
        return currentIndex < recordings.count - 1
    }
}
