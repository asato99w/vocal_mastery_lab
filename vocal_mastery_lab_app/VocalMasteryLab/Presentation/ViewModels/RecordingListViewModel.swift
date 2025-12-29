import Foundation
import VocalisDomain
import Combine

/// ViewModel for recording list screen
@MainActor
public class RecordingListViewModel: ObservableObject {
    @Published public private(set) var recordings: [Recording] = []
    @Published public private(set) var isLoading: Bool = false
    @Published public private(set) var errorMessage: String?
    @Published public private(set) var isPlaying: Bool = false
    @Published public private(set) var currentTime: Double = 0.0
    @Published public private(set) var currentPlaybackPosition: [RecordingId: TimeInterval] = [:]
    @Published public private(set) var selectedRecording: Recording?
    @Published public private(set) var extractedRecordingIds: Set<RecordingId> = []
    @Published public var selectedAudioSource: AudioSourceType = .original
    @Published public private(set) var extractedAudios: [RecordingId: [ExtractedAudio]] = [:]
    @Published public private(set) var currentPlayingSource: AudioSourceType = .original

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
        guard let recording = selectedRecording else { return }
        isPlaying = false
        stopPositionTracking()
        currentPlaybackPosition[recording.id] = 0.0
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
            var audiosMap: [RecordingId: [ExtractedAudio]] = [:]

            for extracted in allExtracted {
                extractedIds.insert(extracted.sourceRecordingId)
                if audiosMap[extracted.sourceRecordingId] == nil {
                    audiosMap[extracted.sourceRecordingId] = []
                }
                audiosMap[extracted.sourceRecordingId]?.append(extracted)
            }

            extractedRecordingIds = extractedIds
            extractedAudios = audiosMap
        } catch {
            // Silently ignore extraction status errors
        }
    }

    /// Check if a recording has extracted audio
    public func hasExtractedAudio(_ recording: Recording) -> Bool {
        extractedRecordingIds.contains(recording.id)
    }

    /// Get available audio sources for a recording
    public func availableSources(for recording: Recording) -> [AudioSourceType] {
        var sources: [AudioSourceType] = [.original]

        if let audios = extractedAudios[recording.id] {
            if audios.contains(where: { $0.type == .vocal }) {
                sources.append(.vocal)
            }
            if audios.contains(where: { $0.type == .instrumental }) {
                sources.append(.instrumental)
            }
        }

        return sources
    }

    /// Check if audio source is available for selected recording
    public func isSourceAvailable(_ source: AudioSourceType) -> Bool {
        guard let recording = selectedRecording else { return source == .original }
        return availableSources(for: recording).contains(source)
    }

    /// Get extracted audio for a specific type
    public func getExtractedAudio(for recording: Recording, type: ExtractionType) -> ExtractedAudio? {
        extractedAudios[recording.id]?.first { $0.type == type }
    }

    /// Get file URL for the selected audio source
    private func getPlaybackURL(for recording: Recording, source: AudioSourceType) -> URL? {
        switch source {
        case .original:
            return recording.fileURL
        case .vocal:
            return getExtractedAudio(for: recording, type: .vocal)?.fileURL
        case .instrumental:
            return getExtractedAudio(for: recording, type: .instrumental)?.fileURL
        }
    }

    /// Get duration for the selected audio source
    public func getDuration(for recording: Recording, source: AudioSourceType) -> Double {
        switch source {
        case .original:
            return recording.duration.seconds
        case .vocal:
            return getExtractedAudio(for: recording, type: .vocal)?.duration.seconds ?? recording.duration.seconds
        case .instrumental:
            return getExtractedAudio(for: recording, type: .instrumental)?.duration.seconds ?? recording.duration.seconds
        }
    }

    /// Play a recording with specified audio source
    public func playRecording(_ recording: Recording, source: AudioSourceType? = nil) async {
        let audioSource = source ?? selectedAudioSource

        // Validate source is available
        guard let url = getPlaybackURL(for: recording, source: audioSource) else {
            // Fallback to original if selected source is not available
            if audioSource != .original {
                await playRecording(recording, source: .original)
            }
            return
        }

        isPlaying = true
        currentPlayingSource = audioSource

        // Start playback without waiting for completion
        Task {
            do {
                try await audioPlayer.play(url: url)
                // Playback finished naturally
                await MainActor.run {
                    if selectedRecording?.id == recording.id {
                        isPlaying = false
                        stopPositionTracking()
                        // Reset position to beginning
                        currentPlaybackPosition[recording.id] = 0.0
                        currentTime = 0.0
                    }
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    if selectedRecording?.id == recording.id {
                        isPlaying = false
                        stopPositionTracking()
                        // Reset position to beginning on error too
                        currentPlaybackPosition[recording.id] = 0.0
                        currentTime = 0.0
                    }
                }
            }
        }
    }

    /// Switch audio source and restart playback
    public func switchAudioSource(to source: AudioSourceType) async {
        guard let recording = selectedRecording else { return }

        // Only switch if the source is available
        guard isSourceAvailable(source) else { return }

        // Don't do anything if already on this source
        guard selectedAudioSource != source else { return }

        selectedAudioSource = source

        // Stop current playback and start with new source
        // This works whether currently playing or paused
        await audioPlayer.stop()
        currentPlaybackPosition[recording.id] = 0.0
        currentTime = 0.0
        await playRecording(recording, source: source)
        await startPositionTracking()
    }

    /// Pause playback (keeps position)
    public func pausePlayback() {
        audioPlayer.pause()
        isPlaying = false
    }

    /// Resume playback from current position
    public func resumePlayback() {
        guard selectedRecording != nil else { return }
        audioPlayer.resume()
        isPlaying = true
    }

    /// Stop playback completely (resets position and selection)
    public func stopPlayback() async {
        await audioPlayer.stop()
        stopPositionTracking()
        isPlaying = false
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
            if selectedRecording?.id == recording.id {
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
                if let recording = selectedRecording, isPlaying {
                    let position = audioPlayer.currentTime
                    currentTime = position
                    currentPlaybackPosition[recording.id] = position
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
        // Allow seeking if this recording is selected (playing or paused)
        guard selectedRecording?.id == recordingId else { return }
        audioPlayer.seek(to: position)
        currentPlaybackPosition[recordingId] = position
        currentTime = position
    }

    /// Update position immediately (synchronous) for responsive UI during slider drag
    public func updatePositionImmediate(_ position: TimeInterval, for recordingId: RecordingId) {
        guard selectedRecording?.id == recordingId else { return }
        currentPlaybackPosition[recordingId] = position
        currentTime = position
    }

    /// Seek audio to position (async) - call after updatePositionImmediate for actual audio seek
    public func seekAudio(to position: TimeInterval) {
        audioPlayer.seek(to: position)
    }

    // MARK: - Selection and Playback Control

    /// Select a recording and start playback
    public func selectAndPlay(_ recording: Recording) async {
        // If same recording is already selected, toggle playback instead
        if selectedRecording?.id == recording.id {
            await togglePlayback()
            return
        }

        // Different recording selected - stop current playback completely
        // Must use stop() instead of pause() to properly release the continuation
        // in AVAudioPlayerWrapper, otherwise the old continuation remains and
        // prevents new playback from starting
        if isPlaying {
            await audioPlayer.stop()
            isPlaying = false
            stopPositionTracking()
        }

        // Reset audio source to original when selecting new recording
        selectedAudioSource = .original

        // Select recording and update UI immediately
        selectedRecording = recording

        // Pre-configure audio session to reduce latency
        try? AudioSessionManager.shared.configureForPlayback()
        try? AudioSessionManager.shared.activate()

        await playRecording(recording, source: .original)
        await startPositionTracking()
    }

    /// Toggle playback for selected recording (pause/resume)
    public func togglePlayback() async {
        guard selectedRecording != nil else { return }

        if isPlaying {
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
        selectedAudioSource = .original  // Reset to original when navigating
        await playRecording(previousRecording, source: .original)
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
        selectedAudioSource = .original  // Reset to original when navigating
        await playRecording(nextRecording, source: .original)
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
