import Foundation
import VocalisDomain

/// Protocol for audio file analysis
public protocol AudioFileAnalyzerProtocol {
    /// Analyze audio file and return pitch and spectrogram data
    /// - Parameters:
    ///   - fileURL: URL of audio file to analyze
    ///   - progress: Callback for progress updates (0.0 to 1.0), called on MainActor
    func analyze(fileURL: URL, progress: @escaping @MainActor (Double) async -> Void) async throws -> (pitchData: PitchAnalysisData, spectrogramData: SpectrogramData)

    /// Analyze spectrogram only (when pitch data is cached)
    /// - Parameters:
    ///   - fileURL: URL of audio file to analyze
    ///   - progress: Callback for progress updates (0.0 to 1.0), called on MainActor
    func analyzeSpectrogramOnly(fileURL: URL, progress: @escaping @MainActor (Double) async -> Void) async throws -> SpectrogramData
}

/// Protocol for analysis result caching (in-memory)
public protocol AnalysisCacheProtocol {
    /// Get cached analysis result
    func get(_ id: RecordingId) -> AnalysisResult?

    /// Set analysis result to cache
    func set(_ id: RecordingId, result: AnalysisResult)

    /// Clear all cache
    func clear()
}

/// Protocol for file-based pitch data caching (persistent)
public protocol PitchDataCacheProtocol {
    /// Get cached pitch data
    func get(_ id: RecordingId) -> PitchAnalysisData?

    /// Set pitch data to cache
    func set(_ id: RecordingId, pitchData: PitchAnalysisData)

    /// Delete cached pitch data
    func delete(_ id: RecordingId)

    /// Check if pitch data exists
    func exists(_ id: RecordingId) -> Bool
}

/// Factory protocol to create AudioFileAnalyzer instances with current settings
public protocol AudioFileAnalyzerFactoryProtocol {
    /// Create a new audio file analyzer with current settings
    func makeAnalyzer() -> AudioFileAnalyzerProtocol
}

/// Simple factory that always returns the same analyzer (for backward compatibility)
private class StaticAnalyzerFactory: AudioFileAnalyzerFactoryProtocol {
    private let analyzer: AudioFileAnalyzerProtocol

    init(analyzer: AudioFileAnalyzerProtocol) {
        self.analyzer = analyzer
    }

    func makeAnalyzer() -> AudioFileAnalyzerProtocol {
        return analyzer
    }
}

/// Use case for analyzing recorded audio files
/// Analyzes pitch and spectrogram data from audio files
///
/// Cache hierarchy:
/// 1. In-memory cache (AnalysisCache) - Full analysis results, fast access
/// 2. File cache (FilePitchDataCache) - Pitch data only, persists across app restarts
/// 3. Full analysis - When no cache exists or algorithm changed
@MainActor
public class AnalyzeRecordingUseCase {
    private let analyzerFactory: AudioFileAnalyzerFactoryProtocol
    private let analysisCache: AnalysisCacheProtocol
    private let pitchDataCache: PitchDataCacheProtocol?
    private let audioSettingsRepository: AudioSettingsRepositoryProtocol?
    private let recordingRepository: RecordingRepositoryProtocol?
    private let logger: LoggerProtocol

    /// Get a new analyzer instance with current settings
    private var audioFileAnalyzer: AudioFileAnalyzerProtocol {
        analyzerFactory.makeAnalyzer()
    }

    /// Current pitch detection algorithm from settings
    private var currentAlgorithm: PitchDetectionAlgorithm {
        audioSettingsRepository?.get().pitchAlgorithm ?? .yin
    }

    public init(
        analyzerFactory: AudioFileAnalyzerFactoryProtocol,
        analysisCache: AnalysisCacheProtocol,
        pitchDataCache: PitchDataCacheProtocol? = nil,
        audioSettingsRepository: AudioSettingsRepositoryProtocol? = nil,
        recordingRepository: RecordingRepositoryProtocol? = nil,
        logger: LoggerProtocol
    ) {
        self.analyzerFactory = analyzerFactory
        self.analysisCache = analysisCache
        self.pitchDataCache = pitchDataCache
        self.audioSettingsRepository = audioSettingsRepository
        self.recordingRepository = recordingRepository
        self.logger = logger
    }

    /// Legacy initializer for backward compatibility
    public convenience init(
        audioFileAnalyzer: AudioFileAnalyzerProtocol,
        analysisCache: AnalysisCacheProtocol,
        pitchDataCache: PitchDataCacheProtocol? = nil,
        logger: LoggerProtocol
    ) {
        // Create a simple factory that always returns the same analyzer
        let factory = StaticAnalyzerFactory(analyzer: audioFileAnalyzer)
        self.init(
            analyzerFactory: factory,
            analysisCache: analysisCache,
            pitchDataCache: pitchDataCache,
            logger: logger
        )
    }

    /// Analyze recording and return analysis result
    /// - Parameters:
    ///   - recording: Recording to analyze
    ///   - progress: Callback for progress updates (0.0 to 1.0), called on MainActor
    /// - Returns: Analysis result with pitch and spectrogram data
    /// - Throws: Error if file reading or analysis fails
    public func execute(recording: Recording, progress: @escaping @MainActor (Double) -> Void = { _ in }) async throws -> AnalysisResult {
        logger.info("Starting analysis for recording: \(recording.id.value.uuidString)", category: "useCase")

        let algorithm = currentAlgorithm
        let algorithmChanged = recording.analysisAlgorithm != nil && recording.analysisAlgorithm != algorithm

        if algorithmChanged {
            logger.info("Algorithm changed from \(recording.analysisAlgorithm?.rawValue ?? "nil") to \(algorithm.rawValue), invalidating cache", category: "useCase")
            // Clear in-memory cache for this recording
            analysisCache.clear()
            // Clear file cache for this recording
            pitchDataCache?.delete(recording.id)
        }

        // Layer 1: Check in-memory cache first (full result) - only if algorithm hasn't changed
        if !algorithmChanged, let cachedResult = analysisCache.get(recording.id) {
            logger.info("In-memory cache hit for recording: \(recording.id.value.uuidString)", category: "useCase")
            await progress(1.0)
            return cachedResult
        }

        // Layer 2: Check file cache for pitch data - only if algorithm hasn't changed
        if !algorithmChanged,
           let pitchDataCache = pitchDataCache,
           let cachedPitchData = pitchDataCache.get(recording.id) {
            logger.info("File cache hit for pitch data, analyzing spectrogram only: \(recording.id.value.uuidString)", category: "useCase")

            // Analyze spectrogram only (faster - skips YIN algorithm)
            let spectrogramData = try await audioFileAnalyzer.analyzeSpectrogramOnly(
                fileURL: recording.fileURL,
                progress: progress
            )

            // Create analysis result
            let result = AnalysisResult(
                pitchData: cachedPitchData,
                spectrogramData: spectrogramData
            )

            // Update in-memory cache
            analysisCache.set(recording.id, result: result)

            logger.info("Analysis completed (pitch from cache) for recording: \(recording.id.value.uuidString)", category: "useCase")

            return result
        }

        // Layer 3: Full analysis required
        logger.info("Cache miss - full analysis for file: \(recording.fileURL.path) with algorithm: \(algorithm.rawValue)", category: "useCase")

        // Analyze audio file with progress reporting
        let (pitchData, spectrogramData) = try await audioFileAnalyzer.analyze(
            fileURL: recording.fileURL,
            progress: progress
        )

        // Save pitch data to file cache for persistence
        pitchDataCache?.set(recording.id, pitchData: pitchData)

        // Create analysis result
        let result = AnalysisResult(
            pitchData: pitchData,
            spectrogramData: spectrogramData
        )

        // Cache the corrected results in memory
        analysisCache.set(recording.id, result: result)

        // Update recording with the algorithm used for analysis
        await updateRecordingAlgorithm(recording: recording, algorithm: algorithm)

        logger.info("Analysis completed (full) for recording: \(recording.id.value.uuidString)", category: "useCase")

        return result
    }

    /// Update the recording's analysisAlgorithm property
    private func updateRecordingAlgorithm(recording: Recording, algorithm: PitchDetectionAlgorithm) async {
        guard let repository = recordingRepository else { return }

        var updatedRecording = recording
        updatedRecording.analysisAlgorithm = algorithm

        do {
            try await repository.update(updatedRecording)
            logger.info("Updated recording analysisAlgorithm to \(algorithm.rawValue)", category: "useCase")
        } catch {
            logger.error("Failed to update recording analysisAlgorithm: \(error.localizedDescription)", category: "useCase")
        }
    }

    /// Check if valid cached data exists for a recording
    /// Returns false if algorithm has changed since last analysis
    /// - Parameter recording: Recording to check
    /// - Returns: true if valid cached data exists with matching algorithm
    public func hasCachedData(for recording: Recording) -> Bool {
        // If recording was analyzed with a different algorithm, cache is invalid
        if let analyzedWith = recording.analysisAlgorithm,
           analyzedWith != currentAlgorithm {
            return false
        }

        let inMemory = analysisCache.get(recording.id) != nil
        let inFile = pitchDataCache?.exists(recording.id) ?? false
        return inMemory || inFile
    }

    /// Legacy method for backward compatibility
    /// - Parameter recordingId: Recording identifier
    /// - Returns: true if cached data exists (does not check algorithm)
    public func hasCachedData(for recordingId: RecordingId) -> Bool {
        let inMemory = analysisCache.get(recordingId) != nil
        let inFile = pitchDataCache?.exists(recordingId) ?? false
        return inMemory || inFile
    }
}
