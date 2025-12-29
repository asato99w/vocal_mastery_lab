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
    ///   - recording: Recording to analyze (used for cache key)
    ///   - audioURL: Optional URL to analyze (if nil, uses recording.fileURL)
    ///   - progress: Callback for progress updates (0.0 to 1.0), called on MainActor
    /// - Returns: Analysis result with pitch and spectrogram data
    /// - Throws: Error if file reading or analysis fails
    public func execute(recording: Recording, audioURL: URL? = nil, progress: @escaping @MainActor (Double) -> Void = { _ in }) async throws -> AnalysisResult {
        let fileURL = audioURL ?? recording.fileURL
        logger.info("Starting analysis for recording: \(recording.id.value.uuidString), file: \(fileURL.lastPathComponent)", category: "useCase")

        // DEBUG: Log cache state for diagnosis
        logger.debug("========================================", category: "cache")
        logger.debug("Recording ID: \(recording.id.value.uuidString)", category: "cache")
        logger.debug("Recording analysisAlgorithm: \(recording.analysisAlgorithm?.rawValue ?? "nil")", category: "cache")
        logger.debug("Audio URL: \(fileURL.lastPathComponent)", category: "cache")

        let algorithm = currentAlgorithm
        logger.debug("Current algorithm: \(algorithm.rawValue)", category: "cache")

        let algorithmChanged = recording.analysisAlgorithm != nil && recording.analysisAlgorithm != algorithm
        logger.debug("Algorithm changed: \(algorithmChanged)", category: "cache")

        if algorithmChanged {
            logger.info("Algorithm changed from \(recording.analysisAlgorithm?.rawValue ?? "nil") to \(algorithm.rawValue), invalidating cache", category: "useCase")
            logger.debug(">>> INVALIDATING CACHE due to algorithm change", category: "cache")
            // Clear in-memory cache for this recording
            analysisCache.clear()
            // Clear file cache for this recording
            pitchDataCache?.delete(recording.id)
        }

        // Layer 1: Check in-memory cache first (full result) - only if algorithm hasn't changed
        let inMemoryCached = analysisCache.get(recording.id)
        logger.debug("In-memory cache exists: \(inMemoryCached != nil)", category: "cache")
        if !algorithmChanged, let cachedResult = inMemoryCached {
            logger.info("In-memory cache hit for recording: \(recording.id.value.uuidString)", category: "useCase")
            logger.debug(">>> IN-MEMORY CACHE HIT - returning cached result", category: "cache")
            logger.debug("    pitchData.timeStamps.count: \(cachedResult.pitchData.timeStamps.count)", category: "cache")
            await progress(1.0)
            return cachedResult
        }

        // Layer 2: Check file cache for pitch data - only if algorithm hasn't changed
        let fileCacheExists = pitchDataCache?.exists(recording.id) ?? false
        logger.debug("File cache exists: \(fileCacheExists)", category: "cache")
        if !algorithmChanged,
           let pitchDataCache = pitchDataCache,
           let cachedPitchData = pitchDataCache.get(recording.id) {
            logger.info("File cache hit for pitch data, analyzing spectrogram only: \(recording.id.value.uuidString)", category: "useCase")
            logger.debug(">>> FILE CACHE HIT - analyzing spectrogram only", category: "cache")
            logger.debug("    cachedPitchData.timeStamps.count: \(cachedPitchData.timeStamps.count)", category: "cache")

            // Analyze spectrogram only (faster - skips YIN algorithm)
            let spectrogramData = try await audioFileAnalyzer.analyzeSpectrogramOnly(
                fileURL: fileURL,
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
            logger.debug("    spectrogramData.timeStamps.count: \(spectrogramData.timeStamps.count)", category: "cache")
            logger.debug("========================================", category: "cache")

            return result
        }

        // Layer 3: Full analysis required
        logger.info("Cache miss - full analysis for file: \(fileURL.path) with algorithm: \(algorithm.rawValue)", category: "useCase")
        logger.debug(">>> CACHE MISS - performing FULL ANALYSIS", category: "cache")
        logger.debug("    Algorithm: \(algorithm.rawValue)", category: "cache")

        // Analyze audio file with progress reporting
        let (pitchData, spectrogramData) = try await audioFileAnalyzer.analyze(
            fileURL: fileURL,
            progress: progress
        )

        logger.debug(">>> FULL ANALYSIS COMPLETED", category: "cache")
        logger.debug("    pitchData.timeStamps.count: \(pitchData.timeStamps.count)", category: "cache")
        logger.debug("    spectrogramData.timeStamps.count: \(spectrogramData.timeStamps.count)", category: "cache")

        // Save pitch data to file cache for persistence
        pitchDataCache?.set(recording.id, pitchData: pitchData)
        logger.debug("    Saved to file cache", category: "cache")

        // Create analysis result
        let result = AnalysisResult(
            pitchData: pitchData,
            spectrogramData: spectrogramData
        )

        // Cache the corrected results in memory
        analysisCache.set(recording.id, result: result)
        logger.debug("    Saved to in-memory cache", category: "cache")

        // Update recording with the algorithm used for analysis
        await updateRecordingAlgorithm(recording: recording, algorithm: algorithm)
        logger.debug("    Updated recording algorithm to: \(algorithm.rawValue)", category: "cache")
        logger.debug("========================================", category: "cache")

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
