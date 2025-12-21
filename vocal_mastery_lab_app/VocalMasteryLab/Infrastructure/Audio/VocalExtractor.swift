import Foundation
import VocalisDomain

/// Result of vocal extraction
public struct VocalExtractionResult {
    public let vocalFileURL: URL
    public let instrumentalFileURL: URL
    public let duration: Duration

    public init(vocalFileURL: URL, instrumentalFileURL: URL, duration: Duration) {
        self.vocalFileURL = vocalFileURL
        self.instrumentalFileURL = instrumentalFileURL
        self.duration = duration
    }
}

/// Errors that can occur during vocal extraction
public enum VocalExtractionError: Error {
    case sourceFileNotFound
    case extractionFailed(String)
    case cancelled
}

/// Protocol for vocal extraction service
public protocol VocalExtractorProtocol {
    /// Extract vocals from a recording
    /// - Parameters:
    ///   - sourceURL: URL of the source audio file
    ///   - progressHandler: Called with progress updates (0.0 to 1.0)
    /// - Returns: Extraction result containing URLs for vocal and instrumental tracks
    func extract(
        from sourceURL: URL,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> VocalExtractionResult
}

/// Mock implementation of vocal extractor
/// This simply copies the source file as both vocal and instrumental tracks
public class MockVocalExtractor: VocalExtractorProtocol {

    private let fileManager = FileManager.default

    public init() {}

    public func extract(
        from sourceURL: URL,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> VocalExtractionResult {

        // Check source file exists
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            throw VocalExtractionError.sourceFileNotFound
        }

        // Simulate extraction progress
        progressHandler(0.1, "モデルを読み込み中...")
        try await Task.sleep(nanoseconds: 300_000_000) // 0.3 seconds

        progressHandler(0.2, "音声を解析中...")
        try await Task.sleep(nanoseconds: 300_000_000)

        progressHandler(0.5, "ボーカルを分離中...")
        try await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds

        progressHandler(0.8, "出力ファイルを生成中...")
        try await Task.sleep(nanoseconds: 300_000_000)

        // Get documents directory for extracted files
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let extractedDir = documentsURL.appendingPathComponent("ExtractedAudio", isDirectory: true)

        // Create directory if it doesn't exist
        if !fileManager.fileExists(atPath: extractedDir.path) {
            try fileManager.createDirectory(at: extractedDir, withIntermediateDirectories: true)
        }

        // Generate unique filenames
        let timestamp = Int(Date().timeIntervalSince1970)
        let sourceFileName = sourceURL.deletingPathExtension().lastPathComponent
        let fileExtension = sourceURL.pathExtension

        let vocalFileName = "\(sourceFileName)_vocal_\(timestamp).\(fileExtension)"
        let instrumentalFileName = "\(sourceFileName)_instrumental_\(timestamp).\(fileExtension)"

        let vocalURL = extractedDir.appendingPathComponent(vocalFileName)
        let instrumentalURL = extractedDir.appendingPathComponent(instrumentalFileName)

        // Mock: Copy source file as both vocal and instrumental
        try fileManager.copyItem(at: sourceURL, to: vocalURL)
        try fileManager.copyItem(at: sourceURL, to: instrumentalURL)

        progressHandler(1.0, "完了")

        // Get duration from source file
        let duration = try await getAudioDuration(from: sourceURL)

        return VocalExtractionResult(
            vocalFileURL: vocalURL,
            instrumentalFileURL: instrumentalURL,
            duration: duration
        )
    }

    private func getAudioDuration(from url: URL) async throws -> Duration {
        // Use AVAsset to get duration
        let asset = AVURLAsset(url: url)
        let durationSeconds = try await asset.load(.duration).seconds
        return Duration(seconds: durationSeconds)
    }
}

import AVFoundation
