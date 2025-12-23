import Foundation
import VocalisDomain

/// CoreML-based vocal extractor using UVR-MDX-NET model
///
/// This is the production implementation of VocalExtractorProtocol
/// that uses the VocalSeparatorEngine for actual vocal separation.
public final class CoreMLVocalExtractor: VocalExtractorProtocol {

    private let fileManager = FileManager.default
    private var engine: VocalSeparatorEngine?
    private let modelURL: URL

    public init(modelURL: URL) {
        self.modelURL = modelURL
    }

    public func extract(
        from sourceURL: URL,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> VocalExtractionResult {

        // Check source file exists
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            throw VocalExtractionError.sourceFileNotFound
        }

        // Initialize engine lazily
        if engine == nil {
            progressHandler(0.05, "モデルを読み込み中...")
            do {
                engine = try VocalSeparatorEngine(modelURL: modelURL)
            } catch {
                throw VocalExtractionError.extractionFailed("モデル読み込み失敗: \(error.localizedDescription)")
            }
        }

        guard let engine = engine else {
            throw VocalExtractionError.extractionFailed("エンジン初期化失敗")
        }

        // Run separation
        let result: VocalSeparatorEngine.SeparationResult
        do {
            result = try engine.separate(audioURL: sourceURL) { progress, stage in
                progressHandler(progress, stage)
            }
        } catch {
            throw VocalExtractionError.extractionFailed(error.localizedDescription)
        }

        // Prepare output directory
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let extractedDir = documentsURL.appendingPathComponent("ExtractedAudio", isDirectory: true)

        if !fileManager.fileExists(atPath: extractedDir.path) {
            try fileManager.createDirectory(at: extractedDir, withIntermediateDirectories: true)
        }

        // Generate unique filenames
        let timestamp = Int(Date().timeIntervalSince1970)
        let sourceFileName = sourceURL.deletingPathExtension().lastPathComponent
        let vocalFileName = "\(sourceFileName)_vocal_\(timestamp).wav"
        let instrumentalFileName = "\(sourceFileName)_instrumental_\(timestamp).wav"
        let vocalURL = extractedDir.appendingPathComponent(vocalFileName)
        let instrumentalURL = extractedDir.appendingPathComponent(instrumentalFileName)

        // Save vocals and instrumental
        do {
            try engine.save(result: result, vocalsURL: vocalURL, instrumentalURL: instrumentalURL)
        } catch {
            throw VocalExtractionError.extractionFailed("保存失敗: \(error.localizedDescription)")
        }

        // Calculate duration
        let duration = Duration(seconds: Double(result.vocals.frameCount) / result.vocals.sampleRate)

        return VocalExtractionResult(
            vocalFileURL: vocalURL,
            instrumentalFileURL: instrumentalURL,
            duration: duration
        )
    }
}
