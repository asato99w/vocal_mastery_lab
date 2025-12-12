import Foundation
import VocalisDomain

/// In-memory cache for analysis results
/// Implements LRU (Least Recently Used) eviction policy
public class AnalysisCache: AnalysisCacheProtocol {
    private var cache: [RecordingId: CacheEntry] = [:]
    private var accessOrder: [RecordingId] = []
    private let maxCacheSize: Int
    private let lock = NSLock()

    private struct CacheEntry {
        let result: AnalysisResult
        let timestamp: Date
    }

    public init(maxCacheSize: Int = 10) {
        self.maxCacheSize = maxCacheSize
    }

    public func get(_ id: RecordingId) -> AnalysisResult? {
        lock.lock()
        defer { lock.unlock() }

        guard let entry = cache[id] else {
            return nil
        }

        // Update access order (move to end = most recently used)
        if let index = accessOrder.firstIndex(of: id) {
            accessOrder.remove(at: index)
        }
        accessOrder.append(id)

        return entry.result
    }

    public func set(_ id: RecordingId, result: AnalysisResult) {
        lock.lock()
        defer { lock.unlock() }

        // Create new cache entry
        let entry = CacheEntry(result: result, timestamp: Date())
        cache[id] = entry

        // Update access order
        if let index = accessOrder.firstIndex(of: id) {
            accessOrder.remove(at: index)
        }
        accessOrder.append(id)

        // Evict oldest if cache is full
        if accessOrder.count > maxCacheSize {
            let oldestId = accessOrder.removeFirst()
            cache.removeValue(forKey: oldestId)
        }
    }

    public func clear() {
        lock.lock()
        defer { lock.unlock() }

        cache.removeAll()
        accessOrder.removeAll()
    }

    /// Get current cache size (for testing/debugging)
    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return cache.count
    }
}
