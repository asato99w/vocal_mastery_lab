import Foundation
import VocalisDomain
@testable import VocalMasteryLab

/// Mock implementation of PitchDataCacheProtocol for testing
final class MockPitchDataCache: PitchDataCacheProtocol {
    var cachedData: [RecordingId: PitchAnalysisData] = [:]
    var getCallCount = 0
    var setCallCount = 0
    var deleteCallCount = 0
    var existsCallCount = 0
    var lastDeletedId: RecordingId?

    func get(_ id: RecordingId) -> PitchAnalysisData? {
        getCallCount += 1
        return cachedData[id]
    }

    func set(_ id: RecordingId, pitchData: PitchAnalysisData) {
        setCallCount += 1
        cachedData[id] = pitchData
    }

    func delete(_ id: RecordingId) {
        deleteCallCount += 1
        lastDeletedId = id
        cachedData.removeValue(forKey: id)
    }

    func exists(_ id: RecordingId) -> Bool {
        existsCallCount += 1
        return cachedData[id] != nil
    }

    func reset() {
        cachedData = [:]
        getCallCount = 0
        setCallCount = 0
        deleteCallCount = 0
        existsCallCount = 0
        lastDeletedId = nil
    }
}
