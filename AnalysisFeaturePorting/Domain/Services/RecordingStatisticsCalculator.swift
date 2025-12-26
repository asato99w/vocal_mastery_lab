import Foundation

/// Calculator for objective recording statistics
/// Analyzes pitch data against target notes to produce deviation metrics
public final class RecordingStatisticsCalculator {
    /// Minimum confidence for including a sample in analysis
    private let minConfidence: Float

    /// Minimum confidence for vibrato analysis (may differ from general minConfidence)
    private let vibratoMinConfidence: Float

    /// Vibrato analyzer for detecting vibrato characteristics
    private let vibratoAnalyzer: VibratoAnalyzer

    /// Singer's Formant analyzer for detecting SF presence
    private let singersFormantAnalyzer: SingersFormantAnalyzer

    /// High frequency analyzer for brightness and airiness
    private let highFrequencyAnalyzer: HighFrequencyAnalyzer

    /// Minimum samples required for vibrato analysis per segment
    private let minVibratoSamples = 10

    // MARK: - Stable Region Detection Parameters

    /// Maximum pitch variation (cents) between adjacent frames to be considered stable
    private let stabilityThresholdCents: Double = 30.0

    /// Minimum duration (seconds) for a stable region to be valid
    private let minStableDurationSeconds: Double = 0.1

    // MARK: - Onset Detection Parameters

    /// Pitch tolerance for matching detected note to target (cents)
    private let onsetPitchToleranceCents: Double = 50.0

    /// Onset tolerance for timing accuracy (seconds)
    private let onsetToleranceSeconds: Double = 0.05  // 50ms

    /// Pitch change threshold for note boundary detection (cents)
    private let noteBoundaryThresholdCents: Double = 50.0

    /// Minimum note duration (seconds)
    private let minNoteDurationSeconds: Double = 0.05

    /// Initialize with algorithm-specific parameters
    /// - Parameter algorithm: Pitch detection algorithm used (determines vibrato detection thresholds)
    public init(algorithm: PitchDetectionAlgorithm) {
        self.minConfidence = 0.5
        self.vibratoMinConfidence = algorithm.vibratoMinConfidence
        self.vibratoAnalyzer = VibratoAnalyzer(minimumRegularity: algorithm.vibratoMinRegularity)
        self.singersFormantAnalyzer = SingersFormantAnalyzer()
        self.highFrequencyAnalyzer = HighFrequencyAnalyzer()
    }

    /// Legacy initializer for backward compatibility
    public init(minConfidence: Float = 0.5) {
        self.minConfidence = minConfidence
        self.vibratoMinConfidence = minConfidence
        self.vibratoAnalyzer = VibratoAnalyzer()
        self.singersFormantAnalyzer = SingersFormantAnalyzer()
        self.highFrequencyAnalyzer = HighFrequencyAnalyzer()
    }

    /// Calculate statistics from pitch data and playback timeline
    /// - Parameters:
    ///   - pitchData: Detected pitch analysis data
    ///   - playbackTimeline: Timeline of target notes played
    ///   - scaleSettings: Scale settings used for recording
    /// - Returns: Calculated statistics or nil if insufficient data
    public func calculate(
        pitchData: PitchAnalysisData,
        playbackTimeline: ScalePlaybackTimeline?,
        scaleSettings: ScaleSettings?
    ) -> RecordingStatistics? {
        calculate(
            pitchData: pitchData,
            playbackTimeline: playbackTimeline,
            scaleSettings: scaleSettings,
            spectrogramData: nil
        )
    }

    /// Calculate statistics from pitch data, playback timeline, and spectrogram
    /// - Parameters:
    ///   - pitchData: Detected pitch analysis data
    ///   - playbackTimeline: Timeline of target notes played
    ///   - scaleSettings: Scale settings used for recording
    ///   - spectrogramData: Optional spectrogram for Singer's Formant analysis
    /// - Returns: Calculated statistics or nil if insufficient data
    public func calculate(
        pitchData: PitchAnalysisData,
        playbackTimeline: ScalePlaybackTimeline?,
        scaleSettings: ScaleSettings?,
        spectrogramData: SpectrogramData?
    ) -> RecordingStatistics? {
        let segments = playbackTimeline?.noteSegments ?? []
        let totalDuration = pitchData.timeStamps.last ?? 0

        guard !pitchData.timeStamps.isEmpty else { return nil }

        // Calculate overall statistics
        let overallStats = calculateOverallStatistics(
            pitchData: pitchData,
            segments: segments
        )

        // Calculate position statistics (only if scale was used)
        let positionStats: [RecordingStatistics.PositionStatistics]
        if let settings = scaleSettings, !segments.isEmpty {
            positionStats = calculatePositionStatistics(
                pitchData: pitchData,
                segments: segments,
                notePattern: settings.notePattern
            )
        } else {
            positionStats = []
        }

        // Calculate pitch statistics (sorted by frequency descending)
        // FIXED: Pass scaleSettings to calculate ALL unique notes from key progression
        let pitchStats = calculatePitchStatistics(
            pitchData: pitchData,
            segments: segments,
            scaleSettings: scaleSettings
        )

        // Calculate vibrato statistics
        let vibratoStats = calculateVibratoStatistics(
            pitchData: pitchData,
            segments: segments
        )

        // Calculate Singer's Formant statistics (if spectrogram available)
        let sfStats = calculateSingersFormantStatistics(
            spectrogramData: spectrogramData,
            pitchData: pitchData
        )

        // Calculate high frequency statistics (Brightness and Airiness)
        let hfStats = calculateHighFrequencyStatistics(spectrogramData: spectrogramData)

        return RecordingStatistics(
            overall: overallStats,
            positionStatistics: positionStats,
            pitchStatistics: pitchStats,
            vibratoStatistics: vibratoStats,
            singersFormantStatistics: sfStats,
            highFrequencyStatistics: hfStats,
            totalDuration: totalDuration
        )
    }

    // MARK: - Private Methods

    private func calculateOverallStatistics(
        pitchData: PitchAnalysisData,
        segments: [NoteSegment]
    ) -> RecordingStatistics.OverallStatistics {
        var allDeviations: [Double] = []
        var validFrequencies: [Double] = []
        var samplesInSegments = 0

        for (index, timestamp) in pitchData.timeStamps.enumerated() {
            let confidence = pitchData.confidences[index]
            guard confidence >= minConfidence else { continue }

            let frequency = Double(pitchData.frequencies[index])

            // Track all valid frequencies for range
            if frequency > 50 && frequency < 2000 {
                validFrequencies.append(frequency)
            }

            // Count samples in segments (for detection rate)
            if segments.first(where: { timestamp >= $0.startTime && timestamp < $0.endTime }) != nil {
                samplesInSegments += 1
            }
        }

        // Calculate deviations from stable regions only (not all frames)
        // This avoids penalizing legato transitions
        for segment in segments {
            if let stableRegion = detectStableRegion(pitchData: pitchData, segment: segment) {
                let deviations = calculateStableRegionDeviations(
                    pitchData: pitchData,
                    segment: segment,
                    stableRegion: stableRegion
                )
                allDeviations.append(contentsOf: deviations.map { abs($0) })
            }
        }

        let totalSamples = validFrequencies.count
        let detectionRate = totalSamples > 0 ? Double(samplesInSegments) / Double(totalSamples) : 0

        return RecordingStatistics.OverallStatistics(
            averageDeviationCents: average(allDeviations),
            deviationStdDev: standardDeviation(allDeviations),
            medianDeviationCents: median(allDeviations),
            detectionRate: detectionRate,
            totalSamples: totalSamples,
            lowestFrequency: validFrequencies.min(),
            highestFrequency: validFrequencies.max()
        )
    }

    private func calculatePositionStatistics(
        pitchData: PitchAnalysisData,
        segments: [NoteSegment],
        notePattern: NotePattern
    ) -> [RecordingStatistics.PositionStatistics] {
        // FIXED: Use playbackPattern.count for position count
        // fiveToneScale: [0, 2, 4, 5, 7, 5, 4, 2, 0] = 9 positions
        // octaveRepeat: [0, 4, 7, 12, 12, 12, 12, 7, 4, 0] = 10 positions
        let patternLength = notePattern.playbackPattern.count

        // Group segments by position within pattern
        var positionDeviations: [Int: [Double]] = [:]
        var positionOccurrences: [Int: Int] = [:]  // Total occurrences per position
        var positionDetections: [Int: Int] = [:]   // Successfully detected notes per position
        var positionOnsetErrors: [Int: [Double]] = [:]  // Onset errors per position

        // Assign position to each segment based on order
        var segmentPositions: [(segment: NoteSegment, position: Int)] = []
        var currentPosition = 1

        for segment in segments {
            segmentPositions.append((segment, currentPosition))
            // Count total occurrences for this position
            positionOccurrences[currentPosition, default: 0] += 1
            currentPosition += 1
            if currentPosition > patternLength {
                currentPosition = 1
            }
        }

        // Collect deviations and onset errors for each position
        for (segment, position) in segmentPositions {
            if let stableRegion = detectStableRegion(pitchData: pitchData, segment: segment) {
                let deviations = calculateStableRegionDeviations(
                    pitchData: pitchData,
                    segment: segment,
                    stableRegion: stableRegion
                )
                if !deviations.isEmpty {
                    positionDeviations[position, default: []].append(contentsOf: deviations)
                    // Count as detected if we have stable region with deviations
                    positionDetections[position, default: 0] += 1

                    // Calculate onset error for this segment
                    if let onsetError = calculateOnsetError(pitchData: pitchData, segment: segment) {
                        positionOnsetErrors[position, default: []].append(onsetError)
                    }
                }
            }
        }

        // FIXED: Return ALL positions, even those without detections
        // This is required by specification - statistics are based on scale settings, not detection results
        var results: [RecordingStatistics.PositionStatistics] = []

        for position in 1...patternLength {
            let deviations = positionDeviations[position] ?? []
            let occurrences = positionOccurrences[position] ?? 0
            let detected = positionDetections[position] ?? 0
            let detectionRate = occurrences > 0 ? Double(detected) / Double(occurrences) : 0

            // Calculate onset timing metrics
            let onsetErrors = positionOnsetErrors[position] ?? []
            let avgOnsetError = average(onsetErrors)
            let onsetsWithinTolerance = onsetErrors.filter { abs($0) <= onsetToleranceSeconds * 1000 }.count
            let onsetAcc = onsetErrors.isEmpty ? 0 : Double(onsetsWithinTolerance) / Double(onsetErrors.count)

            results.append(RecordingStatistics.PositionStatistics(
                position: position,
                averageDeviationCents: average(deviations),
                deviationStdDev: standardDeviation(deviations.map { abs($0) }),
                sampleCount: deviations.count,
                noteDetectionRate: detectionRate,
                noteOccurrences: occurrences,
                notesDetected: detected,
                averageOnsetErrorMs: avgOnsetError,
                onsetAccuracy: onsetAcc
            ))
        }

        return results.sorted { $0.position < $1.position }
    }

    private func calculatePitchStatistics(
        pitchData: PitchAnalysisData,
        segments: [NoteSegment],
        scaleSettings: ScaleSettings?
    ) -> [RecordingStatistics.PitchStatistics] {
        // FIXED: Calculate ALL unique notes from scale settings (key progression + note pattern)
        // Not just detected notes - specification requires all notes based on scale settings

        // Calculate all unique MIDI notes from scale settings
        let allUniqueMIDINotes: Set<Int>
        if let settings = scaleSettings {
            allUniqueMIDINotes = calculateAllUniqueNotesFromSettings(settings)
        } else {
            // Fallback: use notes from segments
            allUniqueMIDINotes = Set(segments.map { Int($0.note.value) })
        }

        // Group segments by note for deviation calculation
        var noteGroups: [Int: (note: MIDINote, segments: [NoteSegment])] = [:]

        for segment in segments {
            let midiNumber = Int(segment.note.value)
            if noteGroups[midiNumber] == nil {
                noteGroups[midiNumber] = (segment.note, [segment])
            } else {
                noteGroups[midiNumber]?.segments.append(segment)
            }
        }

        var results: [RecordingStatistics.PitchStatistics] = []

        // Return statistics for ALL unique notes, not just detected ones
        for midiNumber in allUniqueMIDINotes {
            let group = noteGroups[midiNumber]
            var deviations: [Double] = []

            // Use stable regions only to avoid penalizing legato transitions
            if let group = group {
                for segment in group.segments {
                    if let stableRegion = detectStableRegion(pitchData: pitchData, segment: segment) {
                        let segmentDeviations = calculateStableRegionDeviations(
                            pitchData: pitchData,
                            segment: segment,
                            stableRegion: stableRegion
                        )
                        deviations.append(contentsOf: segmentDeviations)
                    }
                }
            }

            // Create MIDINote for this MIDI number
            guard let note = try? MIDINote(UInt8(midiNumber)) else { continue }

            // Always add the note - even if no detections
            results.append(RecordingStatistics.PitchStatistics(
                noteName: note.noteName,
                midiNoteNumber: midiNumber,
                frequency: note.frequency,
                averageDeviationCents: average(deviations),
                deviationStdDev: standardDeviation(deviations.map { abs($0) }),
                occurrenceCount: group?.segments.count ?? 0,
                sampleCount: deviations.count
            ))
        }

        // Sort by frequency descending (high to low)
        return results.sorted { $0.frequency > $1.frequency }
    }

    /// Calculate all unique MIDI notes from scale settings
    /// Considers key progression pattern and note pattern intervals
    private func calculateAllUniqueNotesFromSettings(_ settings: ScaleSettings) -> Set<Int> {
        var uniqueNotes = Set<Int>()

        // Get all key roots from the progression
        let keyRoots = settings.generateKeyRoots()

        // For each key root, add all notes from the note pattern intervals
        for root in keyRoots {
            for interval in settings.notePattern.intervals {
                let midiNote = Int(root) + interval
                if midiNote >= 0 && midiNote <= 127 {
                    uniqueNotes.insert(midiNote)
                }
            }
        }

        return uniqueNotes
    }

    private func calculateVibratoStatistics(
        pitchData: PitchAnalysisData,
        segments: [NoteSegment]
    ) -> RecordingStatistics.VibratoStatistics? {
        // Collect valid frequencies and timestamps for vibrato analysis
        var allFrequencies: [Float] = []
        var allTimeStamps: [Double] = []

        for (index, _) in pitchData.timeStamps.enumerated() {
            // Use vibratoMinConfidence for vibrato analysis (may be lower than general minConfidence)
            guard pitchData.confidences[index] >= vibratoMinConfidence else { continue }
            let frequency = pitchData.frequencies[index]
            // Filter out invalid frequencies
            if frequency > 50 && frequency < 2000 {
                allFrequencies.append(frequency)
                allTimeStamps.append(pitchData.timeStamps[index])
            }
        }

        guard allFrequencies.count >= minVibratoSamples else { return nil }

        // Use sliding window analysis to get multiple vibrato assessments
        // This provides more accurate presence rate calculation
        // Window: 0.5s (captures ~2-3 vibrato cycles at 5-6Hz)
        // Hop: 0.25s (50% overlap for better coverage)
        let vibratoAnalyses = vibratoAnalyzer.analyzeAllWindows(
            frequencies: allFrequencies,
            timeStamps: allTimeStamps,
            windowDuration: 0.5,
            hopRatio: 0.5
        )

        guard !vibratoAnalyses.isEmpty else { return nil }

        // Calculate aggregate statistics
        let presentAnalyses = vibratoAnalyses.filter { $0.isPresent }
        let presenceRate = Float(presentAnalyses.count) / Float(vibratoAnalyses.count)

        // Calculate averages only from windows where vibrato was detected
        let averageRate: Float
        let averageExtent: Float
        let averageRegularity: Float

        if presentAnalyses.isEmpty {
            averageRate = 0
            averageExtent = 0
            averageRegularity = 0
        } else {
            averageRate = presentAnalyses.map { $0.rate }.reduce(0, +) / Float(presentAnalyses.count)
            averageExtent = presentAnalyses.map { $0.extent }.reduce(0, +) / Float(presentAnalyses.count)
            averageRegularity = presentAnalyses.map { $0.regularity }.reduce(0, +) / Float(presentAnalyses.count)
        }

        return RecordingStatistics.VibratoStatistics(
            averageRate: averageRate,
            averageExtent: averageExtent,
            averageRegularity: averageRegularity,
            presenceRate: presenceRate,
            segmentsAnalyzed: vibratoAnalyses.count
        )
    }

    private func calculateSingersFormantStatistics(
        spectrogramData: SpectrogramData?,
        pitchData: PitchAnalysisData
    ) -> RecordingStatistics.SingersFormantStatistics? {
        guard let spectrogram = spectrogramData,
              !spectrogram.magnitudes.isEmpty else {
            return nil
        }

        // Calculate average pitch for confidence adjustment
        let validFrequencies = pitchData.frequencies.enumerated()
            .filter { pitchData.confidences[$0.offset] >= minConfidence }
            .map { $0.element }

        let averagePitch: Float? = validFrequencies.isEmpty
            ? nil
            : validFrequencies.reduce(0, +) / Float(validFrequencies.count)

        // Analyze spectrogram
        let analysis = singersFormantAnalyzer.analyzeSpectrogram(
            spectrogramData: spectrogram,
            averagePitch: averagePitch
        )

        // Return nil if no valid analysis
        guard analysis.ratio > 0 else { return nil }

        return RecordingStatistics.SingersFormantStatistics(
            averageRatio: analysis.ratio,
            averageIntensity: analysis.intensity,
            isPresent: analysis.isPresent,
            confidence: analysis.confidence
        )
    }

    private func calculateHighFrequencyStatistics(
        spectrogramData: SpectrogramData?
    ) -> RecordingStatistics.HighFrequencyStatistics? {
        guard let spectrogram = spectrogramData,
              !spectrogram.magnitudes.isEmpty else {
            return nil
        }

        // Analyze spectrogram for brightness and airiness
        let analysis = highFrequencyAnalyzer.analyzeSpectrogram(spectrogramData: spectrogram)

        // Return nil if no valid analysis
        guard analysis.brightnessRatio > 0 || analysis.airinessRatio > 0 else { return nil }

        return RecordingStatistics.HighFrequencyStatistics(
            brightnessRatio: analysis.brightnessRatio,
            airinessRatio: analysis.airinessRatio
        )
    }

    // MARK: - Helper Functions

    /// Convert frequency ratio to cents
    /// Positive = sharp (higher), Negative = flat (lower)
    private func frequencyToCents(_ frequency: Double, relativeTo reference: Double) -> Double {
        guard reference > 0, frequency > 0 else { return 0 }
        return 1200 * log2(frequency / reference)
    }

    private func average(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private func standardDeviation(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        let avg = average(values)
        let variance = values.map { pow($0 - avg, 2) }.reduce(0, +) / Double(values.count - 1)
        return sqrt(variance)
    }

    private func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 0 {
            return (sorted[mid - 1] + sorted[mid]) / 2
        } else {
            return sorted[mid]
        }
    }

    // MARK: - Onset Detection

    /// Represents a detected note from pitch data
    private struct DetectedNote {
        let onsetTime: Double
        let offsetTime: Double
        let frequency: Float

        var duration: Double { offsetTime - onsetTime }

        var midiNote: Int {
            Int(round(69 + 12 * log2(frequency / 440.0)))
        }
    }

    /// Detect actual note onset time for a target segment
    /// Searches for the first stable pitch matching the target note within segment's time range
    /// - Parameters:
    ///   - pitchData: The pitch analysis data
    ///   - segment: The target note segment
    ///   - searchWindow: Time window before/after segment start to search (default 0.2s)
    /// - Returns: Detected onset time, or nil if not found
    private func detectOnsetTime(
        pitchData: PitchAnalysisData,
        segment: NoteSegment,
        searchWindow: Double = 0.2
    ) -> Double? {
        let targetFreq = segment.frequency
        let searchStart = max(0, segment.startTime - searchWindow)
        let searchEnd = segment.endTime

        // Find first valid pitch frame that matches target note
        for (index, timestamp) in pitchData.timeStamps.enumerated() {
            guard timestamp >= searchStart && timestamp <= searchEnd else { continue }
            guard pitchData.confidences[index] >= minConfidence else { continue }

            let freq = pitchData.frequencies[index]
            guard freq > 50 && freq < 2000 else { continue }

            // Check if pitch matches target within tolerance
            let centsError = abs(1200 * log2(Double(freq) / targetFreq))
            if centsError <= onsetPitchToleranceCents {
                return timestamp
            }
        }

        return nil
    }

    /// Calculate onset error in milliseconds
    /// Positive = late (detected after expected), Negative = early (detected before expected)
    private func calculateOnsetError(
        pitchData: PitchAnalysisData,
        segment: NoteSegment
    ) -> Double? {
        guard let detectedOnset = detectOnsetTime(pitchData: pitchData, segment: segment) else {
            return nil
        }

        let expectedOnset = segment.startTime
        let errorSeconds = detectedOnset - expectedOnset
        return errorSeconds * 1000  // Convert to milliseconds
    }

    // MARK: - Stable Region Detection

    /// Represents a stable pitch region within a segment
    private struct StableRegion {
        let startIndex: Int
        let endIndex: Int
        let startTime: Double
        let endTime: Double

        var duration: Double { endTime - startTime }
    }

    /// Detect stable region within a segment's time range
    /// A stable region is where pitch variation between adjacent frames is within threshold
    /// - Parameters:
    ///   - pitchData: The pitch analysis data
    ///   - segment: The note segment to analyze
    /// - Returns: StableRegion if found, nil otherwise
    private func detectStableRegion(
        pitchData: PitchAnalysisData,
        segment: NoteSegment
    ) -> StableRegion? {
        // Find frames within segment time range
        var frameIndices: [Int] = []
        for (index, timestamp) in pitchData.timeStamps.enumerated() {
            guard timestamp >= segment.startTime && timestamp < segment.endTime else { continue }
            guard pitchData.confidences[index] >= minConfidence else { continue }
            let freq = pitchData.frequencies[index]
            guard freq > 50 && freq < 2000 else { continue }
            frameIndices.append(index)
        }

        guard frameIndices.count >= 2 else { return nil }

        // Find the longest stable region
        var bestRegion: StableRegion?
        var currentStart = frameIndices[0]
        var previousFreq = pitchData.frequencies[currentStart]

        for i in 1..<frameIndices.count {
            let index = frameIndices[i]
            let freq = pitchData.frequencies[index]
            let centsChange = abs(1200 * log2(Double(freq) / Double(previousFreq)))

            if centsChange > stabilityThresholdCents {
                // Check if current region is valid
                let endIndex = frameIndices[i - 1]
                let startTime = pitchData.timeStamps[currentStart]
                let endTime = pitchData.timeStamps[endIndex]
                let duration = endTime - startTime

                if duration >= minStableDurationSeconds {
                    let region = StableRegion(
                        startIndex: currentStart,
                        endIndex: endIndex,
                        startTime: startTime,
                        endTime: endTime
                    )
                    if bestRegion == nil || region.duration > bestRegion!.duration {
                        bestRegion = region
                    }
                }
                // Start new region
                currentStart = index
            }
            previousFreq = freq
        }

        // Check final region
        let lastIndex = frameIndices.last!
        let startTime = pitchData.timeStamps[currentStart]
        let endTime = pitchData.timeStamps[lastIndex]
        let duration = endTime - startTime

        if duration >= minStableDurationSeconds {
            let region = StableRegion(
                startIndex: currentStart,
                endIndex: lastIndex,
                startTime: startTime,
                endTime: endTime
            )
            if bestRegion == nil || region.duration > bestRegion!.duration {
                bestRegion = region
            }
        }

        return bestRegion
    }

    /// Calculate deviations only from stable regions within segments
    /// - Parameters:
    ///   - pitchData: The pitch analysis data
    ///   - segment: The note segment
    ///   - stableRegion: The stable region within the segment
    /// - Returns: Array of deviation values in cents
    private func calculateStableRegionDeviations(
        pitchData: PitchAnalysisData,
        segment: NoteSegment,
        stableRegion: StableRegion
    ) -> [Double] {
        var deviations: [Double] = []
        let targetFreq = segment.frequency

        for index in stableRegion.startIndex...stableRegion.endIndex {
            guard pitchData.confidences[index] >= minConfidence else { continue }
            let frequency = Double(pitchData.frequencies[index])
            let deviationCents = frequencyToCents(frequency, relativeTo: targetFreq)
            deviations.append(deviationCents)
        }

        return deviations
    }
}
