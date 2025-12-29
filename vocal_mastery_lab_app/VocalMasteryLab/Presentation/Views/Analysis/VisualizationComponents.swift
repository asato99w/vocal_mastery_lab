//
//  VisualizationComponents.swift
//  VocalMasteryLab
//
//  Visualization UI components for AnalysisView
//  Extracted from AnalysisView.swift for better code organization
//

import SwiftUI
import VocalisDomain

// MARK: - Spectrogram View

struct SpectrogramView: View {
    let currentTime: Double
    let spectrogramData: SpectrogramData?
    var isPlaying: Bool = false
    var isExpanded: Bool = false
    var onExpand: (() -> Void)? = nil
    var onCollapse: (() -> Void)? = nil
    var onPlayPause: (() -> Void)? = nil
    var onSeek: ((Double) -> Void)? = nil

    // MARK: - Dependencies
    private let coordinateSystem = SpectrogramCoordinateSystem()
    private var renderer: SpectrogramRenderer {
        SpectrogramRenderer(coordinateSystem: coordinateSystem)
    }
    @State private var scrollManager = SpectrogramScrollManager()

    // MARK: - Drag State
    @State private var dragStartTime: Double = 0.0
    @State private var isDraggingVertically: Bool? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            GeometryReader { geometry in
                let viewportWidth = geometry.size.width
                let viewportHeight = geometry.size.height
                let maxFreq = coordinateSystem.getMaxFrequency()
                let canvasHeight = coordinateSystem.calculateCanvasHeight(maxFreq: maxFreq, viewportHeight: viewportHeight)

                // Calculate canvas width based on data duration, NOT viewport
                let pixelsPerSecond: CGFloat = 300  // Ultra-high density for time axis zoom (6x from 50)
                // Add extra space at the beginning of canvas to ensure frequency labels
                // are always within canvas bounds even at initial scroll position
                let canvasLeftPadding: CGFloat = viewportWidth / 2  // Same as playhead offset
                let canvasWidth: CGFloat = {
                    if let data = spectrogramData, !data.timeStamps.isEmpty {
                        let dataDuration = data.timeStamps.last ?? 0.0
                        let dataWidth = CGFloat(dataDuration) * pixelsPerSecond
                        return max(dataWidth + canvasLeftPadding, 100)  // Include left padding
                    }
                    return viewportWidth + canvasLeftPadding  // fallback with padding
                }()

                let cellWidth = pixelsPerSecond * 0.1

                // Initialize scroll position to bottom when expanded (low frequency visible)
                let scrollableRange = canvasHeight - viewportHeight

                // Debug log
                let _ = {
                    FileLogger.shared.log(level: "INFO", category: "viewport_debug",
                        message: "🔍 VIEWPORT DEBUG: isExpanded=\(isExpanded), viewportW=\(viewportWidth), viewportH=\(viewportHeight), canvasW=\(canvasWidth), canvasH=\(canvasHeight), pixelsPerSecond=\(pixelsPerSecond), cellWidth=\(cellWidth), scrollableRange=\(scrollableRange)")
                }()

                // Calculate visible time range for viewport culling
                let visibleTimeRange: ClosedRange<Double> = {
                    let visibleStartX = -scrollManager.canvasOffsetX
                    let startTime = max(0, Double(visibleStartX - canvasLeftPadding) / Double(pixelsPerSecond))
                    let visibleEndX = visibleStartX + viewportWidth
                    let endTime = Double(visibleEndX - canvasLeftPadding) / Double(pixelsPerSecond)
                    // Add margin for smooth scrolling
                    let margin = 2.0
                    let marginedStart = max(0, startTime - margin)
                    let marginedEnd = endTime + margin
                    return marginedStart...marginedEnd
                }()

                // Canvas: Contains the entire frequency range (0Hz ~ maxFreq)
                Canvas { context, size in
                    if let data = spectrogramData, !data.timeStamps.isEmpty {
                        // Draw everything in canvas coordinates
                        // size here is the canvas size, not viewport size

                        // 1. Draw spectrogram (background) - SCROLLABLE with viewport culling
                        renderer.drawSpectrogram(
                            context: context,
                            canvasWidth: size.width,
                            canvasHeight: canvasHeight,
                            maxFreq: maxFreq,
                            data: data,
                            leftPadding: canvasLeftPadding,
                            visibleTimeRange: visibleTimeRange
                        )

                        // 2. Draw Y-axis labels - Y-SCROLLABLE, X-FIXED
                        // Compensate for X scroll offset to keep labels in viewport
                        var yAxisContext = context
                        yAxisContext.translateBy(x: -scrollManager.canvasOffsetX, y: 0)
                        renderer.drawFrequencyLabels(
                            context: yAxisContext,
                            canvasHeight: canvasHeight,
                            maxFreq: maxFreq,
                            viewportHeight: viewportHeight,
                            paperTop: scrollManager.paperTop
                        )

                        // 3. Draw time axis (X-axis) - X-SCROLLABLE, Y-FIXED
                        // Compensate for Y scroll offset to keep labels at viewport bottom
                        var timeAxisContext = context
                        timeAxisContext.translateBy(x: 0, y: -scrollManager.paperTop)
                        renderer.drawTimeAxis(
                            context: timeAxisContext,
                            size: CGSize(width: size.width, height: viewportHeight),
                            leftPadding: canvasLeftPadding
                        )

                        // 4. Draw playback position (red line) - FULLY FIXED
                        // Compensate for both X and Y scroll offsets
                        var playheadContext = context
                        playheadContext.translateBy(x: -scrollManager.canvasOffsetX, y: -scrollManager.paperTop)
                        renderer.drawPlaybackPosition(context: playheadContext, size: CGSize(width: viewportWidth, height: viewportHeight))
                    } else {
                        renderer.drawPlaceholder(context: context, size: size)
                    }
                }
                .frame(width: canvasWidth, height: canvasHeight)  // Fixed canvas size based on data
                .offset(x: scrollManager.canvasOffsetX, y: scrollManager.paperTop)  // Scroll by moving canvas
                // - canvasOffsetX: X offset to keep currentTime position under red line (playhead)
                // - paperTop: Y offset for frequency axis scrolling (canvas top edge Y in viewport space)
                .frame(width: viewportWidth, height: viewportHeight, alignment: .topLeading)  // Viewport window
                .clipped()  // Viewport clips to visible area
                .accessibilityIdentifier("SpectrogramCanvas")
                .overlay(alignment: .topTrailing) {
                    if !isExpanded, let onExpand = onExpand {
                        Button(action: onExpand) {
                            Image(systemName: "arrow.down.left.and.arrow.up.right")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundColor(.white)
                                .padding(8)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(6)
                        }
                        .padding(8)
                        .accessibilityLabel("analysis.fullscreen".localized)
                        .accessibilityIdentifier("SpectrogramExpandButton")
                    } else if isExpanded, let onCollapse = onCollapse {
                        Button(action: onCollapse) {
                            Image(systemName: "arrow.up.right.and.arrow.down.left")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundColor(.white)
                                .padding(8)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(6)
                        }
                        .padding(8)
                        .accessibilityLabel("analysis.close".localized)
                        .accessibilityIdentifier("SpectrogramCollapseButton")
                    }
                }
                .onAppear {
                    // Wait for layout to be ready, then initialize position
                    DispatchQueue.main.async {
                        scrollManager.initializePosition(
                            viewportWidth: viewportWidth,
                            viewportHeight: viewportHeight,
                            canvasHeight: canvasHeight,
                            currentTime: currentTime,
                            pixelsPerSecond: pixelsPerSecond,
                            canvasLeftPadding: canvasLeftPadding
                        )
                    }
                }
                .onChange(of: isExpanded) { _, newValue in
                    // Re-initialize position when expanding (for non-fullScreenCover transitions)
                    if newValue {
                        scrollManager.initializePosition(
                            viewportWidth: viewportWidth,
                            viewportHeight: viewportHeight,
                            canvasHeight: canvasHeight,
                            currentTime: currentTime,
                            pixelsPerSecond: pixelsPerSecond,
                            canvasLeftPadding: canvasLeftPadding
                        )
                    }
                }
                .gesture(
                    DragGesture(minimumDistance: 1)
                        .onChanged { value in
                            if isPlaying {
                                // During playback: direction lock (vertical only)
                                if isDraggingVertically == nil {
                                    let dx = abs(value.translation.width)
                                    let dy = abs(value.translation.height)
                                    if dx > 5 || dy > 5 {
                                        isDraggingVertically = dy > dx
                                    }
                                }
                                if isDraggingVertically == true {
                                    scrollManager.handleVerticalDrag(
                                        translation: value.translation.height,
                                        viewportHeight: viewportHeight,
                                        canvasHeight: canvasHeight
                                    )
                                }
                            } else {
                                // When paused: diagonal swipe (both vertical and horizontal simultaneously)
                                // Store start time on first drag
                                if isDraggingVertically == nil {
                                    dragStartTime = currentTime
                                    isDraggingVertically = true // Mark as started
                                }

                                // Vertical: frequency axis scrolling
                                scrollManager.handleVerticalDrag(
                                    translation: value.translation.height,
                                    viewportHeight: viewportHeight,
                                    canvasHeight: canvasHeight
                                )

                                // Horizontal: time axis seeking
                                if let onSeek = onSeek {
                                    let timeChange = -Double(value.translation.width) / Double(pixelsPerSecond)
                                    let dataDuration = spectrogramData?.timeStamps.last ?? 10.0
                                    let newTime = max(0, min(dataDuration, dragStartTime + timeChange))
                                    onSeek(newTime)
                                }
                            }
                        }
                        .onEnded { _ in
                            scrollManager.endDrag()
                            isDraggingVertically = nil
                        }
                )
                .onChange(of: currentTime) { _, newTime in
                    scrollManager.updateTimeScroll(
                        currentTime: newTime,
                        viewportWidth: viewportWidth,
                        pixelsPerSecond: pixelsPerSecond,
                        canvasLeftPadding: canvasLeftPadding
                    )
                }
            }
            .background(Color.black.opacity(0.1))
            .cornerRadius(8)
        }
        .contentShape(Rectangle())
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("SpectrogramView")
        .accessibilityValue(String(format: "%.3f", currentTime))
        .onTapGesture {
            onPlayPause?()
        }
    }

}

// MARK: - Pitch Analysis View

struct PitchAnalysisView: View {
    let currentTime: Double
    let pitchData: PitchAnalysisData?
    var isPlaying: Bool = false
    var isExpanded: Bool = false
    @Binding var autoPitchFollow: Bool  // Auto-follow pitch during playback (toggled via UI)
    var onExpand: (() -> Void)? = nil
    var onCollapse: (() -> Void)? = nil
    var onPlayPause: (() -> Void)? = nil
    var onSeek: ((Double) -> Void)? = nil

    // Scroll manager for 2D scrolling (reusing SpectrogramScrollManager)
    @State private var scrollManager = SpectrogramScrollManager()

    // Drag state for horizontal seek
    @State private var lastDragTranslation: CGSize = .zero

    // Drag direction state (for playback mode)
    @State private var isDraggingVertically: Bool? = nil

    // Auto-follow state
    @State private var isUserDragging: Bool = false
    @State private var lastFollowedFrequency: Double? = nil  // Track last followed pitch to avoid redundant updates

    // Coordinate system and renderer
    private let coordinateSystem = PitchGraphCoordinateSystem()
    private let renderer = PitchGraphRenderer()

    // Drag gesture state
    @State private var dragStartLocation: CGPoint = .zero

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {

            GeometryReader { geometry in
                let viewportWidth = geometry.size.width
                let viewportHeight = geometry.size.height
                let canvasHeight = coordinateSystem.calculateCanvasHeight()
                let leftPadding = coordinateSystem.calculateLeftPadding(viewportWidth: viewportWidth)
                let dataDuration = pitchData?.timeStamps.last ?? 10.0
                let canvasWidth = coordinateSystem.calculateCanvasWidth(dataDuration: dataDuration, leftPadding: leftPadding)

                Canvas { context, size in
                    var mutableContext = context
                    if let data = pitchData, !data.timeStamps.isEmpty {
                        drawPitchGraphCanvas(
                            context: &mutableContext,
                            viewportSize: size,
                            canvasHeight: canvasHeight,
                            canvasWidth: canvasWidth,
                            leftPadding: leftPadding,
                            data: data
                        )
                    } else {
                        renderer.drawPlaceholder(context: mutableContext, size: size)
                    }
                }
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            isUserDragging = true
                            handleDrag(value: value, viewportHeight: viewportHeight, canvasHeight: canvasHeight)
                        }
                        .onEnded { _ in
                            scrollManager.endDrag()
                            isDraggingVertically = nil
                            lastDragTranslation = .zero
                            isUserDragging = false
                        }
                )
                .overlay(alignment: .topLeading) {
                    // Auto-follow toggle button (inside graph, top-left)
                    Button {
                        autoPitchFollow.toggle()
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "scope")
                                .font(.system(size: 12))
                            Text(autoPitchFollow ? "analysis.auto_follow_on".localized : "analysis.auto_follow_off".localized)
                                .font(.caption2)
                        }
                        .foregroundColor(autoPitchFollow ? ColorPalette.primary : .white.opacity(0.7))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(4)
                    }
                    .padding(8)
                    .accessibilityIdentifier("AutoFollowToggle")
                }
                .overlay(alignment: .topTrailing) {
                    if !isExpanded, let onExpand = onExpand {
                        Button(action: onExpand) {
                            Image(systemName: "arrow.down.left.and.arrow.up.right")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundColor(.white)
                                .padding(8)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(6)
                        }
                        .padding(8)
                        .accessibilityLabel("analysis.fullscreen".localized)
                        .accessibilityIdentifier("PitchGraphExpandButton")
                    } else if isExpanded, let onCollapse = onCollapse {
                        Button(action: onCollapse) {
                            Image(systemName: "arrow.up.right.and.arrow.down.left")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundColor(.white)
                                .padding(8)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(6)
                        }
                        .padding(8)
                        .accessibilityLabel("analysis.close".localized)
                        .accessibilityIdentifier("PitchGraphCollapseButton")
                    }
                }
                .onAppear {
                    // Wait for layout to be ready, then initialize position
                    DispatchQueue.main.async {
                        initializeScrollPosition(
                            viewportWidth: viewportWidth,
                            viewportHeight: viewportHeight,
                            canvasHeight: canvasHeight,
                            leftPadding: leftPadding
                        )
                    }
                }
                .onChange(of: isExpanded) { _, _ in
                    initializeScrollPosition(
                        viewportWidth: viewportWidth,
                        viewportHeight: viewportHeight,
                        canvasHeight: canvasHeight,
                        leftPadding: leftPadding
                    )
                }
                .onChange(of: pitchData?.frequencies.count) { _, newCount in
                    // Re-initialize scroll position when pitch data is loaded
                    if let count = newCount, count > 0 {
                        initializeScrollPosition(
                            viewportWidth: viewportWidth,
                            viewportHeight: viewportHeight,
                            canvasHeight: canvasHeight,
                            leftPadding: leftPadding
                        )
                    }
                }
                .onChange(of: currentTime) { _, newTime in
                    scrollManager.updateTimeScroll(
                        currentTime: newTime,
                        viewportWidth: viewportWidth,
                        pixelsPerSecond: PitchGraphConstants.pixelsPerSecond,
                        canvasLeftPadding: leftPadding
                    )

                    // Auto-follow detected pitch during playback (when enabled and not user-dragging)
                    if isPlaying && autoPitchFollow && !isUserDragging {
                        if let targetFreq = currentPitchFrequency(at: newTime) {
                            // Only update when pitch changes significantly (avoid constant small movements)
                            if lastFollowedFrequency == nil || abs(targetFreq - lastFollowedFrequency!) > 10.0 {
                                lastFollowedFrequency = targetFreq
                            }
                            // Always apply gentle easing towards target
                            updateVerticalScrollForPitch(
                                targetFrequency: targetFreq,
                                viewportHeight: viewportHeight,
                                canvasHeight: canvasHeight
                            )
                        }
                    }
                }
            }
            .background(ColorPalette.secondary)
            .cornerRadius(8)
        }
        .contentShape(Rectangle())
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("PitchAnalysisView")
        .onTapGesture {
            onPlayPause?()
        }
    }

    // MARK: - Scroll Position Management

    private func initializeScrollPosition(
        viewportWidth: CGFloat,
        viewportHeight: CGFloat,
        canvasHeight: CGFloat,
        leftPadding: CGFloat
    ) {
        // Calculate target frequency to center (based on initial pitch data)
        var targetFrequency: Double? = nil
        if let data = pitchData, !data.frequencies.isEmpty {
            // Use pitch data from the first 3 seconds for initial positioning
            let initialDuration = 3.0
            var initialFrequencies: [Float] = []

            for i in 0..<data.timeStamps.count {
                if data.timeStamps[i] <= initialDuration {
                    initialFrequencies.append(data.frequencies[i])
                } else {
                    break
                }
            }

            // Fall back to all frequencies if no data in first 3 seconds
            let frequenciesToUse = initialFrequencies.isEmpty ? data.frequencies : initialFrequencies

            if let minFreq = frequenciesToUse.min(), let maxFreq = frequenciesToUse.max() {
                targetFrequency = (Double(minFreq) + Double(maxFreq)) / 2
            }
        }

        scrollManager.initializePosition(
            viewportWidth: viewportWidth,
            viewportHeight: viewportHeight,
            canvasHeight: canvasHeight,
            currentTime: currentTime,
            pixelsPerSecond: PitchGraphConstants.pixelsPerSecond,
            canvasLeftPadding: leftPadding
        )

        // Adjust Y position to center on detected pitch range
        if let targetFreq = targetFrequency {
            let coordinateSystem = PitchGraphCoordinateSystem()
            let targetCanvasY = coordinateSystem.frequencyToCanvasY(frequency: targetFreq, canvasHeight: canvasHeight)

            // Calculate paperTop to center targetCanvasY in viewport
            let idealPaperTop = viewportHeight / 2 - targetCanvasY

            // Clamp to valid range
            let maxPaperTop: CGFloat = 0
            let minPaperTop = viewportHeight - canvasHeight
            let clampedPaperTop = max(minPaperTop, min(maxPaperTop, idealPaperTop))

            scrollManager.paperTop = clampedPaperTop
            scrollManager.lastPaperTop = scrollManager.paperTop
        }
    }

    private func handleDrag(value: DragGesture.Value, viewportHeight: CGFloat, canvasHeight: CGFloat) {
        if isPlaying {
            // During playback: vertical only (direction locking)
            if isDraggingVertically == nil {
                let dx = abs(value.translation.width)
                let dy = abs(value.translation.height)
                if dx > 5 || dy > 5 {
                    isDraggingVertically = dy > dx
                }
            }

            if isDraggingVertically == true {
                scrollManager.handleVerticalDrag(
                    translation: value.translation.height,
                    viewportHeight: viewportHeight,
                    canvasHeight: canvasHeight
                )
            }
            // Ignore horizontal drag during playback
        } else {
            // When paused: allow diagonal swipe (both vertical + horizontal)
            // Determine drag direction on first movement
            if isDraggingVertically == nil {
                let dx = abs(value.translation.width)
                let dy = abs(value.translation.height)
                if dx > 5 || dy > 5 {
                    isDraggingVertically = dy > dx
                }
            }

            // Handle vertical drag for frequency scrolling
            if isDraggingVertically == true {
                scrollManager.handleVerticalDrag(
                    translation: value.translation.height,
                    viewportHeight: viewportHeight,
                    canvasHeight: canvasHeight
                )
            } else if isDraggingVertically == false {
                // Handle horizontal drag for seek
                let dataDuration = pitchData?.timeStamps.last ?? 10.0
                let deltaX = value.translation.width - lastDragTranslation.width
                let deltaTime = -Double(deltaX) / Double(PitchGraphConstants.pixelsPerSecond)
                let newTime = max(0, min(dataDuration, currentTime + deltaTime))

                lastDragTranslation = value.translation
                onSeek?(newTime)
            }
        }
    }

    // MARK: - Pitch Lookup

    /// Find the detected pitch frequency at the given time
    /// Uses binary search for efficient lookup
    private func currentPitchFrequency(at time: Double) -> Double? {
        guard let data = pitchData, !data.timeStamps.isEmpty else { return nil }

        // Find the closest timestamp using binary search
        var low = 0
        var high = data.timeStamps.count - 1

        while low < high {
            let mid = (low + high) / 2
            if data.timeStamps[mid] < time {
                low = mid + 1
            } else {
                high = mid
            }
        }

        // Get the frequency at the closest timestamp
        let frequency = data.frequencies[low]

        // Skip very low frequencies (likely silence or noise)
        if frequency < 50.0 {
            return nil
        }

        return Double(frequency)
    }

    // MARK: - Auto-Follow Scrolling

    /// Update vertical scroll to keep the target pitch frequency in view
    private func updateVerticalScrollForPitch(
        targetFrequency: Double,
        viewportHeight: CGFloat,
        canvasHeight: CGFloat
    ) {
        // Convert target frequency to canvas Y position (in canvas coordinates)
        let targetCanvasY = coordinateSystem.frequencyToCanvasY(frequency: targetFrequency, canvasHeight: canvasHeight)

        // Convert to viewport Y position
        let targetViewportY = targetCanvasY + scrollManager.paperTop

        // Define safe zone margins (20% from top and bottom)
        let topMargin = viewportHeight * 0.2
        let bottomMargin = viewportHeight * 0.8

        // Check if target is within safe zone - no scrolling needed
        if targetViewportY >= topMargin && targetViewportY <= bottomMargin {
            return
        }

        // Target is outside safe zone - scroll to bring it back to center
        let idealPaperTop = viewportHeight / 2 - targetCanvasY

        // Clamp to valid range
        let maxPaperTop: CGFloat = 0
        let minPaperTop = viewportHeight - canvasHeight
        let targetPaperTop = max(minPaperTop, min(maxPaperTop, idealPaperTop))

        // Fast easing when scrolling is actually needed (0.3 = responsive)
        let easingFactor: CGFloat = 0.3
        let newPaperTop = scrollManager.paperTop + (targetPaperTop - scrollManager.paperTop) * easingFactor

        scrollManager.paperTop = newPaperTop
        scrollManager.lastPaperTop = newPaperTop
    }

    // MARK: - Canvas Drawing

    private func drawPitchGraphCanvas(
        context: inout GraphicsContext,
        viewportSize: CGSize,
        canvasHeight: CGFloat,
        canvasWidth: CGFloat,
        leftPadding: CGFloat,
        data: PitchAnalysisData
    ) {
        let viewportWidth = viewportSize.width
        let viewportHeight = viewportSize.height

        // Create clipping region for graph area
        let clipRect = CGRect(
            x: PitchGraphConstants.leftMargin,
            y: 0,
            width: viewportWidth - PitchGraphConstants.leftMargin - PitchGraphConstants.rightMargin,
            height: viewportHeight - PitchGraphConstants.bottomMargin
        )

        // Draw main graph content with clipping (using a copy of context)
        var clippedContext = context
        clippedContext.clip(to: Path(clipRect))

        // Apply canvas offset for scrolling
        clippedContext.translateBy(x: scrollManager.canvasOffsetX, y: scrollManager.paperTop)

        // Draw background grid
        renderer.drawBackground(
            context: clippedContext,
            canvasHeight: canvasHeight,
            canvasWidth: canvasWidth,
            leftPadding: leftPadding
        )

        // Prepare pitch data for renderer
        let pitchPoints = preparePitchData(from: data)
        renderer.drawPitchData(
            context: clippedContext,
            canvasHeight: canvasHeight,
            pitchData: pitchPoints,
            leftPadding: leftPadding,
            targetSegments: nil
        )

        // Draw frequency labels (fixed X, scrolling Y) - use original context without clipping
        renderer.drawFrequencyLabels(
            context: context,
            canvasHeight: canvasHeight,
            viewportHeight: viewportHeight,
            paperTop: scrollManager.paperTop
        )

        // Draw time labels (scrolling X, fixed Y)
        let dataDuration = data.timeStamps.last ?? 10.0
        renderer.drawTimeLabels(
            context: context,
            dataDuration: dataDuration,
            leftPadding: leftPadding,
            viewportWidth: viewportWidth,
            viewportHeight: viewportHeight,
            canvasOffsetX: scrollManager.canvasOffsetX
        )

        // Draw playback position line (fully fixed)
        renderer.drawPlaybackPosition(
            context: context,
            viewportWidth: viewportWidth,
            viewportHeight: viewportHeight
        )
    }

    private func preparePitchData(from data: PitchAnalysisData) -> [(time: Double, frequency: Double, confidence: Float, amplitude: Float)] {
        var result: [(Double, Double, Float, Float)] = []

        for (index, timestamp) in data.timeStamps.enumerated() {
            let frequency = Double(data.frequencies[index])
            let confidence = data.confidences[index]
            // Get amplitude (default to 0.5 if not available for backward compatibility)
            let amplitude = index < data.amplitudes.count ? data.amplitudes[index] : 0.5

            // Filter out frequencies outside display range
            guard frequency >= PitchGraphConstants.minFrequency &&
                  frequency <= PitchGraphConstants.maxFrequency else { continue }

            result.append((timestamp, frequency, confidence, amplitude))
        }

        return result
    }
}
