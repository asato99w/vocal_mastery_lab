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

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("analysis.spectrogram_title".localized)
                .font(.subheadline)
                .fontWeight(.semibold)
                .accessibilityIdentifier("SpectrogramTitle")

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

                // Canvas: Contains the entire frequency range (0Hz ~ maxFreq)
                Canvas { context, size in
                    if let data = spectrogramData, !data.timeStamps.isEmpty {
                        // Draw everything in canvas coordinates
                        // size here is the canvas size, not viewport size

                        // 1. Draw spectrogram (background) - SCROLLABLE
                        renderer.drawSpectrogram(
                            context: context,
                            canvasWidth: size.width,
                            canvasHeight: canvasHeight,
                            maxFreq: maxFreq,
                            data: data,
                            leftPadding: canvasLeftPadding
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
                    DragGesture()
                        .onChanged { value in
                            let translation = value.translation

                            // Detect drag direction
                            let angle = atan2(abs(translation.height), abs(translation.width))

                            if angle > .pi / 4 {
                                // Vertical-dominant drag: frequency axis scrolling
                                scrollManager.handleVerticalDrag(
                                    translation: translation.height,
                                    viewportHeight: viewportHeight,
                                    canvasHeight: canvasHeight
                                )
                            } else if let onSeek = onSeek {
                                // Horizontal-dominant drag: time axis seeking
                                // Calculate time change from horizontal translation (reduced sensitivity)
                                let seekSensitivity = 3.0  // Lower sensitivity: 3x more drag needed
                                let timeChange = -Double(translation.width) / (Double(pixelsPerSecond) * seekSensitivity)
                                let newTime = max(0, currentTime + timeChange)

                                // Seek to new time
                                onSeek(newTime)
                            }
                        }
                        .onEnded { _ in
                            scrollManager.endDrag()
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
        .onTapGesture {
            onPlayPause?()
        }
    }

}

// MARK: - Pitch Analysis View

struct PitchAnalysisView: View {
    let currentTime: Double
    let pitchData: PitchAnalysisData?
    var isExpanded: Bool = false
    var onExpand: (() -> Void)? = nil
    var onCollapse: (() -> Void)? = nil
    var onPlayPause: (() -> Void)? = nil
    var onSeek: ((Double) -> Void)? = nil

    // Scroll manager for 2D scrolling (reusing SpectrogramScrollManager)
    @State private var scrollManager = SpectrogramScrollManager()

    // Drag state for horizontal seek
    @State private var lastDragTranslation: CGSize = .zero

    // Coordinate system and renderer
    private let coordinateSystem = PitchGraphCoordinateSystem()
    private let renderer = PitchGraphRenderer()

    // Drag gesture state
    @State private var dragStartLocation: CGPoint = .zero
    @State private var isDraggingVertically: Bool? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("analysis.pitch_graph_title".localized)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(ColorPalette.text)
                .accessibilityIdentifier("PitchGraphTitle")

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
                    drawLegend(context: mutableContext, size: size)
                }
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            handleDrag(value: value, viewportHeight: viewportHeight, canvasHeight: canvasHeight)
                        }
                        .onEnded { _ in
                            scrollManager.endDrag()
                            isDraggingVertically = nil
                            lastDragTranslation = .zero
                        }
                )
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

    private func drawLegend(context: GraphicsContext, size: CGSize) {
        let legendY: CGFloat = 20

        // Detected pitch legend only (target scale legend removed)
        var path = Path()
        path.move(to: CGPoint(x: 10, y: legendY))
        path.addLine(to: CGPoint(x: 40, y: legendY))
        context.stroke(path, with: .color(.blue), lineWidth: 1.5)
        context.draw(Text("analysis.legend_detected".localized).font(.caption2), at: CGPoint(x: 45, y: legendY), anchor: .leading)
    }
}
