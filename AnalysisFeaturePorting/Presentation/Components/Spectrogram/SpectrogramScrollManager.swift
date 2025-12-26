import SwiftUI
import os.log

/// Spectrogram scroll management
/// Handles 2D scrolling logic for canvas positioning
@Observable
public class SpectrogramScrollManager {
    // MARK: - Scroll State

    // Y-axis scroll (vertical - frequency axis)
    // paperTop: Y coordinate of paper top relative to viewport top
    //   - paperTop = 0 (maxPaperTop): paper top aligned with viewport top (cannot push down further)
    //   - paperTop = viewportH - canvasH (minPaperTop): paper bottom aligned with viewport bottom (cannot push up further)
    //   - Initial: paperTop = minPaperTop (bottom-aligned, low frequency visible)
    public var paperTop: CGFloat = 0
    public var lastPaperTop: CGFloat = 0

    // X-axis scroll (horizontal - time axis)
    // canvasOffsetX: X offset to apply to canvas (positive = move canvas right)
    //   - Formula: canvasOffsetX = playheadX - currentTimeCanvasX
    //   - Initial (currentTime=0): canvasOffsetX = playheadX (positive, canvas shifts right, 0s at center)
    //   - During playback: canvasOffsetX decreases (canvas shifts left, spectrogram flows left)
    // NOTE: With data positioned at leftPadding, initial offset = 0 aligns Canvas left edge with screen left edge
    // This puts 0s data (at Canvas x=leftPadding) at screen center (playheadX)
    public var canvasOffsetX: CGFloat = 0

    // MARK: - Initialization

    public init() {}

    // MARK: - Scroll Position Management

    /// Initialize scroll position (called on appear and expand)
    /// - Parameters:
    ///   - viewportWidth: Viewport width
    ///   - viewportHeight: Viewport height
    ///   - canvasHeight: Canvas height
    ///   - currentTime: Current playback time
    ///   - pixelsPerSecond: Time axis density
    ///   - canvasLeftPadding: Left padding for canvas
    public func initializePosition(
        viewportWidth: CGFloat,
        viewportHeight: CGFloat,
        canvasHeight: CGFloat,
        currentTime: Double,
        pixelsPerSecond: CGFloat,
        canvasLeftPadding: CGFloat
    ) {
        // Y-axis scroll initialization (both Normal and Expanded views)
        let viewportH = viewportHeight
        let canvasH = canvasHeight
        let maxPaperTop: CGFloat = 0
        let minPaperTop = viewportH - canvasH

        // Initial placement: bottom-aligned (low frequency visible)
        paperTop = minPaperTop

        // Clamp (mandatory after any movement)
        paperTop = max(minPaperTop, min(maxPaperTop, paperTop))

        lastPaperTop = paperTop

        // X-axis scroll initialization
        let playheadX = viewportWidth / 2
        let currentTimeCanvasX = CGFloat(currentTime) * pixelsPerSecond + canvasLeftPadding
        canvasOffsetX = playheadX - currentTimeCanvasX

        os_log(.debug, log: OSLog(subsystem: "com.kazuasato.VocalisStudio", category: "scroll_init"),
               "📍 Initial: paperTop=%{public}f, minPaperTop=%{public}f, maxPaperTop=%{public}f, canvasOffsetX=%{public}f",
               paperTop, minPaperTop, maxPaperTop, canvasOffsetX)
        FileLogger.shared.log(level: "INFO", category: "scroll_init",
            message: "📍 Initial placement: paperTop=\(paperTop), viewportH=\(viewportH), canvasH=\(canvasH), canvasOffsetX=\(canvasOffsetX), playheadX=\(playheadX), currentTime=\(currentTime)")
    }

    /// Handle vertical drag (frequency axis scrolling)
    /// - Parameters:
    ///   - translation: Drag translation height
    ///   - viewportHeight: Viewport height
    ///   - canvasHeight: Canvas height
    public func handleVerticalDrag(
        translation: CGFloat,
        viewportHeight: CGFloat,
        canvasHeight: CGFloat
    ) {
        let viewportH = viewportHeight
        let canvasH = canvasHeight
        let maxPaperTop: CGFloat = 0
        let minPaperTop = viewportH - canvasH

        // Calculate candidate position
        let candidate = lastPaperTop + translation

        // Clamp immediately (mandatory after any movement)
        paperTop = max(minPaperTop, min(maxPaperTop, candidate))
    }

    /// Finish drag gesture
    public func endDrag() {
        lastPaperTop = paperTop
    }

    /// Update X-axis scroll position based on current time
    /// - Parameters:
    ///   - currentTime: Current playback time
    ///   - viewportWidth: Viewport width
    ///   - pixelsPerSecond: Time axis density
    ///   - canvasLeftPadding: Left padding for canvas
    public func updateTimeScroll(
        currentTime: Double,
        viewportWidth: CGFloat,
        pixelsPerSecond: CGFloat,
        canvasLeftPadding: CGFloat
    ) {
        // Update canvasOffsetX to keep currentTime position under red line (playheadX)
        let playheadX = viewportWidth / 2
        let currentTimeCanvasX = CGFloat(currentTime) * pixelsPerSecond + canvasLeftPadding
        canvasOffsetX = playheadX - currentTimeCanvasX

        // Calculate label position in viewport coordinates for verification
        let labelViewportX = currentTimeCanvasX + canvasOffsetX
        let alignmentError = abs(labelViewportX - playheadX)

        FileLogger.shared.log(level: "DEBUG", category: "time_axis_scroll",
            message: "⏩ Time axis scroll: currentTime=\(String(format: "%.2f", currentTime))s, playheadX=\(String(format: "%.1f", playheadX)), labelCanvasX=\(String(format: "%.1f", currentTimeCanvasX)), canvasOffsetX=\(String(format: "%.1f", canvasOffsetX)), labelViewportX=\(String(format: "%.1f", labelViewportX)), alignmentError=\(String(format: "%.2f", alignmentError))px")
    }
}
