import SwiftUI
import VocalisDomain

/// Helper struct for target note bar position calculations
/// Used by PitchBarView to render target note segments
public struct TargetNoteBarRenderer {

    // MARK: - Position Calculations

    /// Calculate the width of a target note bar in pixels
    /// - Parameter segment: The note segment to calculate width for
    /// - Returns: Width in pixels based on duration and pixelsPerSecond
    public static func calculateBarWidth(for segment: NoteSegment) -> CGFloat {
        return segment.duration * PitchBarConstants.pixelsPerSecond
    }

    /// Calculate the X position of a target note bar
    /// - Parameters:
    ///   - segment: The note segment to position
    ///   - leftPadding: Left padding offset for the canvas
    /// - Returns: X coordinate for the bar's left edge
    public static func calculateBarXPosition(for segment: NoteSegment, leftPadding: CGFloat) -> CGFloat {
        return PitchBarConstants.timeToX(time: segment.startTime, leftPadding: leftPadding)
    }

    /// Calculate the Y position of a target note bar (center Y)
    /// Uses logarithmic scale for frequency-based positioning
    /// - Parameters:
    ///   - segment: The note segment to position
    ///   - canvasHeight: Total canvas height for frequency mapping
    /// - Returns: Y coordinate for the bar's center
    public static func calculateBarYPosition(for segment: NoteSegment, canvasHeight: CGFloat) -> CGFloat {
        return PitchBarConstants.frequencyToY(frequency: segment.frequency, canvasHeight: canvasHeight)
    }

    // MARK: - Canvas Drawing

    /// Draw a single target note bar on the canvas
    /// - Parameters:
    ///   - context: Graphics context to draw in
    ///   - segment: The note segment to render
    ///   - canvasHeight: Total canvas height
    ///   - leftPadding: Left padding offset
    public static func drawTargetBar(
        context: inout GraphicsContext,
        segment: NoteSegment,
        canvasHeight: CGFloat,
        leftPadding: CGFloat
    ) {
        let x = calculateBarXPosition(for: segment, leftPadding: leftPadding)
        let y = calculateBarYPosition(for: segment, canvasHeight: canvasHeight)
        let width = calculateBarWidth(for: segment)
        let height = PitchBarConstants.noteBarHeight

        // Draw rounded rectangle for the target bar
        let rect = CGRect(
            x: x,
            y: y - height / 2,  // Center vertically on the frequency
            width: width,
            height: height
        )

        let path = Path(roundedRect: rect, cornerRadius: 4)
        context.fill(path, with: .color(PitchBarConstants.targetBarColor))
    }

    /// Draw all target note bars from a timeline
    /// - Parameters:
    ///   - context: Graphics context to draw in
    ///   - segments: Array of note segments to render
    ///   - canvasHeight: Total canvas height
    ///   - leftPadding: Left padding offset
    public static func drawAllTargetBars(
        context: inout GraphicsContext,
        segments: [NoteSegment],
        canvasHeight: CGFloat,
        leftPadding: CGFloat
    ) {
        for segment in segments {
            drawTargetBar(
                context: &context,
                segment: segment,
                canvasHeight: canvasHeight,
                leftPadding: leftPadding
            )
        }
    }

    /// Draw note name labels for target bars
    /// - Parameters:
    ///   - context: Graphics context to draw in
    ///   - segments: Array of note segments
    ///   - canvasHeight: Total canvas height
    ///   - leftPadding: Left padding offset
    public static func drawNoteLabels(
        context: inout GraphicsContext,
        segments: [NoteSegment],
        canvasHeight: CGFloat,
        leftPadding: CGFloat
    ) {
        // Group segments by note to avoid duplicate labels
        var drawnNotes: Set<UInt8> = []

        for segment in segments {
            let noteValue = segment.note.value
            guard !drawnNotes.contains(noteValue) else { continue }
            drawnNotes.insert(noteValue)

            let y = calculateBarYPosition(for: segment, canvasHeight: canvasHeight)
            let noteName = segment.note.noteName

            // Draw note name at the left margin
            context.draw(
                Text(noteName)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.gray),
                at: CGPoint(x: 25, y: y),
                anchor: .center
            )
        }
    }
}

/// SwiftUI View for displaying target note bars in karaoke-style UI
/// This view renders the gray background bars representing target notes
public struct TargetNoteBarView: View {
    let segments: [NoteSegment]
    let canvasHeight: CGFloat
    let leftPadding: CGFloat

    public init(segments: [NoteSegment], canvasHeight: CGFloat, leftPadding: CGFloat) {
        self.segments = segments
        self.canvasHeight = canvasHeight
        self.leftPadding = leftPadding
    }

    public var body: some View {
        Canvas { context, size in
            var mutableContext = context
            TargetNoteBarRenderer.drawAllTargetBars(
                context: &mutableContext,
                segments: segments,
                canvasHeight: canvasHeight,
                leftPadding: leftPadding
            )
        }
    }
}

#if DEBUG
struct TargetNoteBarView_Previews: PreviewProvider {
    static var previews: some View {
        let segments: [NoteSegment] = {
            do {
                return [
                    NoteSegment(note: try MIDINote(60), startTime: 0.0, endTime: 1.0),   // C4
                    NoteSegment(note: try MIDINote(62), startTime: 1.0, endTime: 2.0),   // D4
                    NoteSegment(note: try MIDINote(64), startTime: 2.0, endTime: 3.0),   // E4
                    NoteSegment(note: try MIDINote(65), startTime: 3.0, endTime: 4.0),   // F4
                    NoteSegment(note: try MIDINote(67), startTime: 4.0, endTime: 5.0)    // G4
                ]
            } catch {
                return []
            }
        }()

        VStack {
            Text("Target Note Bars Preview")
                .font(.headline)

            TargetNoteBarView(
                segments: segments,
                canvasHeight: 300,
                leftPadding: 50
            )
            .frame(width: 600, height: 300)
            .background(Color.black.opacity(0.9))
        }
        .padding()
    }
}
#endif
