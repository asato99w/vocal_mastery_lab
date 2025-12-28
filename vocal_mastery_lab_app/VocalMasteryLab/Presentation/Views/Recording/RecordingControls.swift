import SwiftUI
import VocalisDomain
import OSLog

/// Recording control buttons (start, stop, cancel, analyze)
struct RecordingControls: View {
    let recordingState: RecordingState
    let hasLastRecording: Bool
    let canStartRecording: Bool
    let countdownValue: Int
    let onStart: () -> Void
    let onStop: () -> Void
    let onCancel: () -> Void
    let onAnalyze: (() -> Void)?

    /// Whether to use compact horizontal layout (for landscape mode)
    var isCompactLayout: Bool = false

    init(
        recordingState: RecordingState,
        hasLastRecording: Bool,
        canStartRecording: Bool,
        countdownValue: Int = 3,
        onStart: @escaping () -> Void,
        onStop: @escaping () -> Void,
        onCancel: @escaping () -> Void,
        onAnalyze: (() -> Void)? = nil,
        isCompactLayout: Bool = false
    ) {
        self.recordingState = recordingState
        self.hasLastRecording = hasLastRecording
        self.canStartRecording = canStartRecording
        self.countdownValue = countdownValue
        self.onStart = onStart
        self.onStop = onStop
        self.onCancel = onCancel
        self.onAnalyze = onAnalyze
        self.isCompactLayout = isCompactLayout
    }

    // Logger for diagnostic purposes
    private static let logger = Logger(subsystem: "com.kazuasato.VocalMasteryLab", category: "RecordingControls")

    var body: some View {
        VStack(spacing: 10) {
            switch recordingState {
            case .idle:
                idleControls

            case .preparing:
                preparingControls

            case .countdown:
                countdownControls

            case .recording:
                recordingControls
            }
        }
    }

    // MARK: - Preparing State Controls

    private var preparingControls: some View {
        VStack(spacing: 8) {
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle(tint: ColorPalette.primary))
                .accessibilityIdentifier("RecordingLoadingIndicator")

            Text("recording.preparing".localized)
                .font(.subheadline)
                .foregroundColor(ColorPalette.text.opacity(0.6))
        }
    }

    // MARK: - Idle State Controls

    private var idleControls: some View {
        Group {
            if isCompactLayout && hasLastRecording && onAnalyze != nil {
                // Horizontal layout for landscape mode
                HStack(spacing: 12) {
                    startRecordingButton
                    analyzeButton
                }
            } else {
                // Vertical layout for portrait mode
                VStack(spacing: 8) {
                    startRecordingButton
                    if hasLastRecording && onAnalyze != nil {
                        analyzeButton
                    }
                }
            }
        }
    }

    private var startRecordingButton: some View {
        Button(action: {
            Self.logger.error("UI_TEST_MARK: StartRecordingButton action called")
            onStart()
        }) {
            HStack {
                Image(systemName: "mic.fill")
                Text("recording.start_button".localized)
            }
        }
        .buttonStyle(AlertButtonStyle())
        .disabled(!canStartRecording)
        .opacity(canStartRecording ? 1.0 : 0.5)
        .accessibilityIdentifier("StartRecordingButton")
    }

    private var analyzeButton: some View {
        Button(action: {
            Self.logger.info("AnalyzeRecordingButton action called")
            onAnalyze?()
        }) {
            HStack {
                Image(systemName: "waveform.and.magnifyingglass")
                Text("recording.analyze_button".localized)
            }
        }
        .buttonStyle(SecondaryButtonStyle())
        .accessibilityIdentifier("AnalyzeRecordingButton")
    }

    // MARK: - Countdown State Controls

    private var countdownControls: some View {
        VStack(spacing: 16) {
            // Countdown number display (consistent with timer: 48pt monospaced)
            Text("\(countdownValue)")
                .font(.system(size: 48, weight: .light, design: .monospaced))
                .foregroundColor(ColorPalette.primary)
                .accessibilityIdentifier("CountdownNumber")
                .accessibilityLabel("カウントダウン \(countdownValue)")

            Button(action: onCancel) {
                Text("cancel".localized)
            }
            .buttonStyle(SecondaryButtonStyle())
        }
    }

    // MARK: - Recording State Controls

    private var recordingControls: some View {
        Button(action: onStop) {
            HStack {
                Image(systemName: "stop.fill")
                Text("recording.stop_button".localized)
            }
        }
        .buttonStyle(SecondaryButtonStyle())
        .accessibilityIdentifier("StopRecordingButton")
    }
}
