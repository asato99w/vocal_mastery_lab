import SwiftUI
import VocalisDomain

/// Vocal extraction screen
public struct VocalExtractionView: View {
    @StateObject private var viewModel: VocalExtractionViewModel
    @Environment(\.dismiss) private var dismiss

    public init(viewModel: VocalExtractionViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    public var body: some View {
        VStack(spacing: 24) {
            // Recording info section
            recordingInfoSection

            Spacer()

            // Main content based on state
            stateContent

            Spacer()

            // Action buttons
            actionButtons
        }
        .padding()
        .navigationTitle("ボーカル抽出")
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear {
            Task {
                await viewModel.stopPlayback()
            }
        }
    }

    // MARK: - Recording Info Section

    private var recordingInfoSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(viewModel.recordingTitle)
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(ColorPalette.text)

            HStack {
                Label(viewModel.recordingDate, systemImage: "calendar")
                    .font(.caption)
                    .foregroundColor(ColorPalette.text.opacity(0.6))

                Text("•")
                    .foregroundColor(ColorPalette.text.opacity(0.4))

                Label(viewModel.recordingDuration, systemImage: "clock")
                    .font(.caption)
                    .foregroundColor(ColorPalette.text.opacity(0.6))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(ColorPalette.secondary)
        )
    }

    // MARK: - State Content

    @ViewBuilder
    private var stateContent: some View {
        switch viewModel.state {
        case .idle:
            idleContent
        case .processing(let progress, let stage):
            processingContent(progress: progress, stage: stage)
        case .completed(let result):
            completedContent(result: result)
        case .error(let message):
            errorContent(message: message)
        }
    }

    private var idleContent: some View {
        VStack(spacing: 20) {
            Image(systemName: "waveform.path.ecg")
                .font(.system(size: 60))
                .foregroundColor(ColorPalette.primary)

            Text("録音からボーカルを抽出します")
                .font(.body)
                .foregroundColor(ColorPalette.text.opacity(0.8))
                .multilineTextAlignment(.center)
        }
    }

    private func processingContent(progress: Double, stage: String) -> some View {
        VStack(spacing: 20) {
            ProgressView(value: progress)
                .progressViewStyle(LinearProgressViewStyle(tint: ColorPalette.primary))
                .scaleEffect(y: 2)

            Text("\(Int(progress * 100))%")
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(ColorPalette.primary)

            Text(stage)
                .font(.body)
                .foregroundColor(ColorPalette.text.opacity(0.8))
        }
        .padding(.horizontal, 40)
    }

    private func completedContent(result: ExtractionResultData) -> some View {
        VStack(spacing: 20) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 60))
                .foregroundColor(.green)

            Text("抽出完了")
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(ColorPalette.text)

            // Preview section
            VStack(spacing: 12) {
                Text("プレビュー")
                    .font(.headline)
                    .foregroundColor(ColorPalette.text)
                    .frame(maxWidth: .infinity, alignment: .leading)

                PreviewButton(
                    title: "元の音声",
                    icon: "waveform",
                    action: { Task { await viewModel.playOriginal() } }
                )

                PreviewButton(
                    title: "ボーカル",
                    icon: "person.wave.2",
                    action: { Task { await viewModel.playVocal() } }
                )

                if result.instrumentalURL != nil {
                    PreviewButton(
                        title: "伴奏",
                        icon: "music.note.list",
                        action: { Task { await viewModel.playInstrumental() } }
                    )
                }

                Button("停止") {
                    Task { await viewModel.stopPlayback() }
                }
                .font(.caption)
                .foregroundColor(ColorPalette.text.opacity(0.6))
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(ColorPalette.secondary)
            )
        }
    }

    private func errorContent(message: String) -> some View {
        VStack(spacing: 20) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 60))
                .foregroundColor(.red)

            Text("エラー")
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(ColorPalette.text)

            Text(message)
                .font(.body)
                .foregroundColor(ColorPalette.text.opacity(0.8))
                .multilineTextAlignment(.center)
        }
    }

    // MARK: - Action Buttons

    @ViewBuilder
    private var actionButtons: some View {
        switch viewModel.state {
        case .idle:
            Button(action: {
                Task { await viewModel.startExtraction() }
            }) {
                Text("抽出開始")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(ColorPalette.primary)
                    .cornerRadius(12)
            }

        case .processing:
            EmptyView()

        case .completed:
            HStack(spacing: 16) {
                Button(action: {
                    viewModel.reset()
                }) {
                    Text("やり直し")
                        .font(.headline)
                        .foregroundColor(ColorPalette.text)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(ColorPalette.secondary)
                        .cornerRadius(12)
                }

                Button(action: {
                    Task {
                        let success = await viewModel.saveExtraction()
                        if success {
                            dismiss()
                        }
                    }
                }) {
                    if viewModel.isSaving {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(ColorPalette.primary)
                            .cornerRadius(12)
                    } else {
                        Text("保存")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(ColorPalette.primary)
                            .cornerRadius(12)
                    }
                }
                .disabled(viewModel.isSaving)
            }

        case .error:
            Button(action: {
                viewModel.reset()
            }) {
                Text("戻る")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(ColorPalette.primary)
                    .cornerRadius(12)
            }
        }
    }
}

// MARK: - Preview Button

private struct PreviewButton: View {
    let title: String
    let icon: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: icon)
                    .frame(width: 24)
                Text(title)
                Spacer()
                Image(systemName: "play.fill")
                    .font(.caption)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(ColorPalette.background)
            )
        }
        .foregroundColor(ColorPalette.text)
    }
}
