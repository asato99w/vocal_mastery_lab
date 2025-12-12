import SwiftUI
import VocalisDomain

/// Audio output settings configuration view (recording playback volume)
struct AudioOutputSettingsView: View {

    @StateObject private var viewModel: AudioOutputSettingsViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var showResetAlert = false
    @State private var showSaveError = false
    @State private var saveErrorMessage = ""

    init(viewModel: AudioOutputSettingsViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        NavigationView {
            Form {
                // Volume Settings Section
                Section {
                    VStack(alignment: .leading, spacing: 16) {
                        // Recording Playback Volume
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("output.recording_volume".localized)
                                    .font(.body)
                                Spacer()
                                Text("\(Int(viewModel.recordingPlaybackVolume * 100))%")
                                    .foregroundColor(.secondary)
                            }
                            Slider(value: $viewModel.recordingPlaybackVolume, in: 0...1, step: 0.05)
                        }
                    }
                } header: {
                    Text("output.volume_settings".localized)
                } footer: {
                    Text("output.volume_description".localized)
                }

                // Reset Button Section
                Section {
                    Button(role: .destructive) {
                        showResetAlert = true
                    } label: {
                        HStack {
                            Spacer()
                            Text("output.reset_default".localized)
                            Spacer()
                        }
                    }
                }
            }
            .navigationTitle("output.title".localized)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("cancel".localized) {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .confirmationAction) {
                    Button("save".localized) {
                        saveSettings()
                    }
                    .disabled(!viewModel.hasChanges)
                }
            }
            .alert("output.reset_default".localized, isPresented: $showResetAlert) {
                Button("cancel".localized, role: .cancel) { }
                Button("output.reset".localized, role: .destructive) {
                    resetSettings()
                }
            } message: {
                Text("output.reset_confirmation".localized)
            }
            .alert("output.save_error".localized, isPresented: $showSaveError) {
                Button("ok".localized, role: .cancel) { }
            } message: {
                Text(saveErrorMessage)
            }
        }
    }

    // MARK: - Private Methods

    private func saveSettings() {
        do {
            try viewModel.saveSettings()
            dismiss()
        } catch {
            saveErrorMessage = "output.save_failed".localized + ": \(error.localizedDescription)"
            showSaveError = true
        }
    }

    private func resetSettings() {
        do {
            try viewModel.resetSettings()
        } catch {
            saveErrorMessage = "output.reset_failed".localized + ": \(error.localizedDescription)"
            showSaveError = true
        }
    }
}

// MARK: - Preview

#Preview {
    AudioOutputSettingsView(
        viewModel: AudioOutputSettingsViewModel(
            repository: PreviewAudioSettingsRepository()
        )
    )
}

/// Preview用のリポジトリ実装
private class PreviewAudioSettingsRepository: AudioSettingsRepositoryProtocol {
    func get() -> AudioDetectionSettings {
        .default
    }

    func save(_ settings: AudioDetectionSettings) throws {
        // Preview用なので何もしない
    }

    func reset() throws {
        // Preview用なので何もしない
    }
}
