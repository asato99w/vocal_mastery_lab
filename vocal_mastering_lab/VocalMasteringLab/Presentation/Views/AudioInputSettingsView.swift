import SwiftUI
import VocalisDomain

/// Audio input settings configuration view (detection sensitivity and confidence)
struct AudioInputSettingsView: View {

    @StateObject private var viewModel: AudioInputSettingsViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var showResetAlert = false
    @State private var showSaveError = false
    @State private var saveErrorMessage = ""

    init(viewModel: AudioInputSettingsViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        NavigationView {
            Form {
                // Detection Sensitivity Section
                Section {
                    Picker("input.detection_sensitivity".localized, selection: $viewModel.detectionSensitivity) {
                        Text("input.sensitivity_low".localized).tag(AudioDetectionSettings.DetectionSensitivity.low)
                        Text("input.sensitivity_normal".localized).tag(AudioDetectionSettings.DetectionSensitivity.normal)
                        Text("input.sensitivity_high".localized).tag(AudioDetectionSettings.DetectionSensitivity.high)
                    }
                    .pickerStyle(.segmented)
                } header: {
                    Text("input.pitch_detection_sensitivity".localized)
                } footer: {
                    Text("input.sensitivity_description".localized)
                }

                // Confidence Threshold Section
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("input.detection_accuracy".localized)
                                .font(.body)
                            Spacer()
                            Text("\(Int(viewModel.confidenceThreshold * 100))%")
                                .foregroundColor(.secondary)
                        }

                        Slider(value: $viewModel.confidenceThreshold, in: 0.1...1.0, step: 0.05)
                    }
                } header: {
                    Text("input.pitch_detection_accuracy".localized)
                } footer: {
                    Text("input.accuracy_description".localized)
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
            .navigationTitle("input.title".localized)
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
                Text("input.reset_confirmation".localized)
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
            saveErrorMessage = "input.save_failed".localized + ": \(error.localizedDescription)"
            showSaveError = true
        }
    }

    private func resetSettings() {
        do {
            try viewModel.resetSettings()
        } catch {
            saveErrorMessage = "input.reset_failed".localized + ": \(error.localizedDescription)"
            showSaveError = true
        }
    }
}

// MARK: - Preview

#Preview {
    AudioInputSettingsView(
        viewModel: AudioInputSettingsViewModel(
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
