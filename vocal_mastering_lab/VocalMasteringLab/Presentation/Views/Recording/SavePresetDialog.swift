import SwiftUI

/// Dialog view for saving a new preset
struct SavePresetDialog: View {
    @ObservedObject var presetViewModel: ScalePresetViewModel
    @ObservedObject var settingsViewModel: RecordingSettingsViewModel
    @FocusState private var isNameFieldFocused: Bool

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                TextField("preset.name_placeholder".localized, text: $presetViewModel.newPresetName)
                    .textFieldStyle(.roundedBorder)
                    .focused($isNameFieldFocused)
                    .padding(.horizontal)
                    .onSubmit {
                        savePresetIfValid()
                    }

                if let errorMessage = presetViewModel.errorMessage {
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundColor(.red)
                }

                Spacer()
            }
            .padding(.top, 20)
            .navigationTitle("preset.save_dialog_title".localized)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("cancel".localized) {
                        presetViewModel.newPresetName = ""
                        presetViewModel.isShowingSaveDialog = false
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("preset.save".localized) {
                        savePresetIfValid()
                    }
                    .disabled(!presetViewModel.isValidPresetName(presetViewModel.newPresetName))
                }
            }
            .onAppear {
                isNameFieldFocused = true
            }
        }
    }

    private func savePresetIfValid() {
        guard presetViewModel.isValidPresetName(presetViewModel.newPresetName) else { return }
        presetViewModel.savePreset(name: presetViewModel.newPresetName.trimmingCharacters(in: .whitespacesAndNewlines), from: settingsViewModel)
    }
}
