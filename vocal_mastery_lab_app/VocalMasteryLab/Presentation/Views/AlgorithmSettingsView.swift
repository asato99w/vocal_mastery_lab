import SwiftUI
import VocalisDomain

/// Algorithm settings configuration view (pitch detection algorithm selection)
struct AlgorithmSettingsView: View {

    @StateObject private var viewModel: AlgorithmSettingsViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var showResetAlert = false
    @State private var showSaveError = false
    @State private var saveErrorMessage = ""

    init(viewModel: AlgorithmSettingsViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        NavigationView {
            Form {
                // Pitch Detection Algorithm Section
                Section {
                    ForEach(PitchDetectionAlgorithm.displayCases, id: \.self) { algorithm in
                        AlgorithmRow(
                            algorithm: algorithm,
                            isSelected: viewModel.pitchAlgorithm == algorithm
                        ) {
                            viewModel.pitchAlgorithm = algorithm
                        }
                    }
                } header: {
                    Text("algorithm.pitch_analysis".localized)
                } footer: {
                    Text("algorithm.section_description".localized)
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
            .navigationTitle("algorithm.title".localized)
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
                Text("algorithm.reset_confirmation".localized)
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
            saveErrorMessage = "algorithm.save_failed".localized + ": \(error.localizedDescription)"
            showSaveError = true
        }
    }

    private func resetSettings() {
        do {
            try viewModel.resetSettings()
        } catch {
            saveErrorMessage = "algorithm.reset_failed".localized + ": \(error.localizedDescription)"
            showSaveError = true
        }
    }
}

// MARK: - Algorithm Row Component

private struct AlgorithmRow: View {
    let algorithm: PitchDetectionAlgorithm
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        Button {
            onSelect()
        } label: {
            HStack(alignment: .top, spacing: 12) {
                // Selection indicator
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(isSelected ? .accentColor : .secondary)
                    .font(.title3)

                // Algorithm info
                VStack(alignment: .leading, spacing: 4) {
                    Text(algorithm.displayNameKey.localized)
                        .font(.body)
                        .fontWeight(isSelected ? .semibold : .regular)
                        .foregroundColor(.primary)

                    Text(algorithm.descriptionKey.localized)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(.vertical, 4)
        .accessibilityIdentifier("AlgorithmRow_\(algorithm.rawValue)")
    }
}

// MARK: - Preview

#Preview {
    AlgorithmSettingsView(
        viewModel: AlgorithmSettingsViewModel(
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
