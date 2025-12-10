import SwiftUI
import VocalisDomain

/// View for displaying and selecting scale presets
struct PresetListView: View {
    @ObservedObject var presetViewModel: ScalePresetViewModel
    @ObservedObject var settingsViewModel: RecordingSettingsViewModel
    @State private var presetToDelete: ScalePreset?
    @State private var showDeleteConfirmation = false

    var body: some View {
        NavigationView {
            Group {
                if presetViewModel.presets.isEmpty {
                    emptyStateView
                } else {
                    presetListContent
                }
            }
            .navigationTitle("preset.title".localized)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("cancel".localized) {
                        presetViewModel.isShowingPresetList = false
                    }
                }
            }
            .alert("preset.delete_confirmation_title".localized, isPresented: $showDeleteConfirmation, presenting: presetToDelete) { preset in
                Button("cancel".localized, role: .cancel) {}
                Button("delete".localized, role: .destructive) {
                    presetViewModel.deletePreset(id: preset.id)
                }
            } message: { _ in
                Text("preset.delete_confirmation_message".localized)
            }
        }
    }

    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "bookmark.slash")
                .font(.system(size: 48))
                .foregroundColor(.secondary)

            Text("preset.empty_title".localized)
                .font(.headline)

            Text("preset.empty_message".localized)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
    }

    private var presetListContent: some View {
        List {
            ForEach(presetViewModel.presets) { preset in
                PresetRow(preset: preset)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        presetViewModel.applyPreset(preset, to: settingsViewModel)
                    }
                    .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                        Button(role: .destructive) {
                            presetToDelete = preset
                            showDeleteConfirmation = true
                        } label: {
                            Label("delete".localized, systemImage: "trash")
                        }
                    }
            }
        }
        .listStyle(.insetGrouped)
    }
}

/// Row view for displaying a single preset
private struct PresetRow: View {
    let preset: ScalePreset

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(preset.name)
                .font(.headline)

            HStack(spacing: 8) {
                // Scale type
                Text(scaleTypeLabel)
                    .font(.caption)
                    .foregroundColor(.secondary)

                Text("•")
                    .foregroundColor(.secondary)

                // Start pitch
                Text(startPitchLabel)
                    .font(.caption)
                    .foregroundColor(.secondary)

                Text("•")
                    .foregroundColor(.secondary)

                // Tempo
                Text("\(preset.settings.tempo) BPM")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    private var scaleTypeLabel: String {
        switch preset.settings.scaleType {
        case .fiveTone:
            return "recording.scale_five_tone".localized
        case .fiveToneDown:
            return "recording.scale_five_tone_down".localized
        case .octaveRepeat:
            return "recording.scale_octave_repeat".localized
        case .brokenScale:
            return "recording.scale_broken".localized
        case .brokenScaleDouble:
            return "recording.scale_broken_double".localized
        case .rossiniScale:
            return "recording.scale_rossini".localized
        case .arpeggioDownTriple:
            return "recording.scale_arpeggio_down_triple".localized
        case .off:
            return "recording.scale_off".localized
        }
    }

    private var startPitchLabel: String {
        let pitches = [
            "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2",
            "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
            "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
            "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
            "C6"
        ]
        let index = preset.settings.startPitchIndex
        if index >= 0 && index < pitches.count {
            return pitches[index]
        }
        return "C3"
    }
}
