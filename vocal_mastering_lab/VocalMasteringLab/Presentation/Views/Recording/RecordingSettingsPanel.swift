import SwiftUI
import VocalisDomain

/// Full recording settings panel for landscape layout
struct RecordingSettingsPanel: View {
    @ObservedObject var viewModel: RecordingSettingsViewModel
    @ObservedObject var presetViewModel: ScalePresetViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Text("recording.settings_title".localized)
                        .font(.headline)

                    Spacer()

                    // Preset buttons
                    HStack(spacing: 8) {
                        Button {
                            presetViewModel.isShowingPresetList = true
                        } label: {
                            Image(systemName: "bookmark")
                                .font(.subheadline)
                        }
                        .accessibilityIdentifier("LoadPresetButton")

                        Button {
                            presetViewModel.isShowingSaveDialog = true
                        } label: {
                            Image(systemName: "bookmark.fill")
                                .font(.subheadline)
                        }
                        .accessibilityIdentifier("SavePresetButton")
                    }
                }
                .padding(.bottom, 4)

                // Scale selection
                VStack(alignment: .leading, spacing: 6) {
                    Text("recording.scale_label".localized)
                        .font(.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.6))

                    Picker("recording.scale_label".localized, selection: $viewModel.scaleType) {
                        Text("recording.scale_five_tone".localized).tag(ScaleType.fiveTone)
                        Text("recording.scale_five_tone_down".localized).tag(ScaleType.fiveToneDown)
                        Text("recording.scale_rossini".localized).tag(ScaleType.rossiniScale)
                        Text("recording.scale_octave_repeat".localized).tag(ScaleType.octaveRepeat)
                        Text("recording.scale_broken".localized).tag(ScaleType.brokenScale)
                        Text("recording.scale_broken_double".localized).tag(ScaleType.brokenScaleDouble)
                        Text("recording.scale_arpeggio_down_triple".localized).tag(ScaleType.arpeggioDownTriple)
                        Text("recording.scale_off".localized).tag(ScaleType.off)
                    }
                    .pickerStyle(.menu)
                    .accessibilityIdentifier("ScaleTypePicker")
                }

                // Start pitch
                VStack(alignment: .leading, spacing: 6) {
                    Text("recording.start_pitch_label".localized)
                        .font(.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.6))

                    Picker("recording.start_pitch_label".localized, selection: $viewModel.startPitchIndex) {
                        ForEach(0..<viewModel.availablePitches.count, id: \.self) { index in
                            Text(viewModel.availablePitches[index]).tag(index)
                        }
                    }
                    .pickerStyle(.menu)
                    .disabled(!viewModel.isSettingsEnabled)
                    .accessibilityIdentifier("StartPitchPicker")
                }

                // Tempo
                VStack(alignment: .leading, spacing: 6) {
                    Text("recording.tempo_label".localized)
                        .font(.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.6))

                    HStack(spacing: 8) {
                        Text("\(viewModel.tempo)")
                            .font(.callout)
                            .monospacedDigit()
                            .frame(width: 50, alignment: .leading)

                        Slider(value: Binding(
                            get: { Double(viewModel.tempo) },
                            set: { viewModel.tempo = Int($0) }
                        ), in: 30...240, step: 1)

                        Text("recording.tempo_unit".localized)
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                    }
                    .disabled(!viewModel.isSettingsEnabled)
                }

                // Key progression pattern
                VStack(alignment: .leading, spacing: 6) {
                    Text("recording.pattern_label".localized)
                        .font(.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.6))

                    Picker("recording.pattern_label".localized, selection: $viewModel.keyProgressionPattern) {
                        Text("recording.pattern_ascending_only".localized).tag(KeyProgressionPattern.ascendingOnly)
                        Text("recording.pattern_descending_only".localized).tag(KeyProgressionPattern.descendingOnly)
                        Text("recording.pattern_ascending_then_descending".localized).tag(KeyProgressionPattern.ascendingThenDescending)
                        Text("recording.pattern_descending_then_ascending".localized).tag(KeyProgressionPattern.descendingThenAscending)
                    }
                    .pickerStyle(.menu)
                    .disabled(!viewModel.isSettingsEnabled)
                    .accessibilityIdentifier("KeyProgressionPatternPicker")
                }

                // Ascending key settings (count and interval grouped together)
                if viewModel.showsAscendingKeyCount {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("recording.key_ascending_count".localized)
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.6))

                        VStack(spacing: 8) {
                            // Count row
                            HStack(spacing: 8) {
                                Text("recording.count_label".localized)
                                    .font(.callout)

                                Picker("recording.key_ascending_count".localized, selection: $viewModel.ascendingKeyCount) {
                                    ForEach(1...12, id: \.self) { count in
                                        Text("\(count) " + "recording.key_count_unit".localized).tag(count)
                                    }
                                }
                                .pickerStyle(.menu)
                                .disabled(!viewModel.isSettingsEnabled)
                                .accessibilityIdentifier("AscendingKeyCountPicker")

                                Spacer()
                            }

                            // Interval row
                            HStack(spacing: 8) {
                                Text("recording.interval_label".localized)
                                    .font(.callout)

                                Picker("recording.ascending_interval".localized, selection: $viewModel.ascendingKeyStepInterval) {
                                    Text("recording.interval_semitone".localized).tag(1)
                                    Text("recording.interval_whole_tone".localized).tag(2)
                                    Text("recording.interval_minor_third".localized).tag(3)
                                    Text("recording.interval_major_third".localized).tag(4)
                                }
                                .pickerStyle(.menu)
                                .disabled(!viewModel.isSettingsEnabled)
                                .accessibilityIdentifier("AscendingKeyStepIntervalPicker")

                                Spacer()
                            }
                        }
                        .padding(8)
                        .background(ColorPalette.background.opacity(0.5))
                        .cornerRadius(8)
                    }
                }

                // Descending key settings (count and interval grouped together)
                if viewModel.showsDescendingKeyCount {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("recording.key_descending_count".localized)
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.6))

                        VStack(spacing: 8) {
                            // Count row
                            HStack(spacing: 8) {
                                Text("recording.count_label".localized)
                                    .font(.callout)

                                Picker("recording.key_descending_count".localized, selection: $viewModel.descendingKeyCount) {
                                    ForEach(1...12, id: \.self) { count in
                                        Text("\(count) " + "recording.key_count_unit".localized).tag(count)
                                    }
                                }
                                .pickerStyle(.menu)
                                .disabled(!viewModel.isSettingsEnabled)
                                .accessibilityIdentifier("DescendingKeyCountPicker")

                                Spacer()
                            }

                            // Interval row
                            HStack(spacing: 8) {
                                Text("recording.interval_label".localized)
                                    .font(.callout)

                                Picker("recording.descending_interval".localized, selection: $viewModel.descendingKeyStepInterval) {
                                    Text("recording.interval_semitone".localized).tag(1)
                                    Text("recording.interval_whole_tone".localized).tag(2)
                                    Text("recording.interval_minor_third".localized).tag(3)
                                    Text("recording.interval_major_third".localized).tag(4)
                                }
                                .pickerStyle(.menu)
                                .disabled(!viewModel.isSettingsEnabled)
                                .accessibilityIdentifier("DescendingKeyStepIntervalPicker")

                                Spacer()
                            }
                        }
                        .padding(8)
                        .background(ColorPalette.background.opacity(0.5))
                        .cornerRadius(8)
                    }
                }

                // MIDI Range Warning
                if let warning = viewModel.midiRangeWarning {
                    MIDIRangeWarningView(message: warning)
                }
            }
            .padding(12)
        }
        .background(ColorPalette.secondary)
        .sheet(isPresented: $presetViewModel.isShowingPresetList) {
            PresetListView(presetViewModel: presetViewModel, settingsViewModel: viewModel)
        }
        .sheet(isPresented: $presetViewModel.isShowingSaveDialog) {
            SavePresetDialog(presetViewModel: presetViewModel, settingsViewModel: viewModel)
        }
    }
}

/// Compact recording settings panel for portrait layout
struct RecordingSettingsCompact: View {
    @ObservedObject var viewModel: RecordingSettingsViewModel
    @ObservedObject var presetViewModel: ScalePresetViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("recording.settings_title".localized)
                    .font(.headline)

                Spacer()

                // Preset buttons
                HStack(spacing: 8) {
                    Button {
                        presetViewModel.isShowingPresetList = true
                    } label: {
                        Image(systemName: "bookmark")
                            .font(.subheadline)
                    }

                    Button {
                        presetViewModel.isShowingSaveDialog = true
                    } label: {
                        Image(systemName: "bookmark.fill")
                            .font(.subheadline)
                    }
                }
            }

            HStack {
                Text("recording.scale_label".localized + ":")
                Picker("", selection: $viewModel.scaleType) {
                    Text("recording.scale_five_tone".localized).tag(ScaleType.fiveTone)
                    Text("recording.scale_five_tone_down".localized).tag(ScaleType.fiveToneDown)
                    Text("recording.scale_rossini".localized).tag(ScaleType.rossiniScale)
                    Text("recording.scale_octave_repeat".localized).tag(ScaleType.octaveRepeat)
                    Text("recording.scale_broken".localized).tag(ScaleType.brokenScale)
                    Text("recording.scale_broken_double".localized).tag(ScaleType.brokenScaleDouble)
                    Text("recording.scale_arpeggio_down_triple".localized).tag(ScaleType.arpeggioDownTriple)
                    Text("recording.scale_off".localized).tag(ScaleType.off)
                }
                .pickerStyle(.menu)
                .accessibilityIdentifier("ScaleTypePicker")
            }

            if viewModel.isSettingsEnabled {
                HStack {
                    Text("recording.start_pitch_label".localized + ":")
                    Picker("", selection: $viewModel.startPitchIndex) {
                        ForEach(0..<viewModel.availablePitches.count, id: \.self) { index in
                            Text(viewModel.availablePitches[index])
                                .tag(index)
                                .accessibilityIdentifier("Pitch_\(viewModel.availablePitches[index])")
                        }
                    }
                    .pickerStyle(.menu)
                    .accessibilityIdentifier("StartPitchPicker")
                }

                VStack(alignment: .leading) {
                    Text("recording.tempo_label".localized + ": \(viewModel.tempo) " + "recording.tempo_unit".localized)
                    Slider(value: Binding(
                        get: { Double(viewModel.tempo) },
                        set: { viewModel.tempo = Int($0) }
                    ), in: 30...240, step: 1)
                }

                HStack {
                    Text("recording.pattern_label".localized + ":")
                    Picker("", selection: $viewModel.keyProgressionPattern) {
                        Text("recording.pattern_ascending_only".localized).tag(KeyProgressionPattern.ascendingOnly)
                        Text("recording.pattern_descending_only".localized).tag(KeyProgressionPattern.descendingOnly)
                        Text("recording.pattern_ascending_then_descending".localized).tag(KeyProgressionPattern.ascendingThenDescending)
                        Text("recording.pattern_descending_then_ascending".localized).tag(KeyProgressionPattern.descendingThenAscending)
                    }
                    .pickerStyle(.menu)
                }

                if viewModel.showsAscendingKeyCount {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("recording.key_ascending_count".localized)
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                        VStack(spacing: 6) {
                            HStack {
                                Text("recording.count_label".localized)
                                Picker("", selection: $viewModel.ascendingKeyCount) {
                                    ForEach(1...12, id: \.self) { count in
                                        Text("\(count) " + "recording.key_count_unit".localized).tag(count)
                                    }
                                }
                                .pickerStyle(.menu)
                                Spacer()
                            }
                            HStack {
                                Text("recording.interval_label".localized)
                                Picker("", selection: $viewModel.ascendingKeyStepInterval) {
                                    Text("recording.interval_semitone".localized).tag(1)
                                    Text("recording.interval_whole_tone".localized).tag(2)
                                    Text("recording.interval_minor_third".localized).tag(3)
                                    Text("recording.interval_major_third".localized).tag(4)
                                }
                                .pickerStyle(.menu)
                                Spacer()
                            }
                        }
                        .padding(6)
                        .background(ColorPalette.background.opacity(0.5))
                        .cornerRadius(6)
                    }
                }

                if viewModel.showsDescendingKeyCount {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("recording.key_descending_count".localized)
                            .font(.caption)
                            .foregroundColor(ColorPalette.text.opacity(0.6))
                        VStack(spacing: 6) {
                            HStack {
                                Text("recording.count_label".localized)
                                Picker("", selection: $viewModel.descendingKeyCount) {
                                    ForEach(1...12, id: \.self) { count in
                                        Text("\(count) " + "recording.key_count_unit".localized).tag(count)
                                    }
                                }
                                .pickerStyle(.menu)
                                Spacer()
                            }
                            HStack {
                                Text("recording.interval_label".localized)
                                Picker("", selection: $viewModel.descendingKeyStepInterval) {
                                    Text("recording.interval_semitone".localized).tag(1)
                                    Text("recording.interval_whole_tone".localized).tag(2)
                                    Text("recording.interval_minor_third".localized).tag(3)
                                    Text("recording.interval_major_third".localized).tag(4)
                                }
                                .pickerStyle(.menu)
                                Spacer()
                            }
                        }
                        .padding(6)
                        .background(ColorPalette.background.opacity(0.5))
                        .cornerRadius(6)
                    }
                }

                // MIDI Range Warning
                if let warning = viewModel.midiRangeWarning {
                    MIDIRangeWarningView(message: warning)
                }
            }
        }
        .padding()
        .background(ColorPalette.secondary)
        .cornerRadius(12)
        .sheet(isPresented: $presetViewModel.isShowingPresetList) {
            PresetListView(presetViewModel: presetViewModel, settingsViewModel: viewModel)
        }
        .sheet(isPresented: $presetViewModel.isShowingSaveDialog) {
            SavePresetDialog(presetViewModel: presetViewModel, settingsViewModel: viewModel)
        }
    }
}
