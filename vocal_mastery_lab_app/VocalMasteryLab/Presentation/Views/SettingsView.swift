import SwiftUI

/// Settings screen - language selection and app information
public struct SettingsView: View {
    @StateObject private var localization = LocalizationManager.shared
    @EnvironmentObject private var subscriptionViewModel: SubscriptionViewModel

    public init() {}

    public var body: some View {
        Form {
            // Subscription Section
            Section("settings.subscription_section".localized) {
                NavigationLink {
                    SubscriptionManagementView(viewModel: subscriptionViewModel)
                } label: {
                    Label("settings.manage_subscription".localized, systemImage: "gear")
                }
            }

            // Audio Settings Section
            Section("settings.audio_section".localized) {
                NavigationLink {
                    AudioInputSettingsView(
                        viewModel: DependencyContainer.shared.makeAudioInputSettingsViewModel()
                    )
                } label: {
                    Label("settings.input_settings".localized, systemImage: "mic")
                }

                NavigationLink {
                    AudioOutputSettingsView(
                        viewModel: DependencyContainer.shared.makeAudioOutputSettingsViewModel()
                    )
                } label: {
                    Label("settings.output_settings".localized, systemImage: "speaker.wave.2")
                }
                .accessibilityIdentifier("AudioOutputSettingsRow")
            }

            // Algorithm Settings Section
            Section("settings.algorithm_section".localized) {
                NavigationLink {
                    AlgorithmSettingsView(
                        viewModel: DependencyContainer.shared.makeAlgorithmSettingsViewModel()
                    )
                } label: {
                    Label("settings.algorithm_settings".localized, systemImage: "waveform.path.ecg")
                }
                .accessibilityIdentifier("AlgorithmSettingsRow")
            }

            Section("settings.language_section".localized) {
                Picker("settings.language_label".localized, selection: $localization.currentLanguage) {
                    Text("settings.language_chinese_simplified".localized).tag("zh-Hans")
                    Text("settings.language_chinese_traditional".localized).tag("zh-Hant")
                    Text("settings.language_english".localized).tag("en")
                    Text("settings.language_french".localized).tag("fr")
                    Text("settings.language_german".localized).tag("de")
                    Text("settings.language_japanese".localized).tag("ja")
                    Text("settings.language_korean".localized).tag("ko")
                    Text("settings.language_spanish".localized).tag("es")
                }
            }

            Section("settings.info_section".localized) {
                HStack {
                    Text("settings.version_label".localized)
                    Spacer()
                    Text("1.4.0")
                        .foregroundColor(ColorPalette.text.opacity(0.6))
                }
            }

            Section("settings.terms_section".localized) {
                Link(destination: URL(string: "settings.terms_url".localized)!) {
                    HStack {
                        Text("settings.terms".localized)
                            .foregroundColor(ColorPalette.text)
                        Spacer()
                        Image(systemName: "arrow.up.right.square")
                            .foregroundColor(.secondary)
                            .font(.caption)
                    }
                }

                Link(destination: URL(string: "settings.privacy_url".localized)!) {
                    HStack {
                        Text("settings.privacy".localized)
                            .foregroundColor(ColorPalette.text)
                        Spacer()
                        Image(systemName: "arrow.up.right.square")
                            .foregroundColor(.secondary)
                            .font(.caption)
                    }
                }
            }
        }
        .navigationTitle("settings.title".localized)
        .navigationBarTitleDisplayMode(.large)
        .task {
            await subscriptionViewModel.loadStatus()
        }
    }
}

// MARK: - Preview

#if DEBUG
struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationStack {
            SettingsView()
                .environmentObject(DependencyContainer.shared.subscriptionViewModel)
        }
    }
}
#endif
