import SwiftUI
import VocalisDomain
import AVFoundation

/// Home screen - main entry point with navigation to all features
/// Design: "Precision in Silence" - calm, professional, studio-like atmosphere
public struct HomeView: View {
    @StateObject private var localization = LocalizationManager.shared
    @EnvironmentObject private var subscriptionViewModel: SubscriptionViewModel

    public init() {}

    public var body: some View {
        NavigationStack {
            ZStack {
                // Background: Adaptive background color for light/dark mode
                ColorPalette.background
                    .ignoresSafeArea()

                VStack(spacing: 32) {
                    Spacer()

                    // App Logo and Title
                    VStack(spacing: 10) {
                        Image("LogoImage")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 160, height: 160)
                            .shadow(color: Color.black.opacity(0.1), radius: 10, x: 0, y: 5)

                        // 1行 + "Lab"にロゴ波線と同じグラデーション
                        HStack(spacing: 0) {
                            Text("Vocal Mastery ")
                                .font(.system(size: 24, weight: .regular, design: .rounded))
                                .foregroundColor(ColorPalette.text)
                            Text("Lab")
                                .font(.system(size: 24, weight: .semibold, design: .rounded))
                                .foregroundStyle(
                                    LinearGradient(
                                        colors: [
                                            Color(red: 0.68, green: 0.72, blue: 0.42), // 黄緑/オリーブ
                                            Color(red: 0.90, green: 0.62, blue: 0.35)  // オレンジ
                                        ],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                        }
                    }
                    .padding(.bottom, 24)

                    // Menu Buttons
                    VStack(spacing: 20) {
                        NavigationLink(destination: RecordingView(
                            viewModel: DependencyContainer.shared.recordingViewModel,
                            vocalExtractor: DependencyContainer.shared.vocalExtractor,
                            extractedAudioRepository: DependencyContainer.shared.extractedAudioRepository,
                            audioPlayer: DependencyContainer.shared.audioPlayer
                        )) {
                            MenuButton(title: "home.record_button".localized, icon: "mic.fill")
                        }
                        .accessibilityIdentifier("HomeRecordButton")

                        NavigationLink(destination: RecordingListView(
                            viewModel: RecordingListViewModel(
                                recordingRepository: DependencyContainer.shared.recordingRepository,
                                extractedAudioRepository: DependencyContainer.shared.extractedAudioRepository,
                                audioPlayer: DependencyContainer.shared.audioPlayer
                            ),
                            audioPlayer: DependencyContainer.shared.audioPlayer,
                            analyzeRecordingUseCase: DependencyContainer.shared.analyzeRecordingUseCase,
                            extractedAudioRepository: DependencyContainer.shared.extractedAudioRepository,
                            vocalExtractor: DependencyContainer.shared.vocalExtractor
                        )) {
                            MenuButton(title: "home.list_button".localized, icon: "list.bullet")
                        }
                        .accessibilityIdentifier("HomeListButton")

                        NavigationLink(destination: SettingsView()) {
                            MenuButton(title: "home.settings_button".localized, icon: "gearshape")
                        }
                        .accessibilityIdentifier("HomeSettingsButton")
                    }
                    .padding(.horizontal, 40)

                    Spacer()

                    // Note: Upgrade banner hidden while all features are free
                    // Preserved for future paid plan restoration

                    // Debug button (only in debug builds) - unobtrusive
                    #if DEBUG
                    NavigationLink(destination: DebugMenuView()) {
                        HStack(spacing: 4) {
                            Image(systemName: "ant.fill")
                                .font(.system(size: 10))
                            Text("Debug")
                                .font(.system(size: 10))
                        }
                        .foregroundColor(ColorPalette.text.opacity(0.3))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(ColorPalette.text.opacity(0.05))
                        .cornerRadius(6)
                    }
                    .padding(.bottom, 8)
                    #endif
                }
            }
            .navigationBarHidden(true)
            .task {
                await subscriptionViewModel.loadStatus()
                await requestMicrophonePermissionIfNeeded()
            }
        }
    }

    // MARK: - Microphone Permission

    /// Request microphone permission on first launch
    /// This allows users to grant permission upfront rather than during recording
    private func requestMicrophonePermissionIfNeeded() async {
        if #available(iOS 17.0, *) {
            let currentStatus = AVAudioApplication.shared.recordPermission
            if currentStatus == .undetermined {
                _ = await AVAudioApplication.requestRecordPermission()
            }
        } else {
            let audioSession = AVAudioSession.sharedInstance()
            let currentStatus = audioSession.recordPermission
            if currentStatus == .undetermined {
                await withCheckedContinuation { continuation in
                    audioSession.requestRecordPermission { _ in
                        continuation.resume()
                    }
                }
            }
        }
    }
}

/// Upgrade banner component for free users
private struct UpgradeBanner: View {
    @EnvironmentObject private var subscriptionViewModel: SubscriptionViewModel
    @State private var showPaywall = false

    var body: some View {
        Button(action: {
            showPaywall = true
        }) {
            HStack(spacing: 12) {
                Image(systemName: "crown.fill")
                    .font(.system(size: 24))
                    .foregroundColor(.yellow)

                VStack(alignment: .leading, spacing: 4) {
                    Text("home.unlock_unlimited".localized)
                        .font(Typography.body)
                        .fontWeight(.semibold)
                    Text("home.premium_no_limit".localized)
                        .font(.system(size: 12))
                        .foregroundColor(ColorPalette.text.opacity(0.7))
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .foregroundColor(ColorPalette.text.opacity(0.5))
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(ColorPalette.alertActive.opacity(0.1))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(ColorPalette.alertActive.opacity(0.3), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(.plain)
        .sheet(isPresented: $showPaywall) {
            PaywallView(viewModel: DependencyContainer.shared.paywallViewModel)
        }
    }
}

/// Custom menu button component
/// Design: Primary color with subtle shadow for professional look
struct MenuButton: View {
    let title: String
    let icon: String

    var body: some View {
        HStack {
            Image(systemName: icon)
                .font(Typography.bodyLarge)
            Text(title)
                .font(Typography.bodyLarge)
        }
        .foregroundColor(.white)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 20)
        .background(ColorPalette.primary)  // Use design system primary color
        .cornerRadius(10)
        .shadow(color: Color.black.opacity(0.1), radius: 4, x: 0, y: 2)
    }
}

// MARK: - Preview

#if DEBUG
struct HomeView_Previews: PreviewProvider {
    static var previews: some View {
        HomeView()
            .environmentObject(DependencyContainer.shared.subscriptionViewModel)
    }
}
#endif
