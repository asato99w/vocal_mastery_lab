//
//  SubscriptionManagementView.swift
//  VocalMasteryLab
//
//  Subscription management screen for existing subscribers
//

import SwiftUI
import SubscriptionDomain

public struct SubscriptionManagementView: View {

    @StateObject private var viewModel: SubscriptionViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var showPaywall = false

    public init(viewModel: SubscriptionViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    public var body: some View {
        NavigationView {
            ZStack {
                ScrollView {
                    VStack(spacing: 24) {
                        if let status = viewModel.currentStatus {
                            // Current Status Card
                            currentStatusCard(status: status)

                            // Management Actions
                            actionsSection(for: status)
                        } else {
                            noSubscriptionView
                        }
                    }
                    .padding()
                }

                // Loading Overlay
                if viewModel.isLoading {
                    Color.black.opacity(0.3)
                        .ignoresSafeArea()
                    ProgressView()
                        .scaleEffect(1.5)
                        .tint(.white)
                }
            }
            .navigationTitle("subscription.title".localized)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("subscription.close".localized) {
                        dismiss()
                    }
                }
            }
            .alert("error".localized, isPresented: .constant(viewModel.errorMessage != nil)) {
                Button("ok".localized) {
                    viewModel.clearError()
                }
            } message: {
                if let error = viewModel.errorMessage {
                    Text(error)
                }
            }
        }
        .task {
            await viewModel.loadStatus()
        }
    }

    // MARK: - Sections

    private func currentStatusCard(status: SubscriptionStatus) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(status.tier.displayName)
                        .font(.title2)
                        .fontWeight(.bold)

                    if status.tier != .free {
                        Text(status.isActive ? "subscription.active".localized : "subscription.inactive".localized)
                            .font(.subheadline)
                            .foregroundColor(status.isActive ? .green : .secondary)
                    }
                }

                Spacer()

                Image(systemName: tierIcon(for: status.tier))
                    .font(.largeTitle)
                    .foregroundColor(tierColor(for: status.tier))
            }

            Divider()

            // Details
            VStack(alignment: .leading, spacing: 12) {
                if status.tier != .free {
                    DetailRow(
                        title: "subscription.purchase_date".localized,
                        value: status.purchaseDate.map { formatDate($0) } ?? "—"
                    )

                    DetailRow(
                        title: "subscription.expiration_date".localized,
                        value: status.expirationDate.map { formatDate($0) } ?? "—"
                    )

                    DetailRow(
                        title: "subscription.auto_renewal".localized,
                        value: status.willAutoRenew ? "ON" : "OFF"
                    )
                }

                DetailRow(
                    title: "subscription.version".localized,
                    value: "v1.0"
                )
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: Color.black.opacity(0.1), radius: 4)
        )
    }

    private var featuresSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("subscription.available_features".localized)
                .font(.headline)

            VStack(alignment: .leading, spacing: 12) {
                ForEach(Feature.allCases, id: \.self) { feature in
                    FeatureRow(
                        feature: feature,
                        hasAccess: viewModel.hasAccessTo(feature)
                    )
                }
            }
        }
    }

    private func actionsSection(for status: SubscriptionStatus) -> some View {
        VStack(spacing: 16) {
            // Upgrade Button - Show only for free users
            if status.tier == .free {
                Button {
                    showPaywall = true
                } label: {
                    Label("subscription.upgrade".localized, systemImage: "crown.fill")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.accentColor)
                        )
                }
                .disabled(viewModel.isLoading)
                .sheet(isPresented: $showPaywall) {
                    PaywallView(viewModel: DependencyContainer.shared.paywallViewModel)
                }
            }

            // Restore Button - Only show for free users (to restore previous purchases)
            if status.tier == .free {
                Button {
                    Task {
                        await viewModel.restorePurchases()
                    }
                } label: {
                    Label("subscription.restore".localized, systemImage: "arrow.clockwise")
                        .font(.headline)
                        .foregroundColor(.accentColor)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.accentColor, lineWidth: 1)
                        )
                }
                .disabled(viewModel.isLoading)
            }

            // Subscription Management Link - Always show for App Store subscription management
            Button {
                openAppStoreManagement()
            } label: {
                Text("subscription.cancel".localized)
                    .foregroundColor(.blue)
            }
            .padding(.top, 8)
        }
    }

    private var noSubscriptionView: some View {
        VStack(spacing: 24) {
            Image(systemName: "exclamationmark.circle")
                .font(.system(size: 60))
                .foregroundColor(.secondary)

            Text("subscription.not_found".localized)
                .font(.headline)

            Button {
                Task {
                    await viewModel.loadStatus()
                }
            } label: {
                Text("subscription.reload".localized)
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.accentColor)
                    .cornerRadius(12)
            }
        }
        .padding()
    }

    // MARK: - Helper Methods

    private func tierIcon(for tier: SubscriptionTier) -> String {
        switch tier {
        case .free:
            return "person.circle"
        case .premium:
            return "star.circle.fill"
        case .premiumPlus:
            return "crown.fill"
        }
    }

    private func tierColor(for tier: SubscriptionTier) -> Color {
        switch tier {
        case .free:
            return .secondary
        case .premium:
            return .blue
        case .premiumPlus:
            return .yellow
        }
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .none
        formatter.locale = Locale(identifier: "ja_JP")
        return formatter.string(from: date)
    }

    private func openAppStoreManagement() {
        // iOS設定アプリのサブスクリプション管理画面に遷移
        // Note: シミュレータでは動作しないが、実機では正しく動作する
        if let url = URL(string: "itms-apps://apps.apple.com/account/subscriptions") {
            UIApplication.shared.open(url)
        }
    }
}

// MARK: - Detail Row

private struct DetailRow: View {
    let title: String
    let value: String

    var body: some View {
        HStack {
            Text(title)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Spacer()

            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
        }
    }
}

// MARK: - Feature Row

private struct FeatureRow: View {
    let feature: Feature
    let hasAccess: Bool

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: hasAccess ? "checkmark.circle.fill" : "lock.circle.fill")
                .foregroundColor(hasAccess ? .green : .secondary)

            VStack(alignment: .leading, spacing: 2) {
                Text(feature.displayName)
                    .font(.subheadline)

                Text(feature.minimumTier.displayName + " " + "subscription.minimum_tier".localized)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .opacity(hasAccess ? 1.0 : 0.6)
    }
}

// MARK: - Preview

#Preview {
    SubscriptionManagementView(
        viewModel: SubscriptionViewModel(
            getStatusUseCase: PreviewGetStatusUseCase(),
            purchaseUseCase: PreviewPurchaseUseCase(),
            restoreUseCase: PreviewRestoreUseCase()
        )
    )
}

// MARK: - Preview Use Cases

private final class PreviewGetStatusUseCase: GetSubscriptionStatusUseCaseProtocol {
    func execute() async throws -> SubscriptionStatus {
        return SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true,
            expirationDate: Date().addingTimeInterval(30 * 24 * 60 * 60),
            purchaseDate: Date(),
            willAutoRenew: true
        )
    }
}

private final class PreviewPurchaseUseCase: PurchaseSubscriptionUseCaseProtocol {
    func execute(tier: SubscriptionTier) async throws {
        // Preview mock
    }
}

private final class PreviewRestoreUseCase: RestorePurchasesUseCaseProtocol {
    func execute() async throws {
        // Preview mock
    }
}
