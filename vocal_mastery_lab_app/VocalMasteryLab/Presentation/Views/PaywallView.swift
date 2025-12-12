//
//  PaywallView.swift
//  VocalMasteryLab
//
//  Paywall screen for subscription tiers
//

import SwiftUI
import SubscriptionDomain

public struct PaywallView: View {

    @StateObject private var viewModel: PaywallViewModel
    @EnvironmentObject private var subscriptionViewModel: SubscriptionViewModel
    @Environment(\.dismiss) private var dismiss

    public init(viewModel: PaywallViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    public var body: some View {
        NavigationView {
            ZStack {
                ScrollView {
                    VStack(spacing: 32) {
                        // Header
                        headerSection

                        // Simple comparison: Free → Premium
                        comparisonSection

                        // Purchase Button
                        purchaseButton

                        // Restore Button
                        restoreButton

                        // Terms and Privacy
                        termsSection
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
            .navigationTitle("paywall.title".localized)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if viewModel.hasActiveSubscription || viewModel.isPurchaseSuccessful {
                        Button("paywall.close".localized) {
                            dismiss()
                        }
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
        .onChange(of: viewModel.isPurchaseSuccessful) { newValue in
            if newValue {
                // Purchase successful - update subscription status before dismissing
                // This ensures HomeView's banner updates immediately
                Task {
                    await subscriptionViewModel.loadStatus()
                }
                viewModel.resetPurchaseSuccess()
                dismiss()
            }
        }
        .task {
            await viewModel.loadStatus()
        }
    }

    // MARK: - Sections

    private var headerSection: some View {
        VStack(spacing: 16) {
            Image(systemName: "crown.fill")
                .font(.system(size: 60))
                .foregroundColor(.yellow)

            Text("paywall.unlock_unlimited".localized)
                .font(Typography.headingLarge)
                .fontWeight(.bold)

            Text("paywall.unlimited_description".localized)
                .font(Typography.body)
                .foregroundColor(ColorPalette.text.opacity(0.7))
                .multilineTextAlignment(.center)
        }
        .padding(.vertical)
    }

    private var comparisonSection: some View {
        VStack(spacing: 24) {
            // Current limitation (Free)
            VStack(spacing: 12) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 40))
                    .foregroundColor(.orange)

                Text("paywall.current".localized)
                    .font(Typography.caption)
                    .foregroundColor(ColorPalette.text.opacity(0.6))

                Text("paywall.free_limit".localized)
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(ColorPalette.text)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 24)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(ColorPalette.text.opacity(0.05))
            )

            // Arrow down
            Image(systemName: "arrow.down")
                .font(.system(size: 32, weight: .bold))
                .foregroundColor(ColorPalette.primary)

            // Premium unlimited
            VStack(spacing: 12) {
                Image(systemName: "infinity")
                    .font(.system(size: 40))
                    .foregroundColor(.green)

                if let product = viewModel.product(for: .premium) {
                    // Product name from StoreKit
                    Text(product.displayName)
                        .font(Typography.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.6))

                    Text("paywall.premium_limit".localized)
                        .font(.system(size: 20, weight: .bold))
                        .foregroundColor(ColorPalette.text)

                    // Price with period from StoreKit
                    Text(priceWithPeriod(for: product))
                        .font(Typography.body)
                        .foregroundColor(ColorPalette.primary)

                    // Subscription period disclosure (Apple requirement)
                    if let period = product.subscriptionPeriod {
                        Text(period.autoRenewalDescriptionKey.localized)
                            .font(.caption2)
                            .foregroundColor(ColorPalette.text.opacity(0.5))
                    }
                } else {
                    Text("paywall.loading".localized)
                        .font(Typography.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.6))

                    Text("paywall.premium_limit".localized)
                        .font(.system(size: 20, weight: .bold))
                        .foregroundColor(ColorPalette.text)

                    Text("paywall.loading".localized)
                        .font(Typography.body)
                        .foregroundColor(ColorPalette.primary)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 24)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(ColorPalette.primary.opacity(0.1))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(ColorPalette.primary, lineWidth: 2)
                    )
            )
        }
    }

    private var purchaseButton: some View {
        Button {
            Task {
                await viewModel.purchaseSelectedTier()
            }
        } label: {
            Text("paywall.purchase".localized)
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.accentColor)
                .cornerRadius(12)
        }
        .disabled(viewModel.isLoading)
    }

    private var restoreButton: some View {
        Button {
            Task {
                await viewModel.restorePurchases()
            }
        } label: {
            Text("paywall.restore".localized)
                .font(.subheadline)
                .foregroundColor(.accentColor)
        }
        .disabled(viewModel.isLoading)
    }

    private var termsSection: some View {
        VStack(spacing: 8) {
            Text("paywall.terms_agreement".localized)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            HStack(spacing: 16) {
                Link("paywall.terms".localized, destination: URL(string: "paywall.terms_url".localized)!)
                    .font(.caption)

                Link("paywall.privacy".localized, destination: URL(string: "paywall.privacy_url".localized)!)
                    .font(.caption)
            }
        }
        .padding(.top)
    }

    // MARK: - Helper Methods

    /// Format price with subscription period (e.g., "¥500/月")
    private func priceWithPeriod(for product: SubscriptionProduct) -> String {
        guard let period = product.subscriptionPeriod else {
            return product.displayPrice
        }

        let periodKey: String
        switch period.unit {
        case .day:
            periodKey = period.value == 1 ? "subscription.period.per_day" : "subscription.period.per_days"
        case .week:
            periodKey = period.value == 1 ? "subscription.period.per_week" : "subscription.period.per_weeks"
        case .month:
            periodKey = period.value == 1 ? "subscription.period.per_month" : "subscription.period.per_months"
        case .year:
            periodKey = period.value == 1 ? "subscription.period.per_year" : "subscription.period.per_years"
        }

        if period.value == 1 {
            return periodKey.localized(with: product.displayPrice)
        } else {
            return periodKey.localized(with: product.displayPrice, period.value)
        }
    }
}

// MARK: - Preview

#Preview {
    PaywallView(
        viewModel: PaywallViewModel(
            getStatusUseCase: PreviewGetStatusUseCase(),
            purchaseUseCase: PreviewPurchaseUseCase(),
            restoreUseCase: PreviewRestoreUseCase(),
            getProductsUseCase: PreviewGetProductsUseCase()
        )
    )
}

// MARK: - Preview Use Cases

private final class PreviewGetStatusUseCase: GetSubscriptionStatusUseCaseProtocol {
    func execute() async throws -> SubscriptionStatus {
        return SubscriptionStatus.defaultFree(cohort: .v2_0)
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

private final class PreviewGetProductsUseCase: GetAvailableProductsUseCaseProtocol {
    func execute() async throws -> [SubscriptionProduct] {
        return [
            SubscriptionProduct(
                productId: "com.vocalisstudio.premium.monthly",
                displayName: "Premium (Monthly)",
                description: "Unlimited recordings",
                displayPrice: "¥500",
                subscriptionPeriod: SubscriptionPeriod(value: 1, unit: .month),
                tier: .premium
            ),
            SubscriptionProduct(
                productId: "com.vocalisstudio.premiumplus.monthly",
                displayName: "Premium Plus (Monthly)",
                description: "Advanced features",
                displayPrice: "¥980",
                subscriptionPeriod: SubscriptionPeriod(value: 1, unit: .month),
                tier: .premiumPlus
            )
        ]
    }
}
