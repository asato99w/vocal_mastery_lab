import SwiftUI

/// Design system button styles
/// Provides consistent button appearance following "Precision in Silence" design principles

/// Primary action button style (e.g., Start Recording, Play)
struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(Typography.bodyLarge)
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(ColorPalette.primary)
            .cornerRadius(10)
            .shadow(color: Color.black.opacity(0.1), radius: 4, x: 0, y: 2)
            .opacity(configuration.isPressed ? 0.8 : 1.0)
    }
}

/// Secondary action button style (e.g., Cancel, Settings)
struct SecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(Typography.body)
            .foregroundColor(ColorPalette.text)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(ColorPalette.secondary)
            .cornerRadius(8)
            .opacity(configuration.isPressed ? 0.7 : 1.0)
    }
}

/// Alert/Warning action button style (e.g., Record button in active state)
struct AlertButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(Typography.bodyLarge)
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(ColorPalette.alertActive)
            .cornerRadius(10)
            .shadow(color: Color.black.opacity(0.1), radius: 4, x: 0, y: 2)
            .opacity(configuration.isPressed ? 0.8 : 1.0)
    }
}

/// Compact button style for toolbar/navigation items
struct CompactButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(Typography.caption)
            .foregroundColor(ColorPalette.primary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(ColorPalette.secondary)
            .cornerRadius(8)
            .opacity(configuration.isPressed ? 0.7 : 1.0)
    }
}
