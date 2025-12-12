import SwiftUI

/// Design system typography
/// Provides consistent font styles across the app following "Precision in Silence" design principles
enum Typography {
    // MARK: - Heading: 見出し用

    /// Large heading for main titles (24pt, bold)
    static let headingLarge = Font.system(size: 24, weight: .bold)

    /// Standard heading for section titles (20pt, bold)
    static let heading = Font.system(size: 20, weight: .bold)

    // MARK: - Body: 本文用

    /// Large body text for important content (16pt, regular)
    static let bodyLarge = Font.system(size: 16, weight: .regular)

    /// Standard body text for general content (14pt, regular)
    static let body = Font.system(size: 14, weight: .regular)

    // MARK: - Caption: キャプション用

    /// Caption text for secondary information (12pt, regular)
    static let caption = Font.system(size: 12, weight: .regular)

    // MARK: - Data display: 数値・タイムコード用

    /// Monospaced font for numeric data and timecodes (14pt, monospaced)
    static let data = Font.system(size: 14, weight: .regular, design: .monospaced)
}
