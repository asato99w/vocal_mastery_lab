import SwiftUI

/// Design system color palette
/// Based on "Precision in Silence" concept defined in docs/DESIGN_SYSTEM.md
/// Supports both light and dark mode with adaptive colors
public enum ColorPalette {
    // MARK: - Core Colors

    /// Primary: ロゴのコアブルーに統一
    /// Light mode: #4A9FD8 (Logo Blue), Dark mode: #5CA9DD (Logo Highlight Blue)
    /// Usage: メインアクションボタン、選択状態の強調、重要なナビゲーション要素
    public static let primary = Color(
        light: Color(red: 0x4A / 255.0, green: 0x9F / 255.0, blue: 0xD8 / 255.0),
        dark: Color(red: 0x5C / 255.0, green: 0xA9 / 255.0, blue: 0xDD / 255.0)
    )

    /// Secondary: ブルーティントを加えた柔らかい背景色
    /// Light mode: #E8F1F8 (淡ブルーグレー), Dark mode: #1A2633 (ダークネイビーブルー)
    /// Usage: カード背景、セクション区切り、非アクティブ状態の背景
    public static let secondary = Color(
        light: Color(red: 0xE8 / 255.0, green: 0xF1 / 255.0, blue: 0xF8 / 255.0),
        dark: Color(red: 0x1A / 255.0, green: 0x26 / 255.0, blue: 0x33 / 255.0)
    )

    /// Background: 画面全体の背景色
    /// Light mode: #FFFFFF (白), Dark mode: #0A0F14 (ダークネイビーグレー)
    /// Usage: 画面の基本背景
    public static let background = Color(
        light: Color.white,
        dark: Color(red: 0x0A / 255.0, green: 0x0F / 255.0, blue: 0x14 / 255.0)
    )

    /// Text: ロゴのネイビーブルーを基調としたテキスト色
    /// Light mode: #1E3A5F (ネイビーブルー), Dark mode: #D8E6F2 (淡ブルーグレー)
    /// Usage: 本文テキスト、ラベル、説明文
    public static let text = Color(
        light: Color(red: 0x1E / 255.0, green: 0x3A / 255.0, blue: 0x5F / 255.0),
        dark: Color(red: 0xD8 / 255.0, green: 0xE6 / 255.0, blue: 0xF2 / 255.0)
    )

    /// Accent: ロゴのシャドウブルーから抽出した分析用カラー
    /// Light mode: #3A8EC5 (Logo Shadow Blue), Dark mode: #6DB8E8 (明るいブルー)
    /// Usage: ピッチグラフのライン、リアルタイム分析表示、データ可視化要素
    public static let accent = Color(
        light: Color(red: 0x3A / 255.0, green: 0x8E / 255.0, blue: 0xC5 / 255.0),
        dark: Color(red: 0x6D / 255.0, green: 0xB8 / 255.0, blue: 0xE8 / 255.0)
    )

    /// Alert/Active: アナログ計器的な警告色
    /// Light mode: #F2B705, Dark mode: #FFD60A (明度を上げて視認性向上)
    /// Usage: 録音中インジケーター、警告メッセージ、注意を引く必要がある状態表示
    public static let alertActive = Color(
        light: Color(red: 0xF2 / 255.0, green: 0xB7 / 255.0, blue: 0x05 / 255.0),
        dark: Color(red: 0xFF / 255.0, green: 0xD6 / 255.0, blue: 0x0A / 255.0)
    )
}

// MARK: - Color Extension for Adaptive Colors

extension Color {
    /// Create an adaptive color that changes based on the color scheme
    init(light: Color, dark: Color) {
        self.init(UIColor { traitCollection in
            switch traitCollection.userInterfaceStyle {
            case .dark:
                return UIColor(dark)
            default:
                return UIColor(light)
            }
        })
    }
}
