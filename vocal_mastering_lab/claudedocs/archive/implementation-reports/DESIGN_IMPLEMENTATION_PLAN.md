# デザインシステム実装プラン

## 概要

`VocalisStudio/docs/DESIGN_SYSTEM.md` で定義されたデザインシステム「**静かな精密 (Precision in Silence)**」に基づき、既存UIを段階的に改善する実装プランです。

作成日: 2025-11-04

---

## 🎯 全体目標

デザインシステムのコンセプトに沿ったUI/UX実現:
- **無駄を削ぎ落とす**: 音声波形やデータが主役、UIは透明感と秩序を重視
- **計測器＋スタジオの中間デザイン**: 分析ツールの精密さとスタジオの落ち着き
- **安心して声を預けられる感覚**: 信頼感、明確なフィードバック、予測可能な操作感

---

## 📋 実装フェーズ

### Phase 0: HomeView リデザイン (最優先) ✅

**目的**: 最も目立つエントランス画面を「静かな精密」に再設計

#### 現在の問題点

| 要素 | 現状 | 問題 |
|------|------|------|
| 背景 | 紫→青のビビッドグラデーション | 装飾的すぎる、「静的で落ち着いた」に反する |
| ロゴ | `music.mic` (システムアイコン) | 独自アプリアイコンを活用していない |
| テキスト色 | すべて白色 | カラーパレット未使用 |
| ボタン | 半透明白色オーバーレイ | 「計測器＋スタジオ」の印象がない |

#### 実装内容

**0.1 カラーパレット定義ファイル作成**

- ファイル: `VocalisStudio/VocalisStudio/Presentation/Theme/ColorPalette.swift`
- 内容:
```swift
import SwiftUI

/// Design system color palette
/// Based on "Precision in Silence" concept
enum ColorPalette {
    // Primary: 精度・信頼を象徴する淡いブルー
    static let primary = Color(red: 0x3A/255, green: 0x6E/255, blue: 0xA5/255)

    // Secondary: 柔らかく主張しない淡グレー
    static let secondary = Color(red: 0xD8/255, green: 0xE1/255, blue: 0xE8/255)

    // Text: 読みやすく温度感を抑えた深灰
    static let text = Color(red: 0x1E/255, green: 0x1E/255, blue: 0x1E/255)

    // Accent: ピッチラインなど分析用のサインブルー
    static let accent = Color(red: 0x00/255, green: 0xA6/255, blue: 0xD6/255)

    // Alert/Active: アナログ計器的な警告色
    static let alertActive = Color(red: 0xF2/255, green: 0xB7/255, blue: 0x05/255)
}
```

**0.2 HomeView背景の変更**

Before:
```swift
LinearGradient(
    colors: [Color(red: 0.42, green: 0.36, blue: 0.90),
             Color(red: 0.58, green: 0.29, blue: 0.76)],
    startPoint: .topLeading,
    endPoint: .bottomTrailing
)
```

After:
```swift
ColorPalette.secondary  // 淡グレー背景
```

**0.3 アプリロゴの活用**

Before:
```swift
Image(systemName: "music.mic")
    .font(.system(size: 80))
    .foregroundColor(.white)
```

After:
```swift
Image("AppIcon")
    .resizable()
    .aspectRatio(contentMode: .fit)
    .frame(width: 120, height: 120)
    .cornerRadius(24)
    .shadow(color: Color.black.opacity(0.1), radius: 8, x: 0, y: 4)
```

**0.4 タイトルの色・フォント調整**

Before:
```swift
Text("app_name".localized)
    .font(.system(size: 36, weight: .bold))
    .foregroundColor(.white)
```

After:
```swift
Text("app_name".localized)
    .font(.system(size: 28, weight: .semibold))
    .foregroundColor(ColorPalette.text)
```

**0.5 MenuButtonの再設計**

Before:
```swift
struct MenuButton: View {
    var body: some View {
        HStack { ... }
            .foregroundColor(.white)
            .background(Color.white.opacity(0.2))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.white.opacity(0.3), lineWidth: 1)
            )
    }
}
```

After:
```swift
struct MenuButton: View {
    var body: some View {
        HStack { ... }
            .foregroundColor(.white)
            .background(ColorPalette.primary)  // #3A6EA5
            .cornerRadius(10)
            .shadow(color: Color.black.opacity(0.1), radius: 4, x: 0, y: 2)
    }
}
```

#### 成果物

- [x] `Presentation/Theme/ColorPalette.swift` 作成
- [x] `HomeView.swift` 背景・ロゴ・タイトル・ボタン変更
- [x] ビルド確認
- [x] テスト実行 (既存テスト影響なし確認)
- [x] コミット

---

### Phase 1: RecordingView カラー適用

**目的**: 使用頻度が高い録音画面にカラーパレット適用

#### 実装内容

**1.1 RecordingControlsボタン色変更**

ファイル: `VocalisStudio/VocalisStudio/Presentation/Views/Recording/RecordingControls.swift`

変更箇所:
- 録音開始ボタン: `.red` → `ColorPalette.alertActive` (#F2B705)
- 停止ボタン: `.gray` → `ColorPalette.secondary` + `ColorPalette.text`
- 再生ボタン: `.blue` → `ColorPalette.primary` (#3A6EA5)

**1.2 RecordingView細部調整**

ファイル: `VocalisStudio/VocalisStudio/Presentation/Views/Recording/RecordingView.swift`

変更箇所:
- 設定トグルボタン背景: `Color(.systemGray6)` → `ColorPalette.secondary`
- リンク色: `.blue` → `ColorPalette.primary`

#### 成果物

- [ ] `RecordingControls.swift` カラー適用
- [ ] `RecordingView.swift` カラー適用
- [ ] ビルド&テスト確認
- [ ] コミット

---

### Phase 2: タイポグラフィ調整

**目的**: デザインシステムのタイポグラフィルール適用

#### 実装内容

**2.1 フォントスタイル定義**

ファイル: `VocalisStudio/VocalisStudio/Presentation/Theme/Typography.swift`

内容:
```swift
import SwiftUI

/// Design system typography
enum Typography {
    // Heading: 見出し用
    static let heading = Font.system(size: 20, weight: .bold)
    static let headingLarge = Font.system(size: 24, weight: .bold)

    // Body: 本文用
    static let body = Font.system(size: 14, weight: .regular)
    static let bodyLarge = Font.system(size: 16, weight: .regular)

    // Caption: キャプション用
    static let caption = Font.system(size: 12, weight: .regular)

    // Data display: 数値・タイムコード用 (monospace)
    static let data = Font.system(size: 14, weight: .regular, design: .monospaced)
}
```

**2.2 既存View適用**

- `HomeView.swift`: タイトル、ボタンテキスト
- `RecordingView.swift`: ナビゲーションタイトル、ボタンテキスト
- `RecordingControls.swift`: ボタン内テキスト

#### 成果物

- [ ] `Presentation/Theme/Typography.swift` 作成
- [ ] 各Viewにフォント適用
- [ ] レイアウトテスト (横向き・縦向き)
- [ ] テスト実行
- [ ] コミット

---

### Phase 3: ボタンスタイルの統一

**目的**: 一貫したボタンデザイン実現

#### 実装内容

**3.1 ボタンスタイル定義**

ファイル: `VocalisStudio/VocalisStudio/Presentation/Theme/ButtonStyles.swift`

内容:
```swift
import SwiftUI

/// Primary action button style
struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.callout)
            .fontWeight(.semibold)
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(ColorPalette.primary)
            .cornerRadius(10)
            .opacity(configuration.isPressed ? 0.8 : 1.0)
    }
}

/// Secondary action button style
struct SecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.caption)
            .foregroundColor(ColorPalette.text)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
            .background(ColorPalette.secondary)
            .cornerRadius(8)
            .opacity(configuration.isPressed ? 0.8 : 1.0)
    }
}

/// Alert/Active button style (for recording start)
struct AlertButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.callout)
            .fontWeight(.semibold)
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(ColorPalette.alertActive)
            .cornerRadius(10)
            .opacity(configuration.isPressed ? 0.8 : 1.0)
    }
}
```

**3.2 適用**

- `HomeView.swift`: MenuButton
- `RecordingControls.swift`: 録音開始・停止・再生ボタン
- `RecordingView.swift`: 設定トグルボタン

#### 成果物

- [ ] `Presentation/Theme/ButtonStyles.swift` 作成
- [ ] 各ボタンにスタイル適用
- [ ] タップ反応確認
- [ ] テスト実行
- [ ] コミット

---

### Phase 4: アニメーション調整

**目的**: デザインシステムのアニメーション原則適用

#### 実装内容

**4.1 アニメーション定義**

ファイル: `VocalisStudio/VocalisStudio/Presentation/Theme/Animations.swift`

内容:
```swift
import SwiftUI

/// Design system animations
enum Animations {
    // Button tap: 即時フィードバック
    static let buttonTap = Animation.easeOut(duration: 0.1)

    // Screen transition: 画面切り替え
    static let screenTransition = Animation.easeInOut(duration: 0.3)

    // Panel display: オーバーレイ表示
    static let panelDisplay = Animation.easeOut(duration: 0.4)

    // Waveform update: リアルタイム更新 (spring)
    static let waveformUpdate = Animation.spring(response: 0.5, dampingFraction: 0.7)
}
```

**4.2 適用**

ファイル: `RecordingView.swift`

変更箇所:
- Settings panel表示/非表示: `withAnimation()` → `withAnimation(Animations.panelDisplay)`
- 録音開始時の自動非表示: `withAnimation()` → `withAnimation(Animations.screenTransition)`

#### 成果物

- [ ] `Presentation/Theme/Animations.swift` 作成
- [ ] `RecordingView.swift` アニメーション適用
- [ ] UI Test確認 (`uiTestAnimationsDisabled`分岐維持)
- [ ] テスト実行
- [ ] コミット

---

### Phase 5: レイアウト・間隔の調整

**目的**: 8ptグリッドシステム適用

#### 実装内容

**5.1 スペーシング定義**

ファイル: `VocalisStudio/VocalisStudio/Presentation/Theme/Layout.swift`

内容:
```swift
import SwiftUI

/// Design system layout spacing (8pt grid system)
enum Spacing {
    static let xs: CGFloat = 4
    static let sm: CGFloat = 8
    static let md: CGFloat = 16
    static let lg: CGFloat = 24
    static let xl: CGFloat = 32
}

/// Design system padding
enum Padding {
    static let card: CGFloat = 12
    static let section: CGFloat = 16
    static let screen: CGFloat = 20
}
```

**5.2 適用**

- `HomeView.swift`: VStack spacing: `40` → `Spacing.xl`
- `RecordingView.swift`: VStack spacing: `8`, `16` → `Spacing.sm`, `Spacing.md`
- `RecordingControls.swift`: ボタン間spacing: `10`, `8` → `Spacing.sm`

#### 成果物

- [ ] `Presentation/Theme/Layout.swift` 作成
- [ ] 各Viewにスペーシング適用
- [ ] 横向き・縦向きレイアウト確認
- [ ] テスト実行
- [ ] コミット

---

### Phase 6: テキスト表現の改善

**目的**: トーン&マナーに沿った表現へ変更

#### 実装内容

**6.1 テキスト見直し**

ファイル: `VocalisStudio/VocalisStudio/Resources/Localizable.strings` (日本語・英語)

変更例:
- ❌ 「録音します」→ ✅ 「声を記録」
- ❌ 「エラーが発生しました」→ ✅ 「うまく保存できませんでした」
- ❌ 「削除しますか?」→ ✅ 「この録音を削除してもよろしいですか?」

#### 成果物

- [ ] `Localizable.strings` 更新
- [ ] アプリ実行確認
- [ ] テスト実行
- [ ] コミット

---

## 📊 実行順序と優先度

### ✅ 推奨順序

1. **Phase 0**: HomeView リデザイン (完了) ✅
2. **Phase 1**: RecordingView カラー適用
3. **Phase 3**: ボタンスタイル統一 (Phase 1完了後)
4. **Phase 2**: タイポグラフィ調整
5. **Phase 6**: テキスト表現改善 (独立タスク)
6. **Phase 5**: レイアウト調整
7. **Phase 4**: アニメーション調整 (最後に微調整)

### ⚠️ 各Phase後の確認事項

- [ ] ビルド成功
- [ ] 既存テストすべてパス
- [ ] シミュレータでUI確認 (横向き・縦向き両方)
- [ ] コミット作成

---

## 📝 進捗トラッキング

| Phase | ステータス | 完了日 | コミットハッシュ |
|-------|-----------|--------|--------------|
| Phase 0 | 完了 ✅ | 2025-11-04 | [ハッシュ] |
| Phase 1 | 未着手 | - | - |
| Phase 2 | 未着手 | - | - |
| Phase 3 | 未着手 | - | - |
| Phase 4 | 未着手 | - | - |
| Phase 5 | 未着手 | - | - |
| Phase 6 | 未着手 | - | - |

---

## 🔗 参照ドキュメント

- **デザインシステム仕様**: `VocalisStudio/docs/DESIGN_SYSTEM.md`
- **カラーパレット定義**: Line 36-74
- **タイポグラフィ原則**: Line 77-108
- **アニメーション原則**: Line 174-200
- **トーン&マナー**: Line 203-226

---

最終更新: 2025-11-04
