# App Store リリースガイド - VocalisStudio

**作成日**: 2024-11-20
**最終更新**: 2025-11-25
**ステータス**: TestFlight完了 → App Store審査準備中

---

## 目次

1. [進捗サマリー](#進捗サマリー)
2. [プロジェクト現状](#プロジェクト現状)
3. [事前準備チェックリスト](#事前準備チェックリスト)
4. [App Store Connect設定ガイド](#app-store-connect設定ガイド)
5. [Xcode設定・アーカイブ手順](#xcode設定アーカイブ手順)
6. [審査提出チェックリスト](#審査提出チェックリスト)
7. [リリース後タスク](#リリース後タスク)

---

## 進捗サマリー

### 全体進捗: 80%

```
[████████████████████████░░░░░░] 80%
```

### フェーズ別進捗

| フェーズ | ステータス | 進捗 | 備考 |
|----------|-----------|------|------|
| 0. 事前準備 | ✅ 完了 | 100% | PrivacyInfo.xcprivacy作成済み |
| 1. Xcode設定 | ✅ 完了 | 100% | 署名・バージョン設定済み |
| 2. App Store Connect | ✅ 完了 | 100% | App作成・サブスク1プラン登録済み |
| 3. TestFlight | ✅ 完了 | 100% | 外部テスト設定完了（審査待ち） |
| 4. 審査提出準備 | 🟡 進行中 | 60% | スクリーンショット2サイズ完了 |
| 5. リリース | 🔴 未着手 | 0% | 審査承認後 |

---

## プロジェクト現状

### 確認済み設定

| 項目 | 値 | ステータス |
|------|-----|-----------|
| Bundle ID | `com.kazuasato.VocalisStudio` | ✅ |
| MARKETING_VERSION | 1.0 | ✅ |
| CURRENT_PROJECT_VERSION | 1 | ✅ |
| DEVELOPMENT_TEAM | 76359B69WG | ✅ |
| CODE_SIGN_STYLE | Automatic | ✅ |
| アプリアイコン | 1024x1024 (通常+ダーク) | ✅ |
| マイク権限説明 | 日本語+英語ローカライズ | ✅ |

### App内課金プラン（StoreKit設定済み）

| プラン名 | Product ID | 価格 |
|---------|------------|------|
| プレミアム（月額） | `com.vocalisstudio.premium.monthly` | ¥500 |
| プレミアム（年額） | `com.vocalisstudio.premium.yearly` | ¥4,800 |
| プレミアムプラス（月額） | `com.vocalisstudio.premiumplus.monthly` | ¥980 |
| プレミアムプラス（年額） | `com.vocalisstudio.premiumplus.yearly` | ¥9,800 |

### 使用SDK/ライブラリ

| ライブラリ | 用途 | プライバシー影響 |
|-----------|------|-----------------|
| AudioKit | オーディオ処理 | なし（ローカル処理のみ） |
| VocalisDomain | ドメインロジック | なし |
| SubscriptionDomain | サブスク管理 | なし |

---

## 事前準備チェックリスト

### 0. 事前準備（1回やればOK）

- [x] **Apple Developer Program登録**
  - Team ID: 76359B69WG

- [ ] **App Store Connectの組織設定**
  - [ ] ユーザー権限設定
  - [ ] 財務/税務情報の設定
  - [ ] 銀行口座情報の登録

- [x] **バンドルID作成**
  - `com.kazuasato.VocalisStudio`

- [ ] **App内課金プロダクト作成**（App Store Connect）
  - [ ] サブスクリプショングループ作成
  - [ ] 4つのプランを登録
  - [ ] 価格設定
  - [ ] ローカライズ

- [x] **プライバシーポリシーURL**
  - 日本語: `https://vocalisstudio.github.io/web/privacy.html`
  - 英語: `https://vocalisstudio.github.io/web/privacy_en.html`

- [x] **PrivacyInfo.xcprivacy作成**
  - ファイル: `VocalisStudio/PrivacyInfo.xcprivacy`
  - 申告内容:
    - オーディオデータ収集（App機能用）
    - UserDefaults API (CA92.1)
    - File Timestamp API (C617.1)

- [x] **暗号化の確認**
  - 独自暗号化なし
  - HTTPS/ATSのみ（免除対象）

---

## App Store Connect設定ガイド

### 1. 新規Appの作成

**URL**: https://appstoreconnect.apple.com

**入力情報**:

| 項目 | 値 |
|------|-----|
| プラットフォーム | iOS |
| 名前 | Vocalis Studio |
| プライマリ言語 | 日本語 |
| バンドルID | `com.kazuasato.VocalisStudio` |
| SKU | `vocalisstudio001` |
| ユーザーアクセス | フルアクセス |

**進捗**: [ ] 未完了

---

### 2. App情報の設定

#### 「App情報」タブ

| 項目 | 値 |
|------|-----|
| 名前 | Vocalis Studio |
| サブタイトル | ボーカルトレーニング |
| プライマリカテゴリ | ミュージック |
| セカンダリカテゴリ | 教育 |

#### 「価格および配信状況」

| 項目 | 値 |
|------|-----|
| 価格 | 無料 |
| App内課金 | あり |
| 配信地域 | 日本（+ 希望地域） |

**進捗**: [ ] 未完了

---

### 3. サブスクリプション設定

#### 3.1 グループ作成

- **グループ名**: VocalisStudio Premium
- **グループID**: 21527443

#### 3.2 プラン登録（4つ）

**順位設定（アップグレード/ダウングレード用）**:
1. Premium Plus Yearly（最上位）
2. Premium Plus Monthly
3. Premium Yearly
4. Premium Monthly（最下位）

##### プレミアム（月額）
| 項目 | 値 |
|------|-----|
| 参照名 | Premium Monthly |
| 製品ID | `com.vocalisstudio.premium.monthly` |
| 期間 | 1ヶ月 |
| 価格 | ¥480 |
| 表示名 | プレミアム（月額） |
| 説明 | 無制限録音・最大5分 |

##### プレミアム（年額）
| 項目 | 値 |
|------|-----|
| 参照名 | Premium Yearly |
| 製品ID | `com.vocalisstudio.premium.yearly` |
| 期間 | 1年 |
| 価格 | ¥4,800 |
| 表示名 | プレミアム（年額） |
| 説明 | 無制限録音・最大5分（年間プラン） |

##### プレミアムプラス（月額）
| 項目 | 値 |
|------|-----|
| 参照名 | Premium Plus Monthly |
| 製品ID | `com.vocalisstudio.premiumplus.monthly` |
| 期間 | 1ヶ月 |
| 価格 | ¥980 |
| 表示名 | プレミアムプラス（月額） |
| 説明 | 無制限録音・最大10分 |

##### プレミアムプラス（年額）
| 項目 | 値 |
|------|-----|
| 参照名 | Premium Plus Yearly |
| 製品ID | `com.vocalisstudio.premiumplus.yearly` |
| 期間 | 1年 |
| 価格 | ¥9,800 |
| 表示名 | プレミアムプラス（年額） |
| 説明 | 無制限録音・最大10分（年間プラン） |

**進捗**: [ ] 未完了

---

### 4. Appプライバシー設定

#### プライバシーポリシーURL
（ユーザーが用意したURLを入力）

#### データ収集申告

**収集するデータ**:

| データタイプ | オーディオデータ |
|-------------|----------------|
| 使用目的 | App機能 |
| ユーザーにリンク | いいえ |
| トラッキングに使用 | いいえ |

**SDKについて**:
- AudioKit: ローカル処理のみ、データ送信なし

**進捗**: [ ] 未完了

---

### 5. 年齢レーティング

全ての質問に「なし」と回答 → **結果: 4+（全年齢）**

| 質問 | 回答 |
|------|------|
| 暴力的なコンテンツ | なし |
| 性的/ヌードコンテンツ | なし |
| 冒涜的/下品な表現 | なし |
| 医療/治療情報 | なし |
| アルコール/タバコ/ドラッグ | なし |
| ギャンブル | なし |
| ホラー/恐怖 | なし |
| ユーザー生成コンテンツ | なし |
| 無制限のウェブアクセス | なし |

**進捗**: [ ] 未完了

---

### 6. バージョン情報（メタデータ）

#### 概要（説明文）

```
Vocalis Studioは、ボーカリストのための練習アプリです。

【主な機能】
• スケール練習：5音スケールで正確な音程を練習
• 録音機能：自分の声を録音して確認
• 再生機能：練習成果を聴き直し

【プレミアム機能】
• 無制限録音
• 最大5分/10分の長時間録音
• 月額/年額プランから選択

初心者からプロまで、効果的なボーカルトレーニングをサポートします。
```

#### キーワード

```
ボーカル,トレーニング,練習,歌,声,録音,スケール,音程,カラオケ,ボイトレ
```

**進捗**: [ ] 未完了

---

### 7. スクリーンショット

#### 必須サイズ

| デバイス | サイズ | ステータス |
|----------|--------|-----------|
| 6.7インチ（iPhone 15 Pro Max等） | 1290 x 2796 | [x] 完了（5枚） |
| 6.5インチ（iPhone 11 Pro Max等） | 1284 x 2778 | [x] 完了（5枚） |
| 5.5インチ（iPhone 8 Plus等） | 1242 x 2208 | [ ] 未作成 |

#### 撮影済みシーン（6.7/6.5インチ 各5枚）

1. [x] ホーム画面 (`01_home.png`)
2. [x] 録音中の画面 (`02_recording_screen.png`)
3. [x] 録音リスト (`03_recording_list.png`)
4. [x] 分析画面 (`04_analysis.png`)
5. [x] 設定画面 (`05_settings.png`)

**保存場所**: `assets/app-store/screenshots/`

**進捗**: [x] 6.7/6.5インチ完了、5.5インチ未作成

---

### 8. 審査メモ

```
【アプリの使い方】
1. ホーム画面で「録音開始」をタップ
2. マイク権限を許可
3. スケール音を聴きながら歌を録音
4. 録音完了後、再生ボタンで確認

【サブスクリプションについて】
- 設定画面の「プレミアムにアップグレード」からサブスク画面に遷移
- 無料体験期間はありません
- サブスク解約後も期間終了まで利用可能

【マイク使用について】
- 録音機能のためにマイクを使用
- 音声データはデバイスにのみ保存
- 外部サーバーへの送信なし
```

**進捗**: [ ] 未完了

---

## Xcode設定・アーカイブ手順

### 事前確認

- [x] Version: 1.0
- [x] Build: 1
- [x] 署名: Automatic
- [x] Team: 76359B69WG
- [x] PrivacyInfo.xcprivacy追加済み

### アーカイブ手順

1. **Xcode でプロジェクトを開く**
   ```bash
   open VocalisStudio/VocalisStudio.xcodeproj
   ```

2. **ビルドターゲット確認**
   - Scheme: VocalisStudio
   - Destination: Any iOS Device (arm64)

3. **アーカイブ作成**
   - `Product` → `Archive`
   - ビルド完了を待つ

4. **App Store Connect にアップロード**
   - Organizer が自動で開く
   - `Distribute App` をクリック
   - `App Store Connect` を選択
   - `Upload` を選択
   - オプション確認（通常はデフォルトでOK）
   - `Upload` 実行

5. **処理完了を待つ**
   - App Store Connect でビルド処理（15-30分）
   - 完了メール通知あり

**進捗**: [ ] 未完了

---

## 審査提出チェックリスト

### 必須項目

- [ ] App Store Connect で新規App作成
- [ ] App情報（名前、カテゴリ、説明）入力
- [ ] スクリーンショット全サイズアップロード
- [ ] プライバシーポリシーURL設定
- [ ] Appプライバシー申告完了
- [ ] 年齢レーティング設定
- [ ] サブスクリプション4プラン作成
- [ ] 輸出コンプライアンス回答
- [ ] 審査メモ記入
- [ ] ビルドをApp Store Connectにアップロード
- [ ] ビルドをバージョンに関連付け

### 輸出コンプライアンス

**Info.plist に追加（毎回の質問を省略）**:

```xml
<key>ITSAppUsesNonExemptEncryption</key>
<false/>
```

### 審査用連絡先

- 名前: （入力必要）
- メール: （入力必要）
- 電話番号: （入力必要）

---

## リリース後タスク

### 即時対応

- [ ] リリース通知の確認
- [ ] App Store での表示確認
- [ ] ダウンロードテスト
- [ ] サブスク購入テスト（Sandbox）

### 継続対応

- [ ] ユーザーレビュー監視
- [ ] クラッシュレポート確認
- [ ] アナリティクス設定（任意）
- [ ] ASO（App Store最適化）検討

---

## 参考リンク

- [App Store Connect](https://appstoreconnect.apple.com)
- [Apple Developer](https://developer.apple.com)
- [App Store Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)
- [Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)

---

## 更新履歴

| 日付 | 内容 |
|------|------|
| 2024-11-20 | 初版作成。プロジェクト状況確認完了。 |
| 2024-11-20 | PrivacyInfo.xcprivacy作成完了。進捗40%に更新。 |
| 2024-11-21 | TestFlight外部テスト設定完了。進捗70%に更新。App Store Connect設定・アーカイブ・アップロード完了。 |
| 2025-11-25 | スクリーンショット自動撮影機能実装。6.7/6.5インチ各5枚作成完了。プライバシーポリシーURL更新（言語別対応）。進捗80%に更新。 |

---

## 次のアクション

### TestFlight段階（現在）

1. **TestFlight外部テスト審査待ち**（1-2日）
2. **テスターからのフィードバック収集**
3. **必要に応じてビルド更新**

### App Store審査提出準備

1. **スクリーンショット作成**
   - [x] 6.7インチ: 1290 x 2796（5枚完了）
   - [x] 6.5インチ: 1284 x 2778（5枚完了）
   - [ ] 5.5インチ: 1242 x 2208（オプション）

2. **メタデータ設定**
   - App説明文作成
   - キーワード設定
   - 年齢レーティング設定
   - Appプライバシー申告

3. **追加サブスクリプションプラン**（任意）
   - Premium Yearly
   - Premium Plus Monthly/Yearly
