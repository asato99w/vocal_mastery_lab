# ドキュメント整理計画

## 概要

リポジトリ内のドキュメント（68ファイル）が5箇所に分散しており、整理が必要な状態です。
このドキュメントでは現状分析と整理計画をまとめます。

**作成日**: 2025-11-20
**ステータス**: 計画段階

---

## 1. 現状分析

### 1.1 配置場所別ファイル数

| 場所 | ファイル数 | 内容 |
|------|-----------|------|
| `vocalis_studio/docs/` | 15 | プロジェクト公式ドキュメント |
| `vocalis_studio/claudedocs/` | 8 | 古いClaudeドキュメント |
| `vocalis_studio/VocalisStudio/docs/` | 4 | Xcode内ドキュメント（重複配置） |
| `vocalis_studio/VocalisStudio/claudedocs/` | 35 | Xcode内Claudeドキュメント |
| `vocalis_studio/pitch_detection_poc/` | 3 | POC関連 |
| その他（README, CLAUDE.md等） | 3 | ルートファイル |
| **合計** | **68** | |

### 1.2 主な問題点

1. **docsが2箇所に分散**
   - `vocalis_studio/docs/` - 15ファイル（メイン）
   - `vocalis_studio/VocalisStudio/docs/` - 4ファイル（重複配置）

2. **claudedocsが2箇所に分散**
   - `vocalis_studio/claudedocs/` - 8ファイル（古い）
   - `vocalis_studio/VocalisStudio/claudedocs/` - 35ファイル（新しい）

3. **古いPOCが残存**
   - `pitch_detection_poc/` - 3ファイル（プロジェクト初期のPOC、本体に実装済み）

4. **完了済みドキュメントの残存**
   - 実装完了した計画書・調査報告が多数残っている

---

## 2. 全ファイルリスト

### 2.1 vocalis_studio/docs/（15ファイル）

| ファイル | カテゴリ | 状態 |
|---------|---------|------|
| ARCHITECTURE.md | アーキテクチャ | ✅ 保持 |
| PROJECT_OVERVIEW.md | プロジェクト基盤 | ✅ 保持 |
| TECHNICAL_SPEC.md | 技術仕様 | ✅ 保持 |
| ROADMAP.md | プロジェクト基盤 | ✅ 保持 |
| MVP_ARCHITECTURE.md | アーキテクチャ | ✅ 保持 |
| MVP_SPECIFICATION.md | 機能仕様 | ✅ 保持 |
| TDD_PRINCIPLES.md | 開発ガイド | ✅ 保持 |
| TEST_MANAGEMENT.md | テスト | ✅ 保持 |
| SCREEN_DESIGN_V2.md | UI仕様 | ✅ 保持 |
| ANALYSIS_VIEW_SPECIFICATION.md | 機能仕様 | ✅ 保持 |
| SUBSCRIPTION_DESIGN.md | 機能仕様 | ✅ 保持 |
| MONETIZATION_STRATEGY.md | ビジネス | ✅ 保持 |
| PITCH_DETECTION_IMPROVEMENT.md | 機能仕様 | ✅ 保持 |

### 2.2 vocalis_studio/claudedocs/（8ファイル）

| ファイル | カテゴリ | 状態 | 備考 |
|---------|---------|------|------|
| phase1a_improvement_report.md | 報告 | ⚠️ 要検討 | 完了済み |
| phase2_improvement_report.md | 報告 | ⚠️ 要検討 | 完了済み |
| phase2d_improvement_report.md | 報告 | ⚠️ 要検討 | 完了済み |
| phase3_improvement_report.md | 報告 | ⚠️ 要検討 | 完了済み |
| pitch_detection_parameter_tuning.md | 技術 | ⚠️ 要検討 | 実装済み |
| pitch_detection_testing_summary.md | テスト | ⚠️ 要検討 | 完了済み |
| vocadito_accuracy_evaluation_report.md | 報告 | ⚠️ 要検討 | 評価完了 |
| DESIGN_IMPLEMENTATION_PLAN.md | 計画 | ⚠️ 要検討 | 実装済み？ |

### 2.3 vocalis_studio/VocalisStudio/docs/（4ファイル）

| ファイル | カテゴリ | 状態 | 備考 |
|---------|---------|------|------|
| CODE_SMELLS_REFERENCE.md | 品質ガイド | ✅ 保持 | docs/へ移動候補 |
| SOLID_PRINCIPLES_ANALYSIS.md | 設計ガイド | ✅ 保持 | docs/へ移動候補 |
| DESIGN_SYSTEM.md | UI仕様 | ✅ 保持 | docs/へ移動候補 |
| RECORDING_QUALITY_GUIDE.md | 品質ガイド | ✅ 保持 | docs/へ移動候補 |

### 2.4 vocalis_studio/VocalisStudio/claudedocs/（35ファイル）

#### UIテスト関連（6ファイル）
| ファイル | 状態 | 備考 |
|---------|------|------|
| ui-test-analysis-and-optimization.md | ✅ 保持 | 最新の最適化ガイド |
| test-scheme-management.md | ✅ 保持 | テストスキーム管理 |
| UI_TEST_FAILURE_INVESTIGATION_REPORT.md | ✅ 保持 | シミュレータ設定ガイド |
| UITEST_SCREENSHOT_EXTRACTION.md | ✅ 保持 | スクリーンショット抽出方法 |
| UI_TEST_WAIT_TIME_ANALYSIS.md | ⚠️ 要検討 | 最適化ドキュメントに統合済み？ |
| paywall_uitest_flaky_analysis.md | ⚠️ 要検討 | 一時的な調査報告 |

#### スペクトログラム関連（10ファイル）
| ファイル | 状態 | 備考 |
|---------|------|------|
| spectrogram-resolution-zoom-guide.md | ✅ 保持 | 解像度・ズーム調整の包括的ガイド |
| spectrogram-refactoring-plan.md | ⚠️ 要検討 | 完了済み？ |
| spectrogram_canvas_architecture_plan.md | ⚠️ 要検討 | 実装済み？ |
| spectrogram_canvas_implementation_result.md | ⚠️ 要検討 | 完了報告 |
| spectrogram_time_axis_requirements.md | ⚠️ 要検討 | 実装済み？ |
| time_axis_gap_analysis_and_implementation_plan.md | ⚠️ 要検討 | 完了済み？ |
| time_label_alignment_specification.md | ⚠️ 要検討 | 実装済み？ |
| vertical_scroll_analysis_and_time_axis_plan.md | ⚠️ 要検討 | 実装済み？ |
| frequency_label_visibility_investigation.md | ⚠️ 要検討 | 調査完了？ |
| spectrogram-resolution-improvement.md | ⚠️ 要検討 | zoom-guideに統合可能？ |

#### グラフ表示関連（5ファイル）
| ファイル | 状態 | 備考 |
|---------|------|------|
| fullscreen-graph-implementation-plan.md | ⚠️ 要検討 | 実装済み？ |
| graph_expansion_discrepancy_analysis.md | ⚠️ 要検討 | 調査完了？ |
| vertical_axis_expansion_analysis.md | ⚠️ 要検討 | 調査完了？ |
| fullscreen_viewport_implementation.md | ⚠️ 要検討 | 実装済み？ |
| scroll_position_issue_report.md | ⚠️ 要検討 | 問題解決済み？ |

#### ログ・デバッグ（3ファイル）
| ファイル | 状態 | 備考 |
|---------|------|------|
| log_capture_guide_v2.md | ✅ 保持 | 最新版 |
| log_capture_guide.md | ❌ 削除 | v2に置き換え済み |
| LOGGING_SYSTEM_ANALYSIS.md | ⚠️ 要検討 | 分析完了？ |

#### アーキテクチャ・設計（3ファイル）
| ファイル | 状態 | 備考 |
|---------|------|------|
| DOMAIN_ARCHITECTURE_CURRENT_STATE.md | ⚠️ 要検討 | 現状分析 |
| DOMAIN_ENRICHMENT_PLAN.md | ⚠️ 要検討 | 計画実装済み？ |
| RECORDING_ARCHITECTURE_REFACTORING_PROPOSAL.md | ⚠️ 要検討 | 実装済み？ |

#### UI/UX設計（2ファイル）
| ファイル | 状態 | 備考 |
|---------|------|------|
| darkmode-color-adjustment-plan.md | ⚠️ 要検討 | 実装完了（2025-11-19） |
| recording_list_ui_redesign.md | ⚠️ 要検討 | 実装済み？ |

#### 機能仕様（3ファイル）
| ファイル | 状態 | 備考 |
|---------|------|------|
| scale-sound-selection-spec.md | ⚠️ 要検討 | 仕様書 |
| scale-sound-selection-test-spec.md | ⚠️ 要検討 | テスト仕様 |
| recording-limit-system-analysis.md | ✅ 保持 | システム分析 |

#### テスト関連（2ファイル）
| ファイル | 状態 | 備考 |
|---------|------|------|
| failing_unit_tests_investigation.md | ⚠️ 要検討 | 調査完了？ |
| AUDIO_ACCURACY_TESTS_INVESTIGATION.md | ⚠️ 要検討 | 調査完了？ |

#### 運用（2ファイル）
| ファイル | 状態 | 備考 |
|---------|------|------|
| APP_STORE_RELEASE_GUIDE.md | ✅ 保持 | リリースガイド |
| ADD_FILES_TO_XCODE.md | ⚠️ 要検討 | 手順書 |

### 2.5 vocalis_studio/pitch_detection_poc/（3ファイル）

| ファイル | 状態 | 備考 |
|---------|------|------|
| PITCH_DETECTION_METHODS.md | ❌ 削除候補 | 本体に実装済み |
| README.md | ❌ 削除候補 | POC説明 |
| SETUP_INSTRUCTIONS.md | ❌ 削除候補 | POCセットアップ |

### 2.6 その他

| ファイル | 場所 | 状態 |
|---------|------|------|
| README.md | ルート | ✅ 保持 |
| CLAUDE.md | ルート | ✅ 保持 |
| web/README.md | web/ | ✅ 保持 |
| VocalisStudioTests/Resources/CSD/README.md | テスト | ✅ 保持 |
| Packages/SubscriptionDomain/README.md | パッケージ | ✅ 保持 |
| VocalisStudioTests/Infrastructure/StoreKit/STOREKIT_TESTING_GUIDE.md | テスト | ✅ 保持 |

---

## 3. 整理計画

### 3.1 推奨するディレクトリ構造

```
vocalis_studio/
├── CLAUDE.md
├── README.md
├── docs/                              # 公式ドキュメント（統合）
│   ├── architecture/                  # アーキテクチャ関連
│   │   ├── ARCHITECTURE.md
│   │   ├── MVP_ARCHITECTURE.md
│   │   └── SOLID_PRINCIPLES_ANALYSIS.md
│   ├── specifications/                # 機能仕様
│   │   ├── MVP_SPECIFICATION.md
│   │   ├── SCREEN_DESIGN_V2.md
│   │   ├── ANALYSIS_VIEW_SPECIFICATION.md
│   │   ├── SUBSCRIPTION_DESIGN.md
│   │   └── PITCH_DETECTION_IMPROVEMENT.md
│   ├── guides/                        # ガイド・手順書
│   │   ├── TDD_PRINCIPLES.md
│   │   ├── DESIGN_SYSTEM.md
│   │   ├── CODE_SMELLS_REFERENCE.md
│   │   └── RECORDING_QUALITY_GUIDE.md
│   ├── testing/                       # テスト関連
│   │   ├── TEST_MANAGEMENT.md
│   │   └── STOREKIT_TESTING_GUIDE.md
│   └── business/                      # ビジネス関連
│       ├── PROJECT_OVERVIEW.md
│       ├── ROADMAP.md
│       ├── MONETIZATION_STRATEGY.md
│       └── TECHNICAL_SPEC.md
└── VocalisStudio/
    └── claudedocs/                    # Claude作業ドキュメント（統合）
        ├── active/                    # 進行中・参照頻度高
        │   ├── ui-test-analysis-and-optimization.md
        │   ├── test-scheme-management.md
        │   ├── UI_TEST_FAILURE_INVESTIGATION_REPORT.md
        │   ├── UITEST_SCREENSHOT_EXTRACTION.md
        │   ├── log_capture_guide_v2.md
        │   ├── spectrogram-resolution-zoom-guide.md
        │   ├── recording-limit-system-analysis.md
        │   └── APP_STORE_RELEASE_GUIDE.md
        └── archive/                   # 完了済み・参照頻度低
            ├── implementation-reports/
            ├── investigation-reports/
            └── deprecated/
```

### 3.2 アクション項目

#### Phase 1: 即時削除（明確に不要なもの）

| ファイル | 理由 |
|---------|------|
| `log_capture_guide.md` | v2に置き換え済み |
| `pitch_detection_poc/` 全体 | 本体に実装済み |

#### Phase 2: 統合・移動

1. **VocalisStudio/docs/ → docs/ へ移動**
   - CODE_SMELLS_REFERENCE.md
   - SOLID_PRINCIPLES_ANALYSIS.md
   - DESIGN_SYSTEM.md
   - RECORDING_QUALITY_GUIDE.md

2. **vocalis_studio/claudedocs/ → VocalisStudio/claudedocs/archive/ へ移動**
   - phase1a_improvement_report.md
   - phase2_improvement_report.md
   - phase2d_improvement_report.md
   - phase3_improvement_report.md
   - pitch_detection_parameter_tuning.md
   - pitch_detection_testing_summary.md
   - vocadito_accuracy_evaluation_report.md
   - DESIGN_IMPLEMENTATION_PLAN.md

#### Phase 3: 要検討ファイルの精査

以下のファイルは内容を確認し、実装済みなら削除/アーカイブ：

**スペクトログラム関連（9ファイル）**
- spectrogram-refactoring-plan.md
- spectrogram_canvas_architecture_plan.md
- spectrogram_canvas_implementation_result.md
- spectrogram_time_axis_requirements.md
- time_axis_gap_analysis_and_implementation_plan.md
- time_label_alignment_specification.md
- vertical_scroll_analysis_and_time_axis_plan.md
- frequency_label_visibility_investigation.md
- spectrogram-resolution-improvement.md

**グラフ表示関連（5ファイル）**
- fullscreen-graph-implementation-plan.md
- graph_expansion_discrepancy_analysis.md
- vertical_axis_expansion_analysis.md
- fullscreen_viewport_implementation.md
- scroll_position_issue_report.md

**その他（8ファイル）**
- UI_TEST_WAIT_TIME_ANALYSIS.md
- paywall_uitest_flaky_analysis.md
- LOGGING_SYSTEM_ANALYSIS.md
- DOMAIN_ARCHITECTURE_CURRENT_STATE.md
- DOMAIN_ENRICHMENT_PLAN.md
- RECORDING_ARCHITECTURE_REFACTORING_PROPOSAL.md
- darkmode-color-adjustment-plan.md
- recording_list_ui_redesign.md

#### Phase 4: ディレクトリ再編成

docs/配下をカテゴリ別サブディレクトリに整理

---

## 4. 期待される効果

### Before
- 68ファイルが5箇所に分散
- 重複・旧バージョンが混在
- 完了済み計画書が多数残存
- 目的のドキュメントを探しにくい

### After
- docs/: 公式ドキュメント（カテゴリ別整理）
- claudedocs/active/: 現在参照するドキュメント
- claudedocs/archive/: 過去の作業記録
- 不要ファイル削除でリポジトリ軽量化
- 目的別にすぐ見つかる構造

---

## 5. 次のステップ

1. [ ] Phase 1: 即時削除の実行
2. [ ] Phase 2: 統合・移動の実行
3. [ ] Phase 3: 要検討ファイルの内容確認と判断
4. [ ] Phase 4: ディレクトリ再編成
5. [ ] CLAUDE.mdの参照パス更新
6. [ ] このドキュメント自体のアーカイブ

---

## 6. 参考情報

### 保持必須ファイル（確定）

以下のファイルは削除・移動時に注意が必要（CLAUDE.mdから参照されている可能性）：

- docs/TDD_PRINCIPLES.md
- docs/MVP_SPECIFICATION.md
- docs/MVP_ARCHITECTURE.md
- docs/TECHNICAL_SPEC.md
- docs/ARCHITECTURE.md
- docs/PROJECT_OVERVIEW.md
- docs/ROADMAP.md

### claudedocs保持ファイル（現在アクティブ）

- ui-test-analysis-and-optimization.md
- test-scheme-management.md
- UI_TEST_FAILURE_INVESTIGATION_REPORT.md
- UITEST_SCREENSHOT_EXTRACTION.md
- log_capture_guide_v2.md
- spectrogram-resolution-zoom-guide.md
- recording-limit-system-analysis.md
- APP_STORE_RELEASE_GUIDE.md

---

**最終更新**: 2025-11-20
