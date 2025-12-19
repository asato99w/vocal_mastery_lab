//
//  Feature.swift
//  VocalisStudio
//
//  Feature enum representing app features that can be restricted by subscription
//

import Foundation

/// App features that can be restricted by subscription tier
public enum Feature: String, Codable, CaseIterable {
    // MARK: - Basic Features (Free tier)

    /// Basic audio recording functionality
    case basicRecording = "basic_recording"

    /// Real-time pitch detection
    case realtimePitchDetection = "realtime_pitch_detection"

    /// 5-tone scale playback
    case fiveToneScale = "five_tone_scale"

    // MARK: - Premium Features

    /// Spectrum visualization
    case spectrumVisualization = "spectrum_visualization"

    /// Pitch accuracy analysis
    case pitchAccuracyAnalysis = "pitch_accuracy_analysis"

    /// Pitch trend graph
    case pitchTrendGraph = "pitch_trend_graph"

    /// Unlimited local recording storage
    case unlimitedLocalStorage = "unlimited_local_storage"

    /// Custom scale creation
    case customScale = "custom_scale"

    /// Practice history tracking
    case practiceHistory = "practice_history"

    /// WAV export
    case wavExport = "wav_export"

    /// Theme customization (10 themes)
    case standardThemes = "standard_themes"

    // MARK: - Premium Plus Features

    /// AI pitch improvement suggestions
    case aiPitchSuggestions = "ai_pitch_suggestions"

    /// Unlimited cloud backup
    case cloudBackup = "cloud_backup"

    /// MP3/AAC export
    case advancedExport = "advanced_export"

    /// Professional analysis (harmonics, formants)
    case professionalAnalysis = "professional_analysis"

    /// Multi-device sync
    case multiDeviceSync = "multi_device_sync"

    /// Unlimited theme customization
    case unlimitedThemes = "unlimited_themes"

    // MARK: - Helper Properties

    /// Localized display name
    public var displayName: String {
        switch self {
        case .basicRecording:
            return "基本録音"
        case .realtimePitchDetection:
            return "リアルタイムピッチ検出"
        case .fiveToneScale:
            return "5音スケール再生"
        case .spectrumVisualization:
            return "スペクトル表示"
        case .pitchAccuracyAnalysis:
            return "ピッチ精度分析"
        case .pitchTrendGraph:
            return "音程推移グラフ"
        case .unlimitedLocalStorage:
            return "無制限ローカル保存"
        case .customScale:
            return "カスタムスケール"
        case .practiceHistory:
            return "練習履歴"
        case .wavExport:
            return "WAVエクスポート"
        case .standardThemes:
            return "テーマカスタマイズ（10種類）"
        case .aiPitchSuggestions:
            return "AI音程改善提案"
        case .cloudBackup:
            return "クラウドバックアップ"
        case .advancedExport:
            return "高度なエクスポート（MP3/AAC）"
        case .professionalAnalysis:
            return "プロフェッショナル分析"
        case .multiDeviceSync:
            return "マルチデバイス同期"
        case .unlimitedThemes:
            return "無制限テーマカスタマイズ"
        }
    }

    /// Feature description for Paywall
    public var description: String {
        switch self {
        case .basicRecording:
            return "音声の録音・再生機能"
        case .realtimePitchDetection:
            return "リアルタイムで音程を検出"
        case .fiveToneScale:
            return "5音階で練習できる"
        case .spectrumVisualization:
            return "周波数スペクトルを可視化"
        case .pitchAccuracyAnalysis:
            return "音程の精度を詳細に分析"
        case .pitchTrendGraph:
            return "音程の推移をグラフで確認"
        case .unlimitedLocalStorage:
            return "録音を無制限に保存"
        case .customScale:
            return "自由に音階を設定できる"
        case .practiceHistory:
            return "練習の履歴を記録・確認"
        case .wavExport:
            return "高音質WAV形式で書き出し"
        case .standardThemes:
            return "10種類のテーマから選択"
        case .aiPitchSuggestions:
            return "AIが音程改善をアドバイス"
        case .cloudBackup:
            return "クラウドに自動バックアップ"
        case .advancedExport:
            return "MP3/AAC形式で書き出し"
        case .professionalAnalysis:
            return "倍音・フォルマント分析"
        case .multiDeviceSync:
            return "複数デバイスで同期"
        case .unlimitedThemes:
            return "テーマを自由にカスタマイズ"
        }
    }

    /// Minimum required tier for this feature
    public var minimumTier: SubscriptionTier {
        switch self {
        case .basicRecording, .realtimePitchDetection, .fiveToneScale:
            return .free
        case .spectrumVisualization, .pitchAccuracyAnalysis, .pitchTrendGraph,
             .unlimitedLocalStorage, .customScale, .practiceHistory,
             .wavExport, .standardThemes:
            return .premium
        case .aiPitchSuggestions, .cloudBackup, .advancedExport,
             .professionalAnalysis, .multiDeviceSync, .unlimitedThemes:
            return .premiumPlus
        }
    }
}
