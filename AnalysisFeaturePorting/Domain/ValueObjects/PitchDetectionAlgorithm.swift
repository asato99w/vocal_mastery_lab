import Foundation

/// ピッチ検出アルゴリズムの種類
public enum PitchDetectionAlgorithm: String, CaseIterable, Codable, Equatable {
    /// FCPE - 機械学習モデル (CoreML) - 推奨
    case fcpe = "FCPE"

    /// YINアルゴリズム (高速・安定)
    case yin = "YIN"

    /// pYIN - デフォルト設定
    case pyinDefault = "pYIN"

    /// pYIN - 高検出設定 (低い無音閾値)
    case pyinHighDetection = "pYIN-highDetection"

    /// pYIN - バランス設定 (voicedBias強化)
    case pyinBalanced = "pYIN-balanced"

    /// pYIN - アグレッシブ設定 (最大voicedBias)
    case pyinAggressive = "pYIN-aggressive"

    // MARK: - Display Properties

    /// ローカライズ用キー
    public var displayNameKey: String {
        switch self {
        case .yin:
            return "algorithm.yin"
        case .pyinDefault:
            return "algorithm.pyin"
        case .pyinHighDetection:
            return "algorithm.pyin_high_detection"
        case .pyinBalanced:
            return "algorithm.pyin_balanced"
        case .pyinAggressive:
            return "algorithm.pyin_aggressive"
        case .fcpe:
            return "algorithm.fcpe"
        }
    }

    /// 説明文のローカライズ用キー
    public var descriptionKey: String {
        switch self {
        case .yin:
            return "algorithm.yin.desc"
        case .pyinDefault:
            return "algorithm.pyin.desc"
        case .pyinHighDetection:
            return "algorithm.pyin_high_detection.desc"
        case .pyinBalanced:
            return "algorithm.pyin_balanced.desc"
        case .pyinAggressive:
            return "algorithm.pyin_aggressive.desc"
        case .fcpe:
            return "algorithm.fcpe.desc"
        }
    }

    /// pYINアルゴリズムかどうか
    public var isPYIN: Bool {
        switch self {
        case .yin, .fcpe:
            return false
        case .pyinDefault, .pyinHighDetection, .pyinBalanced, .pyinAggressive:
            return true
        }
    }

    /// FCPEアルゴリズムかどうか
    public var isFCPE: Bool {
        self == .fcpe
    }

    // MARK: - Vibrato Detection Parameters

    /// ビブラート検出用の最小信頼度閾値
    /// - FCPE: 0.5 (高解像度のため標準閾値)
    /// - YIN/pYIN: 0.3 (信頼度変動が大きいため緩和)
    public var vibratoMinConfidence: Float {
        isFCPE ? 0.5 : 0.3
    }

    /// ビブラート検出用の最小規則性閾値
    /// - FCPE: 0.3 (100Hzサンプリングで高精度)
    /// - YIN/pYIN: 0.15 (20Hzサンプリングのため緩和)
    public var vibratoMinRegularity: Float {
        isFCPE ? 0.3 : 0.15
    }

    // MARK: - Default

    /// デフォルトアルゴリズム
    public static let `default`: PitchDetectionAlgorithm = .fcpe

    /// 設定画面に表示するアルゴリズム一覧
    /// pYINのバリエーション（balanced, aggressive等）は非表示
    public static let displayCases: [PitchDetectionAlgorithm] = [.fcpe, .yin, .pyinDefault]
}
