import Foundation

/// Errors related to recording permissions
public enum RecordingPermissionError: Error, Equatable {
    /// User exceeded daily recording limit
    case dailyLimitExceeded

    /// Premium subscription required for this feature
    case premiumRequired

    /// Invalid recording settings
    case invalidSettings(String)

    /// Unexpected permission state
    case unexpectedState

    /// Convert from RecordingPermission.DenialReason
    public static func from(_ reason: RecordingPermission.DenialReason) -> RecordingPermissionError {
        switch reason {
        case .dailyLimitExceeded:
            return .dailyLimitExceeded
        case .premiumRequired:
            return .premiumRequired
        case .invalidSettings(let message):
            return .invalidSettings(message)
        }
    }
}

// MARK: - LocalizedError

extension RecordingPermissionError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .dailyLimitExceeded:
            return "1日の録音回数制限に達しました"
        case .premiumRequired:
            return "この機能を使用するにはプレミアムプランが必要です"
        case .invalidSettings(let message):
            return "録音設定が無効です: \(message)"
        case .unexpectedState:
            return "予期しないエラーが発生しました"
        }
    }

    public var failureReason: String? {
        switch self {
        case .dailyLimitExceeded:
            return "無料プランでは1日3回まで録音できます"
        case .premiumRequired:
            return "スケール録音機能はプレミアムプラン限定です"
        case .invalidSettings:
            return "録音設定を確認してください"
        case .unexpectedState:
            return "録音権限の確認中に予期しない状態が発生しました"
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .dailyLimitExceeded:
            return "明日もう一度お試しいただくか、プレミアムプランにアップグレードしてください"
        case .premiumRequired:
            return "プレミアムプランにアップグレードしてすべての機能をご利用ください"
        case .invalidSettings:
            return "録音設定を修正してください"
        case .unexpectedState:
            return "アプリを再起動するか、サポートにお問い合わせください"
        }
    }
}
