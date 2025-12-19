import Foundation

/// ピッチ検出のオーディオ設定
public struct AudioDetectionSettings: Equatable, Codable {
    /// 録音再生音量 (0.0 - 1.0, 1.0が標準音量)
    public let recordingPlaybackVolume: Float

    /// RMS silence threshold (0.001 - 0.1)
    /// 内部パラメータ: ユーザーには「検出感度」として表示
    public let rmsSilenceThreshold: Float

    /// Confidence threshold (0.1 - 1.0)
    /// ピッチ検出の信頼度閾値
    public let confidenceThreshold: Float

    /// ピッチ検出アルゴリズム
    public let pitchAlgorithm: PitchDetectionAlgorithm

    // MARK: - Initialization

    public init(
        recordingPlaybackVolume: Float = 0.8,
        rmsSilenceThreshold: Float = 0.02,
        confidenceThreshold: Float = 0.4,
        pitchAlgorithm: PitchDetectionAlgorithm = .default
    ) {
        // Validation: clamp values to valid ranges
        self.recordingPlaybackVolume = max(0.0, min(1.0, recordingPlaybackVolume))
        self.rmsSilenceThreshold = max(0.001, min(0.1, rmsSilenceThreshold))
        self.confidenceThreshold = max(0.1, min(1.0, confidenceThreshold))
        self.pitchAlgorithm = pitchAlgorithm
    }

    // MARK: - Default Settings

    /// デフォルト設定 (実機用)
    public static let `default` = AudioDetectionSettings(
        recordingPlaybackVolume: 0.8,
        rmsSilenceThreshold: 0.02,
        confidenceThreshold: 0.4,
        pitchAlgorithm: .default
    )

    /// シミュレーター用設定 (低感度閾値)
    public static let simulator = AudioDetectionSettings(
        recordingPlaybackVolume: 0.8,
        rmsSilenceThreshold: 0.005,
        confidenceThreshold: 0.3,
        pitchAlgorithm: .default
    )

    // MARK: - Detection Sensitivity

    /// 検出感度の enum 表現
    public enum DetectionSensitivity: Equatable, Codable {
        case low     // RMS: 0.05 (大きい音のみ検出)
        case normal  // RMS: 0.02 (標準)
        case high    // RMS: 0.005 (小さい音も検出)

        /// 検出感度に対応するRMS閾値
        public var rmsThreshold: Float {
            switch self {
            case .low: return 0.05
            case .normal: return 0.02
            case .high: return 0.005
            }
        }

        /// RMS閾値から検出感度を推定
        /// - Parameter threshold: RMS閾値
        public init(fromRMSThreshold threshold: Float) {
            if threshold <= 0.01 {
                // 0.01以下は「高感度」
                self = .high
            } else if threshold >= 0.035 {
                // 0.035以上は「低感度」
                self = .low
            } else {
                // 0.01 < threshold < 0.035は「標準」
                self = .normal
            }
        }
    }

    /// 現在の設定に対応する検出感度
    public var sensitivity: DetectionSensitivity {
        DetectionSensitivity(fromRMSThreshold: rmsSilenceThreshold)
    }
}

// MARK: - Codable Implementation (Backward Compatibility)

extension AudioDetectionSettings {
    enum CodingKeys: String, CodingKey {
        case recordingPlaybackVolume
        case rmsSilenceThreshold
        case confidenceThreshold
        case pitchAlgorithm
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let recordingPlaybackVolume = try container.decode(Float.self, forKey: .recordingPlaybackVolume)
        let rmsSilenceThreshold = try container.decode(Float.self, forKey: .rmsSilenceThreshold)
        let confidenceThreshold = try container.decode(Float.self, forKey: .confidenceThreshold)

        // Backward compatibility: use default if pitchAlgorithm is missing
        let pitchAlgorithm = try container.decodeIfPresent(PitchDetectionAlgorithm.self, forKey: .pitchAlgorithm) ?? .default

        self.init(
            recordingPlaybackVolume: recordingPlaybackVolume,
            rmsSilenceThreshold: rmsSilenceThreshold,
            confidenceThreshold: confidenceThreshold,
            pitchAlgorithm: pitchAlgorithm
        )
    }
}
