import Foundation

/// オーディオ設定のリポジトリインターフェース
public protocol AudioSettingsRepositoryProtocol {
    /// 現在の設定を取得
    func get() -> AudioDetectionSettings

    /// 設定を保存
    /// - Parameter settings: 保存する設定
    /// - Throws: 保存に失敗した場合
    func save(_ settings: AudioDetectionSettings) throws

    /// デフォルトに戻す
    /// - Throws: リセットに失敗した場合
    func reset() throws
}
