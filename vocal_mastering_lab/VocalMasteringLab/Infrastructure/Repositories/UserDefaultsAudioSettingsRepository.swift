import Foundation
import VocalisDomain

/// UserDefaultsを使用したオーディオ設定のリポジトリ実装
final class UserDefaultsAudioSettingsRepository: AudioSettingsRepositoryProtocol {

    private let userDefaults: UserDefaults
    private let settingsKey = "audioDetectionSettings"

    init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
    }

    func get() -> AudioDetectionSettings {
        // UserDefaultsからJSON文字列を取得
        guard let data = userDefaults.data(forKey: settingsKey) else {
            // 保存された設定がない場合はデフォルトを返す
            #if targetEnvironment(simulator)
            return AudioDetectionSettings.simulator
            #else
            return AudioDetectionSettings.default
            #endif
        }

        // JSONデコード
        do {
            let settings = try JSONDecoder().decode(AudioDetectionSettings.self, from: data)
            return settings
        } catch {
            // デコード失敗時もデフォルトを返す
            #if targetEnvironment(simulator)
            return AudioDetectionSettings.simulator
            #else
            return AudioDetectionSettings.default
            #endif
        }
    }

    func save(_ settings: AudioDetectionSettings) throws {
        // JSONエンコード
        let data = try JSONEncoder().encode(settings)

        // UserDefaultsに保存
        userDefaults.set(data, forKey: settingsKey)
    }

    func reset() throws {
        // 保存された設定を削除
        userDefaults.removeObject(forKey: settingsKey)
    }
}
