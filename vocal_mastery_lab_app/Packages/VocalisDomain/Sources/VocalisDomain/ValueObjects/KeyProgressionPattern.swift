import Foundation

/// Key progression pattern for scale exercises
public enum KeyProgressionPattern: Equatable, Codable, Hashable {
    case ascendingOnly           // 上昇のみ
    case descendingOnly          // 下降のみ
    case ascendingThenDescending // 上昇→下降
    case descendingThenAscending // 下降→上昇

    /// Display name in Japanese
    public var displayName: String {
        switch self {
        case .ascendingOnly:
            return "上昇のみ"
        case .descendingOnly:
            return "下降のみ"
        case .ascendingThenDescending:
            return "上昇→下降"
        case .descendingThenAscending:
            return "下降→上昇"
        }
    }

    /// Whether this pattern requires ascending key count setting
    public var showsAscendingCount: Bool {
        switch self {
        case .ascendingOnly, .ascendingThenDescending, .descendingThenAscending:
            return true
        case .descendingOnly:
            return false
        }
    }

    /// Whether this pattern requires descending key count setting
    public var showsDescendingCount: Bool {
        switch self {
        case .descendingOnly, .ascendingThenDescending, .descendingThenAscending:
            return true
        case .ascendingOnly:
            return false
        }
    }
}
