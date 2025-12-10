import SwiftUI

/// Environment key for UI test animation disabling
struct UITestAnimationsDisabledKey: EnvironmentKey {
    static let defaultValue: Bool = false
}

extension EnvironmentValues {
    var uiTestAnimationsDisabled: Bool {
        get { self[UITestAnimationsDisabledKey.self] }
        set { self[UITestAnimationsDisabledKey.self] = newValue }
    }
}
