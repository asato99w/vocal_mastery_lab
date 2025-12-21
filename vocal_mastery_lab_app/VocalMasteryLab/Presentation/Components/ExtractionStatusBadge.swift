import SwiftUI

/// Badge component showing extraction status
public struct ExtractionStatusBadge: View {
    let isExtracted: Bool

    public init(isExtracted: Bool) {
        self.isExtracted = isExtracted
    }

    public var body: some View {
        HStack(spacing: 4) {
            if isExtracted {
                Image(systemName: "checkmark.circle.fill")
                    .font(.caption2)
                Text("抽出済み")
                    .font(.caption2)
            } else {
                Text("未抽出")
                    .font(.caption2)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            Capsule()
                .fill(isExtracted ? Color.green.opacity(0.2) : Color.gray.opacity(0.2))
        )
        .foregroundColor(isExtracted ? .green : .gray)
    }
}
