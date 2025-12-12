import SwiftUI

/// Warning view for MIDI range validation errors
struct MIDIRangeWarningView: View {
    let message: String

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.orange)
                .font(.title3)

            VStack(alignment: .leading, spacing: 4) {
                Text("warning.midi_range.title".localized)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                Text(message)
                    .font(.caption)
                    .foregroundColor(ColorPalette.text.opacity(0.8))

                Text("warning.midi_range.hint".localized)
                    .font(.caption2)
                    .foregroundColor(ColorPalette.text.opacity(0.6))
            }
        }
        .padding(12)
        .background(Color.orange.opacity(0.1))
        .cornerRadius(8)
        .accessibilityIdentifier("MIDIRangeWarning")
    }
}
