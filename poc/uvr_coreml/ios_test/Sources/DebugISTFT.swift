import Foundation
import Accelerate

func debugISTFTDetails() {
    print("\n" + String(repeating: "=", count: 80))
    print("🔍 Debug: Detailed ISTFT Analysis")
    print(String(repeating: "=", count: 80))

    let processor = STFTProcessorV2(fftSize: 8, hopSize: 2)

    // Simple DC signal
    let signal = [Float](repeating: 1.0, count: 16)

    print("\n📊 Input signal (DC):")
    print("   \(signal.map { String(format: "%.3f", $0) }.joined(separator: ", "))")

    // STFT
    let (real, imag) = processor.stft(signal)

    print("\n📊 STFT output:")
    print("   Frames: \(real.count)")
    print("   Bins per frame: \(real[0].count)")

    for (frameIdx, (r, i)) in zip(real, imag).enumerated() {
        print("\n   Frame \(frameIdx):")
        print("     Real: \(r.map { String(format: "%.3f", $0) }.joined(separator: ", "))")
        print("     Imag: \(i.map { String(format: "%.3f", $0) }.joined(separator: ", "))")

        var mag = [Float](repeating: 0, count: r.count)
        for j in 0..<r.count {
            mag[j] = sqrtf(r[j] * r[j] + i[j] * i[j])
        }
        print("     Magnitude: \(mag.map { String(format: "%.3f", $0) }.joined(separator: ", "))")
    }

    // ISTFT
    let reconstructed = processor.istft(real: real, imag: imag)

    print("\n📊 Reconstructed signal:")
    print("   Length: \(reconstructed.count)")
    print("   Values: \(reconstructed.map { String(format: "%.3f", $0) }.joined(separator: ", "))")

    // Error
    let minLen = min(signal.count, reconstructed.count)
    var errors = [Float](repeating: 0, count: minLen)
    for i in 0..<minLen {
        errors[i] = signal[i] - reconstructed[i]
    }

    print("\n📊 Reconstruction errors:")
    print("   \(errors.map { String(format: "%.6f", $0) }.joined(separator: ", "))")

    let rmsError = sqrtf(errors.map { $0 * $0 }.reduce(0, +) / Float(minLen))
    print("   RMS Error: \(rmsError)")

    print("\n" + String(repeating: "=", count: 80))
}
