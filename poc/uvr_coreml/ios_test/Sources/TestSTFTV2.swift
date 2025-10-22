import Foundation
import Accelerate

/// Test STFTProcessorV2 with round-trip verification
func testSTFTV2RoundTrip() {
    print("\n" + String(repeating: "=", count: 80))
    print("🧪 STFTProcessorV2 Round-Trip Test")
    print(String(repeating: "=", count: 80))

    let processor = STFTProcessorV2(fftSize: 4096, hopSize: 1024)

    // Test 1: Simple sine wave
    print("\n📊 Test 1: 440Hz Sine Wave")
    let sampleRate: Float = 44100.0
    let frequency: Float = 440.0
    let duration: Float = 1.0
    let numSamples = Int(sampleRate * duration)

    var sineWave = [Float](repeating: 0, count: numSamples)
    for i in 0..<numSamples {
        let t = Float(i) / sampleRate
        sineWave[i] = sin(2.0 * Float.pi * frequency * t)
    }

    let error1 = processor.roundTripTest(sineWave)
    print("   RMS Error: \(error1)")
    print("   Status: \(error1 < 0.01 ? "✅ PASS" : "❌ FAIL")")

    // Test 2: Complex signal (multiple frequencies)
    print("\n📊 Test 2: Multi-Frequency Signal (440Hz + 880Hz + 1320Hz)")
    var complexSignal = [Float](repeating: 0, count: numSamples)
    for i in 0..<numSamples {
        let t = Float(i) / sampleRate
        complexSignal[i] = sin(2.0 * Float.pi * 440.0 * t) * 0.5 +
                          sin(2.0 * Float.pi * 880.0 * t) * 0.3 +
                          sin(2.0 * Float.pi * 1320.0 * t) * 0.2
    }

    let error2 = processor.roundTripTest(complexSignal)
    print("   RMS Error: \(error2)")
    print("   Status: \(error2 < 0.01 ? "✅ PASS" : "❌ FAIL")")

    // Test 3: Random noise
    print("\n📊 Test 3: Random Noise")
    var noise = [Float](repeating: 0, count: numSamples)
    for i in 0..<numSamples {
        noise[i] = Float.random(in: -1.0...1.0)
    }

    let error3 = processor.roundTripTest(noise)
    print("   RMS Error: \(error3)")
    print("   Status: \(error3 < 0.01 ? "✅ PASS" : "❌ FAIL")")

    print("\n" + String(repeating: "=", count: 80))
    print("✅ All STFTProcessorV2 tests completed")
    print(String(repeating: "=", count: 80))
}

/// Compare STFTProcessorV2 with Python librosa STFT
func compareSTFTV2WithPython() {
    print("\n" + String(repeating: "=", count: 80))
    print("🔍 STFTProcessorV2 vs Python librosa STFT Comparison")
    print(String(repeating: "=", count: 80))

    let processor = STFTProcessorV2(fftSize: 4096, hopSize: 1024)

    // Generate 440Hz test signal
    let sampleRate: Float = 44100.0
    let frequency: Float = 440.0
    let duration: Float = 0.5  // 0.5 seconds
    let numSamples = Int(sampleRate * duration)

    var signal = [Float](repeating: 0, count: numSamples)
    for i in 0..<numSamples {
        let t = Float(i) / sampleRate
        signal[i] = sin(2.0 * Float.pi * frequency * t)
    }

    // Compute STFT
    let (real, imag) = processor.stft(signal)

    print("📊 STFT Results:")
    print("   Number of frames: \(real.count)")
    print("   Frequency bins per frame: \(real[0].count)")

    // Save first frame for Python comparison
    if real.count > 0 {
        let outputDir = "../tests/swift_output"
        try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        // Save first frame real and imaginary parts
        let realStr = real[0].map { String(format: "%.10f", $0) }.joined(separator: "\n")
        let imagStr = imag[0].map { String(format: "%.10f", $0) }.joined(separator: "\n")

        try? realStr.write(toFile: "\(outputDir)/stft_real_swift.txt", atomically: true, encoding: .utf8)
        try? imagStr.write(toFile: "\(outputDir)/stft_imag_swift.txt", atomically: true, encoding: .utf8)

        // Calculate and save magnitude
        var magnitude = [Float](repeating: 0, count: real[0].count)
        for i in 0..<real[0].count {
            let re = real[0][i]
            let im = imag[0][i]
            magnitude[i] = sqrtf(re * re + im * im)
        }

        let magStr = magnitude.map { String(format: "%.10f", $0) }.joined(separator: "\n")
        try? magStr.write(toFile: "\(outputDir)/stft_magnitude_swift.txt", atomically: true, encoding: .utf8)

        print("   First frame saved to: \(outputDir)/stft_*_swift.txt")

        // Print some key values
        let expectedBin = Int(round(frequency * Float(4096) / sampleRate))
        print("\n🎯 440Hz Analysis:")
        print("   Expected bin: \(expectedBin)")
        print("   Magnitude at bin[\(expectedBin)]: \(magnitude[expectedBin])")
        print("   Real part: \(real[0][expectedBin])")
        print("   Imaginary part: \(imag[0][expectedBin])")
    }

    print("\n" + String(repeating: "=", count: 80))
    print("✅ STFT comparison data saved - ready for Python verification")
    print(String(repeating: "=", count: 80))
}
