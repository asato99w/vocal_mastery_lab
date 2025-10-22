import Foundation
import Accelerate

/// vDSP_fft_zripの基本動作とスケーリングをテスト
func testSimpleFFT() {
    print("\n" + String(repeating: "=", count: 80))
    print("🧪 Simple FFT Test - Scaling Verification")
    print(String(repeating: "=", count: 80))

    // 簡単なテスト信号: DC component only [1, 1, 1, 1, 1, 1, 1, 1]
    let N = 8
    let log2n = vDSP_Length(log2(Float(N)))
    let halfN = N / 2

    guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
        print("❌ FFT setup failed")
        return
    }
    defer { vDSP_destroy_fftsetup(setup) }

    // Test 1: DC component (all ones)
    print("\n📊 Test 1: DC Component [1,1,1,1,1,1,1,1]")
    var input1: [Float] = [1, 1, 1, 1, 1, 1, 1, 1]

    // Pack into split complex
    var realp1 = [Float](repeating: 0, count: halfN)
    var imagp1 = [Float](repeating: 0, count: halfN)
    for i in 0..<halfN {
        realp1[i] = input1[2*i]
        imagp1[i] = input1[2*i + 1]
    }

    print("  Before FFT - realp: \(realp1)")
    print("  Before FFT - imagp: \(imagp1)")

    // FFT
    realp1.withUnsafeMutableBufferPointer { realBuffer in
        imagp1.withUnsafeMutableBufferPointer { imagBuffer in
            var splitComplex = DSPSplitComplex(realp: realBuffer.baseAddress!, imagp: imagBuffer.baseAddress!)
            vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
        }
    }

    print("  After FFT - realp: \(realp1)")
    print("  After FFT - imagp: \(imagp1)")
    print("  DC component (realp[0]): \(realp1[0])")
    print("  Expected (without scaling): 8.0")
    print("  Expected (with 2x scaling): 16.0")
    print("  Actual ratio to expected: \(realp1[0] / 8.0)x")

    // Test 2: Simple sine wave component
    print("\n📊 Test 2: Single Frequency")
    var input2: [Float] = [0, 1, 0, -1, 0, 1, 0, -1]  // 2 cycles

    var realp2 = [Float](repeating: 0, count: halfN)
    var imagp2 = [Float](repeating: 0, count: halfN)
    for i in 0..<halfN {
        realp2[i] = input2[2*i]
        imagp2[i] = input2[2*i + 1]
    }

    print("  Input: \(input2)")
    print("  Before FFT - realp: \(realp2)")
    print("  Before FFT - imagp: \(imagp2)")

    realp2.withUnsafeMutableBufferPointer { realBuffer in
        imagp2.withUnsafeMutableBufferPointer { imagBuffer in
            var splitComplex = DSPSplitComplex(realp: realBuffer.baseAddress!, imagp: imagBuffer.baseAddress!)
            vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
        }
    }

    print("  After FFT - realp: \(realp2)")
    print("  After FFT - imagp: \(imagp2)")

    // Calculate magnitudes
    print("\n  Magnitudes:")
    for i in 0..<halfN {
        let mag = sqrtf(realp2[i] * realp2[i] + imagp2[i] * imagp2[i])
        print("    Bin \(i): \(mag)")
    }

    print("\n" + String(repeating: "=", count: 80))
}
