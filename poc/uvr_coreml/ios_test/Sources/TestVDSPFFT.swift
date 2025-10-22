import Foundation
import Accelerate

/// vDSP_fft_zripの正しい使い方をテストする
func testVDSPFFT() {
    print("=" * 80)
    print("🧪 vDSP_fft_zrip 使い方テスト")
    print("=" * 80)

    // 簡単なテスト: [1, 2, 3, 4, 5, 6, 7, 8]
    let n = 8
    let log2n = vDSP_Length(log2(Float(n)))

    guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
        print("❌ FFT setup failed")
        return
    }

    defer {
        vDSP_destroy_fftsetup(setup)
    }

    // 入力信号
    var input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
    print("\n📥 入力信号: \\(input)")

    // vDSP_fft_zripは入力をinterleaved complexとして解釈
    // つまり [re0, im0, re1, im1, ...] という形式
    // 実数信号の場合、虚部は0なので [re0, 0, re1, 0, ...]

    // 方法1: Interleaved complex形式に変換
    var complexInput: [Float] = []
    for val in input {
        complexInput.append(val)
        complexInput.append(0)  // 虚部は0
    }

    print("📊 Interleaved complex形式: \\(complexInput)")

    // Split complex用バッファ
    let halfSize = n / 2
    var realPart = [Float](repeating: 0, count: halfSize)
    var imagPart = [Float](repeating: 0, count: halfSize)

    // Interleaved → Split complex変換
    realPart.withUnsafeMutableBufferPointer { realBuffer in
        imagPart.withUnsafeMutableBufferPointer { imagBuffer in
            var splitComplex = DSPSplitComplex(realp: realBuffer.baseAddress!, imagp: imagBuffer.baseAddress!)

            complexInput.withUnsafeBytes { complexBytes in
                let complexPtr = complexBytes.baseAddress!.assumingMemoryBound(to: DSPComplex.self)
                vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfSize))
            }

            print("\\n📊 変換後のSplit Complex:")
            print("  Real: \\(Array(realBuffer))")
            print("  Imag: \\(Array(imagBuffer))")

            // FFT実行
            vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

            print("\\n🔄 FFT後のSplit Complex:")
            print("  Real: \\(Array(realBuffer))")
            print("  Imag: \\(Array(imagBuffer))")
        }
    }

    print("\\n" + "=" * 80)
}

private func *(string: String, count: Int) -> String {
    return String(repeating: string, count: count)
}
