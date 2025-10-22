import Foundation
import Accelerate

/// vDSP_DFT（離散フーリエ変換）の基本動作テスト
///
/// vDSP_DFTはApple推奨のDFTルーチン
/// 複素数入力・複素数出力で、データレイアウトが明確で標準的
/// numpy/librosaと互換性が高いはず
func testComplexDFT() {
    print("\n" + String(repeating: "=", count: 80))
    print("🧪 Complex DFT Test (vDSP_DFT) - v2 Implementation")
    print(String(repeating: "=", count: 80))

    let N = 8

    // DFTセットアップ作成（Forward）
    guard let dftSetupForward = vDSP_DFT_zop_CreateSetup(
        nil,  // 前のセットアップ（nil）
        vDSP_Length(N),  // FFT長
        vDSP_DFT_Direction.FORWARD  // Forward方向
    ) else {
        print("❌ DFT setup (Forward) failed")
        return
    }
    defer { vDSP_DFT_DestroySetup(dftSetupForward) }

    // DFTセットアップ作成（Inverse）
    guard let dftSetupInverse = vDSP_DFT_zop_CreateSetup(
        nil,
        vDSP_Length(N),
        vDSP_DFT_Direction.INVERSE
    ) else {
        print("❌ DFT setup (Inverse) failed")
        return
    }
    defer { vDSP_DFT_DestroySetup(dftSetupInverse) }

    // テスト1: DC成分（全て1）
    print("\n📊 Test 1: DC Component [1+0j, 1+0j, ...]")

    var realInput1 = [Float](repeating: 1.0, count: N)
    var imagInput1 = [Float](repeating: 0.0, count: N)
    var realOutput1 = [Float](repeating: 0.0, count: N)
    var imagOutput1 = [Float](repeating: 0.0, count: N)

    print("  Input:")
    print("    Real: \(realInput1)")
    print("    Imag: \(imagInput1)")

    // DFT実行
    vDSP_DFT_Execute(dftSetupForward, &realInput1, &imagInput1, &realOutput1, &imagOutput1)

    print("  After DFT:")
    print("    Real: \(realOutput1)")
    print("    Imag: \(imagOutput1)")

    // DC成分確認
    let dcMag = sqrtf(realOutput1[0] * realOutput1[0] + imagOutput1[0] * imagOutput1[0])
    print("  DC magnitude (bin 0): \(dcMag)")
    print("  Expected: \(Float(N))")

    // 他のビン
    print("  Other bins:")
    for i in 1..<N {
        let mag = sqrtf(realOutput1[i] * realOutput1[i] + imagOutput1[i] * imagOutput1[i])
        print("    Bin \(i): \(mag)")
    }

    // テスト2: 単一周波数
    print("\n📊 Test 2: Single Frequency (bin 2, 2 cycles)")

    var realInput2 = [Float](repeating: 0, count: N)
    var imagInput2 = [Float](repeating: 0, count: N)
    var realOutput2 = [Float](repeating: 0, count: N)
    var imagOutput2 = [Float](repeating: 0, count: N)

    // 2サイクルの正弦波
    for i in 0..<N {
        let t = Float(i)
        realInput2[i] = sin(2.0 * Float.pi * 2.0 * t / Float(N))
    }

    print("  Input: \(realInput2.map { String(format: "%.3f", $0) })")

    vDSP_DFT_Execute(dftSetupForward, &realInput2, &imagInput2, &realOutput2, &imagOutput2)

    print("  After DFT (magnitudes):")
    for i in 0..<N {
        let mag = sqrtf(realOutput2[i] * realOutput2[i] + imagOutput2[i] * imagOutput2[i])
        print("    Bin \(i): \(String(format: "%.4f", mag))")
    }

    // テスト3: 往復（Forward → Inverse）
    print("\n📊 Test 3: Round-trip (Forward → Inverse)")

    var realInput3 = [Float](repeating: 1.0, count: N)
    var imagInput3 = [Float](repeating: 0.0, count: N)
    var realIntermediate = [Float](repeating: 0, count: N)
    var imagIntermediate = [Float](repeating: 0, count: N)
    var realReconstructed = [Float](repeating: 0, count: N)
    var imagReconstructed = [Float](repeating: 0, count: N)

    // Forward DFT
    vDSP_DFT_Execute(dftSetupForward, &realInput3, &imagInput3, &realIntermediate, &imagIntermediate)

    print("  After Forward DFT:")
    print("    Real[0] = \(realIntermediate[0])")
    print("    Imag[0] = \(imagIntermediate[0])")

    // Inverse DFT
    vDSP_DFT_Execute(dftSetupInverse, &realIntermediate, &imagIntermediate, &realReconstructed, &imagReconstructed)

    // スケーリング（DFTは1/Nスケールが必要）
    var scale = Float(1.0) / Float(N)
    vDSP_vsmul(realReconstructed, 1, &scale, &realReconstructed, 1, vDSP_Length(N))
    vDSP_vsmul(imagReconstructed, 1, &scale, &imagReconstructed, 1, vDSP_Length(N))

    print("  After Inverse DFT (with 1/N scaling):")
    print("    Real: \(realReconstructed.map { String(format: "%.4f", $0) })")
    print("    Imag: \(imagReconstructed.map { String(format: "%.4f", $0) })")

    // 誤差計算
    var maxErrorReal: Float = 0
    var maxErrorImag: Float = 0
    for i in 0..<N {
        maxErrorReal = max(maxErrorReal, abs(realReconstructed[i] - realInput3[i]))
        maxErrorImag = max(maxErrorImag, abs(imagReconstructed[i] - imagInput3[i]))
    }

    print("  Reconstruction error:")
    print("    Real: \(maxErrorReal)")
    print("    Imag: \(maxErrorImag)")

    if maxErrorReal < 0.0001 && maxErrorImag < 0.0001 {
        print("  ✅ Perfect reconstruction!")
    } else {
        print("  ⚠️  Some error detected")
    }

    print("\n" + String(repeating: "=", count: 80))
}
