import Foundation
import CoreML
import AVFoundation
import Accelerate

// v2実装: 複素数DFT（vDSP_DFT）の基本テスト
print("================================================================================")
print("🧪 V2 Implementation: Complex DFT (vDSP_DFT) Tests")
print("================================================================================")
testComplexDFT()
print("\n✅ Complex DFT tests completed")

// Python librosa との1フレーム比較
testDFTComparison()

// ISTFT debug
debugISTFTDetails()

// STFTProcessorV2のround-tripテスト
testSTFTV2RoundTrip()

// STFTProcessorV2とPython librosaの比較
compareSTFTV2WithPython()

print("")  // 空行を追加して見やすく

// 本格的な音源分離テストを実行
print("\n" + String(repeating: "=", count: 80))
print("🎵 Main Test: Vocal Separation")
print(String(repeating: "=", count: 80))
do {
    try runProperSeparationTest()
} catch {
    print("❌ エラー: \(error.localizedDescription)")
    if let nsError = error as NSError? {
        print("詳細: \(nsError.debugDescription)")
    }
    exit(1)
}
