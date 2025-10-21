import Foundation
import CoreML
import AVFoundation
import Accelerate

// デバッグ: STFT/iSTFT 中間データ比較
print(String(repeating: "=", count: 80))
print("🔍 Debug Mode: STFT/iSTFT Intermediate Data Comparison")
print(String(repeating: "=", count: 80))
debugSTFTComparison()

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
