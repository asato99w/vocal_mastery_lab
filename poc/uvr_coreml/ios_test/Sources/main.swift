import Foundation
import CoreML
import AVFoundation
import Accelerate

// MDXNet Voc_FT ボーカル分離テスト
do {
    try runMDXNetVocFTTest()
} catch {
    print("❌ エラー: \(error.localizedDescription)")
    if let nsError = error as NSError? {
        print("詳細: \(nsError.debugDescription)")
    }
    exit(1)
}
