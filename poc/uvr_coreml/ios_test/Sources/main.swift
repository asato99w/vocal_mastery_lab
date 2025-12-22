import Foundation
import CoreML
import AVFoundation
import Accelerate

// MDXNet ボーカル分離テスト
do {
    try runMDXNetTest()
} catch {
    print("❌ エラー: \(error.localizedDescription)")
    if let nsError = error as NSError? {
        print("詳細: \(nsError.debugDescription)")
    }
    exit(1)
}
