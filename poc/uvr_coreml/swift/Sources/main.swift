import Foundation

/// メイン処理
func main() {
    print("=" * 60)
    print("ボーカル分離 (Swift/CoreML)")
    print("=" * 60)

    let args = CommandLine.arguments
    let baseDir = URL(fileURLWithPath: #file)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    let modelURL = baseDir
        .appendingPathComponent("models/coreml/UVR-MDX-NET-Voc_FT.mlpackage")

    // コマンドライン引数でバッチ処理
    if args.count >= 3 {
        let inputDirPath = args[1]
        let outputDirPath = args[2]

        let inputDir = inputDirPath.hasPrefix("/")
            ? URL(fileURLWithPath: inputDirPath)
            : baseDir.appendingPathComponent(inputDirPath)
        let outputDir = outputDirPath.hasPrefix("/")
            ? URL(fileURLWithPath: outputDirPath)
            : baseDir.appendingPathComponent(outputDirPath)

        print("入力: \(inputDir.path)")
        print("出力: \(outputDir.path)")

        do {
            print("\nモデル読み込み中...")
            let separator = try VocalSeparator(modelURL: modelURL)

            // サンプルディレクトリ一覧
            let fm = FileManager.default
            let contents = try fm.contentsOfDirectory(at: inputDir, includingPropertiesForKeys: nil)
            let sampleDirs = contents.filter { url in
                var isDir: ObjCBool = false
                return fm.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
                    && fm.fileExists(atPath: url.appendingPathComponent("mix.wav").path)
            }.sorted { $0.lastPathComponent < $1.lastPathComponent }

            print("サンプル数: \(sampleDirs.count)")
            print("-" * 60)

            for sampleDir in sampleDirs {
                let sampleName = sampleDir.lastPathComponent
                let mixPath = sampleDir.appendingPathComponent("mix.wav")
                let sampleOutput = outputDir.appendingPathComponent(sampleName)

                try fm.createDirectory(at: sampleOutput, withIntermediateDirectories: true)

                print("\n\(sampleName)")
                let (vocals, instrumental) = try separator.separate(audioURL: mixPath)

                try separator.saveAudio(vocals, to: sampleOutput.appendingPathComponent("vocals.wav"))
                try separator.saveAudio(instrumental, to: sampleOutput.appendingPathComponent("instrumental.wav"))
            }

            print("\n完了: \(outputDir.path)")
        } catch {
            print("エラー: \(error)")
        }
    } else {
        // 単一ファイル処理（デフォルト）
        let inputURL = baseDir.appendingPathComponent("test_audio/hollow_crown/mix.wav")
        let outputDir = baseDir.appendingPathComponent("output/coreml")

        try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        do {
            print("モデル読み込み中...")
            let separator = try VocalSeparator(modelURL: modelURL)

            print("分離実行中...")
            let (vocals, instrumental) = try separator.separate(audioURL: inputURL)

            print("保存中...")
            try separator.saveAudio(vocals, to: outputDir.appendingPathComponent("vocals.wav"))
            try separator.saveAudio(instrumental, to: outputDir.appendingPathComponent("instrumental.wav"))

            let vocalsRMS = sqrt(vocals.map { $0 * $0 }.reduce(0, +) / Float(vocals.count))
            let instRMS = sqrt(instrumental.map { $0 * $0 }.reduce(0, +) / Float(instrumental.count))

            print("\n出力:")
            print("  Vocals: \(outputDir.appendingPathComponent("vocals.wav").path)")
            print("  Instrumental: \(outputDir.appendingPathComponent("instrumental.wav").path)")
            print("\n統計:")
            print("  Vocals RMS: \(String(format: "%.6f", vocalsRMS))")
            print("  Instrumental RMS: \(String(format: "%.6f", instRMS))")
        } catch {
            print("エラー: \(error)")
        }
    }
}

extension String {
    static func *(lhs: String, rhs: Int) -> String {
        return String(repeating: lhs, count: rhs)
    }
}

main()
