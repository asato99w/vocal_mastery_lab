// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VocalSeparator",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    targets: [
        .executableTarget(
            name: "VocalSeparator",
            path: "Sources"
        )
    ]
)
