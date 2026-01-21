// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "coreml-cli",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "coreml", targets: ["CoreMLCLI"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0")
    ],
    targets: [
        .executableTarget(
            name: "CoreMLCLI",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                "CoreMLToolkit"
            ]
        ),
        .target(
            name: "CoreMLToolkit",
            dependencies: []
        ),
        .testTarget(
            name: "CoreMLCLITests",
            dependencies: ["CoreMLToolkit", "CoreMLCLI"]
        )
    ]
)
