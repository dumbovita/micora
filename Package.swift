// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Micora",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "MicoraCore", targets: ["MicoraCore"]),
        .executable(name: "micora-app", targets: ["MicoraApp"]),
        .executable(name: "micora-cli", targets: ["MicoraCli"]),
        .executable(name: "micora-tests", targets: ["MicoraTests"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "MicoraCore",
            dependencies: [],
            path: "Sources/MicoraCore"
        ),
        .executableTarget(
            name: "MicoraApp",
            dependencies: ["MicoraCore"],
            path: "Sources/MicoraApp"
        ),
        .executableTarget(
            name: "MicoraCli",
            dependencies: ["MicoraCore"],
            path: "Sources/MicoraCli"
        ),
        .executableTarget(
            name: "MicoraTests",
            dependencies: ["MicoraCore"],
            path: "Tests/MicoraTests"
        )
    ]
)
