import ArgumentParser
import Foundation
import CoreMLToolkit

struct Inspect: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "inspect",
        abstract: "Inspect a Core ML model and display its structure"
    )

    @Argument(help: "Path to the Core ML model (.mlmodel, .mlpackage, or .mlmodelc)")
    var modelPath: String

    @Flag(name: .shortAndLong, help: "Output in JSON format")
    var json: Bool = false

    @Flag(name: .shortAndLong, help: "Show verbose output including all metadata")
    var verbose: Bool = false

    func run() throws {
        let inspector = ModelInspector()
        let info = try inspector.inspect(modelPath: modelPath)

        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(info)
            if let jsonString = String(data: data, encoding: .utf8) {
                print(jsonString)
            }
        } else {
            printHumanReadable(info)
        }
    }

    private func printHumanReadable(_ info: ModelInfo) {
        print("Model: \(info.name)")
        print("Size: \(formatBytes(info.fileSize))")
        print("Compiled: \(info.isCompiled ? "Yes" : "No")")
        print()

        print("Inputs:")
        for input in info.inputs {
            printFeature(input, indent: "  ")
        }
        print()

        print("Outputs:")
        for output in info.outputs {
            printFeature(output, indent: "  ")
        }

        if verbose || hasMetadata(info.metadata) {
            print()
            print("Metadata:")
            if let author = info.metadata.author {
                print("  Author: \(author)")
            }
            if let description = info.metadata.description {
                print("  Description: \(description)")
            }
            if let license = info.metadata.license {
                print("  License: \(license)")
            }
            if let version = info.metadata.version {
                print("  Version: \(version)")
            }
            if verbose {
                for (key, value) in info.metadata.additionalInfo {
                    print("  \(key): \(value)")
                }
            }
        }
    }

    private func printFeature(_ feature: FeatureInfo, indent: String) {
        var parts = ["\(indent)\(feature.name): \(feature.type)"]

        if let shape = feature.shape {
            parts.append("shape=[\(shape.map { String($0) }.joined(separator: ", "))]")
        }

        if let dataType = feature.multiArrayDataType {
            parts.append("dtype=\(dataType)")
        }

        if let img = feature.imageConstraint {
            parts.append("\(img.width)x\(img.height) \(img.pixelFormat)")
        }

        print(parts.joined(separator: " "))
    }

    private func hasMetadata(_ metadata: ModelMetadata) -> Bool {
        return metadata.author != nil ||
               metadata.description != nil ||
               metadata.license != nil ||
               metadata.version != nil
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}
