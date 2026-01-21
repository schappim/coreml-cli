import ArgumentParser
import Foundation
import CoreMLToolkit

struct Meta: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "meta",
        abstract: "View and manage model metadata",
        subcommands: [Get.self, Set.self]
    )

    struct Get: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Get metadata from a model"
        )

        @Argument(help: "Path to the Core ML model")
        var modelPath: String

        @Flag(name: .shortAndLong, help: "Output in JSON format")
        var json: Bool = false

        func run() throws {
            let manager = MetadataManager()
            let metadata = try manager.getMetadata(modelPath: modelPath)

            if json {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let data = try encoder.encode(metadata)
                if let jsonString = String(data: data, encoding: .utf8) {
                    print(jsonString)
                }
            } else {
                print("Metadata for: \(modelPath)")
                print()

                if let author = metadata.author {
                    print("  Author:      \(author)")
                }
                if let description = metadata.description {
                    print("  Description: \(description)")
                }
                if let license = metadata.license {
                    print("  License:     \(license)")
                }
                if let version = metadata.version {
                    print("  Version:     \(version)")
                }

                if !metadata.additionalInfo.isEmpty {
                    print()
                    print("Additional Info:")
                    for (key, value) in metadata.additionalInfo.sorted(by: { $0.key < $1.key }) {
                        print("  \(key): \(value)")
                    }
                }

                if metadata.author == nil && metadata.description == nil &&
                   metadata.license == nil && metadata.version == nil &&
                   metadata.additionalInfo.isEmpty {
                    print("  (no metadata found)")
                }
            }
        }
    }

    struct Set: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Set metadata on a model (limited support)"
        )

        @Argument(help: "Metadata field: author, description, license, or version")
        var field: String

        @Argument(help: "Value to set")
        var value: String

        @Argument(help: "Path to the Core ML model")
        var modelPath: String

        @Option(name: .shortAndLong, help: "Output path for modified model")
        var output: String?

        func run() throws {
            guard let metaField = MetadataField(rawValue: field.lowercased()) else {
                throw ValidationError("Invalid field '\(field)'. Use: author, description, license, or version")
            }

            let manager = MetadataManager()

            do {
                let result = try manager.setMetadata(
                    modelPath: modelPath,
                    field: metaField,
                    value: value,
                    outputPath: output
                )
                print("Metadata updated: \(result)")
            } catch MetadataError.notImplemented(let reason) {
                print("Note: \(reason)")
                print()
                print("To modify metadata, you can use Python coremltools:")
                print()
                print("  import coremltools as ct")
                print("  model = ct.models.MLModel('\(modelPath)')")
                print("  model.\(field) = '\(value)'")
                print("  model.save('\(output ?? modelPath)')")
            }
        }
    }
}
