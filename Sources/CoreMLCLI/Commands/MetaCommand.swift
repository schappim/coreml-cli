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
            abstract: "Set a metadata field on a model"
        )

        @Argument(help: "Path to the Core ML model (.mlmodel or .mlpackage)")
        var modelPath: String

        @Argument(help: "Field to set: author, description, license, or version")
        var field: String

        @Argument(help: "New value (pass an empty string to clear the field)")
        var value: String

        @Option(name: .shortAndLong, help: "Write the modified model to this path instead of overwriting the source")
        var output: String?

        func run() throws {
            guard let metaField = MetadataField(rawValue: field.lowercased()) else {
                throw ValidationError("Invalid field '\(field)'. Use: author, description, license, or version")
            }

            let manager = MetadataManager()
            let writtenPath = try manager.setMetadata(
                modelPath: modelPath,
                field: metaField,
                value: value,
                outputPath: output
            )

            if value.isEmpty {
                print("Cleared \(metaField.rawValue) in \(writtenPath)")
            } else {
                print("Set \(metaField.rawValue) = \"\(value)\" in \(writtenPath)")
            }
        }
    }
}
