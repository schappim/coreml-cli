import ArgumentParser
import Foundation
import CoreMLToolkit

struct Compile: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "compile",
        abstract: "Compile a Core ML model to optimized format"
    )

    @Argument(help: "Path to the Core ML model (.mlmodel or .mlpackage)")
    var modelPath: String

    @Option(name: .shortAndLong, help: "Output directory for compiled model")
    var outputDir: String?

    @Flag(name: .shortAndLong, help: "Validate the model after compilation")
    var validate: Bool = false

    @Flag(name: .shortAndLong, help: "Output in JSON format")
    var json: Bool = false

    func run() throws {
        let compiler = ModelCompiler()

        if !json {
            print("Compiling model: \(modelPath)")
        }

        let result = try compiler.compile(modelPath: modelPath, outputDirectory: outputDir)

        if validate {
            let isValid = try compiler.validate(modelPath: result.outputPath)
            if !isValid {
                throw ValidationError("Compiled model validation failed")
            }
        }

        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(result)
            if let jsonString = String(data: data, encoding: .utf8) {
                print(jsonString)
            }
        } else {
            print()
            print("Compilation successful!")
            print("  Source: \(result.sourcePath)")
            print("  Output: \(result.outputPath)")
            print("  Original size: \(formatBytes(result.originalSize))")
            print("  Compiled size: \(formatBytes(result.compiledSize))")

            if validate {
                print("  Validation: Passed")
            }
        }
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}
