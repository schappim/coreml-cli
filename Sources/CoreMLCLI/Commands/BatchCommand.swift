import ArgumentParser
import Foundation
import CoreMLToolkit

struct Batch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "batch",
        abstract: "Run batch inference on multiple inputs"
    )

    @Argument(help: "Path to the Core ML model")
    var modelPath: String

    @Option(name: [.customShort("i"), .customLong("dir")], help: "Directory containing input files")
    var dir: String

    @Option(name: .shortAndLong, help: "Output directory for results")
    var out: String

    @Option(name: .shortAndLong, help: "Output format: json or csv")
    var format: String = "json"

    @Option(name: [.customShort("c"), .customLong("concurrency")], help: "Number of concurrent workers")
    var concurrency: Int = 4

    @Option(name: .long, help: "Compute device: cpu, gpu, ane, or all")
    var device: String = "all"

    @Flag(name: .shortAndLong, help: "Show verbose output")
    var verbose: Bool = false

    func run() async throws {
        guard let computeDevice = ComputeDevice(rawValue: device) else {
            throw ValidationError("Invalid device '\(device)'. Use: cpu, gpu, ane, or all")
        }

        guard format == "json" || format == "csv" else {
            throw ValidationError("Invalid format '\(format)'. Use: json or csv")
        }

        let fileManager = FileManager.default

        // Get all input files
        guard let enumerator = fileManager.enumerator(atPath: dir) else {
            throw ValidationError("Cannot read directory: \(dir)")
        }

        var inputPaths: [String] = []
        while let file = enumerator.nextObject() as? String {
            let fullPath = (dir as NSString).appendingPathComponent(file)
            var isDir: ObjCBool = false
            if fileManager.fileExists(atPath: fullPath, isDirectory: &isDir) && !isDir.boolValue {
                // Filter for common input types
                let ext = (file as NSString).pathExtension.lowercased()
                if ["jpg", "jpeg", "png", "heic", "json", "txt", "wav"].contains(ext) {
                    inputPaths.append(fullPath)
                }
            }
        }

        if inputPaths.isEmpty {
            print("No input files found in: \(dir)")
            return
        }

        print("Found \(inputPaths.count) input files")

        // Create output directory
        if !fileManager.fileExists(atPath: out) {
            try fileManager.createDirectory(atPath: out, withIntermediateDirectories: true)
        }

        let predictor = ModelPredictor(device: computeDevice)
        try predictor.loadModel(at: modelPath)

        let startTime = CFAbsoluteTimeGetCurrent()

        let results = try await predictor.batchPredict(inputPaths: inputPaths, concurrency: concurrency)

        let endTime = CFAbsoluteTimeGetCurrent()
        let totalTime = (endTime - startTime) * 1000

        // Write results
        if format == "json" {
            let outputPath = (out as NSString).appendingPathComponent("results.json")
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(results)
            try data.write(to: URL(fileURLWithPath: outputPath))
            print("Results written to: \(outputPath)")
        } else {
            let outputPath = (out as NSString).appendingPathComponent("results.csv")
            try writeCSV(results: results, to: outputPath)
            print("Results written to: \(outputPath)")
        }

        print()
        print("Processed \(results.count) files in \(String(format: "%.2f", totalTime)) ms")
        print("Average inference time: \(String(format: "%.2f", results.map { $0.inferenceTimeMs }.reduce(0, +) / Double(results.count))) ms")
    }

    private func writeCSV(results: [PredictionResult], to path: String) throws {
        var csv = "input_file,inference_time_ms"

        // Get all unique output keys
        var allKeys: Set<String> = []
        for result in results {
            allKeys.formUnion(result.outputs.keys)
        }
        let sortedKeys = allKeys.sorted()

        for key in sortedKeys {
            csv += ",\(key)"
        }
        csv += "\n"

        for result in results {
            csv += "\(result.inputFile),\(String(format: "%.2f", result.inferenceTimeMs))"
            for key in sortedKeys {
                if let value = result.outputs[key] {
                    csv += ",\"\(value.description)\""
                } else {
                    csv += ","
                }
            }
            csv += "\n"
        }

        try csv.write(toFile: path, atomically: true, encoding: .utf8)
    }
}
