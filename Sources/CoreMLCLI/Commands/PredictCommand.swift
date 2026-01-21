import ArgumentParser
import Foundation
import CoreMLToolkit

struct Predict: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "predict",
        abstract: "Run inference on a Core ML model"
    )

    @Argument(help: "Path to the Core ML model")
    var modelPath: String

    @Option(name: .shortAndLong, help: "Path to input file (image, JSON, or text)")
    var input: String

    @Option(name: .shortAndLong, help: "Path to output file (optional)")
    var output: String?

    @Option(name: .shortAndLong, help: "Compute device: cpu, gpu, ane, or all")
    var device: String = "all"

    @Flag(name: .shortAndLong, help: "Output in JSON format")
    var json: Bool = false

    func run() throws {
        guard let computeDevice = ComputeDevice(rawValue: device) else {
            throw ValidationError("Invalid device '\(device)'. Use: cpu, gpu, ane, or all")
        }

        let predictor = ModelPredictor(device: computeDevice)
        try predictor.loadModel(at: modelPath)

        let result = try predictor.predict(inputPath: input)

        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(result)
            let jsonString = String(data: data, encoding: .utf8) ?? "{}"

            if let outputPath = output {
                try jsonString.write(toFile: outputPath, atomically: true, encoding: .utf8)
                print("Results written to: \(outputPath)")
            } else {
                print(jsonString)
            }
        } else {
            printHumanReadable(result)

            if let outputPath = output {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let data = try encoder.encode(result)
                try data.write(to: URL(fileURLWithPath: outputPath))
                print("\nResults written to: \(outputPath)")
            }
        }
    }

    private func printHumanReadable(_ result: PredictionResult) {
        print("Input: \(result.inputFile)")
        print("Inference time: \(String(format: "%.2f", result.inferenceTimeMs)) ms")
        print()
        print("Outputs:")
        for (name, value) in result.outputs.sorted(by: { $0.key < $1.key }) {
            print("  \(name): \(value.description)")
        }
    }
}
