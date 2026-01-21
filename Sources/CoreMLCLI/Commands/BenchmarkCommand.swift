import ArgumentParser
import Foundation
import CoreMLToolkit

struct Benchmark: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "benchmark",
        abstract: "Benchmark model inference performance"
    )

    @Argument(help: "Path to the Core ML model")
    var modelPath: String

    @Option(name: .shortAndLong, help: "Path to sample input file")
    var input: String

    @Option(name: [.customShort("n"), .long], help: "Number of benchmark iterations")
    var iterations: Int = 100

    @Option(name: .long, help: "Number of warmup iterations")
    var warmup: Int = 10

    @Option(name: .shortAndLong, help: "Compute device: cpu, gpu, ane, or all")
    var device: String = "all"

    @Flag(name: .shortAndLong, help: "Output in JSON format")
    var json: Bool = false

    func run() throws {
        guard let computeDevice = ComputeDevice(rawValue: device) else {
            throw ValidationError("Invalid device '\(device)'. Use: cpu, gpu, ane, or all")
        }

        if !json {
            print("Benchmarking model: \(modelPath)")
            print("Device: \(device)")
            print("Warmup iterations: \(warmup)")
            print("Benchmark iterations: \(iterations)")
            print()
            print("Running warmup...")
        }

        let benchmark = ModelBenchmark(device: computeDevice)
        let result = try benchmark.benchmark(
            modelPath: modelPath,
            inputPath: input,
            iterations: iterations,
            warmupIterations: warmup
        )

        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(result)
            if let jsonString = String(data: data, encoding: .utf8) {
                print(jsonString)
            }
        } else {
            printHumanReadable(result)
        }
    }

    private func printHumanReadable(_ result: BenchmarkResult) {
        print("Benchmark Results for: \(result.modelName)")
        print("=" .repeating(50))
        print()
        print("Configuration:")
        print("  Device: \(result.device)")
        print("  Iterations: \(result.iterations)")
        print("  Warmup: \(result.warmupIterations)")
        print()
        print("Latency (ms):")
        print("  Mean:   \(String(format: "%8.3f", result.meanLatencyMs))")
        print("  Min:    \(String(format: "%8.3f", result.minLatencyMs))")
        print("  Max:    \(String(format: "%8.3f", result.maxLatencyMs))")
        print("  StdDev: \(String(format: "%8.3f", result.stdDevMs))")
        print()
        print("Percentiles (ms):")
        print("  P50:    \(String(format: "%8.3f", result.p50LatencyMs))")
        print("  P95:    \(String(format: "%8.3f", result.p95LatencyMs))")
        print("  P99:    \(String(format: "%8.3f", result.p99LatencyMs))")
        print()
        print("Throughput: \(String(format: "%.2f", result.throughputPerSecond)) inferences/sec")
    }
}

private extension String {
    func repeating(_ count: Int) -> String {
        return String(repeating: self, count: count)
    }
}
