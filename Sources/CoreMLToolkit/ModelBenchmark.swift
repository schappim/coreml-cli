import Foundation
import CoreML
import CoreImage
import AppKit

/// Benchmark results for a model
public struct BenchmarkResult: Codable {
    public let modelName: String
    public let device: String
    public let iterations: Int
    public let warmupIterations: Int
    public let meanLatencyMs: Double
    public let minLatencyMs: Double
    public let maxLatencyMs: Double
    public let stdDevMs: Double
    public let p50LatencyMs: Double
    public let p95LatencyMs: Double
    public let p99LatencyMs: Double
    public let throughputPerSecond: Double

    public init(modelName: String, device: String, iterations: Int, warmupIterations: Int,
                meanLatencyMs: Double, minLatencyMs: Double, maxLatencyMs: Double,
                stdDevMs: Double, p50LatencyMs: Double, p95LatencyMs: Double,
                p99LatencyMs: Double, throughputPerSecond: Double) {
        self.modelName = modelName
        self.device = device
        self.iterations = iterations
        self.warmupIterations = warmupIterations
        self.meanLatencyMs = meanLatencyMs
        self.minLatencyMs = minLatencyMs
        self.maxLatencyMs = maxLatencyMs
        self.stdDevMs = stdDevMs
        self.p50LatencyMs = p50LatencyMs
        self.p95LatencyMs = p95LatencyMs
        self.p99LatencyMs = p99LatencyMs
        self.throughputPerSecond = throughputPerSecond
    }
}

/// Benchmarks Core ML model performance
public class ModelBenchmark {
    private let device: ComputeDevice

    public init(device: ComputeDevice = .all) {
        self.device = device
    }

    /// Benchmark a model with the given input
    public func benchmark(
        modelPath: String,
        inputPath: String,
        iterations: Int = 100,
        warmupIterations: Int = 10
    ) throws -> BenchmarkResult {
        let modelURL = URL(fileURLWithPath: modelPath)
        let inputURL = URL(fileURLWithPath: inputPath)

        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw BenchmarkError.modelNotFound(path: modelPath)
        }

        guard FileManager.default.fileExists(atPath: inputPath) else {
            throw BenchmarkError.inputNotFound(path: inputPath)
        }

        let config = MLModelConfiguration()
        config.computeUnits = device.mlComputeUnits

        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
        } else {
            compiledURL = try MLModel.compileModel(at: modelURL)
        }

        let model = try MLModel(contentsOf: compiledURL, configuration: config)
        let modelDescription = model.modelDescription

        // Create input feature provider
        let featureProvider = try createFeatureProvider(from: inputURL, description: modelDescription)

        // Warmup runs
        for _ in 0..<warmupIterations {
            _ = try model.prediction(from: featureProvider)
        }

        // Benchmark runs
        var latencies: [Double] = []
        latencies.reserveCapacity(iterations)

        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try model.prediction(from: featureProvider)
            let end = CFAbsoluteTimeGetCurrent()
            latencies.append((end - start) * 1000)  // Convert to ms
        }

        // Calculate statistics
        let sortedLatencies = latencies.sorted()
        let mean = latencies.reduce(0, +) / Double(latencies.count)
        let minLatency = sortedLatencies.first ?? 0
        let maxLatency = sortedLatencies.last ?? 0

        let variance = latencies.map { pow($0 - mean, 2) }.reduce(0, +) / Double(latencies.count)
        let stdDev = sqrt(variance)

        let p50 = percentile(sortedLatencies, 0.50)
        let p95 = percentile(sortedLatencies, 0.95)
        let p99 = percentile(sortedLatencies, 0.99)

        let throughput = 1000.0 / mean  // Inferences per second

        let modelName = modelURL.deletingPathExtension().lastPathComponent

        return BenchmarkResult(
            modelName: modelName,
            device: device.rawValue,
            iterations: iterations,
            warmupIterations: warmupIterations,
            meanLatencyMs: mean,
            minLatencyMs: minLatency,
            maxLatencyMs: maxLatency,
            stdDevMs: stdDev,
            p50LatencyMs: p50,
            p95LatencyMs: p95,
            p99LatencyMs: p99,
            throughputPerSecond: throughput
        )
    }

    private func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let index = Int(Double(sorted.count - 1) * p)
        return sorted[index]
    }

    private func createFeatureProvider(from url: URL, description: MLModelDescription) throws -> MLFeatureProvider {
        // Create a feature provider from the input file
        let inputDescriptions = description.inputDescriptionsByName
        var features: [String: MLFeatureValue] = [:]

        for (name, inputDesc) in inputDescriptions {
            let featureValue: MLFeatureValue

            switch inputDesc.type {
            case .image:
                guard let constraint = inputDesc.imageConstraint else {
                    throw BenchmarkError.missingConstraint
                }
                featureValue = try createImageFeature(from: url, constraint: constraint)

            case .multiArray:
                guard let constraint = inputDesc.multiArrayConstraint else {
                    throw BenchmarkError.missingConstraint
                }
                featureValue = try createMultiArrayFeature(from: url, constraint: constraint)

            case .string:
                let content = try String(contentsOf: url, encoding: .utf8)
                featureValue = MLFeatureValue(string: content.trimmingCharacters(in: .whitespacesAndNewlines))

            case .double:
                let content = try String(contentsOf: url, encoding: .utf8)
                guard let value = Double(content.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                    throw BenchmarkError.invalidInputFormat
                }
                featureValue = MLFeatureValue(double: value)

            case .int64:
                let content = try String(contentsOf: url, encoding: .utf8)
                guard let value = Int64(content.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                    throw BenchmarkError.invalidInputFormat
                }
                featureValue = MLFeatureValue(int64: value)

            default:
                throw BenchmarkError.unsupportedInputType
            }

            features[name] = featureValue
        }

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    private func createImageFeature(from url: URL, constraint: MLImageConstraint) throws -> MLFeatureValue {
        guard let nsImage = NSImage(contentsOf: url) else {
            throw BenchmarkError.invalidImage
        }

        guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw BenchmarkError.invalidImage
        }

        let ciImage = CIImage(cgImage: cgImage)
        let context = CIContext()

        let scaleX = CGFloat(constraint.pixelsWide) / CGFloat(cgImage.width)
        let scaleY = CGFloat(constraint.pixelsHigh) / CGFloat(cgImage.height)
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        var pixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            constraint.pixelsWide,
            constraint.pixelsHigh,
            constraint.pixelFormatType,
            attrs,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw BenchmarkError.pixelBufferCreationFailed
        }

        context.render(scaledImage, to: buffer)

        return MLFeatureValue(pixelBuffer: buffer)
    }

    private func createMultiArrayFeature(from url: URL, constraint: MLMultiArrayConstraint) throws -> MLFeatureValue {
        let data = try Data(contentsOf: url)

        if let jsonArray = try? JSONSerialization.jsonObject(with: data) as? [Any] {
            let shape = constraint.shape.map { $0.intValue }
            let multiArray = try MLMultiArray(shape: shape as [NSNumber], dataType: constraint.dataType)

            let flatValues = flattenArray(jsonArray)
            for (index, value) in flatValues.enumerated() {
                if index < multiArray.count {
                    multiArray[index] = NSNumber(value: value)
                }
            }

            return MLFeatureValue(multiArray: multiArray)
        }

        throw BenchmarkError.invalidInputFormat
    }

    private func flattenArray(_ array: [Any]) -> [Double] {
        var result: [Double] = []
        for element in array {
            if let num = element as? NSNumber {
                result.append(num.doubleValue)
            } else if let nested = element as? [Any] {
                result.append(contentsOf: flattenArray(nested))
            }
        }
        return result
    }
}

public enum BenchmarkError: Error, LocalizedError {
    case modelNotFound(path: String)
    case inputNotFound(path: String)
    case missingConstraint
    case invalidInputFormat
    case unsupportedInputType
    case invalidImage
    case pixelBufferCreationFailed

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "Model not found at: \(path)"
        case .inputNotFound(let path):
            return "Input not found at: \(path)"
        case .missingConstraint:
            return "Missing input constraint"
        case .invalidInputFormat:
            return "Invalid input format"
        case .unsupportedInputType:
            return "Unsupported input type"
        case .invalidImage:
            return "Invalid image file"
        case .pixelBufferCreationFailed:
            return "Failed to create pixel buffer"
        }
    }
}
