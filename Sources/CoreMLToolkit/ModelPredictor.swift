import Foundation
import CoreML
import CoreImage
import AppKit

/// Compute device options for model execution
public enum ComputeDevice: String, CaseIterable {
    case cpu = "cpu"
    case gpu = "gpu"
    case ane = "ane"  // Apple Neural Engine
    case all = "all"  // Let Core ML decide

    public var mlComputeUnits: MLComputeUnits {
        switch self {
        case .cpu: return .cpuOnly
        case .gpu: return .cpuAndGPU
        case .ane: return .cpuAndNeuralEngine
        case .all: return .all
        }
    }
}

/// Result of a prediction
public struct PredictionResult: Codable {
    public let inputFile: String
    public let outputs: [String: PredictionValue]
    public let inferenceTimeMs: Double

    public init(inputFile: String, outputs: [String: PredictionValue], inferenceTimeMs: Double) {
        self.inputFile = inputFile
        self.outputs = outputs
        self.inferenceTimeMs = inferenceTimeMs
    }
}

/// Represents different prediction value types
public enum PredictionValue: Codable {
    case string(String)
    case double(Double)
    case int(Int64)
    case array([Double])
    case dictionary([String: Double])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .string(str)
        } else if let int = try? container.decode(Int64.self) {
            // Check if it's a whole number (no fractional part)
            self = .int(int)
        } else if let dbl = try? container.decode(Double.self) {
            self = .double(dbl)
        } else if let arr = try? container.decode([Double].self) {
            self = .array(arr)
        } else if let dict = try? container.decode([String: Double].self) {
            self = .dictionary(dict)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unknown prediction value type")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .double(let d): try container.encode(d)
        case .int(let i): try container.encode(i)
        case .array(let a): try container.encode(a)
        case .dictionary(let d): try container.encode(d)
        }
    }

    public var description: String {
        switch self {
        case .string(let s): return s
        case .double(let d): return String(format: "%.6f", d)
        case .int(let i): return "\(i)"
        case .array(let a): return "[\(a.prefix(5).map { String(format: "%.4f", $0) }.joined(separator: ", "))\(a.count > 5 ? ", ..." : "")]"
        case .dictionary(let d):
            let sorted = d.sorted { $0.value > $1.value }.prefix(5)
            return sorted.map { "\($0.key): \(String(format: "%.4f", $0.value))" }.joined(separator: ", ")
        }
    }
}

/// Handles model predictions
public class ModelPredictor {
    private var model: MLModel?
    private var modelDescription: MLModelDescription?
    private let device: ComputeDevice

    public init(device: ComputeDevice = .all) {
        self.device = device
    }

    /// Load a model for predictions
    public func loadModel(at path: String) throws {
        let url = URL(fileURLWithPath: path)

        guard FileManager.default.fileExists(atPath: path) else {
            throw PredictorError.modelNotFound(path: path)
        }

        let config = MLModelConfiguration()
        config.computeUnits = device.mlComputeUnits

        let modelURL: URL
        if url.pathExtension == "mlmodelc" {
            modelURL = url
        } else {
            modelURL = try MLModel.compileModel(at: url)
        }

        model = try MLModel(contentsOf: modelURL, configuration: config)
        modelDescription = model?.modelDescription
    }

    /// Run prediction on input data
    public func predict(inputPath: String) throws -> PredictionResult {
        guard let model = model, let description = modelDescription else {
            throw PredictorError.modelNotLoaded
        }

        let inputURL = URL(fileURLWithPath: inputPath)
        guard FileManager.default.fileExists(atPath: inputPath) else {
            throw PredictorError.inputNotFound(path: inputPath)
        }

        let featureProvider = try createFeatureProvider(from: inputURL, description: description)

        let startTime = CFAbsoluteTimeGetCurrent()
        let prediction = try model.prediction(from: featureProvider)
        let endTime = CFAbsoluteTimeGetCurrent()
        let inferenceTimeMs = (endTime - startTime) * 1000

        let outputs = extractOutputs(from: prediction, description: description)

        return PredictionResult(
            inputFile: inputURL.lastPathComponent,
            outputs: outputs,
            inferenceTimeMs: inferenceTimeMs
        )
    }

    /// Run predictions on multiple input files
    public func batchPredict(inputPaths: [String], concurrency: Int = 4) async throws -> [PredictionResult] {
        guard model != nil else {
            throw PredictorError.modelNotLoaded
        }

        return try await withThrowingTaskGroup(of: PredictionResult.self) { group in
            var results: [PredictionResult] = []
            var pendingPaths = inputPaths[...]

            // Add initial batch of tasks
            for _ in 0..<min(concurrency, inputPaths.count) {
                if let path = pendingPaths.popFirst() {
                    group.addTask {
                        try self.predict(inputPath: path)
                    }
                }
            }

            // Process results and add more tasks as they complete
            for try await result in group {
                results.append(result)
                if let path = pendingPaths.popFirst() {
                    group.addTask {
                        try self.predict(inputPath: path)
                    }
                }
            }

            return results
        }
    }

    private func createFeatureProvider(from url: URL, description: MLModelDescription) throws -> MLFeatureProvider {
        let inputDescriptions = description.inputDescriptionsByName

        var features: [String: MLFeatureValue] = [:]

        for (name, inputDesc) in inputDescriptions {
            let featureValue: MLFeatureValue

            switch inputDesc.type {
            case .image:
                guard let constraint = inputDesc.imageConstraint else {
                    throw PredictorError.missingImageConstraint
                }
                featureValue = try createImageFeature(from: url, constraint: constraint)

            case .multiArray:
                featureValue = try createMultiArrayFeature(from: url, description: inputDesc)

            case .string:
                let content = try String(contentsOf: url, encoding: .utf8)
                featureValue = MLFeatureValue(string: content.trimmingCharacters(in: .whitespacesAndNewlines))

            case .double:
                let content = try String(contentsOf: url, encoding: .utf8)
                guard let value = Double(content.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                    throw PredictorError.invalidInputFormat
                }
                featureValue = MLFeatureValue(double: value)

            case .int64:
                let content = try String(contentsOf: url, encoding: .utf8)
                guard let value = Int64(content.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                    throw PredictorError.invalidInputFormat
                }
                featureValue = MLFeatureValue(int64: value)

            default:
                throw PredictorError.unsupportedInputType(String(describing: inputDesc.type))
            }

            features[name] = featureValue
        }

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    private func createImageFeature(from url: URL, constraint: MLImageConstraint) throws -> MLFeatureValue {
        guard let nsImage = NSImage(contentsOf: url) else {
            throw PredictorError.invalidImage(path: url.path)
        }

        guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw PredictorError.invalidImage(path: url.path)
        }

        let ciImage = CIImage(cgImage: cgImage)
        let context = CIContext()

        // Resize image to match constraint
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
            throw PredictorError.pixelBufferCreationFailed
        }

        context.render(scaledImage, to: buffer)

        return MLFeatureValue(pixelBuffer: buffer)
    }

    private func createMultiArrayFeature(from url: URL, description: MLFeatureDescription) throws -> MLFeatureValue {
        let data = try Data(contentsOf: url)

        // Try to parse as JSON array
        if let jsonArray = try? JSONSerialization.jsonObject(with: data) as? [Any] {
            guard let constraint = description.multiArrayConstraint else {
                throw PredictorError.missingMultiArrayConstraint
            }

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

        throw PredictorError.invalidInputFormat
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

    private func extractOutputs(from prediction: MLFeatureProvider, description: MLModelDescription) -> [String: PredictionValue] {
        var outputs: [String: PredictionValue] = [:]

        for name in prediction.featureNames {
            guard let featureValue = prediction.featureValue(for: name) else { continue }

            switch featureValue.type {
            case .string:
                outputs[name] = .string(featureValue.stringValue)

            case .double:
                outputs[name] = .double(featureValue.doubleValue)

            case .int64:
                outputs[name] = .int(featureValue.int64Value)

            case .multiArray:
                if let multiArray = featureValue.multiArrayValue {
                    var values: [Double] = []
                    let count = min(multiArray.count, 100) // Limit output size
                    for i in 0..<count {
                        values.append(multiArray[i].doubleValue)
                    }
                    outputs[name] = .array(values)
                }

            case .dictionary:
                if let dict = featureValue.dictionaryValue as? [String: NSNumber] {
                    let doubleDict = dict.mapValues { $0.doubleValue }
                    outputs[name] = .dictionary(doubleDict)
                }

            default:
                outputs[name] = .string("<unsupported type>")
            }
        }

        return outputs
    }
}

public enum PredictorError: Error, LocalizedError {
    case modelNotFound(path: String)
    case modelNotLoaded
    case inputNotFound(path: String)
    case invalidImage(path: String)
    case missingImageConstraint
    case missingMultiArrayConstraint
    case pixelBufferCreationFailed
    case unsupportedInputType(String)
    case invalidInputFormat

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "Model not found at: \(path)"
        case .modelNotLoaded:
            return "Model not loaded. Call loadModel() first."
        case .inputNotFound(let path):
            return "Input file not found at: \(path)"
        case .invalidImage(let path):
            return "Invalid image file: \(path)"
        case .missingImageConstraint:
            return "Model input requires image but no constraint specified"
        case .missingMultiArrayConstraint:
            return "Model input requires multiArray but no constraint specified"
        case .pixelBufferCreationFailed:
            return "Failed to create pixel buffer for image"
        case .unsupportedInputType(let type):
            return "Unsupported input type: \(type)"
        case .invalidInputFormat:
            return "Invalid input format"
        }
    }
}
