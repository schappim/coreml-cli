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

    /// Whether a model has been loaded and is ready for predictions
    public var isLoaded: Bool {
        model != nil
    }

    /// Names of the model's inputs, sorted. Empty when no model is loaded.
    public var inputNames: [String] {
        (modelDescription?.inputDescriptionsByName.keys).map { $0.sorted() } ?? []
    }

    /// Names of the model's outputs, sorted. Empty when no model is loaded.
    public var outputNames: [String] {
        (modelDescription?.outputDescriptionsByName.keys).map { $0.sorted() } ?? []
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

    /// Run prediction on an input file
    public func predict(inputPath: String) throws -> PredictionResult {
        guard isLoaded else {
            throw PredictorError.modelNotLoaded
        }
        return try predict(source: try InputSource.file(path: inputPath))
    }

    /// Run prediction on an in-memory input.
    ///
    /// The same bytes are used for every model input, matching the behaviour of
    /// file-based prediction. Use `predict(namedValues:label:)` to give each
    /// input of a multi-input model its own value.
    public func predict(source: InputSource) throws -> PredictionResult {
        guard let model = model, let description = modelDescription else {
            throw PredictorError.modelNotLoaded
        }

        let featureProvider = try createFeatureProvider(from: source, description: description)
        return try run(model: model, description: description, features: featureProvider, label: source.name)
    }

    /// Run prediction with one JSON value per named model input.
    ///
    /// Values follow the model's declared input types: images accept a base64
    /// string, multi-arrays accept a (possibly nested) numeric array, and
    /// string/double/int64 inputs accept their natural JSON counterparts.
    public func predict(namedValues: [String: Any], label: String = "request") throws -> PredictionResult {
        guard let model = model, let description = modelDescription else {
            throw PredictorError.modelNotLoaded
        }

        let featureProvider = try createFeatureProvider(namedValues: namedValues, description: description)
        return try run(model: model, description: description, features: featureProvider, label: label)
    }

    private func run(
        model: MLModel,
        description: MLModelDescription,
        features: MLFeatureProvider,
        label: String
    ) throws -> PredictionResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        let prediction = try model.prediction(from: features)
        let endTime = CFAbsoluteTimeGetCurrent()
        let inferenceTimeMs = (endTime - startTime) * 1000

        return PredictionResult(
            inputFile: label,
            outputs: extractOutputs(from: prediction, description: description),
            inferenceTimeMs: inferenceTimeMs
        )
    }

    /// Run predictions on multiple input files
    public func batchPredict(inputPaths: [String], concurrency: Int = 4) async throws -> [PredictionResult] {
        guard model != nil else {
            throw PredictorError.modelNotLoaded
        }

        let workers = max(1, concurrency)

        return try await withThrowingTaskGroup(of: PredictionResult.self) { group in
            var results: [PredictionResult] = []
            var pendingPaths = inputPaths[...]

            // Add initial batch of tasks
            for _ in 0..<min(workers, inputPaths.count) {
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

    // MARK: - Feature building

    private func createFeatureProvider(from source: InputSource, description: MLModelDescription) throws -> MLFeatureProvider {
        let inputDescriptions = description.inputDescriptionsByName

        var features: [String: MLFeatureValue] = [:]

        for (name, inputDesc) in inputDescriptions {
            features[name] = try featureValue(from: source, description: inputDesc)
        }

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    private func featureValue(from source: InputSource, description inputDesc: MLFeatureDescription) throws -> MLFeatureValue {
        switch inputDesc.type {
        case .image:
            guard let constraint = inputDesc.imageConstraint else {
                throw PredictorError.missingImageConstraint
            }
            return try createImageFeature(from: source.data, label: source.name, constraint: constraint)

        case .multiArray:
            guard let constraint = inputDesc.multiArrayConstraint else {
                throw PredictorError.missingMultiArrayConstraint
            }
            guard let jsonArray = try? JSONSerialization.jsonObject(with: source.data) as? [Any] else {
                throw PredictorError.invalidInputFormat
            }
            return try createMultiArrayFeature(from: jsonArray, constraint: constraint)

        case .string:
            guard let content = source.text else {
                throw PredictorError.invalidInputFormat
            }
            return MLFeatureValue(string: content)

        case .double:
            guard let content = source.text, let value = Double(content) else {
                throw PredictorError.invalidInputFormat
            }
            return MLFeatureValue(double: value)

        case .int64:
            guard let content = source.text, let value = Int64(content) else {
                throw PredictorError.invalidInputFormat
            }
            return MLFeatureValue(int64: value)

        default:
            throw PredictorError.unsupportedInputType(String(describing: inputDesc.type))
        }
    }

    private func createFeatureProvider(namedValues: [String: Any], description: MLModelDescription) throws -> MLFeatureProvider {
        let inputDescriptions = description.inputDescriptionsByName
        let knownNames = inputDescriptions.keys.sorted()

        for name in namedValues.keys where inputDescriptions[name] == nil {
            throw PredictorError.unknownInput(name: name, known: knownNames)
        }

        var features: [String: MLFeatureValue] = [:]

        for (name, inputDesc) in inputDescriptions {
            guard let value = namedValues[name] else {
                throw PredictorError.missingInput(name: name, known: knownNames)
            }
            features[name] = try featureValue(fromJSON: value, name: name, description: inputDesc)
        }

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    private func featureValue(fromJSON value: Any, name: String, description inputDesc: MLFeatureDescription) throws -> MLFeatureValue {
        switch inputDesc.type {
        case .image:
            guard let constraint = inputDesc.imageConstraint else {
                throw PredictorError.missingImageConstraint
            }
            guard let encoded = value as? String, let data = Self.decodeBase64(encoded) else {
                throw PredictorError.invalidInputValue(name: name, expected: "a base64-encoded image")
            }
            return try createImageFeature(from: data, label: name, constraint: constraint)

        case .multiArray:
            guard let constraint = inputDesc.multiArrayConstraint else {
                throw PredictorError.missingMultiArrayConstraint
            }
            guard let array = value as? [Any] else {
                throw PredictorError.invalidInputValue(name: name, expected: "an array of numbers")
            }
            return try createMultiArrayFeature(from: array, constraint: constraint)

        case .string:
            guard let string = value as? String else {
                throw PredictorError.invalidInputValue(name: name, expected: "a string")
            }
            return MLFeatureValue(string: string)

        case .double:
            guard let number = value as? NSNumber else {
                throw PredictorError.invalidInputValue(name: name, expected: "a number")
            }
            return MLFeatureValue(double: number.doubleValue)

        case .int64:
            guard let number = value as? NSNumber else {
                throw PredictorError.invalidInputValue(name: name, expected: "an integer")
            }
            return MLFeatureValue(int64: number.int64Value)

        default:
            throw PredictorError.unsupportedInputType(String(describing: inputDesc.type))
        }
    }

    /// Decode base64, tolerating a `data:` URL prefix and whitespace/newlines.
    static func decodeBase64(_ string: String) -> Data? {
        var payload = string
        if payload.hasPrefix("data:"), let comma = payload.firstIndex(of: ",") {
            payload = String(payload[payload.index(after: comma)...])
        }
        return Data(base64Encoded: payload, options: [.ignoreUnknownCharacters])
    }

    private func createImageFeature(from data: Data, label: String, constraint: MLImageConstraint) throws -> MLFeatureValue {
        guard let nsImage = NSImage(data: data),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw PredictorError.invalidImage(path: label)
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

    private func createMultiArrayFeature(from jsonArray: [Any], constraint: MLMultiArrayConstraint) throws -> MLFeatureValue {
        let shape = constraint.shape.map { $0.intValue }
        let multiArray = try MLMultiArray(shape: shape as [NSNumber], dataType: constraint.dataType)

        let flatValues = flattenArray(jsonArray)
        guard flatValues.count == multiArray.count else {
            throw PredictorError.shapeMismatch(
                expected: multiArray.count,
                got: flatValues.count,
                shape: shape
            )
        }

        fill(multiArray, with: flatValues)

        return MLFeatureValue(multiArray: multiArray)
    }

    /// Copy values into a freshly allocated (contiguous) multi-array.
    private func fill(_ multiArray: MLMultiArray, with values: [Double]) {
        switch multiArray.dataType {
        case .double:
            multiArray.withUnsafeMutableBufferPointer(ofType: Double.self) { pointer, _ in
                for (index, value) in values.enumerated() { pointer[index] = value }
            }
        case .float32:
            multiArray.withUnsafeMutableBufferPointer(ofType: Float.self) { pointer, _ in
                for (index, value) in values.enumerated() { pointer[index] = Float(value) }
            }
        case .int32:
            multiArray.withUnsafeMutableBufferPointer(ofType: Int32.self) { pointer, _ in
                for (index, value) in values.enumerated() { pointer[index] = Self.clampToInt32(value) }
            }
        default:
            // float16 and any future types go through NSNumber bridging
            for (index, value) in values.enumerated() {
                multiArray[index] = NSNumber(value: value)
            }
        }
    }

    /// Convert without trapping on NaN, infinity, or out-of-range values.
    static func clampToInt32(_ value: Double) -> Int32 {
        guard value.isFinite else { return value.isNaN ? 0 : (value < 0 ? .min : .max) }
        let rounded = value.rounded()
        if rounded >= Double(Int32.max) { return .max }
        if rounded <= Double(Int32.min) { return .min }
        return Int32(rounded)
    }

    private func flattenArray(_ array: [Any]) -> [Double] {
        var result: [Double] = []
        for element in array {
            if let nested = element as? [Any] {
                result.append(contentsOf: flattenArray(nested))
            } else if let num = element as? NSNumber {
                result.append(num.doubleValue)
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
                    values.reserveCapacity(multiArray.count)
                    for i in 0..<multiArray.count {
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
    case shapeMismatch(expected: Int, got: Int, shape: [Int])
    case missingInput(name: String, known: [String])
    case unknownInput(name: String, known: [String])
    case invalidInputValue(name: String, expected: String)

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
        case .shapeMismatch(let expected, let got, let shape):
            let shapeText = shape.map(String.init).joined(separator: ", ")
            return "Input has \(got) values but the model expects \(expected) (shape [\(shapeText)])"
        case .missingInput(let name, let known):
            return "Missing value for model input '\(name)'. Model inputs: \(known.joined(separator: ", "))"
        case .unknownInput(let name, let known):
            return "Unknown model input '\(name)'. Model inputs: \(known.joined(separator: ", "))"
        case .invalidInputValue(let name, let expected):
            return "Input '\(name)' expects \(expected)"
        }
    }
}
