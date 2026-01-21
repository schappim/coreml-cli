import Foundation
import CoreML

/// Represents model input/output feature information
public struct FeatureInfo: Codable, Equatable {
    public let name: String
    public let type: String
    public let shape: [Int]?
    public let multiArrayDataType: String?
    public let imageConstraint: ImageConstraintInfo?

    public init(name: String, type: String, shape: [Int]? = nil, multiArrayDataType: String? = nil, imageConstraint: ImageConstraintInfo? = nil) {
        self.name = name
        self.type = type
        self.shape = shape
        self.multiArrayDataType = multiArrayDataType
        self.imageConstraint = imageConstraint
    }
}

/// Image constraint details
public struct ImageConstraintInfo: Codable, Equatable {
    public let width: Int
    public let height: Int
    public let pixelFormat: String

    public init(width: Int, height: Int, pixelFormat: String) {
        self.width = width
        self.height = height
        self.pixelFormat = pixelFormat
    }
}

/// Complete model information
public struct ModelInfo: Codable {
    public let name: String
    public let inputs: [FeatureInfo]
    public let outputs: [FeatureInfo]
    public let metadata: ModelMetadata
    public let fileSize: Int64
    public let isCompiled: Bool

    public init(name: String, inputs: [FeatureInfo], outputs: [FeatureInfo], metadata: ModelMetadata, fileSize: Int64, isCompiled: Bool) {
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.metadata = metadata
        self.fileSize = fileSize
        self.isCompiled = isCompiled
    }
}

/// Model metadata
public struct ModelMetadata: Codable {
    public let author: String?
    public let description: String?
    public let license: String?
    public let version: String?
    public let additionalInfo: [String: String]

    public init(author: String? = nil, description: String? = nil, license: String? = nil, version: String? = nil, additionalInfo: [String: String] = [:]) {
        self.author = author
        self.description = description
        self.license = license
        self.version = version
        self.additionalInfo = additionalInfo
    }
}

/// Inspects Core ML models and extracts information
public class ModelInspector {

    public init() {}

    /// Inspect a Core ML model and return its information
    public func inspect(modelPath: String) throws -> ModelInfo {
        let url = URL(fileURLWithPath: modelPath)
        let fileManager = FileManager.default

        guard fileManager.fileExists(atPath: modelPath) else {
            throw ModelInspectorError.modelNotFound(path: modelPath)
        }

        let isCompiled = url.pathExtension == "mlmodelc"
        let modelURL: URL

        if isCompiled {
            modelURL = url
        } else {
            // Compile the model first if it's not compiled
            modelURL = try MLModel.compileModel(at: url)
        }

        let model = try MLModel(contentsOf: modelURL)
        let modelDescription = model.modelDescription

        let inputs = extractFeatures(from: modelDescription.inputDescriptionsByName)
        let outputs = extractFeatures(from: modelDescription.outputDescriptionsByName)
        let metadata = extractMetadata(from: modelDescription.metadata)
        let fileSize = try calculateFileSize(at: url)

        let modelName = url.deletingPathExtension().lastPathComponent

        return ModelInfo(
            name: modelName,
            inputs: inputs,
            outputs: outputs,
            metadata: metadata,
            fileSize: fileSize,
            isCompiled: isCompiled
        )
    }

    private func extractFeatures(from descriptions: [String: MLFeatureDescription]) -> [FeatureInfo] {
        return descriptions.map { name, desc in
            let typeString = featureTypeString(desc.type)
            var shape: [Int]? = nil
            var multiArrayDataType: String? = nil
            var imageConstraint: ImageConstraintInfo? = nil

            if let constraint = desc.multiArrayConstraint {
                shape = constraint.shape.map { $0.intValue }
                multiArrayDataType = dataTypeString(constraint.dataType)
            }

            if let imgConstraint = desc.imageConstraint {
                imageConstraint = ImageConstraintInfo(
                    width: imgConstraint.pixelsWide,
                    height: imgConstraint.pixelsHigh,
                    pixelFormat: pixelFormatString(imgConstraint.pixelFormatType)
                )
            }

            return FeatureInfo(
                name: name,
                type: typeString,
                shape: shape,
                multiArrayDataType: multiArrayDataType,
                imageConstraint: imageConstraint
            )
        }.sorted { $0.name < $1.name }
    }

    private func extractMetadata(from metadata: [MLModelMetadataKey: Any]) -> ModelMetadata {
        var additionalInfo: [String: String] = [:]

        for (key, value) in metadata {
            if key != .author && key != .description && key != .license && key != .versionString {
                additionalInfo[key.rawValue] = String(describing: value)
            }
        }

        return ModelMetadata(
            author: metadata[.author] as? String,
            description: metadata[.description] as? String,
            license: metadata[.license] as? String,
            version: metadata[.versionString] as? String,
            additionalInfo: additionalInfo
        )
    }

    private func featureTypeString(_ type: MLFeatureType) -> String {
        switch type {
        case .invalid: return "invalid"
        case .int64: return "int64"
        case .double: return "double"
        case .string: return "string"
        case .image: return "image"
        case .multiArray: return "multiArray"
        case .dictionary: return "dictionary"
        case .sequence: return "sequence"
        case .state: return "state"
        @unknown default: return "unknown"
        }
    }

    private func dataTypeString(_ type: MLMultiArrayDataType) -> String {
        switch type {
        case .double: return "double"
        case .float32: return "float32"
        case .float16: return "float16"
        case .int32: return "int32"
        @unknown default: return "unknown"
        }
    }

    private func pixelFormatString(_ format: OSType) -> String {
        switch format {
        case kCVPixelFormatType_32BGRA: return "BGRA32"
        case kCVPixelFormatType_32RGBA: return "RGBA32"
        case kCVPixelFormatType_OneComponent8: return "Grayscale8"
        case kCVPixelFormatType_OneComponent16Half: return "Grayscale16Half"
        default: return "Unknown(\(format))"
        }
    }

    private func calculateFileSize(at url: URL) throws -> Int64 {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false

        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return 0
        }

        if isDirectory.boolValue {
            return try calculateDirectorySize(at: url)
        } else {
            let attributes = try fileManager.attributesOfItem(atPath: url.path)
            return attributes[.size] as? Int64 ?? 0
        }
    }

    private func calculateDirectorySize(at url: URL) throws -> Int64 {
        let fileManager = FileManager.default
        var totalSize: Int64 = 0

        guard let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }

        for case let fileURL as URL in enumerator {
            let attributes = try fileURL.resourceValues(forKeys: [.fileSizeKey])
            totalSize += Int64(attributes.fileSize ?? 0)
        }

        return totalSize
    }
}

public enum ModelInspectorError: Error, LocalizedError {
    case modelNotFound(path: String)
    case invalidModelFormat
    case compilationFailed(Error)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "Model not found at path: \(path)"
        case .invalidModelFormat:
            return "Invalid model format"
        case .compilationFailed(let error):
            return "Model compilation failed: \(error.localizedDescription)"
        }
    }
}
