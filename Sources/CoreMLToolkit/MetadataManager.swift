import Foundation

/// Manages Core ML model metadata
public class MetadataManager {

    public init() {}

    /// Get metadata from a model's specification file
    public func getMetadata(modelPath: String) throws -> ModelMetadata {
        let url = URL(fileURLWithPath: modelPath)

        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw MetadataError.modelNotFound(path: modelPath)
        }

        // For .mlmodel files (which are directories), read the model.mlmodel file
        // For .mlpackage, read from the Manifest.json or Data/com.apple.CoreML/model.mlmodel
        let specURL: URL

        if url.pathExtension == "mlmodel" || url.pathExtension == "mlpackage" {
            // These are directories, need to find the spec file
            specURL = url.appendingPathComponent("model.mlmodel")
            if !FileManager.default.fileExists(atPath: specURL.path) {
                // Try mlpackage structure
                let mlpackageSpec = url.appendingPathComponent("Data/com.apple.CoreML/model.mlmodel")
                if FileManager.default.fileExists(atPath: mlpackageSpec.path) {
                    return try parseMetadataFromSpec(at: mlpackageSpec)
                }
            }
        }

        // Use the inspector to get metadata
        let inspector = ModelInspector()
        let info = try inspector.inspect(modelPath: modelPath)
        return info.metadata
    }

    /// Set metadata in a model (creates a new model with updated metadata)
    public func setMetadata(modelPath: String, field: MetadataField, value: String, outputPath: String? = nil) throws -> String {
        // Note: Core ML models are typically read-only after compilation
        // This function would need to modify the source .mlmodel/.mlpackage before compilation
        // For now, we document this limitation

        let url = URL(fileURLWithPath: modelPath)

        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw MetadataError.modelNotFound(path: modelPath)
        }

        // Check if it's a compiled model
        if url.pathExtension == "mlmodelc" {
            throw MetadataError.cannotModifyCompiled
        }

        // For mlmodel/mlpackage, we need to modify the specification
        // This requires parsing and regenerating the protobuf format
        // which is complex - document the limitation

        throw MetadataError.notImplemented(reason: "Metadata modification requires coremltools Python library. Use: python -m coremltools.utils.metadata_editor")
    }

    private func parseMetadataFromSpec(at url: URL) throws -> ModelMetadata {
        // The spec file is in protobuf format
        // For simplicity, we use the inspector instead
        let inspector = ModelInspector()
        let parentURL = url.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let info = try inspector.inspect(modelPath: parentURL.path)
        return info.metadata
    }
}

/// Metadata fields that can be modified
public enum MetadataField: String, CaseIterable {
    case author
    case description
    case license
    case version
}

public enum MetadataError: Error, LocalizedError {
    case modelNotFound(path: String)
    case cannotModifyCompiled
    case notImplemented(reason: String)
    case invalidField(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "Model not found at: \(path)"
        case .cannotModifyCompiled:
            return "Cannot modify metadata of compiled model (.mlmodelc). Modify the source model instead."
        case .notImplemented(let reason):
            return "Not implemented: \(reason)"
        case .invalidField(let field):
            return "Invalid metadata field: \(field)"
        }
    }
}
