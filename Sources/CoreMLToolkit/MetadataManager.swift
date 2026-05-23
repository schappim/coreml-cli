import Foundation

/// Reads and writes the standard metadata fields of a CoreML model.
public class MetadataManager {

    private let editor: ModelSpecEditor
    private let fileManager: FileManager

    public init(editor: ModelSpecEditor = ModelSpecEditor(), fileManager: FileManager = .default) {
        self.editor = editor
        self.fileManager = fileManager
    }

    /// Read metadata from a `.mlmodel`, `.mlpackage`, or `.mlmodelc` model.
    public func getMetadata(modelPath: String) throws -> ModelMetadata {
        let url = URL(fileURLWithPath: modelPath)

        guard fileManager.fileExists(atPath: modelPath) else {
            throw MetadataError.modelNotFound(path: modelPath)
        }

        switch url.pathExtension {
        case "mlmodel":
            let data = try Data(contentsOf: url)
            return try editor.readMetadata(specData: data)
        case "mlpackage":
            let specURL = try findPackageSpec(packageURL: url)
            let data = try Data(contentsOf: specURL)
            return try editor.readMetadata(specData: data)
        default:
            // .mlmodelc and any other format: delegate to the inspector, which loads
            // the model through CoreML rather than reading the spec directly.
            let inspector = ModelInspector()
            let info = try inspector.inspect(modelPath: modelPath)
            return info.metadata
        }
    }

    /// Set a metadata field on a `.mlmodel` or `.mlpackage`, returning the path written.
    /// Without `outputPath`, the source is modified in place. Compiled `.mlmodelc` models
    /// are rejected — modify the source spec and recompile instead.
    @discardableResult
    public func setMetadata(
        modelPath: String,
        field: MetadataField,
        value: String,
        outputPath: String? = nil
    ) throws -> String {
        let url = URL(fileURLWithPath: modelPath)

        guard fileManager.fileExists(atPath: modelPath) else {
            throw MetadataError.modelNotFound(path: modelPath)
        }

        if url.pathExtension == "mlmodelc" {
            throw MetadataError.cannotModifyCompiled
        }

        let specURL: URL
        let isPackage: Bool
        switch url.pathExtension {
        case "mlmodel":
            specURL = url
            isPackage = false
        case "mlpackage":
            specURL = try findPackageSpec(packageURL: url)
            isPackage = true
        default:
            throw MetadataError.unsupportedModelFormat(extension: url.pathExtension)
        }

        let originalData = try Data(contentsOf: specURL)
        let updatedData = try editor.setMetadataField(
            specData: originalData,
            field: field,
            value: value
        )

        let destination = try resolveOutputDestination(
            sourceModel: url,
            sourceSpec: specURL,
            outputPath: outputPath,
            isPackage: isPackage
        )

        try updatedData.write(to: destination.specURL, options: .atomic)
        return destination.userVisiblePath
    }

    /// Locate the canonical spec file inside an `.mlpackage` directory.
    private func findPackageSpec(packageURL: URL) throws -> URL {
        let canonical = packageURL.appendingPathComponent("Data/com.apple.CoreML/model.mlmodel")
        if fileManager.fileExists(atPath: canonical.path) {
            return canonical
        }
        // Older layouts placed the spec at the package root.
        let topLevel = packageURL.appendingPathComponent("model.mlmodel")
        if fileManager.fileExists(atPath: topLevel.path) {
            return topLevel
        }
        throw MetadataError.specNotFoundInPackage(path: packageURL.path)
    }

    /// Decide where the modified spec bytes should be written and (for packages with
    /// `--output`) clone the package to the destination first.
    private func resolveOutputDestination(
        sourceModel: URL,
        sourceSpec: URL,
        outputPath: String?,
        isPackage: Bool
    ) throws -> (specURL: URL, userVisiblePath: String) {
        guard let outputPath = outputPath else {
            return (sourceSpec, sourceModel.path)
        }

        let outputURL = URL(fileURLWithPath: outputPath)

        if !isPackage {
            return (outputURL, outputURL.path)
        }

        // .mlpackage with --output: copy the whole package, then aim our write at the
        // mirrored spec location inside the copy.
        if fileManager.fileExists(atPath: outputURL.path) {
            try fileManager.removeItem(at: outputURL)
        }
        try fileManager.copyItem(at: sourceModel, to: outputURL)

        let relativeSpecComponents = relativePathComponents(of: sourceSpec, under: sourceModel)
        let mirroredSpecURL = relativeSpecComponents.reduce(outputURL) {
            $0.appendingPathComponent($1)
        }
        return (mirroredSpecURL, outputURL.path)
    }

    private func relativePathComponents(of url: URL, under ancestor: URL) -> [String] {
        let urlComponents = url.standardized.pathComponents
        let ancestorComponents = ancestor.standardized.pathComponents
        guard urlComponents.starts(with: ancestorComponents) else {
            return [url.lastPathComponent]
        }
        return Array(urlComponents.dropFirst(ancestorComponents.count))
    }
}

/// Metadata fields supported by `setMetadata`.
public enum MetadataField: String, CaseIterable {
    case author
    case description
    case license
    case version
}

public enum MetadataError: Error, LocalizedError, Equatable {
    case modelNotFound(path: String)
    case cannotModifyCompiled
    case unsupportedModelFormat(extension: String)
    case specNotFoundInPackage(path: String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "Model not found at: \(path)"
        case .cannotModifyCompiled:
            return "Cannot modify metadata of a compiled model (.mlmodelc). Modify the source .mlmodel or .mlpackage and recompile."
        case .unsupportedModelFormat(let ext):
            return "Cannot write metadata for model format '.\(ext)'. Use .mlmodel or .mlpackage."
        case .specNotFoundInPackage(let path):
            return "Could not locate model.mlmodel inside package at: \(path)"
        }
    }
}
