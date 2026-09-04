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

        do {
            try updatedData.write(to: destination.specURL, options: .atomic)
            try destination.commit()
        } catch {
            destination.discard()
            throw error
        }
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

    /// Where the modified spec bytes go, plus how to finish (or abandon) the write.
    private struct WriteDestination {
        let specURL: URL
        let userVisiblePath: String
        /// Move a staged package copy into its final place. No-op otherwise.
        let commit: () throws -> Void
        /// Clean up staged files when the write failed.
        let discard: () -> Void
    }

    /// Decide where the modified spec bytes should be written and (for packages with
    /// `--output`) clone the package alongside the destination first.
    private func resolveOutputDestination(
        sourceModel: URL,
        sourceSpec: URL,
        outputPath: String?,
        isPackage: Bool
    ) throws -> WriteDestination {
        guard let outputPath = outputPath else {
            return WriteDestination(
                specURL: sourceSpec,
                userVisiblePath: sourceModel.path,
                commit: {},
                discard: {}
            )
        }

        let outputURL = canonical(URL(fileURLWithPath: outputPath))
        let sourceURL = canonical(sourceModel)

        if !isPackage {
            return WriteDestination(
                specURL: outputURL,
                userVisiblePath: outputURL.path,
                commit: {},
                discard: {}
            )
        }

        // Writing a package means replacing a directory, so refuse the shapes where
        // that would destroy the very model being edited.
        guard outputURL != sourceURL else {
            throw MetadataError.outputSameAsSource(path: outputURL.path)
        }
        guard !isDescendant(outputURL, of: sourceURL) else {
            throw MetadataError.outputInsideSource(path: outputURL.path)
        }
        if fileManager.fileExists(atPath: outputURL.path) {
            guard outputURL.pathExtension == "mlpackage" else {
                throw MetadataError.outputExistsAndIsNotAPackage(path: outputURL.path)
            }
        }

        // Clone to a sibling staging directory and only swap it into place once the
        // spec has been written, so a failure part-way through leaves both the source
        // and any existing destination untouched.
        let stagingURL = try uniqueStagingURL(besides: outputURL)
        do {
            try fileManager.copyItem(at: sourceURL, to: stagingURL)
        } catch {
            // Never leave a half-copied package behind.
            try? fileManager.removeItem(at: stagingURL)
            throw error
        }

        let relativeSpecComponents = relativePathComponents(of: sourceSpec, under: sourceModel)
        let mirroredSpecURL = relativeSpecComponents.reduce(stagingURL) {
            $0.appendingPathComponent($1)
        }

        let manager = fileManager
        return WriteDestination(
            specURL: mirroredSpecURL,
            userVisiblePath: outputURL.path,
            commit: {
                if manager.fileExists(atPath: outputURL.path) {
                    _ = try manager.replaceItemAt(outputURL, withItemAt: stagingURL)
                } else {
                    try manager.moveItem(at: stagingURL, to: outputURL)
                }
            },
            discard: {
                try? manager.removeItem(at: stagingURL)
            }
        )
    }

    /// A staging path next to the destination, so the final move stays on one volume.
    private func uniqueStagingURL(besides outputURL: URL) throws -> URL {
        let directory = outputURL.deletingLastPathComponent()
        let base = ".\(outputURL.lastPathComponent).coreml-staging.\(ProcessInfo.processInfo.processIdentifier)"

        for attempt in 0..<100 {
            let candidate = directory.appendingPathComponent(attempt == 0 ? base : "\(base).\(attempt)")
            if !fileManager.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
        throw MetadataError.stagingPathUnavailable(path: directory.path)
    }

    /// Resolve symlinks as far as the path exists, so two spellings of the same
    /// location — /tmp/x and /private/tmp/x on macOS — compare equal.
    private func canonical(_ url: URL) -> URL {
        let standardized = url.standardizedFileURL

        var trailing: [String] = []
        var candidate = standardized
        while !fileManager.fileExists(atPath: candidate.path) {
            let parent = candidate.deletingLastPathComponent()
            // Stop at the root, or anywhere deleting a component makes no progress.
            guard parent.path != candidate.path, parent.path != "/" || candidate.path == "/" else { break }
            trailing.insert(candidate.lastPathComponent, at: 0)
            candidate = parent
        }

        guard fileManager.fileExists(atPath: candidate.path) else { return standardized }
        return trailing.reduce(candidate.resolvingSymlinksInPath()) { $0.appendingPathComponent($1) }
    }

    private func isDescendant(_ url: URL, of ancestor: URL) -> Bool {
        let components = url.standardized.pathComponents
        let ancestorComponents = ancestor.standardized.pathComponents
        return components.count > ancestorComponents.count && components.starts(with: ancestorComponents)
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
    case outputSameAsSource(path: String)
    case outputInsideSource(path: String)
    case outputExistsAndIsNotAPackage(path: String)
    case stagingPathUnavailable(path: String)

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
        case .outputSameAsSource(let path):
            return "--output is the same as the source model (\(path)). Omit --output to edit in place."
        case .outputInsideSource(let path):
            return "--output (\(path)) is inside the source model. Choose a destination outside it."
        case .outputExistsAndIsNotAPackage(let path):
            return "Refusing to replace \(path): it already exists and is not an .mlpackage."
        case .stagingPathUnavailable(let path):
            return "Could not create a temporary staging directory in: \(path)"
        }
    }
}
