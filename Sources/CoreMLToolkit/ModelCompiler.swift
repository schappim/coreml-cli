import Foundation
import CoreML

/// Result of model compilation
public struct CompilationResult: Codable {
    public let sourcePath: String
    public let outputPath: String
    public let originalSize: Int64
    public let compiledSize: Int64
    public let success: Bool
    public let message: String?

    public init(sourcePath: String, outputPath: String, originalSize: Int64, compiledSize: Int64, success: Bool, message: String? = nil) {
        self.sourcePath = sourcePath
        self.outputPath = outputPath
        self.originalSize = originalSize
        self.compiledSize = compiledSize
        self.success = success
        self.message = message
    }
}

/// Compiles Core ML models
public class ModelCompiler {

    public init() {}

    /// Compile a model to .mlmodelc format
    public func compile(modelPath: String, outputDirectory: String? = nil) throws -> CompilationResult {
        let sourceURL = URL(fileURLWithPath: modelPath)
        let fileManager = FileManager.default

        guard fileManager.fileExists(atPath: modelPath) else {
            throw CompilerError.modelNotFound(path: modelPath)
        }

        // Check if already compiled
        if sourceURL.pathExtension == "mlmodelc" {
            throw CompilerError.alreadyCompiled
        }

        // Calculate original size
        let originalSize = try calculateSize(at: sourceURL)

        // Compile the model
        let compiledURL = try MLModel.compileModel(at: sourceURL)

        // Determine output location
        let outputDir: URL
        if let dir = outputDirectory {
            outputDir = URL(fileURLWithPath: dir)
            if !fileManager.fileExists(atPath: dir) {
                try fileManager.createDirectory(at: outputDir, withIntermediateDirectories: true)
            }
        } else {
            outputDir = sourceURL.deletingLastPathComponent()
        }

        let modelName = sourceURL.deletingPathExtension().lastPathComponent
        let destinationURL = outputDir.appendingPathComponent("\(modelName).mlmodelc")

        // Remove existing compiled model if present
        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }

        // Move compiled model to destination
        try fileManager.moveItem(at: compiledURL, to: destinationURL)

        let compiledSize = try calculateSize(at: destinationURL)

        return CompilationResult(
            sourcePath: sourceURL.path,
            outputPath: destinationURL.path,
            originalSize: originalSize,
            compiledSize: compiledSize,
            success: true,
            message: "Model compiled successfully"
        )
    }

    /// Validate a compiled model
    public func validate(modelPath: String) throws -> Bool {
        let url = URL(fileURLWithPath: modelPath)

        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw CompilerError.modelNotFound(path: modelPath)
        }

        // Try to load the model
        do {
            if url.pathExtension == "mlmodelc" {
                _ = try MLModel(contentsOf: url)
            } else {
                let compiledURL = try MLModel.compileModel(at: url)
                _ = try MLModel(contentsOf: compiledURL)
            }
            return true
        } catch {
            return false
        }
    }

    private func calculateSize(at url: URL) throws -> Int64 {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false

        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return 0
        }

        if isDirectory.boolValue {
            var totalSize: Int64 = 0
            guard let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
                return 0
            }

            for case let fileURL as URL in enumerator {
                let attributes = try fileURL.resourceValues(forKeys: [.fileSizeKey])
                totalSize += Int64(attributes.fileSize ?? 0)
            }
            return totalSize
        } else {
            let attributes = try fileManager.attributesOfItem(atPath: url.path)
            return attributes[.size] as? Int64 ?? 0
        }
    }
}

public enum CompilerError: Error, LocalizedError {
    case modelNotFound(path: String)
    case alreadyCompiled
    case compilationFailed(Error)
    case outputDirectoryCreationFailed

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "Model not found at: \(path)"
        case .alreadyCompiled:
            return "Model is already compiled (.mlmodelc)"
        case .compilationFailed(let error):
            return "Compilation failed: \(error.localizedDescription)"
        case .outputDirectoryCreationFailed:
            return "Failed to create output directory"
        }
    }
}
