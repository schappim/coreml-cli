import Foundation

/// A single inference input: raw bytes plus a name used for reporting.
///
/// Inputs can come from a file on disk (`coreml predict`, `coreml batch`) or
/// straight from memory (an HTTP request body served by `coreml serve`), so the
/// feature-building code works from bytes rather than a file path.
public struct InputSource {
    /// Name reported back in results — a filename for file inputs, or a
    /// caller-supplied label such as "request" for in-memory inputs.
    public let name: String

    /// The raw input bytes.
    public let data: Data

    /// Media type hint, when the caller knows it (for example an HTTP
    /// `Content-Type`). Never required: the bytes are matched against the
    /// model's declared input types regardless.
    public let mediaType: String?

    public init(name: String, data: Data, mediaType: String? = nil) {
        self.name = name
        self.data = data
        self.mediaType = mediaType
    }

    /// Load an input from a file on disk.
    public static func file(path: String) throws -> InputSource {
        guard FileManager.default.fileExists(atPath: path) else {
            throw PredictorError.inputNotFound(path: path)
        }
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        return InputSource(name: url.lastPathComponent, data: data)
    }

    /// The bytes decoded as UTF-8 text, trimmed of surrounding whitespace.
    public var text: String? {
        String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
