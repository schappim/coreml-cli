import Foundation

/// One part of a `multipart/form-data` body.
public struct MultipartPart {
    /// The `name` parameter of the part's Content-Disposition, if any.
    public let name: String?
    /// The `filename` parameter of the part's Content-Disposition, if any.
    public let filename: String?
    /// The part's own Content-Type, if it declared one.
    public let contentType: String?
    /// The part's raw content.
    public let data: Data

    public init(name: String?, filename: String?, contentType: String?, data: Data) {
        self.name = name
        self.filename = filename
        self.contentType = contentType
        self.data = data
    }
}

public enum MultipartError: Error, LocalizedError {
    case missingBoundary
    case malformedBody(String)

    public var errorDescription: String? {
        switch self {
        case .missingBoundary:
            return "multipart/form-data request is missing its boundary parameter"
        case .malformedBody(let detail):
            return "Malformed multipart body: \(detail)"
        }
    }
}

/// Minimal `multipart/form-data` reader, enough for file uploads from
/// `curl -F`, HTML forms, and the usual HTTP client libraries.
public enum MultipartParser {

    public static func parse(body: Data, boundary: String) throws -> [MultipartPart] {
        guard !boundary.isEmpty else { throw MultipartError.missingBoundary }

        let bytes = [UInt8](body)
        let delimiter = [UInt8]("\r\n--\(boundary)".utf8)

        // Treat the opening delimiter like any other by pretending a CRLF
        // preceded it, which is what the wire format implies.
        var haystack = [UInt8]("\r\n".utf8)
        haystack.append(contentsOf: bytes)

        var parts: [MultipartPart] = []
        var cursor = 0
        var sawClosingDelimiter = false

        guard var next = find(delimiter, in: haystack, from: cursor) else {
            throw MultipartError.malformedBody("no boundary found")
        }
        cursor = next + delimiter.count

        while true {
            // After a delimiter comes either "--" (end of body) or CRLF (a part).
            if cursor + 1 < haystack.count, haystack[cursor] == 0x2D, haystack[cursor + 1] == 0x2D {
                sawClosingDelimiter = true
                break
            }
            guard cursor + 1 < haystack.count, haystack[cursor] == 0x0D, haystack[cursor + 1] == 0x0A else {
                throw MultipartError.malformedBody("expected CRLF after boundary")
            }
            cursor += 2

            guard let following = find(delimiter, in: haystack, from: cursor) else {
                throw MultipartError.malformedBody("unterminated part")
            }
            next = following

            let section = Array(haystack[cursor..<next])
            parts.append(try parsePart(section))

            cursor = next + delimiter.count
        }

        guard sawClosingDelimiter else {
            throw MultipartError.malformedBody("missing closing boundary")
        }

        return parts
    }

    private static func parsePart(_ section: [UInt8]) throws -> MultipartPart {
        guard let headerEnd = find([0x0D, 0x0A, 0x0D, 0x0A], in: section, from: 0) else {
            throw MultipartError.malformedBody("part is missing its header block")
        }

        let headerText = String(decoding: section[0..<headerEnd], as: UTF8.self)
        let content = Data(section[(headerEnd + 4)...])

        var name: String?
        var filename: String?
        var contentType: String?

        for line in headerText.components(separatedBy: "\r\n") where !line.isEmpty {
            guard let separator = line.firstIndex(of: ":") else { continue }
            let field = line[line.startIndex..<separator].trimmingCharacters(in: .whitespaces).lowercased()
            let value = String(line[line.index(after: separator)...]).trimmingCharacters(in: .whitespaces)

            switch field {
            case "content-disposition":
                name = parameter("name", in: value)
                filename = parameter("filename", in: value)
            case "content-type":
                contentType = value.split(separator: ";").first
                    .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
            default:
                break
            }
        }

        return MultipartPart(name: name, filename: filename, contentType: contentType, data: content)
    }

    /// Read a `key="value"` (or bare `key=value`) parameter out of a header value.
    static func parameter(_ key: String, in headerValue: String) -> String? {
        for parameter in headerValue.split(separator: ";").dropFirst() {
            let pair = parameter.split(separator: "=", maxSplits: 1)
            guard pair.count == 2,
                  pair[0].trimmingCharacters(in: .whitespaces).lowercased() == key.lowercased() else { continue }
            var value = pair[1].trimmingCharacters(in: .whitespaces)
            if value.hasPrefix("\"") && value.hasSuffix("\"") && value.count >= 2 {
                value = String(value.dropFirst().dropLast())
            }
            return value
        }
        return nil
    }

    private static func find(_ needle: [UInt8], in haystack: [UInt8], from start: Int) -> Int? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        let last = haystack.count - needle.count
        guard start <= last else { return nil }

        for index in start...last {
            var matched = true
            for offset in 0..<needle.count {
                if haystack[index + offset] != needle[offset] {
                    matched = false
                    break
                }
            }
            if matched { return index }
        }
        return nil
    }
}
