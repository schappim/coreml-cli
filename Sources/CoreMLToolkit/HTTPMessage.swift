import Foundation

/// A parsed HTTP request.
public struct HTTPRequest {
    /// Uppercased request method, e.g. `GET`.
    public let method: String
    /// Percent-decoded path with the query string removed, e.g. `/v1/predict`.
    public let path: String
    /// Percent-decoded query parameters.
    public let query: [String: String]
    /// Header fields, keyed by lowercased name. Repeated fields are joined with ", ".
    public let headers: [String: String]
    /// Request body. Empty when the request carries none.
    public let body: Data
    /// Protocol version as written on the request line, e.g. `HTTP/1.1`.
    public let version: String

    public init(
        method: String,
        path: String,
        query: [String: String] = [:],
        headers: [String: String] = [:],
        body: Data = Data(),
        version: String = "HTTP/1.1"
    ) {
        self.method = method
        self.path = path
        self.query = query
        self.headers = headers
        self.body = body
        self.version = version
    }

    /// Look up a header by name, case-insensitively.
    public func header(_ name: String) -> String? {
        headers[name.lowercased()]
    }

    /// The media type from `Content-Type`, lowercased and without parameters.
    public var contentType: String? {
        guard let raw = header("content-type") else { return nil }
        return raw.split(separator: ";").first.map {
            $0.trimmingCharacters(in: .whitespaces).lowercased()
        }
    }

    /// The `boundary` parameter of a multipart `Content-Type`, if present.
    public var multipartBoundary: String? {
        guard let raw = header("content-type") else { return nil }
        for parameter in raw.split(separator: ";").dropFirst() {
            let parts = parameter.split(separator: "=", maxSplits: 1)
            guard parts.count == 2,
                  parts[0].trimmingCharacters(in: .whitespaces).lowercased() == "boundary" else { continue }
            var value = parts[1].trimmingCharacters(in: .whitespaces)
            if value.hasPrefix("\"") && value.hasSuffix("\"") && value.count >= 2 {
                value = String(value.dropFirst().dropLast())
            }
            return value.isEmpty ? nil : value
        }
        return nil
    }

    /// Whether the connection should be reused after this request, per RFC 9112.
    public var keepAlive: Bool {
        let connection = header("connection")?.lowercased() ?? ""
        if connection.contains("close") { return false }
        if version == "HTTP/1.0" { return connection.contains("keep-alive") }
        return true
    }

    /// Whether the client is waiting for a `100 Continue` before sending the body.
    public var expectsContinue: Bool {
        header("expect")?.lowercased().contains("100-continue") ?? false
    }
}

/// An HTTP response ready to be written to a connection.
public struct HTTPResponse {
    public var status: Int
    public var headers: [String: String]
    public var body: Data

    public init(status: Int = 200, headers: [String: String] = [:], body: Data = Data()) {
        self.status = status
        self.headers = headers
        self.body = body
    }

    /// A JSON response from an already-encoded payload.
    public static func json(_ body: Data, status: Int = 200) -> HTTPResponse {
        HTTPResponse(
            status: status,
            headers: ["Content-Type": "application/json"],
            body: body
        )
    }

    /// A JSON response encoded from an `Encodable` value.
    public static func json<T: Encodable>(_ value: T, status: Int = 200) -> HTTPResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(value) else {
            return .error(status: 500, message: "Failed to encode response")
        }
        return .json(data, status: status)
    }

    /// A `{"error": {"status": ..., "message": ...}}` response.
    public static func error(status: Int, message: String) -> HTTPResponse {
        let payload: [String: Any] = [
            "error": [
                "status": status,
                "message": message
            ]
        ]
        let data = (try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]))
            ?? Data(#"{"error":{"message":"Internal error"}}"#.utf8)
        return .json(data, status: status)
    }

    public static func text(_ string: String, status: Int = 200) -> HTTPResponse {
        HTTPResponse(
            status: status,
            headers: ["Content-Type": "text/plain; charset=utf-8"],
            body: Data(string.utf8)
        )
    }

    /// Serialise the response, including the framing headers every reply needs.
    ///
    /// - Parameters:
    ///   - keepAlive: whether the connection will be reused.
    ///   - includeBody: false for `HEAD`, which keeps `Content-Length` but sends no body.
    public func serialize(keepAlive: Bool, includeBody: Bool = true) -> Data {
        var head = "HTTP/1.1 \(status) \(Self.reasonPhrase(for: status))\r\n"

        var allHeaders = headers
        allHeaders["Content-Length"] = String(body.count)
        allHeaders["Connection"] = keepAlive ? "keep-alive" : "close"
        allHeaders["Date"] = Self.dateFormatter.string(from: Date())

        for key in allHeaders.keys.sorted() {
            head += "\(key): \(allHeaders[key]!)\r\n"
        }
        head += "\r\n"

        var data = Data(head.utf8)
        if includeBody {
            data.append(body)
        }
        return data
    }

    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "GMT")
        formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss 'GMT'"
        return formatter
    }()

    static func reasonPhrase(for status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 201: return "Created"
        case 204: return "No Content"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 403: return "Forbidden"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 408: return "Request Timeout"
        case 411: return "Length Required"
        case 413: return "Content Too Large"
        case 415: return "Unsupported Media Type"
        case 422: return "Unprocessable Content"
        case 429: return "Too Many Requests"
        case 431: return "Request Header Fields Too Large"
        case 500: return "Internal Server Error"
        case 501: return "Not Implemented"
        case 503: return "Service Unavailable"
        case 505: return "HTTP Version Not Supported"
        default: return status < 400 ? "OK" : "Error"
        }
    }
}
