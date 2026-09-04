import Foundation

/// Incremental HTTP/1.1 request parser.
///
/// Bytes are fed in as they arrive off the socket and complete requests are
/// pulled out one at a time, so a slow client, a split header block, or several
/// pipelined requests in a single read all behave the same way. Framing follows
/// RFC 9112: `Content-Length` and `Transfer-Encoding` are never both honoured,
/// obsolete line folding is rejected, and chunked bodies are decoded in place.
public final class HTTPRequestParser {

    /// What the parser can say after being handed more bytes.
    public enum Outcome: Equatable {
        /// Not enough bytes yet — read more from the socket.
        case needMoreData
        /// A complete request.
        case request(HTTPRequest)
        /// The stream is unusable; reply with this status and close.
        case failure(status: Int, message: String)

        public static func == (lhs: Outcome, rhs: Outcome) -> Bool {
            switch (lhs, rhs) {
            case (.needMoreData, .needMoreData):
                return true
            case let (.failure(ls, lm), .failure(rs, rm)):
                return ls == rs && lm == rm
            case let (.request(l), .request(r)):
                return l.method == r.method && l.path == r.path && l.body == r.body
            default:
                return false
            }
        }
    }

    /// Size ceilings that keep a hostile or broken client from exhausting memory.
    public struct Limits {
        public var maxHeaderBytes: Int
        public var maxBodyBytes: Int

        public init(maxHeaderBytes: Int = 64 * 1024, maxBodyBytes: Int = 32 * 1024 * 1024) {
            self.maxHeaderBytes = maxHeaderBytes
            self.maxBodyBytes = maxBodyBytes
        }
    }

    private enum State {
        case head
        case fixedBody(remaining: Int)
        case chunkSize
        case chunkData(remaining: Int)
        case chunkTerminator
        case trailers
        case failed
    }

    /// Called as soon as a request's headers are known, before its body has
    /// arrived — the hook a server uses to answer `Expect: 100-continue`.
    public var onHeadReceived: ((HTTPRequest) -> Void)?

    private let limits: Limits
    private var buffer: [UInt8] = []
    private var readOffset = 0
    private var state: State = .head
    private var pendingHead: HTTPRequest?
    private var pendingBody = Data()

    public init(limits: Limits = Limits()) {
        self.limits = limits
    }

    /// Feed bytes read from the connection.
    public func append<C: Collection>(_ bytes: C) where C.Element == UInt8 {
        buffer.append(contentsOf: bytes)
    }

    /// Pull the next complete request out of the buffered bytes.
    ///
    /// Call repeatedly until it returns `.needMoreData`: a single read can carry
    /// more than one pipelined request.
    public func next() -> Outcome {
        while true {
            switch state {
            case .failed:
                return .needMoreData

            case .head:
                guard let headerEnd = indexOfHeaderTerminator() else {
                    if available > limits.maxHeaderBytes {
                        return fail(status: 431, message: "Request header fields too large")
                    }
                    compactIfNeeded()
                    return .needMoreData
                }

                if headerEnd - readOffset > limits.maxHeaderBytes {
                    return fail(status: 431, message: "Request header fields too large")
                }

                let headBytes = Array(buffer[readOffset..<headerEnd])
                readOffset = headerEnd + 4

                switch parseHead(headBytes) {
                case .failure(let status, let message):
                    return fail(status: status, message: message)
                case .success(let head, let framing):
                    pendingHead = head
                    pendingBody = Data()

                    switch framing {
                    case .none:
                        return emitRequest()
                    case .fixed(let length):
                        // Decide on the size before answering Expect: 100-continue,
                        // so a request already destined for 413 is never invited to
                        // send its body (RFC 9110 10.1.1).
                        if length > limits.maxBodyBytes {
                            return fail(status: 413, message: "Request body exceeds the \(limits.maxBodyBytes) byte limit")
                        }
                        if length == 0 { return emitRequest() }
                        onHeadReceived?(head)
                        pendingBody.reserveCapacity(min(length, 1024 * 1024))
                        state = .fixedBody(remaining: length)
                    case .chunked:
                        onHeadReceived?(head)
                        state = .chunkSize
                    }
                }

            case .fixedBody(let remaining):
                let take = min(remaining, available)
                if take > 0 {
                    pendingBody.append(contentsOf: buffer[readOffset..<(readOffset + take)])
                    readOffset += take
                }
                if take == remaining {
                    return emitRequest()
                }
                state = .fixedBody(remaining: remaining - take)
                compactIfNeeded()
                return .needMoreData

            case .chunkSize:
                guard let lineEnd = indexOfCRLF(from: readOffset) else {
                    if available > 1024 {
                        return fail(status: 400, message: "Malformed chunked encoding")
                    }
                    compactIfNeeded()
                    return .needMoreData
                }
                let line = String(decoding: buffer[readOffset..<lineEnd], as: UTF8.self)
                readOffset = lineEnd + 2

                let sizeText = line.split(separator: ";", maxSplits: 1).first.map(String.init) ?? ""
                guard let size = Int(sizeText.trimmingCharacters(in: .whitespaces), radix: 16), size >= 0 else {
                    return fail(status: 400, message: "Malformed chunk size")
                }
                // Subtract rather than add: a chunk size near Int.max would
                // overflow the addition, and Swift traps on that.
                guard size <= limits.maxBodyBytes - pendingBody.count else {
                    return fail(status: 413, message: "Request body exceeds the \(limits.maxBodyBytes) byte limit")
                }
                state = size == 0 ? .trailers : .chunkData(remaining: size)

            case .chunkData(let remaining):
                let take = min(remaining, available)
                if take > 0 {
                    pendingBody.append(contentsOf: buffer[readOffset..<(readOffset + take)])
                    readOffset += take
                }
                if take == remaining {
                    state = .chunkTerminator
                } else {
                    state = .chunkData(remaining: remaining - take)
                    compactIfNeeded()
                    return .needMoreData
                }

            case .chunkTerminator:
                guard available >= 2 else {
                    compactIfNeeded()
                    return .needMoreData
                }
                guard buffer[readOffset] == 0x0D, buffer[readOffset + 1] == 0x0A else {
                    return fail(status: 400, message: "Malformed chunked encoding")
                }
                readOffset += 2
                state = .chunkSize

            case .trailers:
                guard let lineEnd = indexOfCRLF(from: readOffset) else {
                    if available > limits.maxHeaderBytes {
                        return fail(status: 431, message: "Trailer fields too large")
                    }
                    compactIfNeeded()
                    return .needMoreData
                }
                let isBlank = lineEnd == readOffset
                readOffset = lineEnd + 2
                if isBlank {
                    return emitRequest()
                }
                // Trailer fields carry no meaning for this server; skip them.
            }
        }
    }

    // MARK: - Head parsing

    private enum Framing {
        case none
        case fixed(Int)
        case chunked
    }

    private enum HeadParseResult {
        case success(HTTPRequest, Framing)
        case failure(status: Int, message: String)
    }

    private enum FramingResult {
        case framing(Framing)
        case failure(status: Int, message: String)
    }

    private func parseHead(_ bytes: [UInt8]) -> HeadParseResult {
        let text = String(decoding: bytes, as: UTF8.self)
        var lines = text.components(separatedBy: "\r\n")

        guard let requestLine = lines.first, !requestLine.isEmpty else {
            return .failure(status: 400, message: "Malformed request line")
        }
        lines.removeFirst()

        let parts = requestLine.split(separator: " ", omittingEmptySubsequences: true)
        guard parts.count == 3 else {
            return .failure(status: 400, message: "Malformed request line")
        }

        let method = parts[0].uppercased()
        let version = parts[2].uppercased()
        guard version == "HTTP/1.1" || version == "HTTP/1.0" else {
            return .failure(status: 505, message: "Only HTTP/1.0 and HTTP/1.1 are supported")
        }

        var headers: [String: String] = [:]
        for line in lines where !line.isEmpty {
            // Obsolete line folding is a request-smuggling vector; reject it.
            if line.hasPrefix(" ") || line.hasPrefix("\t") {
                return .failure(status: 400, message: "Obsolete header line folding is not supported")
            }
            guard let separator = line.firstIndex(of: ":") else {
                return .failure(status: 400, message: "Malformed header line")
            }
            let name = String(line[line.startIndex..<separator]).lowercased()
            guard !name.isEmpty, !name.contains(" "), !name.contains("\t") else {
                return .failure(status: 400, message: "Malformed header name")
            }
            let value = String(line[line.index(after: separator)...]).trimmingCharacters(in: .whitespaces)
            if let existing = headers[name] {
                headers[name] = existing + ", " + value
            } else {
                headers[name] = value
            }
        }

        let (path, query) = Self.splitTarget(String(parts[1]))

        let request = HTTPRequest(
            method: method,
            path: path,
            query: query,
            headers: headers,
            body: Data(),
            version: version
        )

        switch framing(for: headers) {
        case .failure(let status, let message):
            return .failure(status: status, message: message)
        case .framing(let framing):
            return .success(request, framing)
        }
    }

    private func framing(for headers: [String: String]) -> FramingResult {
        let hasTransferEncoding = headers["transfer-encoding"] != nil
        let hasContentLength = headers["content-length"] != nil

        // RFC 9112 6.3: a message with both is a smuggling risk — reject it.
        if hasTransferEncoding && hasContentLength {
            return .failure(status: 400, message: "Content-Length and Transfer-Encoding must not both be present")
        }

        if let encoding = headers["transfer-encoding"]?.lowercased() {
            let codings = encoding.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            guard codings.last == "chunked", codings.filter({ $0 == "chunked" }).count == 1 else {
                return .failure(status: 501, message: "Unsupported Transfer-Encoding: \(encoding)")
            }
            return .framing(.chunked)
        }

        if let lengthText = headers["content-length"] {
            guard let length = Int(lengthText.trimmingCharacters(in: .whitespaces)), length >= 0 else {
                return .failure(status: 400, message: "Invalid Content-Length")
            }
            return .framing(.fixed(length))
        }

        return .framing(.none)
    }

    /// Split a request target into a percent-decoded path and query parameters.
    static func splitTarget(_ target: String) -> (path: String, query: [String: String]) {
        var rest = target

        // Absolute-form targets (proxy style) still name a path after the authority.
        for scheme in ["http://", "https://"] where rest.lowercased().hasPrefix(scheme) {
            let afterScheme = rest.dropFirst(scheme.count)
            if let slash = afterScheme.firstIndex(of: "/") {
                rest = String(afterScheme[slash...])
            } else {
                rest = "/"
            }
        }

        // A fragment is never sent by a compliant client, but strip it if present.
        if let hash = rest.firstIndex(of: "#") {
            rest = String(rest[rest.startIndex..<hash])
        }

        let pathPart: String
        var query: [String: String] = [:]

        if let mark = rest.firstIndex(of: "?") {
            pathPart = String(rest[rest.startIndex..<mark])
            let queryPart = rest[rest.index(after: mark)...]
            for pair in queryPart.split(separator: "&") {
                let kv = pair.split(separator: "=", maxSplits: 1)
                guard let rawKey = kv.first else { continue }
                let key = percentDecode(String(rawKey))
                let value = kv.count > 1 ? percentDecode(String(kv[1])) : ""
                query[key] = value
            }
        } else {
            pathPart = rest
        }

        let path = percentDecode(pathPart)
        return (path.isEmpty ? "/" : path, query)
    }

    static func percentDecode(_ string: String) -> String {
        let plusDecoded = string.replacingOccurrences(of: "+", with: " ")
        return plusDecoded.removingPercentEncoding ?? plusDecoded
    }

    // MARK: - Buffer helpers

    private var available: Int { buffer.count - readOffset }

    private func indexOfCRLF(from start: Int) -> Int? {
        guard buffer.count >= 2 else { return nil }
        var index = start
        while index + 1 < buffer.count {
            if buffer[index] == 0x0D && buffer[index + 1] == 0x0A { return index }
            index += 1
        }
        return nil
    }

    private func indexOfHeaderTerminator() -> Int? {
        guard buffer.count >= 4 else { return nil }
        var index = readOffset
        while index + 3 < buffer.count {
            if buffer[index] == 0x0D, buffer[index + 1] == 0x0A,
               buffer[index + 2] == 0x0D, buffer[index + 3] == 0x0A {
                return index
            }
            index += 1
        }
        return nil
    }

    private func compactIfNeeded() {
        guard readOffset > 0 else { return }
        if readOffset == buffer.count {
            buffer.removeAll(keepingCapacity: true)
            readOffset = 0
        } else if readOffset > 64 * 1024 {
            buffer.removeFirst(readOffset)
            readOffset = 0
        }
    }

    private func emitRequest() -> Outcome {
        guard let head = pendingHead else {
            return fail(status: 500, message: "Parser lost the request head")
        }
        let request = HTTPRequest(
            method: head.method,
            path: head.path,
            query: head.query,
            headers: head.headers,
            body: pendingBody,
            version: head.version
        )
        pendingHead = nil
        pendingBody = Data()
        state = .head
        compactIfNeeded()
        return .request(request)
    }

    private func fail(status: Int, message: String) -> Outcome {
        state = .failed
        return .failure(status: status, message: message)
    }
}
