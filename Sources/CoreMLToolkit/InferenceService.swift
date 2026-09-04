import Foundation
import Network

/// The prediction surface the HTTP layer needs, so routing can be tested
/// without loading a real Core ML model.
public protocol InferenceEngine: AnyObject {
    var inputNames: [String] { get }
    func predict(source: InputSource) throws -> PredictionResult
    func predict(namedValues: [String: Any], label: String) throws -> PredictionResult
}

extension ModelPredictor: InferenceEngine {}

/// Maps HTTP requests onto a loaded model.
///
/// Routes:
/// - `GET /`            — the endpoints on offer
/// - `GET /health`      — liveness, uptime and request counters
/// - `GET /v1/info`     — the model's inputs, outputs and metadata
/// - `POST /v1/predict` — inference from JSON, a raw body, or a file upload
public final class InferenceService {

    public struct Configuration {
        /// Require this key via `Authorization: Bearer` or `X-API-Key`.
        public var apiKey: String?
        /// Answer preflights and add `Access-Control-Allow-*` headers.
        public var allowCORS: Bool
        /// Host header names to accept, alongside any literal IP address.
        /// `nil` accepts any Host.
        public var allowedHostNames: Set<String>?

        public init(
            apiKey: String? = nil,
            allowCORS: Bool = false,
            allowedHostNames: Set<String>? = nil
        ) {
            self.apiKey = apiKey
            self.allowCORS = allowCORS
            self.allowedHostNames = allowedHostNames
        }
    }

    /// Counters reported by `/health`.
    public struct Stats: Codable {
        public var requests: Int
        public var predictions: Int
        public var errors: Int
    }

    private let engine: InferenceEngine
    private let modelInfo: ModelInfo
    private let configuration: Configuration
    private let startedAt = Date()
    private let lock = NSLock()
    private var stats = Stats(requests: 0, predictions: 0, errors: 0)

    public init(engine: InferenceEngine, modelInfo: ModelInfo, configuration: Configuration = Configuration()) {
        self.engine = engine
        self.modelInfo = modelInfo
        self.configuration = configuration
    }

    /// Handle one request. Never throws: every failure becomes a response.
    public func handle(_ request: HTTPRequest) -> HTTPResponse {
        record { $0.requests += 1 }

        if let denial = hostFailure(for: request) {
            return finish(denial, for: request)
        }

        if request.method == "OPTIONS" && configuration.allowCORS {
            return decorate(HTTPResponse(status: 204), for: request)
        }

        if let denial = authorizationFailure(for: request) {
            return finish(denial, for: request)
        }

        switch (request.method, request.path) {
        case ("GET", "/"), ("HEAD", "/"):
            return finish(index(), for: request)

        case ("GET", "/health"), ("HEAD", "/health"):
            return finish(health(), for: request)

        case ("GET", "/v1/info"), ("HEAD", "/v1/info"):
            return finish(.json(modelInfo), for: request)

        case ("POST", "/v1/predict"):
            return finish(predict(request), for: request)

        case (_, "/v1/predict"):
            return finish(.error(status: 405, message: "Use POST for /v1/predict"), for: request)

        case (_, "/"), (_, "/health"), (_, "/v1/info"):
            return finish(.error(status: 405, message: "\(request.method) is not allowed on \(request.path)"), for: request)

        default:
            return finish(.error(status: 404, message: "No route for \(request.method) \(request.path)"), for: request)
        }
    }

    // MARK: - Routes

    private func index() -> HTTPResponse {
        let payload: [String: Any] = [
            "service": "coreml serve",
            "model": modelInfo.name,
            "endpoints": [
                ["method": "GET", "path": "/health", "description": "Liveness, uptime and request counters"],
                ["method": "GET", "path": "/v1/info", "description": "Model inputs, outputs and metadata"],
                ["method": "POST", "path": "/v1/predict", "description": "Run inference on JSON, a raw body, or an uploaded file"]
            ],
            "inputs": modelInfo.inputs.map { $0.name },
            "outputs": modelInfo.outputs.map { $0.name }
        ]
        return jsonResponse(payload)
    }

    private func health() -> HTTPResponse {
        let snapshot = currentStats
        let payload: [String: Any] = [
            "status": "ok",
            "model": modelInfo.name,
            "uptimeSeconds": Int(Date().timeIntervalSince(startedAt)),
            "requests": snapshot.requests,
            "predictions": snapshot.predictions,
            "errors": snapshot.errors
        ]
        return jsonResponse(payload)
    }

    private func predict(_ request: HTTPRequest) -> HTTPResponse {
        guard !request.body.isEmpty else {
            return .error(status: 400, message: "Request body is empty. Send JSON, raw bytes, or a multipart file upload.")
        }

        let top = request.query["top"].flatMap(Int.init)
        if let top, top < 1 {
            return .error(status: 400, message: "top must be a positive integer")
        }

        do {
            let result = try runPrediction(request)
            record { $0.predictions += 1 }
            return jsonResponse(payload(for: result, top: top))
        } catch let error as PredictorError {
            return .error(status: status(for: error), message: error.errorDescription ?? "Prediction failed")
        } catch let error as MultipartError {
            return .error(status: 400, message: error.errorDescription ?? "Malformed upload")
        } catch let error as InferenceRequestError {
            return .error(status: error.status, message: error.message)
        } catch {
            return .error(status: 500, message: "Prediction failed: \(error.localizedDescription)")
        }
    }

    private func runPrediction(_ request: HTTPRequest) throws -> PredictionResult {
        switch request.contentType {
        case "application/json", "text/json":
            return try predictFromJSON(request.body)

        case "multipart/form-data":
            guard let boundary = request.multipartBoundary else {
                throw MultipartError.missingBoundary
            }
            let parts = try MultipartParser.parse(body: request.body, boundary: boundary)
            guard let part = parts.first(where: { $0.filename != nil }) ?? parts.first else {
                throw InferenceRequestError(status: 400, message: "Upload contained no parts")
            }
            guard !part.data.isEmpty else {
                throw InferenceRequestError(status: 400, message: "Uploaded file is empty")
            }
            return try engine.predict(source: InputSource(
                name: part.filename ?? part.name ?? "upload",
                data: part.data,
                mediaType: part.contentType
            ))

        default:
            // Raw bytes: an image, a text prompt, a JSON tensor without the
            // header set. Applied to the model's input as a file would be.
            return try engine.predict(source: InputSource(
                name: "request",
                data: request.body,
                mediaType: request.contentType
            ))
        }
    }

    private func predictFromJSON(_ body: Data) throws -> PredictionResult {
        let json: Any
        do {
            json = try JSONSerialization.jsonObject(with: body, options: [.fragmentsAllowed])
        } catch {
            throw InferenceRequestError(status: 400, message: "Body is not valid JSON: \(error.localizedDescription)")
        }

        let inputNames = engine.inputNames

        // A bare array is a tensor for a single-input model.
        if let array = json as? [Any] {
            guard inputNames.count == 1, let name = inputNames.first else {
                throw InferenceRequestError(
                    status: 400,
                    message: "This model has \(inputNames.count) inputs — send an object keyed by input name: \(inputNames.joined(separator: ", "))"
                )
            }
            return try engine.predict(namedValues: [name: array], label: "request")
        }

        guard let object = json as? [String: Any] else {
            throw InferenceRequestError(status: 400, message: "Body must be a JSON object or array")
        }

        // { "inputs": { "<name>": <value> } } — unless the model really has an
        // input called "inputs", in which case the object is already keyed by name.
        if let inputs = object["inputs"], !inputNames.contains("inputs") {
            guard let named = inputs as? [String: Any] else {
                throw InferenceRequestError(status: 400, message: "\"inputs\" must be an object keyed by input name")
            }
            return try engine.predict(namedValues: named, label: "request")
        }

        // { "input": <value> } for a single-input model.
        if let value = object["input"], !inputNames.contains("input") {
            guard inputNames.count == 1, let name = inputNames.first else {
                throw InferenceRequestError(
                    status: 400,
                    message: "This model has \(inputNames.count) inputs — use {\"inputs\": {…}} with the names: \(inputNames.joined(separator: ", "))"
                )
            }
            return try engine.predict(namedValues: [name: value], label: "request")
        }

        // Otherwise the object is already keyed by input name.
        return try engine.predict(namedValues: object, label: "request")
    }

    // MARK: - Response shaping

    private func payload(for result: PredictionResult, top: Int?) -> [String: Any] {
        var outputs: [String: Any] = [:]
        var ranked: [String: Any] = [:]

        for (name, value) in result.outputs {
            switch value {
            case .string(let string):
                outputs[name] = string
            case .double(let double):
                outputs[name] = Self.jsonSafe(double)
            case .int(let int):
                outputs[name] = int
            case .array(let array):
                outputs[name] = array.map(Self.jsonSafe)
            case .dictionary(let dictionary):
                let sorted = dictionary.sorted { lhs, rhs in
                    lhs.value == rhs.value ? lhs.key < rhs.key : lhs.value > rhs.value
                }
                let limited = top.map { Array(sorted.prefix($0)) } ?? sorted
                outputs[name] = Dictionary(uniqueKeysWithValues: limited.map { ($0.key, Self.jsonSafe($0.value)) })
                // JSON objects have no order, so ranked results get an array too.
                ranked[name] = limited.map { ["label": $0.key, "score": Self.jsonSafe($0.value)] }
            }
        }

        var payload: [String: Any] = [
            "model": modelInfo.name,
            "inferenceTimeMs": result.inferenceTimeMs,
            "outputs": outputs
        ]
        if !ranked.isEmpty {
            payload["ranked"] = ranked
        }
        return payload
    }

    private func status(for error: PredictorError) -> Int {
        switch error {
        case .modelNotLoaded:
            return 503
        case .modelNotFound, .inputNotFound, .pixelBufferCreationFailed:
            return 500
        case .missingImageConstraint, .missingMultiArrayConstraint, .unsupportedInputType:
            return 501
        case .invalidImage, .invalidInputFormat, .shapeMismatch, .missingInput, .unknownInput,
             .invalidInputValue, .nonNumericTensorValue:
            return 422
        }
    }

    /// JSON has no NaN or infinity, and JSONSerialization answers one by raising an
    /// Objective-C exception — which Swift cannot catch, so it takes the process
    /// down rather than failing the request. Models do emit them, so replace any
    /// non-finite value with null before it reaches the encoder.
    static func jsonSafe(_ value: Double) -> Any {
        value.isFinite ? value : NSNull()
    }

    private func jsonResponse(_ payload: [String: Any], status: Int = 200) -> HTTPResponse {
        // Belt and braces: isValidJSONObject is the check that does not throw.
        guard JSONSerialization.isValidJSONObject(payload),
              let data = try? JSONSerialization.data(
                  withJSONObject: payload,
                  options: [.prettyPrinted, .sortedKeys]
              ) else {
            return .error(status: 500, message: "Failed to encode response")
        }
        return .json(data, status: status)
    }

    // MARK: - Auth, CORS, stats

    /// Reject a Host the server was not reached by.
    ///
    /// A page in someone's browser can point a domain it controls at 127.0.0.1
    /// (DNS rebinding) and drive a loopback-bound server from a remote origin.
    /// Requiring the Host to be a literal address — or a name the operator listed —
    /// closes that, since the attack depends on a domain name.
    private func hostFailure(for request: HTTPRequest) -> HTTPResponse? {
        guard let allowed = configuration.allowedHostNames else { return nil }

        guard let raw = request.header("host"), !raw.isEmpty else {
            return .error(status: 400, message: "Missing Host header")
        }

        let name = Self.hostName(from: raw)
        if IPv4Address(name) != nil || IPv6Address(name) != nil { return nil }
        if allowed.contains(name) { return nil }

        return .error(
            status: 421,
            message: "Host '\(name)' is not served here. Reach the server by IP address, or pass --allowed-host \(name)."
        )
    }

    /// Strip the port and any IPv6 brackets from a Host header value.
    static func hostName(from header: String) -> String {
        var value = header.trimmingCharacters(in: .whitespaces).lowercased()

        if value.hasPrefix("[") {
            guard let end = value.firstIndex(of: "]") else { return value }
            return String(value[value.index(after: value.startIndex)..<end])
        }

        // A bare IPv6 literal has several colons; only strip a single trailing port.
        if value.filter({ $0 == ":" }).count == 1, let colon = value.lastIndex(of: ":") {
            value = String(value[value.startIndex..<colon])
        }
        return value
    }

    private func authorizationFailure(for request: HTTPRequest) -> HTTPResponse? {
        guard let expected = configuration.apiKey, !expected.isEmpty else { return nil }

        var presented = request.header("x-api-key")
        if presented == nil, let authorization = request.header("authorization") {
            let parts = authorization.split(separator: " ", maxSplits: 1)
            if parts.count == 2, parts[0].lowercased() == "bearer" {
                presented = String(parts[1])
            }
        }

        guard let presented, Self.constantTimeEquals(presented, expected) else {
            return .error(status: 401, message: "Missing or invalid API key")
        }
        return nil
    }

    /// Compare without leaking the answer through how long it takes.
    static func constantTimeEquals(_ lhs: String, _ rhs: String) -> Bool {
        let left = Array(lhs.utf8)
        let right = Array(rhs.utf8)
        var difference = left.count ^ right.count
        for index in 0..<max(left.count, right.count) {
            let l = index < left.count ? Int(left[index]) : 0
            let r = index < right.count ? Int(right[index]) : 0
            difference |= l ^ r
        }
        return difference == 0
    }

    private func decorate(_ response: HTTPResponse, for request: HTTPRequest) -> HTTPResponse {
        guard configuration.allowCORS else { return response }
        var decorated = response
        decorated.headers["Access-Control-Allow-Origin"] = "*"
        decorated.headers["Access-Control-Allow-Methods"] = "GET, POST, HEAD, OPTIONS"
        decorated.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
        decorated.headers["Access-Control-Max-Age"] = "86400"
        return decorated
    }

    private func finish(_ response: HTTPResponse, for request: HTTPRequest) -> HTTPResponse {
        if response.status >= 400 {
            record { $0.errors += 1 }
        }
        return decorate(response, for: request)
    }

    public var currentStats: Stats {
        lock.lock()
        defer { lock.unlock() }
        return stats
    }

    private func record(_ change: (inout Stats) -> Void) {
        lock.lock()
        change(&stats)
        lock.unlock()
    }
}

/// A request the service can reject before it reaches the model.
struct InferenceRequestError: Error {
    let status: Int
    let message: String
}
