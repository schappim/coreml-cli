import XCTest
@testable import CoreMLToolkit

/// Records what the service asked of the model, and replays a canned result.
private final class FakeEngine: InferenceEngine {
    var inputNames: [String]
    var result: PredictionResult
    var error: Error?

    private(set) var receivedSource: InputSource?
    private(set) var receivedNamedValues: [String: Any]?

    init(
        inputNames: [String] = ["image"],
        outputs: [String: PredictionValue] = ["classLabel": .string("cat")]
    ) {
        self.inputNames = inputNames
        self.result = PredictionResult(inputFile: "request", outputs: outputs, inferenceTimeMs: 1.5)
    }

    func predict(source: InputSource) throws -> PredictionResult {
        if let error { throw error }
        receivedSource = source
        return result
    }

    func predict(namedValues: [String: Any], label: String) throws -> PredictionResult {
        if let error { throw error }
        receivedNamedValues = namedValues
        return result
    }
}

final class InferenceServiceTests: XCTestCase {

    private func makeModelInfo(inputs: [String] = ["image"]) -> ModelInfo {
        ModelInfo(
            name: "TestModel",
            inputs: inputs.map { FeatureInfo(name: $0, type: "image") },
            outputs: [FeatureInfo(name: "classLabel", type: "string")],
            metadata: ModelMetadata(author: "Tester"),
            fileSize: 1024,
            isCompiled: false
        )
    }

    private func makeService(
        engine: FakeEngine = FakeEngine(),
        configuration: InferenceService.Configuration = .init()
    ) -> InferenceService {
        InferenceService(
            engine: engine,
            modelInfo: makeModelInfo(inputs: engine.inputNames),
            configuration: configuration
        )
    }

    private func post(_ body: Data, contentType: String, query: [String: String] = [:], headers: [String: String] = [:]) -> HTTPRequest {
        var allHeaders = headers
        allHeaders["content-type"] = contentType
        return HTTPRequest(method: "POST", path: "/v1/predict", query: query, headers: allHeaders, body: body)
    }

    private func json(_ response: HTTPResponse) throws -> [String: Any] {
        try XCTUnwrap(try JSONSerialization.jsonObject(with: response.body) as? [String: Any])
    }

    // MARK: - Routes

    func testIndexListsEndpoints() throws {
        let response = makeService().handle(HTTPRequest(method: "GET", path: "/"))
        XCTAssertEqual(response.status, 200)

        let body = try json(response)
        XCTAssertEqual(body["model"] as? String, "TestModel")
        XCTAssertEqual((body["endpoints"] as? [[String: String]])?.count, 3)
    }

    func testHealthReportsCounters() throws {
        let service = makeService()
        _ = service.handle(HTTPRequest(method: "GET", path: "/"))

        let response = service.handle(HTTPRequest(method: "GET", path: "/health"))
        let body = try json(response)

        XCTAssertEqual(body["status"] as? String, "ok")
        XCTAssertEqual(body["requests"] as? Int, 2)
        XCTAssertEqual(body["errors"] as? Int, 0)
        XCTAssertNotNil(body["uptimeSeconds"])
    }

    func testInfoReturnsModelDescription() throws {
        let response = makeService().handle(HTTPRequest(method: "GET", path: "/v1/info"))
        XCTAssertEqual(response.status, 200)

        let body = try json(response)
        XCTAssertEqual(body["name"] as? String, "TestModel")
        XCTAssertEqual((body["inputs"] as? [[String: Any]])?.count, 1)
    }

    func testUnknownRouteIs404() {
        let response = makeService().handle(HTTPRequest(method: "GET", path: "/nope"))
        XCTAssertEqual(response.status, 404)
    }

    func testWrongMethodIs405() {
        let response = makeService().handle(HTTPRequest(method: "GET", path: "/v1/predict"))
        XCTAssertEqual(response.status, 405)
    }

    // MARK: - Prediction inputs

    func testRawBodyIsPassedThroughAsBytes() throws {
        let engine = FakeEngine()
        let service = makeService(engine: engine)

        let bytes = Data([0xFF, 0xD8, 0xFF, 0xE0])
        let response = service.handle(post(bytes, contentType: "image/jpeg"))

        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(engine.receivedSource?.data, bytes)
        XCTAssertEqual(engine.receivedSource?.mediaType, "image/jpeg")
    }

    func testBareJSONArrayGoesToTheSingleInput() throws {
        let engine = FakeEngine(inputNames: ["features"])
        let service = makeService(engine: engine)

        let response = service.handle(post(Data("[5.1, 3.5]".utf8), contentType: "application/json"))
        XCTAssertEqual(response.status, 200)

        let values = try XCTUnwrap(engine.receivedNamedValues?["features"] as? [Any])
        XCTAssertEqual(values.count, 2)
    }

    func testBareJSONArrayRejectedForMultiInputModel() throws {
        let engine = FakeEngine(inputNames: ["a", "b"])
        let service = makeService(engine: engine)

        let response = service.handle(post(Data("[1, 2]".utf8), contentType: "application/json"))
        XCTAssertEqual(response.status, 400)

        let message = try XCTUnwrap((try json(response)["error"] as? [String: Any])?["message"] as? String)
        XCTAssertTrue(message.contains("a"), "Error should name the model's inputs")
    }

    func testInputsObjectIsForwarded() throws {
        let engine = FakeEngine(inputNames: ["a", "b"])
        let service = makeService(engine: engine)

        let body = Data(#"{"inputs": {"a": [1], "b": "text"}}"#.utf8)
        let response = service.handle(post(body, contentType: "application/json"))

        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(engine.receivedNamedValues?.count, 2)
        XCTAssertEqual(engine.receivedNamedValues?["b"] as? String, "text")
    }

    func testSingleInputShorthand() throws {
        let engine = FakeEngine(inputNames: ["text"])
        let service = makeService(engine: engine)

        let response = service.handle(post(Data(#"{"input": "hello"}"#.utf8), contentType: "application/json"))
        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(engine.receivedNamedValues?["text"] as? String, "hello")
    }

    func testObjectKeyedByInputNameIsAccepted() throws {
        let engine = FakeEngine(inputNames: ["text"])
        let service = makeService(engine: engine)

        let response = service.handle(post(Data(#"{"text": "hello"}"#.utf8), contentType: "application/json"))
        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(engine.receivedNamedValues?["text"] as? String, "hello")
    }

    func testMultipartUploadUsesTheFilePart() throws {
        let engine = FakeEngine()
        let service = makeService(engine: engine)

        var body = Data("--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"cat.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".utf8)
        body.append(Data([0x01, 0x02, 0x03]))
        body.append(Data("\r\n--B--\r\n".utf8))

        let response = service.handle(post(body, contentType: "multipart/form-data; boundary=B"))

        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(engine.receivedSource?.name, "cat.jpg")
        XCTAssertEqual(engine.receivedSource?.data, Data([0x01, 0x02, 0x03]))
    }

    func testMultipartWithoutBoundaryIsRejected() {
        let response = makeService().handle(post(Data("x".utf8), contentType: "multipart/form-data"))
        XCTAssertEqual(response.status, 400)
    }

    func testEmptyBodyIsRejected() {
        let response = makeService().handle(post(Data(), contentType: "application/json"))
        XCTAssertEqual(response.status, 400)
    }

    func testMalformedJSONIsRejected() {
        let response = makeService().handle(post(Data("{oops".utf8), contentType: "application/json"))
        XCTAssertEqual(response.status, 400)
    }

    // MARK: - Response shaping

    func testDictionaryOutputsCanBeRanked() throws {
        let engine = FakeEngine(outputs: [
            "probs": .dictionary(["cat": 0.7, "dog": 0.2, "fox": 0.1])
        ])
        let service = makeService(engine: engine)

        let response = service.handle(post(Data("[1]".utf8), contentType: "application/json", query: ["top": "2"]))
        XCTAssertEqual(response.status, 200)

        let body = try json(response)
        let outputs = try XCTUnwrap(body["outputs"] as? [String: Any])
        let probs = try XCTUnwrap(outputs["probs"] as? [String: Double])
        XCTAssertEqual(probs.count, 2, "top=2 should trim the dictionary")
        XCTAssertNil(probs["fox"])

        let ranked = try XCTUnwrap(body["ranked"] as? [String: Any])
        let entries = try XCTUnwrap(ranked["probs"] as? [[String: Any]])
        XCTAssertEqual(entries.first?["label"] as? String, "cat", "Ranked output should be ordered")
        XCTAssertEqual(entries.count, 2)
    }

    func testFullDictionaryReturnedWithoutTop() throws {
        let engine = FakeEngine(outputs: ["probs": .dictionary(["cat": 0.7, "dog": 0.3])])
        let service = makeService(engine: engine)

        let response = service.handle(post(Data("[1]".utf8), contentType: "application/json"))
        let outputs = try XCTUnwrap(try json(response)["outputs"] as? [String: Any])
        XCTAssertEqual((outputs["probs"] as? [String: Double])?.count, 2)
    }

    func testInvalidTopIsRejected() {
        let response = makeService().handle(
            post(Data("[1]".utf8), contentType: "application/json", query: ["top": "0"])
        )
        XCTAssertEqual(response.status, 400)
    }

    func testResponseCarriesModelNameAndTiming() throws {
        let response = makeService().handle(post(Data("[1]".utf8), contentType: "application/json"))
        let body = try json(response)
        XCTAssertEqual(body["model"] as? String, "TestModel")
        XCTAssertEqual(body["inferenceTimeMs"] as? Double, 1.5)
    }

    func testNonFiniteOutputsDoNotCrashTheEncoder() throws {
        // JSONSerialization raises an uncatchable ObjC exception on NaN/Inf, so a
        // model emitting one used to abort the whole server process.
        let engine = FakeEngine(outputs: [
            "score": .double(.nan),
            "embedding": .array([1.0, .infinity, -.infinity, 2.0]),
            "probs": .dictionary(["a": .nan, "b": 0.5])
        ])
        let service = makeService(engine: engine)

        let response = service.handle(post(Data("[1]".utf8), contentType: "application/json"))
        XCTAssertEqual(response.status, 200)

        let outputs = try XCTUnwrap(try json(response)["outputs"] as? [String: Any])
        XCTAssertTrue(outputs["score"] is NSNull)

        let embedding = try XCTUnwrap(outputs["embedding"] as? [Any])
        XCTAssertEqual(embedding.count, 4)
        XCTAssertEqual(embedding[0] as? Double, 1.0)
        XCTAssertTrue(embedding[1] is NSNull)
        XCTAssertTrue(embedding[2] is NSNull)

        let probs = try XCTUnwrap(outputs["probs"] as? [String: Any])
        XCTAssertTrue(probs["a"] is NSNull)
        XCTAssertEqual(probs["b"] as? Double, 0.5)
    }

    func testNonFiniteRankedScoresAreAlsoSafe() throws {
        let engine = FakeEngine(outputs: ["probs": .dictionary(["a": .infinity, "b": 0.5])])
        let service = makeService(engine: engine)

        let response = service.handle(post(Data("[1]".utf8), contentType: "application/json", query: ["top": "2"]))
        XCTAssertEqual(response.status, 200)

        let ranked = try XCTUnwrap(try json(response)["ranked"] as? [String: Any])
        let entries = try XCTUnwrap(ranked["probs"] as? [[String: Any]])
        XCTAssertEqual(entries.count, 2)
    }

    // MARK: - Error mapping

    func testShapeMismatchBecomes422() throws {
        let engine = FakeEngine()
        engine.error = PredictorError.shapeMismatch(expected: 4, got: 3, shape: [4])
        let service = makeService(engine: engine)

        let response = service.handle(post(Data("[1,2,3]".utf8), contentType: "application/json"))
        XCTAssertEqual(response.status, 422)

        let message = try XCTUnwrap((try json(response)["error"] as? [String: Any])?["message"] as? String)
        XCTAssertTrue(message.contains("4"))
    }

    func testModelNotLoadedBecomes503() {
        let engine = FakeEngine()
        engine.error = PredictorError.modelNotLoaded
        let service = makeService(engine: engine)

        let response = service.handle(post(Data("[1]".utf8), contentType: "application/json"))
        XCTAssertEqual(response.status, 503)
    }

    func testUnexpectedErrorBecomes500() {
        struct Boom: Error {}
        let engine = FakeEngine()
        engine.error = Boom()
        let service = makeService(engine: engine)

        let response = service.handle(post(Data("[1]".utf8), contentType: "application/json"))
        XCTAssertEqual(response.status, 500)
    }

    func testErrorsAreCounted() throws {
        let service = makeService()
        _ = service.handle(HTTPRequest(method: "GET", path: "/nope"))

        let body = try json(service.handle(HTTPRequest(method: "GET", path: "/health")))
        XCTAssertEqual(body["errors"] as? Int, 1)
    }

    // MARK: - Auth and CORS

    func testApiKeyIsRequiredWhenConfigured() {
        let service = makeService(configuration: .init(apiKey: "secret"))
        XCTAssertEqual(service.handle(HTTPRequest(method: "GET", path: "/health")).status, 401)
    }

    func testBearerTokenIsAccepted() {
        let service = makeService(configuration: .init(apiKey: "secret"))
        let request = HTTPRequest(
            method: "GET",
            path: "/health",
            headers: ["authorization": "Bearer secret"]
        )
        XCTAssertEqual(service.handle(request).status, 200)
    }

    func testApiKeyHeaderIsAccepted() {
        let service = makeService(configuration: .init(apiKey: "secret"))
        let request = HTTPRequest(method: "GET", path: "/health", headers: ["x-api-key": "secret"])
        XCTAssertEqual(service.handle(request).status, 200)
    }

    func testWrongApiKeyIsRejected() {
        let service = makeService(configuration: .init(apiKey: "secret"))
        let request = HTTPRequest(method: "GET", path: "/health", headers: ["x-api-key": "wrong"])
        XCTAssertEqual(service.handle(request).status, 401)
    }

    func testConstantTimeCompare() {
        XCTAssertTrue(InferenceService.constantTimeEquals("abc", "abc"))
        XCTAssertFalse(InferenceService.constantTimeEquals("abc", "abd"))
        XCTAssertFalse(InferenceService.constantTimeEquals("abc", "abcd"))
        XCTAssertFalse(InferenceService.constantTimeEquals("", "a"))
        XCTAssertTrue(InferenceService.constantTimeEquals("", ""))
    }

    func testCORSPreflightAndHeaders() {
        let service = makeService(configuration: .init(allowCORS: true))

        let preflight = service.handle(HTTPRequest(method: "OPTIONS", path: "/v1/predict"))
        XCTAssertEqual(preflight.status, 204)
        XCTAssertEqual(preflight.headers["Access-Control-Allow-Origin"], "*")

        let normal = service.handle(HTTPRequest(method: "GET", path: "/health"))
        XCTAssertEqual(normal.headers["Access-Control-Allow-Origin"], "*")
    }

    func testCORSHeadersAbsentByDefault() {
        let response = makeService().handle(HTTPRequest(method: "GET", path: "/health"))
        XCTAssertNil(response.headers["Access-Control-Allow-Origin"])
    }
}
