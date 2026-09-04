import XCTest
@testable import CoreMLToolkit

/// End-to-end tests: a real socket on loopback, driven with URLSession.
final class HTTPServerTests: XCTestCase {

    private var server: HTTPServer?

    override func tearDown() {
        server?.stop()
        server = nil
        super.tearDown()
    }

    private func startServer(
        maxBodyBytes: Int = 1024 * 1024,
        maxConcurrentRequests: Int = 4,
        handler: @escaping HTTPServer.Handler
    ) throws -> URL {
        let configuration = HTTPServer.Configuration(
            host: "127.0.0.1",
            port: 0,
            maxBodyBytes: maxBodyBytes,
            maxConcurrentRequests: maxConcurrentRequests,
            idleTimeout: 10
        )
        let server = HTTPServer(configuration: configuration, handler: handler)
        try server.start()
        self.server = server

        let port = try XCTUnwrap(server.boundPort)
        XCTAssertGreaterThan(port, 0, "Server should report the port it bound")
        return try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)"))
    }

    private func send(
        _ request: URLRequest,
        timeout: TimeInterval = 10
    ) throws -> (HTTPURLResponse, Data) {
        let expectation = expectation(description: "response for \(request.url?.path ?? "?")")
        var result: (HTTPURLResponse, Data)?
        var failure: Error?

        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error {
                failure = error
            } else if let http = response as? HTTPURLResponse {
                result = (http, data ?? Data())
            }
            expectation.fulfill()
        }
        task.resume()

        wait(for: [expectation], timeout: timeout)

        if let failure { throw failure }
        return try XCTUnwrap(result)
    }

    func testServesGetRequest() throws {
        let base = try startServer { request in
            .text("hello \(request.path)")
        }

        let (response, data) = try send(URLRequest(url: base.appendingPathComponent("world")))
        XCTAssertEqual(response.statusCode, 200)
        XCTAssertEqual(String(decoding: data, as: UTF8.self), "hello /world")
    }

    func testHandlerSeesBodyAndHeaders() throws {
        let base = try startServer { request in
            .json([
                "method": request.method,
                "contentType": request.contentType ?? "",
                "body": String(decoding: request.body, as: UTF8.self)
            ])
        }

        var request = URLRequest(url: base.appendingPathComponent("echo"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"{"x":1}"#.utf8)

        let (response, data) = try send(request)
        XCTAssertEqual(response.statusCode, 200)

        let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: String])
        XCTAssertEqual(json["method"], "POST")
        XCTAssertEqual(json["contentType"], "application/json")
        XCTAssertEqual(json["body"], #"{"x":1}"#)
    }

    func testQueryParametersReachTheHandler() throws {
        let base = try startServer { request in
            .text(request.query["top"] ?? "none")
        }

        let url = try XCTUnwrap(URL(string: base.absoluteString + "/v1/predict?top=3"))
        let (_, data) = try send(URLRequest(url: url))
        XCTAssertEqual(String(decoding: data, as: UTF8.self), "3")
    }

    func testKeepAliveServesSequentialRequestsOnOneConnection() throws {
        let counter = Counter()
        let base = try startServer { _ in
            .text("\(counter.increment())")
        }

        let session = URLSession(configuration: .ephemeral)
        var seen: [String] = []
        for _ in 0..<3 {
            let expectation = expectation(description: "keep-alive request")
            session.dataTask(with: base.appendingPathComponent("ping")) { data, _, _ in
                seen.append(String(decoding: data ?? Data(), as: UTF8.self))
                expectation.fulfill()
            }.resume()
            wait(for: [expectation], timeout: 10)
        }

        XCTAssertEqual(seen, ["1", "2", "3"])
    }

    func testHeadRequestReturnsHeadersWithoutBody() throws {
        let base = try startServer { _ in .text("hello") }

        var request = URLRequest(url: base.appendingPathComponent("health"))
        request.httpMethod = "HEAD"

        let (response, data) = try send(request)
        XCTAssertEqual(response.statusCode, 200)
        XCTAssertEqual(response.value(forHTTPHeaderField: "Content-Length"), "5")
        XCTAssertTrue(data.isEmpty)
    }

    func testLargeBodyIsReceivedIntact() throws {
        let base = try startServer(maxBodyBytes: 8 * 1024 * 1024) { request in
            .text("\(request.body.count)")
        }

        var request = URLRequest(url: base.appendingPathComponent("upload"))
        request.httpMethod = "POST"
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(repeating: 0xAB, count: 3 * 1024 * 1024)

        let (response, data) = try send(request, timeout: 30)
        XCTAssertEqual(response.statusCode, 200)
        XCTAssertEqual(String(decoding: data, as: UTF8.self), "\(3 * 1024 * 1024)")
    }

    func testOversizedBodyIsRejected() throws {
        let base = try startServer(maxBodyBytes: 1024) { _ in .text("should not run") }

        var request = URLRequest(url: base.appendingPathComponent("upload"))
        request.httpMethod = "POST"
        request.httpBody = Data(repeating: 0x41, count: 64 * 1024)

        let (response, _) = try send(request)
        XCTAssertEqual(response.statusCode, 413)
    }

    func testHandlerErrorsBecomeResponses() throws {
        let base = try startServer { _ in .error(status: 422, message: "bad input") }

        let (response, data) = try send(URLRequest(url: base.appendingPathComponent("v1/predict")))
        XCTAssertEqual(response.statusCode, 422)

        let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let error = try XCTUnwrap(json["error"] as? [String: Any])
        XCTAssertEqual(error["message"] as? String, "bad input")
    }

    func testConcurrentRequestsAreAllServed() throws {
        let counter = Counter()
        let base = try startServer(maxConcurrentRequests: 4) { _ in
            Thread.sleep(forTimeInterval: 0.02)
            return .text("\(counter.increment())")
        }

        let session = URLSession(configuration: .ephemeral)
        let expectations = (0..<12).map { expectation(description: "concurrent \($0)") }

        for expectation in expectations {
            session.dataTask(with: base.appendingPathComponent("predict")) { _, _, _ in
                expectation.fulfill()
            }.resume()
        }

        wait(for: expectations, timeout: 30)
        XCTAssertEqual(counter.value, 12)
    }

    func testStopClosesTheListener() throws {
        let base = try startServer { _ in .text("up") }
        _ = try send(URLRequest(url: base))

        server?.stop()
        server = nil

        var request = URLRequest(url: base)
        request.timeoutInterval = 3
        XCTAssertThrowsError(try send(request, timeout: 15), "Requests should fail once the server has stopped")
    }
}

/// Thread-safe counter for assertions across concurrent handlers.
private final class Counter {
    private let lock = NSLock()
    private var count = 0

    func increment() -> Int {
        lock.lock()
        defer { lock.unlock() }
        count += 1
        return count
    }

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }
}
