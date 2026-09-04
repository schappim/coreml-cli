import XCTest
@testable import CoreMLCLI
@testable import CoreMLToolkit

final class ServeCommandTests: XCTestCase {

    // MARK: - Bind host resolution

    func testLoopbackSpellingsResolveToLoopback() {
        XCTAssertTrue(HTTPServer.isLoopback(host: "127.0.0.1"))
        XCTAssertTrue(HTTPServer.isLoopback(host: "localhost"))
        XCTAssertTrue(HTTPServer.isLoopback(host: "::1"))
        XCTAssertTrue(HTTPServer.isLoopback(host: "127.0.0.5"))
    }

    func testWildcardAndRoutableAddressesAreNotLoopback() {
        XCTAssertFalse(HTTPServer.isLoopback(host: "0.0.0.0"))
        XCTAssertFalse(HTTPServer.isLoopback(host: "192.168.1.10"))
        XCTAssertFalse(HTTPServer.isLoopback(host: "::"))
    }

    func testHostnamesAreRejectedRatherThanSilentlyBindingTheWildcard() {
        // A name reaches Network.framework as .name, which does not constrain the
        // bind at all — the listener would take the wildcard address and its own port.
        XCTAssertNil(HTTPServer.bindAddress(for: "example.com"))
        XCTAssertNil(HTTPServer.bindAddress(for: "my-machine.local"))
        XCTAssertNil(HTTPServer.bindAddress(for: ""))
        XCTAssertNotNil(HTTPServer.bindAddress(for: "localhost"))
        XCTAssertNotNil(HTTPServer.bindAddress(for: "127.0.0.1"))
    }

    func testLocalhostBindsLoopbackAndHonoursThePort() throws {
        let server = HTTPServer(
            configuration: .init(host: "localhost", port: 0, idleTimeout: 5),
            handler: { _ in .text("ok") }
        )
        try server.start()
        defer { server.stop() }

        let port = try XCTUnwrap(server.boundPort)
        XCTAssertGreaterThan(port, 0)

        // Reachable on loopback, which is what "localhost" is meant to promise.
        let expectation = expectation(description: "loopback request")
        let url = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/health"))
        URLSession.shared.dataTask(with: url) { data, _, _ in
            XCTAssertEqual(String(decoding: data ?? Data(), as: UTF8.self), "ok")
            expectation.fulfill()
        }.resume()
        wait(for: [expectation], timeout: 10)
    }

    // MARK: - Option validation

    func testInvalidHostIsRejected() {
        XCTAssertThrowsError(try Serve.parse(["model.mlmodel", "--host", "not-a-real-host"]))
    }

    func testOutOfRangeBodySizeIsRejected() {
        XCTAssertThrowsError(try Serve.parse(["model.mlmodel", "--max-body-mb", "9223372036854775807"]))
        XCTAssertThrowsError(try Serve.parse(["model.mlmodel", "--max-body-mb", "0"]))
        XCTAssertNoThrow(try Serve.parse(["model.mlmodel", "--max-body-mb", "64"]))
    }

    func testInvalidPortAndConcurrencyAreRejected() {
        XCTAssertThrowsError(try Serve.parse(["model.mlmodel", "--port", "70000"]))
        XCTAssertThrowsError(try Serve.parse(["model.mlmodel", "--concurrency", "0"]))
        XCTAssertNoThrow(try Serve.parse(["model.mlmodel", "--port", "0"]))
    }

    func testInvalidDeviceIsRejected() {
        XCTAssertThrowsError(try Serve.parse(["model.mlmodel", "--device", "tpu"]))
        XCTAssertNoThrow(try Serve.parse(["model.mlmodel", "--device", "ane"]))
    }

    // MARK: - Access log

    func testLogEscapesControlCharacters() {
        // A request path is attacker-controlled and percent-decoded, so an ESC or a
        // newline would otherwise reach the operator's terminal verbatim.
        let escaped = RequestLog.escaped("/\u{1b}[31mPWNED\n2026-01-01 GET /health 200")
        XCTAssertFalse(escaped.contains("\u{1b}"))
        XCTAssertFalse(escaped.contains("\n"))
        XCTAssertTrue(escaped.contains("\\x1b"))
        XCTAssertTrue(escaped.contains("\\x0a"))
        XCTAssertTrue(escaped.contains("PWNED"), "Printable text should survive")
    }

    func testLogTruncatesVeryLongPaths() {
        let escaped = RequestLog.escaped("/" + String(repeating: "a", count: 5000), limit: 64)
        XCTAssertLessThan(escaped.count, 100)
        XCTAssertTrue(escaped.hasSuffix("…"))
    }

    func testLogLeavesOrdinaryPathsAlone() {
        XCTAssertEqual(RequestLog.escaped("/v1/predict"), "/v1/predict")
    }
}
