import XCTest
@testable import CoreMLToolkit

final class HTTPRequestParserTests: XCTestCase {

    private func parseAll(_ text: String, limits: HTTPRequestParser.Limits = .init()) -> [HTTPRequestParser.Outcome] {
        let parser = HTTPRequestParser(limits: limits)
        parser.append(Array(text.utf8))
        var outcomes: [HTTPRequestParser.Outcome] = []
        while true {
            let outcome = parser.next()
            outcomes.append(outcome)
            if case .request = outcome { continue }
            return outcomes
        }
    }

    private func firstRequest(_ text: String) -> HTTPRequest? {
        for outcome in parseAll(text) {
            if case .request(let request) = outcome { return request }
        }
        return nil
    }

    func testParsesSimpleGet() throws {
        let request = try XCTUnwrap(firstRequest("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n"))
        XCTAssertEqual(request.method, "GET")
        XCTAssertEqual(request.path, "/health")
        XCTAssertEqual(request.header("host"), "localhost")
        XCTAssertTrue(request.body.isEmpty)
        XCTAssertTrue(request.keepAlive)
    }

    func testHeaderLookupIsCaseInsensitive() throws {
        let request = try XCTUnwrap(firstRequest("GET / HTTP/1.1\r\nContent-Type: application/json\r\n\r\n"))
        XCTAssertEqual(request.header("CONTENT-TYPE"), "application/json")
        XCTAssertEqual(request.contentType, "application/json")
    }

    func testParsesBodyWithContentLength() throws {
        let body = #"{"a":1}"#
        let request = try XCTUnwrap(firstRequest(
            "POST /v1/predict HTTP/1.1\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)"
        ))
        XCTAssertEqual(request.method, "POST")
        XCTAssertEqual(String(decoding: request.body, as: UTF8.self), body)
    }

    func testHeadersSplitAcrossReadsAreBuffered() throws {
        let parser = HTTPRequestParser()
        parser.append(Array("GET /health HT".utf8))
        XCTAssertEqual(parser.next(), .needMoreData)
        parser.append(Array("TP/1.1\r\nHost: x\r\n".utf8))
        XCTAssertEqual(parser.next(), .needMoreData)
        parser.append(Array("\r\n".utf8))

        guard case .request(let request) = parser.next() else {
            return XCTFail("Expected a complete request")
        }
        XCTAssertEqual(request.path, "/health")
    }

    func testBodySplitAcrossReadsIsAccumulated() throws {
        let parser = HTTPRequestParser()
        parser.append(Array("POST /p HTTP/1.1\r\nContent-Length: 10\r\n\r\nabcde".utf8))
        XCTAssertEqual(parser.next(), .needMoreData)
        parser.append(Array("fghij".utf8))

        guard case .request(let request) = parser.next() else {
            return XCTFail("Expected a complete request")
        }
        XCTAssertEqual(String(decoding: request.body, as: UTF8.self), "abcdefghij")
    }

    func testPipelinedRequestsAreReturnedInOrder() {
        let outcomes = parseAll("GET /one HTTP/1.1\r\n\r\nGET /two HTTP/1.1\r\n\r\n")
        let paths: [String] = outcomes.compactMap {
            if case .request(let request) = $0 { return request.path }
            return nil
        }
        XCTAssertEqual(paths, ["/one", "/two"])
    }

    func testDecodesChunkedBody() throws {
        let raw = "POST /p HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
            + "5\r\nhello\r\n"
            + "6\r\n world\r\n"
            + "0\r\n\r\n"
        let request = try XCTUnwrap(firstRequest(raw))
        XCTAssertEqual(String(decoding: request.body, as: UTF8.self), "hello world")
    }

    func testChunkedBodyWithExtensionAndTrailers() throws {
        let raw = "POST /p HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
            + "4;name=value\r\nabcd\r\n"
            + "0\r\nX-Checksum: 1234\r\n\r\n"
        let request = try XCTUnwrap(firstRequest(raw))
        XCTAssertEqual(String(decoding: request.body, as: UTF8.self), "abcd")
    }

    func testRejectsBothContentLengthAndTransferEncoding() {
        let outcomes = parseAll(
            "POST /p HTTP/1.1\r\nContent-Length: 4\r\nTransfer-Encoding: chunked\r\n\r\nabcd"
        )
        guard case .failure(let status, _) = outcomes.last else {
            return XCTFail("Expected a failure")
        }
        XCTAssertEqual(status, 400)
    }

    func testRejectsUnsupportedTransferEncoding() {
        let outcomes = parseAll("POST /p HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n")
        guard case .failure(let status, _) = outcomes.last else {
            return XCTFail("Expected a failure")
        }
        XCTAssertEqual(status, 501)
    }

    func testRejectsObsoleteLineFolding() {
        let outcomes = parseAll("GET / HTTP/1.1\r\nX-Long: one\r\n continued\r\n\r\n")
        guard case .failure(let status, _) = outcomes.last else {
            return XCTFail("Expected a failure")
        }
        XCTAssertEqual(status, 400)
    }

    func testRejectsInvalidContentLength() {
        let outcomes = parseAll("POST /p HTTP/1.1\r\nContent-Length: abc\r\n\r\n")
        guard case .failure(let status, _) = outcomes.last else {
            return XCTFail("Expected a failure")
        }
        XCTAssertEqual(status, 400)
    }

    func testRejectsDuplicateConflictingContentLength() {
        let outcomes = parseAll("POST /p HTTP/1.1\r\nContent-Length: 4\r\nContent-Length: 5\r\n\r\nabcd")
        guard case .failure(let status, _) = outcomes.last else {
            return XCTFail("Expected a failure")
        }
        XCTAssertEqual(status, 400)
    }

    func testRejectsUnsupportedHTTPVersion() {
        let outcomes = parseAll("GET / HTTP/2.0\r\n\r\n")
        guard case .failure(let status, _) = outcomes.last else {
            return XCTFail("Expected a failure")
        }
        XCTAssertEqual(status, 505)
    }

    func testRejectsOversizedBody() {
        let limits = HTTPRequestParser.Limits(maxHeaderBytes: 8 * 1024, maxBodyBytes: 16)
        let outcomes = parseAll("POST /p HTTP/1.1\r\nContent-Length: 64\r\n\r\n", limits: limits)
        guard case .failure(let status, _) = outcomes.last else {
            return XCTFail("Expected a failure")
        }
        XCTAssertEqual(status, 413)
    }

    func testRejectsOversizedChunkedBody() {
        let limits = HTTPRequestParser.Limits(maxHeaderBytes: 8 * 1024, maxBodyBytes: 4)
        let raw = "POST /p HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n8\r\nabcdefgh\r\n0\r\n\r\n"
        let outcomes = parseAll(raw, limits: limits)
        guard case .failure(let status, _) = outcomes.last else {
            return XCTFail("Expected a failure")
        }
        XCTAssertEqual(status, 413)
    }

    func testHugeChunkSizeIsRejectedWithoutOverflowing() {
        // A chunk size near Int.max, arriving after a non-empty body, used to
        // overflow the limit check and trap — killing the whole server process.
        let raw = "POST /p HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
            + "1\r\nA\r\n"
            + "7fffffffffffffff\r\n"
        let outcomes = parseAll(raw)
        guard case .failure(let status, _) = outcomes.last else {
            return XCTFail("Expected a failure")
        }
        XCTAssertEqual(status, 413)
    }

    func testHugeChunkSizeAsFirstChunkIsAlsoRejected() {
        let raw = "POST /p HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n7fffffffffffffff\r\n"
        guard case .failure(let status, _) = parseAll(raw).last else {
            return XCTFail("Expected a failure")
        }
        XCTAssertEqual(status, 413)
    }

    func testUnparseableChunkSizeIsRejected() {
        let raw = "POST /p HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nffffffffffffffffff\r\n"
        guard case .failure(let status, _) = parseAll(raw).last else {
            return XCTFail("Expected a failure")
        }
        XCTAssertEqual(status, 400)
    }

    func testOversizedBodyIsNotInvitedWith100Continue() {
        let parser = HTTPRequestParser(limits: .init(maxHeaderBytes: 8 * 1024, maxBodyBytes: 16))
        var invited = false
        parser.onHeadReceived = { _ in invited = true }
        parser.append(Array("POST /p HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 999999\r\n\r\n".utf8))

        guard case .failure(let status, _) = parser.next() else {
            return XCTFail("Expected a failure")
        }
        XCTAssertEqual(status, 413)
        XCTAssertFalse(invited, "A request already destined for 413 must not be told to continue")
    }

    func testRejectsOversizedHeaders() {
        let limits = HTTPRequestParser.Limits(maxHeaderBytes: 64, maxBodyBytes: 1024)
        let padding = String(repeating: "x", count: 400)
        let outcomes = parseAll("GET / HTTP/1.1\r\nX-Pad: \(padding)\r\n\r\n", limits: limits)
        guard case .failure(let status, _) = outcomes.last else {
            return XCTFail("Expected a failure")
        }
        XCTAssertEqual(status, 431)
    }

    func testOnHeadReceivedFiresBeforeBody() {
        let parser = HTTPRequestParser()
        var seenExpectContinue = false
        parser.onHeadReceived = { head in
            seenExpectContinue = head.expectsContinue
        }
        parser.append(Array("POST /p HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 4\r\n\r\n".utf8))
        XCTAssertEqual(parser.next(), .needMoreData)
        XCTAssertTrue(seenExpectContinue, "Head callback should fire before the body arrives")
    }

    func testKeepAliveRules() throws {
        let http11 = try XCTUnwrap(firstRequest("GET / HTTP/1.1\r\n\r\n"))
        XCTAssertTrue(http11.keepAlive)

        let closed = try XCTUnwrap(firstRequest("GET / HTTP/1.1\r\nConnection: close\r\n\r\n"))
        XCTAssertFalse(closed.keepAlive)

        let http10 = try XCTUnwrap(firstRequest("GET / HTTP/1.0\r\n\r\n"))
        XCTAssertFalse(http10.keepAlive)

        let http10KeepAlive = try XCTUnwrap(firstRequest("GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n"))
        XCTAssertTrue(http10KeepAlive.keepAlive)
    }

    func testQueryParsing() {
        let (path, query) = HTTPRequestParser.splitTarget("/v1/predict?top=5&label=hot%20dog&flag")
        XCTAssertEqual(path, "/v1/predict")
        XCTAssertEqual(query["top"], "5")
        XCTAssertEqual(query["label"], "hot dog")
        XCTAssertEqual(query["flag"], "")
    }

    func testPercentDecodedPath() {
        let (path, _) = HTTPRequestParser.splitTarget("/models/My%20Model")
        XCTAssertEqual(path, "/models/My Model")
    }

    func testAbsoluteFormTarget() {
        let (path, query) = HTTPRequestParser.splitTarget("http://example.com/v1/info?x=1")
        XCTAssertEqual(path, "/v1/info")
        XCTAssertEqual(query["x"], "1")
    }

    func testMultipartBoundaryExtraction() {
        let request = HTTPRequest(
            method: "POST",
            path: "/",
            headers: ["content-type": "multipart/form-data; boundary=\"abc123\""]
        )
        XCTAssertEqual(request.multipartBoundary, "abc123")
        XCTAssertEqual(request.contentType, "multipart/form-data")
    }
}

final class HTTPResponseTests: XCTestCase {

    func testSerializeSetsFramingHeaders() {
        let response = HTTPResponse.text("hello")
        let raw = String(decoding: response.serialize(keepAlive: true), as: UTF8.self)

        XCTAssertTrue(raw.hasPrefix("HTTP/1.1 200 OK\r\n"))
        XCTAssertTrue(raw.contains("Content-Length: 5\r\n"))
        XCTAssertTrue(raw.contains("Connection: keep-alive\r\n"))
        XCTAssertTrue(raw.contains("Date: "))
        XCTAssertTrue(raw.hasSuffix("\r\n\r\nhello"))
    }

    func testHeadResponseKeepsLengthButDropsBody() {
        let response = HTTPResponse.text("hello")
        let raw = String(decoding: response.serialize(keepAlive: false, includeBody: false), as: UTF8.self)

        XCTAssertTrue(raw.contains("Content-Length: 5\r\n"))
        XCTAssertTrue(raw.contains("Connection: close\r\n"))
        XCTAssertTrue(raw.hasSuffix("\r\n\r\n"))
    }

    func testErrorResponseShape() throws {
        let response = HTTPResponse.error(status: 404, message: "No such route")
        XCTAssertEqual(response.status, 404)

        let json = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: response.body) as? [String: Any]
        )
        let error = try XCTUnwrap(json["error"] as? [String: Any])
        XCTAssertEqual(error["status"] as? Int, 404)
        XCTAssertEqual(error["message"] as? String, "No such route")
    }

    func testReasonPhrases() {
        XCTAssertEqual(HTTPResponse.reasonPhrase(for: 200), "OK")
        XCTAssertEqual(HTTPResponse.reasonPhrase(for: 413), "Content Too Large")
        XCTAssertEqual(HTTPResponse.reasonPhrase(for: 505), "HTTP Version Not Supported")
    }
}

final class MultipartParserTests: XCTestCase {

    private func body(_ lines: [String]) -> Data {
        Data(lines.joined(separator: "\r\n").utf8)
    }

    func testParsesFileAndFieldParts() throws {
        let raw = body([
            "--X",
            "Content-Disposition: form-data; name=\"file\"; filename=\"cat.jpg\"",
            "Content-Type: image/jpeg",
            "",
            "IMAGEBYTES",
            "--X",
            "Content-Disposition: form-data; name=\"top\"",
            "",
            "5",
            "--X--",
            ""
        ])

        let parts = try MultipartParser.parse(body: raw, boundary: "X")
        XCTAssertEqual(parts.count, 2)

        XCTAssertEqual(parts[0].name, "file")
        XCTAssertEqual(parts[0].filename, "cat.jpg")
        XCTAssertEqual(parts[0].contentType, "image/jpeg")
        XCTAssertEqual(String(decoding: parts[0].data, as: UTF8.self), "IMAGEBYTES")

        XCTAssertEqual(parts[1].name, "top")
        XCTAssertNil(parts[1].filename)
        XCTAssertEqual(String(decoding: parts[1].data, as: UTF8.self), "5")
    }

    func testPreservesBinaryContentIncludingCRLF() throws {
        var raw = Data("--B\r\nContent-Disposition: form-data; name=\"f\"; filename=\"a.bin\"\r\n\r\n".utf8)
        let payload = Data([0x00, 0x0D, 0x0A, 0xFF, 0x2D, 0x2D])
        raw.append(payload)
        raw.append(Data("\r\n--B--\r\n".utf8))

        let parts = try MultipartParser.parse(body: raw, boundary: "B")
        XCTAssertEqual(parts.count, 1)
        XCTAssertEqual(parts[0].data, payload)
    }

    func testRejectsBodyWithoutClosingBoundary() {
        let raw = body([
            "--X",
            "Content-Disposition: form-data; name=\"f\"",
            "",
            "value",
            ""
        ])
        XCTAssertThrowsError(try MultipartParser.parse(body: raw, boundary: "X"))
    }

    func testRejectsEmptyBoundary() {
        XCTAssertThrowsError(try MultipartParser.parse(body: Data(), boundary: ""))
    }

    func testParameterExtraction() {
        let value = "form-data; name=\"file\"; filename=\"my photo.jpg\""
        XCTAssertEqual(MultipartParser.parameter("name", in: value), "file")
        XCTAssertEqual(MultipartParser.parameter("filename", in: value), "my photo.jpg")
        XCTAssertNil(MultipartParser.parameter("missing", in: value))
    }
}
