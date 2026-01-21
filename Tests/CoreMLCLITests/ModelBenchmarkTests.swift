import XCTest
@testable import CoreMLToolkit

final class ModelBenchmarkTests: XCTestCase {

    func testBenchmarkInitialization() {
        let benchmark = ModelBenchmark(device: .cpu)
        XCTAssertNotNil(benchmark)
    }

    func testBenchmarkResultCodable() throws {
        let result = BenchmarkResult(
            modelName: "TestModel",
            device: "cpu",
            iterations: 100,
            warmupIterations: 10,
            meanLatencyMs: 15.5,
            minLatencyMs: 10.0,
            maxLatencyMs: 25.0,
            stdDevMs: 3.2,
            p50LatencyMs: 14.0,
            p95LatencyMs: 22.0,
            p99LatencyMs: 24.5,
            throughputPerSecond: 64.5
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(result)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(BenchmarkResult.self, from: data)

        XCTAssertEqual(decoded.modelName, "TestModel")
        XCTAssertEqual(decoded.device, "cpu")
        XCTAssertEqual(decoded.iterations, 100)
        XCTAssertEqual(decoded.warmupIterations, 10)
        XCTAssertEqual(decoded.meanLatencyMs, 15.5)
        XCTAssertEqual(decoded.minLatencyMs, 10.0)
        XCTAssertEqual(decoded.maxLatencyMs, 25.0)
        XCTAssertEqual(decoded.stdDevMs, 3.2)
        XCTAssertEqual(decoded.p50LatencyMs, 14.0)
        XCTAssertEqual(decoded.p95LatencyMs, 22.0)
        XCTAssertEqual(decoded.p99LatencyMs, 24.5)
        XCTAssertEqual(decoded.throughputPerSecond, 64.5)
    }

    func testBenchmarkNonExistentModel() {
        let benchmark = ModelBenchmark()

        XCTAssertThrowsError(try benchmark.benchmark(
            modelPath: "/nonexistent/model.mlmodel",
            inputPath: "/test/input.jpg"
        )) { error in
            guard let benchmarkError = error as? BenchmarkError else {
                XCTFail("Expected BenchmarkError")
                return
            }

            if case .modelNotFound(let path) = benchmarkError {
                XCTAssertEqual(path, "/nonexistent/model.mlmodel")
            } else {
                XCTFail("Expected modelNotFound error")
            }
        }
    }

    func testBenchmarkErrorDescriptions() {
        XCTAssertTrue(BenchmarkError.modelNotFound(path: "/test").errorDescription?.contains("/test") ?? false)
        XCTAssertTrue(BenchmarkError.inputNotFound(path: "/input").errorDescription?.contains("/input") ?? false)
        XCTAssertNotNil(BenchmarkError.missingConstraint.errorDescription)
        XCTAssertNotNil(BenchmarkError.invalidInputFormat.errorDescription)
        XCTAssertNotNil(BenchmarkError.unsupportedInputType.errorDescription)
        XCTAssertNotNil(BenchmarkError.invalidImage.errorDescription)
        XCTAssertNotNil(BenchmarkError.pixelBufferCreationFailed.errorDescription)
    }
}
