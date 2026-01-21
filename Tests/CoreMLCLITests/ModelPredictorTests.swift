import XCTest
@testable import CoreMLToolkit

final class ModelPredictorTests: XCTestCase {

    func testComputeDeviceValues() {
        XCTAssertEqual(ComputeDevice.cpu.rawValue, "cpu")
        XCTAssertEqual(ComputeDevice.gpu.rawValue, "gpu")
        XCTAssertEqual(ComputeDevice.ane.rawValue, "ane")
        XCTAssertEqual(ComputeDevice.all.rawValue, "all")
    }

    func testComputeDeviceFromRawValue() {
        XCTAssertEqual(ComputeDevice(rawValue: "cpu"), .cpu)
        XCTAssertEqual(ComputeDevice(rawValue: "gpu"), .gpu)
        XCTAssertEqual(ComputeDevice(rawValue: "ane"), .ane)
        XCTAssertEqual(ComputeDevice(rawValue: "all"), .all)
        XCTAssertNil(ComputeDevice(rawValue: "invalid"))
    }

    func testPredictorInitialization() {
        let predictor = ModelPredictor(device: .cpu)
        XCTAssertNotNil(predictor)
    }

    func testPredictWithoutLoadingModel() {
        let predictor = ModelPredictor()

        XCTAssertThrowsError(try predictor.predict(inputPath: "/test/input.jpg")) { error in
            guard let predictorError = error as? PredictorError else {
                XCTFail("Expected PredictorError")
                return
            }

            if case .modelNotLoaded = predictorError {
                // Expected
            } else {
                XCTFail("Expected modelNotLoaded error")
            }
        }
    }

    func testLoadNonExistentModel() {
        let predictor = ModelPredictor()

        XCTAssertThrowsError(try predictor.loadModel(at: "/nonexistent/model.mlmodel")) { error in
            guard let predictorError = error as? PredictorError else {
                XCTFail("Expected PredictorError")
                return
            }

            if case .modelNotFound(let path) = predictorError {
                XCTAssertEqual(path, "/nonexistent/model.mlmodel")
            } else {
                XCTFail("Expected modelNotFound error")
            }
        }
    }

    func testPredictionResultCodable() throws {
        let result = PredictionResult(
            inputFile: "test.jpg",
            outputs: [
                "label": .string("cat"),
                "confidence": .double(0.95)
            ],
            inferenceTimeMs: 15.5
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(result)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(PredictionResult.self, from: data)

        XCTAssertEqual(decoded.inputFile, "test.jpg")
        XCTAssertEqual(decoded.inferenceTimeMs, 15.5)
        XCTAssertEqual(decoded.outputs.count, 2)
    }

    func testPredictionValueString() throws {
        let value = PredictionValue.string("test")

        let encoder = JSONEncoder()
        let data = try encoder.encode(value)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(PredictionValue.self, from: data)

        if case .string(let s) = decoded {
            XCTAssertEqual(s, "test")
        } else {
            XCTFail("Expected string value")
        }

        XCTAssertEqual(value.description, "test")
    }

    func testPredictionValueDouble() throws {
        let value = PredictionValue.double(3.14159)

        let encoder = JSONEncoder()
        let data = try encoder.encode(value)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(PredictionValue.self, from: data)

        if case .double(let d) = decoded {
            XCTAssertEqual(d, 3.14159, accuracy: 0.00001)
        } else {
            XCTFail("Expected double value")
        }
    }

    func testPredictionValueInt() throws {
        let value = PredictionValue.int(42)

        let encoder = JSONEncoder()
        let data = try encoder.encode(value)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(PredictionValue.self, from: data)

        if case .int(let i) = decoded {
            XCTAssertEqual(i, 42)
        } else {
            XCTFail("Expected int value")
        }
    }

    func testPredictionValueArray() throws {
        let value = PredictionValue.array([1.0, 2.0, 3.0])

        let encoder = JSONEncoder()
        let data = try encoder.encode(value)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(PredictionValue.self, from: data)

        if case .array(let arr) = decoded {
            XCTAssertEqual(arr, [1.0, 2.0, 3.0])
        } else {
            XCTFail("Expected array value")
        }
    }

    func testPredictionValueDictionary() throws {
        let value = PredictionValue.dictionary(["a": 0.5, "b": 0.3])

        let encoder = JSONEncoder()
        let data = try encoder.encode(value)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(PredictionValue.self, from: data)

        if case .dictionary(let dict) = decoded {
            XCTAssertEqual(dict["a"], 0.5)
            XCTAssertEqual(dict["b"], 0.3)
        } else {
            XCTFail("Expected dictionary value")
        }
    }

    func testPredictorErrorDescriptions() {
        XCTAssertTrue(PredictorError.modelNotFound(path: "/test").errorDescription?.contains("/test") ?? false)
        XCTAssertNotNil(PredictorError.modelNotLoaded.errorDescription)
        XCTAssertTrue(PredictorError.inputNotFound(path: "/input").errorDescription?.contains("/input") ?? false)
        XCTAssertTrue(PredictorError.invalidImage(path: "/img").errorDescription?.contains("/img") ?? false)
        XCTAssertNotNil(PredictorError.missingImageConstraint.errorDescription)
        XCTAssertNotNil(PredictorError.missingMultiArrayConstraint.errorDescription)
        XCTAssertNotNil(PredictorError.pixelBufferCreationFailed.errorDescription)
        XCTAssertTrue(PredictorError.unsupportedInputType("test").errorDescription?.contains("test") ?? false)
        XCTAssertNotNil(PredictorError.invalidInputFormat.errorDescription)
    }
}
