import XCTest
@testable import CoreMLToolkit

final class ModelInspectorTests: XCTestCase {

    var inspector: ModelInspector!

    override func setUp() {
        super.setUp()
        inspector = ModelInspector()
    }

    override func tearDown() {
        inspector = nil
        super.tearDown()
    }

    func testInspectorInitialization() {
        XCTAssertNotNil(inspector)
    }

    func testInspectNonExistentModel() {
        XCTAssertThrowsError(try inspector.inspect(modelPath: "/nonexistent/model.mlmodel")) { error in
            guard let inspectorError = error as? ModelInspectorError else {
                XCTFail("Expected ModelInspectorError")
                return
            }

            if case .modelNotFound(let path) = inspectorError {
                XCTAssertEqual(path, "/nonexistent/model.mlmodel")
            } else {
                XCTFail("Expected modelNotFound error")
            }
        }
    }

    func testFeatureInfoEquality() {
        let feature1 = FeatureInfo(name: "input", type: "image", shape: [1, 224, 224, 3])
        let feature2 = FeatureInfo(name: "input", type: "image", shape: [1, 224, 224, 3])
        let feature3 = FeatureInfo(name: "output", type: "string")

        XCTAssertEqual(feature1, feature2)
        XCTAssertNotEqual(feature1, feature3)
    }

    func testImageConstraintInfoEquality() {
        let constraint1 = ImageConstraintInfo(width: 224, height: 224, pixelFormat: "BGRA32")
        let constraint2 = ImageConstraintInfo(width: 224, height: 224, pixelFormat: "BGRA32")
        let constraint3 = ImageConstraintInfo(width: 299, height: 299, pixelFormat: "BGRA32")

        XCTAssertEqual(constraint1, constraint2)
        XCTAssertNotEqual(constraint1, constraint3)
    }

    func testModelMetadataInitialization() {
        let metadata = ModelMetadata(
            author: "Test Author",
            description: "Test Description",
            license: "MIT",
            version: "1.0.0",
            additionalInfo: ["key": "value"]
        )

        XCTAssertEqual(metadata.author, "Test Author")
        XCTAssertEqual(metadata.description, "Test Description")
        XCTAssertEqual(metadata.license, "MIT")
        XCTAssertEqual(metadata.version, "1.0.0")
        XCTAssertEqual(metadata.additionalInfo["key"], "value")
    }

    func testModelMetadataCodable() throws {
        let metadata = ModelMetadata(
            author: "Test Author",
            description: "Test Description"
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(metadata)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(ModelMetadata.self, from: data)

        XCTAssertEqual(decoded.author, "Test Author")
        XCTAssertEqual(decoded.description, "Test Description")
        XCTAssertNil(decoded.license)
        XCTAssertNil(decoded.version)
    }

    func testModelInfoCodable() throws {
        let info = ModelInfo(
            name: "TestModel",
            inputs: [FeatureInfo(name: "input", type: "image")],
            outputs: [FeatureInfo(name: "output", type: "string")],
            metadata: ModelMetadata(author: "Test"),
            fileSize: 1024,
            isCompiled: false
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(info)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(ModelInfo.self, from: data)

        XCTAssertEqual(decoded.name, "TestModel")
        XCTAssertEqual(decoded.inputs.count, 1)
        XCTAssertEqual(decoded.outputs.count, 1)
        XCTAssertEqual(decoded.fileSize, 1024)
        XCTAssertFalse(decoded.isCompiled)
    }

    func testModelInspectorErrorDescriptions() {
        let notFoundError = ModelInspectorError.modelNotFound(path: "/test/path")
        XCTAssertTrue(notFoundError.errorDescription?.contains("/test/path") ?? false)

        let invalidFormatError = ModelInspectorError.invalidModelFormat
        XCTAssertNotNil(invalidFormatError.errorDescription)

        let compilationError = ModelInspectorError.compilationFailed(NSError(domain: "test", code: 1))
        XCTAssertTrue(compilationError.errorDescription?.contains("compilation") ?? false)
    }
}
