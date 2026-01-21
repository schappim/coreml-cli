import XCTest
@testable import CoreMLToolkit

final class MetadataManagerTests: XCTestCase {

    func testMetadataManagerInitialization() {
        let manager = MetadataManager()
        XCTAssertNotNil(manager)
    }

    func testMetadataFieldValues() {
        XCTAssertEqual(MetadataField.author.rawValue, "author")
        XCTAssertEqual(MetadataField.description.rawValue, "description")
        XCTAssertEqual(MetadataField.license.rawValue, "license")
        XCTAssertEqual(MetadataField.version.rawValue, "version")
    }

    func testMetadataFieldFromRawValue() {
        XCTAssertEqual(MetadataField(rawValue: "author"), .author)
        XCTAssertEqual(MetadataField(rawValue: "description"), .description)
        XCTAssertEqual(MetadataField(rawValue: "license"), .license)
        XCTAssertEqual(MetadataField(rawValue: "version"), .version)
        XCTAssertNil(MetadataField(rawValue: "invalid"))
    }

    func testGetMetadataNonExistentModel() {
        let manager = MetadataManager()

        XCTAssertThrowsError(try manager.getMetadata(modelPath: "/nonexistent/model.mlmodel")) { error in
            guard let metadataError = error as? MetadataError else {
                XCTFail("Expected MetadataError")
                return
            }

            if case .modelNotFound(let path) = metadataError {
                XCTAssertEqual(path, "/nonexistent/model.mlmodel")
            } else {
                XCTFail("Expected modelNotFound error")
            }
        }
    }

    func testMetadataErrorDescriptions() {
        XCTAssertTrue(MetadataError.modelNotFound(path: "/test").errorDescription?.contains("/test") ?? false)
        XCTAssertNotNil(MetadataError.cannotModifyCompiled.errorDescription)
        XCTAssertTrue(MetadataError.notImplemented(reason: "test reason").errorDescription?.contains("test reason") ?? false)
        XCTAssertTrue(MetadataError.invalidField("test").errorDescription?.contains("test") ?? false)
    }
}
