import XCTest
@testable import CoreMLToolkit

final class MetadataManagerTests: XCTestCase {

    var manager: MetadataManager!
    var tempDirectory: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        manager = MetadataManager()
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("coreml-cli-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if FileManager.default.fileExists(atPath: tempDirectory.path) {
            try FileManager.default.removeItem(at: tempDirectory)
        }
        manager = nil
        tempDirectory = nil
        try super.tearDownWithError()
    }

    // MARK: - MetadataField

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

    // MARK: - getMetadata

    func testGetMetadataNonExistentModel() {
        XCTAssertThrowsError(try manager.getMetadata(modelPath: "/nonexistent/model.mlmodel")) { error in
            XCTAssertEqual(error as? MetadataError, .modelNotFound(path: "/nonexistent/model.mlmodel"))
        }
    }

    func testGetMetadataFromSyntheticMLModel() throws {
        let specURL = try writeSyntheticMLModel(named: "WithAllFields", metadata: [
            .author: "Jane",
            .shortDescription: "A model",
            .license: "MIT",
            .versionString: "2.0",
        ])

        let metadata = try manager.getMetadata(modelPath: specURL.path)
        XCTAssertEqual(metadata.author, "Jane")
        XCTAssertEqual(metadata.description, "A model")
        XCTAssertEqual(metadata.license, "MIT")
        XCTAssertEqual(metadata.version, "2.0")
    }

    func testGetMetadataFromSyntheticMLPackage() throws {
        let packageURL = try writeSyntheticMLPackage(named: "Package", metadata: [
            .author: "Picasso",
        ])

        let metadata = try manager.getMetadata(modelPath: packageURL.path)
        XCTAssertEqual(metadata.author, "Picasso")
    }

    // MARK: - setMetadata

    func testSetMetadataNonExistentModel() {
        XCTAssertThrowsError(try manager.setMetadata(
            modelPath: "/nonexistent/model.mlmodel",
            field: .author,
            value: "Jane"
        )) { error in
            XCTAssertEqual(error as? MetadataError, .modelNotFound(path: "/nonexistent/model.mlmodel"))
        }
    }

    func testSetMetadataRejectsCompiledModel() throws {
        // Create a placeholder .mlmodelc directory; the manager only inspects the extension.
        let compiledURL = tempDirectory.appendingPathComponent("Compiled.mlmodelc")
        try FileManager.default.createDirectory(at: compiledURL, withIntermediateDirectories: true)

        XCTAssertThrowsError(try manager.setMetadata(
            modelPath: compiledURL.path,
            field: .author,
            value: "Jane"
        )) { error in
            XCTAssertEqual(error as? MetadataError, .cannotModifyCompiled)
        }
    }

    func testSetMetadataRejectsUnsupportedFormat() throws {
        let bogusURL = tempDirectory.appendingPathComponent("file.txt")
        try Data().write(to: bogusURL)

        XCTAssertThrowsError(try manager.setMetadata(
            modelPath: bogusURL.path,
            field: .author,
            value: "Jane"
        )) { error in
            XCTAssertEqual(error as? MetadataError, .unsupportedModelFormat(extension: "txt"))
        }
    }

    func testSetMetadataMLPackageWithMissingSpec() throws {
        // An .mlpackage directory that doesn't contain model.mlmodel anywhere.
        let packageURL = tempDirectory.appendingPathComponent("Empty.mlpackage")
        try FileManager.default.createDirectory(at: packageURL, withIntermediateDirectories: true)

        XCTAssertThrowsError(try manager.setMetadata(
            modelPath: packageURL.path,
            field: .author,
            value: "Jane"
        )) { error in
            XCTAssertEqual(error as? MetadataError, .specNotFoundInPackage(path: packageURL.path))
        }
    }

    func testSetMetadataInPlaceRoundTripOnMLModel() throws {
        let modelURL = try writeSyntheticMLModel(named: "InPlace", metadata: [.author: "Old"])

        let writtenPath = try manager.setMetadata(
            modelPath: modelURL.path,
            field: .author,
            value: "New"
        )
        XCTAssertEqual(writtenPath, modelURL.path)

        let metadata = try manager.getMetadata(modelPath: modelURL.path)
        XCTAssertEqual(metadata.author, "New")
    }

    func testSetMetadataWithOutputLeavesSourceUntouched() throws {
        let modelURL = try writeSyntheticMLModel(named: "Source", metadata: [.author: "Original"])
        let outputURL = tempDirectory.appendingPathComponent("Renamed.mlmodel")

        let writtenPath = try manager.setMetadata(
            modelPath: modelURL.path,
            field: .author,
            value: "Updated",
            outputPath: outputURL.path
        )
        XCTAssertEqual(writtenPath, outputURL.path)

        let originalMetadata = try manager.getMetadata(modelPath: modelURL.path)
        XCTAssertEqual(originalMetadata.author, "Original")

        let outputMetadata = try manager.getMetadata(modelPath: outputURL.path)
        XCTAssertEqual(outputMetadata.author, "Updated")
    }

    func testSetMetadataPreservesOtherFieldsOnRoundTrip() throws {
        let modelURL = try writeSyntheticMLModel(named: "Multi", metadata: [
            .author: "Jane",
            .license: "MIT",
            .versionString: "1.0",
            .shortDescription: "Original",
        ])

        _ = try manager.setMetadata(modelPath: modelURL.path, field: .description, value: "Updated")

        let metadata = try manager.getMetadata(modelPath: modelURL.path)
        XCTAssertEqual(metadata.description, "Updated")
        XCTAssertEqual(metadata.author, "Jane")
        XCTAssertEqual(metadata.license, "MIT")
        XCTAssertEqual(metadata.version, "1.0")
    }

    func testSetMetadataMLPackageInPlace() throws {
        let packageURL = try writeSyntheticMLPackage(named: "Pkg", metadata: [.author: "Old"])

        let writtenPath = try manager.setMetadata(
            modelPath: packageURL.path,
            field: .author,
            value: "New"
        )
        XCTAssertEqual(writtenPath, packageURL.path)

        let metadata = try manager.getMetadata(modelPath: packageURL.path)
        XCTAssertEqual(metadata.author, "New")
    }

    func testSetMetadataMLPackageWithOutputClonesPackage() throws {
        let sourceURL = try writeSyntheticMLPackage(named: "Source", metadata: [.author: "Old"])
        let outputURL = tempDirectory.appendingPathComponent("Clone.mlpackage")

        let writtenPath = try manager.setMetadata(
            modelPath: sourceURL.path,
            field: .author,
            value: "New",
            outputPath: outputURL.path
        )
        XCTAssertEqual(writtenPath, outputURL.path)

        // Source unchanged.
        XCTAssertEqual(try manager.getMetadata(modelPath: sourceURL.path).author, "Old")

        // Clone has the new metadata.
        XCTAssertEqual(try manager.getMetadata(modelPath: outputURL.path).author, "New")

        // And the canonical spec location exists inside the clone.
        let clonedSpec = outputURL.appendingPathComponent("Data/com.apple.CoreML/model.mlmodel")
        XCTAssertTrue(FileManager.default.fileExists(atPath: clonedSpec.path))
    }

    // MARK: - Error descriptions

    func testMetadataErrorDescriptions() {
        XCTAssertTrue(MetadataError.modelNotFound(path: "/test").errorDescription?.contains("/test") ?? false)
        XCTAssertNotNil(MetadataError.cannotModifyCompiled.errorDescription)
        XCTAssertTrue(MetadataError.unsupportedModelFormat(extension: "csv").errorDescription?.contains("csv") ?? false)
        XCTAssertTrue(MetadataError.specNotFoundInPackage(path: "/x").errorDescription?.contains("/x") ?? false)
    }

    // MARK: - Helpers

    /// Write a synthetic .mlmodel file containing only a metadata block. The result is
    /// not a runnable CoreML model but it has the wire-format structure the editor expects.
    private func writeSyntheticMLModel(
        named name: String,
        metadata: [ModelSpecEditor.MetadataFieldTag: String]
    ) throws -> URL {
        let url = tempDirectory.appendingPathComponent("\(name).mlmodel")
        try buildSpecBytes(metadata: metadata).write(to: url)
        return url
    }

    /// Write a synthetic .mlpackage directory containing model.mlmodel at the canonical
    /// path under Data/com.apple.CoreML/.
    private func writeSyntheticMLPackage(
        named name: String,
        metadata: [ModelSpecEditor.MetadataFieldTag: String]
    ) throws -> URL {
        let packageURL = tempDirectory.appendingPathComponent("\(name).mlpackage")
        let dataDirectory = packageURL.appendingPathComponent("Data/com.apple.CoreML")
        try FileManager.default.createDirectory(at: dataDirectory, withIntermediateDirectories: true)
        try buildSpecBytes(metadata: metadata).write(to: dataDirectory.appendingPathComponent("model.mlmodel"))
        return packageURL
    }

    private func buildSpecBytes(metadata: [ModelSpecEditor.MetadataFieldTag: String]) -> Data {
        var metadataWriter = ProtobufWriter()
        for tag in metadata.keys.sorted(by: { $0.rawValue < $1.rawValue }) {
            metadataWriter.writeString(fieldNumber: tag.rawValue, value: metadata[tag]!)
        }

        var descriptionWriter = ProtobufWriter()
        descriptionWriter.writeLengthDelimited(fieldNumber: 100, payload: metadataWriter.data)

        var modelWriter = ProtobufWriter()
        modelWriter.writeTag(fieldNumber: 1, wireType: .varint)
        modelWriter.writeVarint(5)
        modelWriter.writeLengthDelimited(fieldNumber: 2, payload: descriptionWriter.data)
        return modelWriter.data
    }
}
