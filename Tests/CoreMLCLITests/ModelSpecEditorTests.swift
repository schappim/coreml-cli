import XCTest
@testable import CoreMLToolkit

final class ModelSpecEditorTests: XCTestCase {

    var editor: ModelSpecEditor!

    override func setUp() {
        super.setUp()
        editor = ModelSpecEditor()
    }

    override func tearDown() {
        editor = nil
        super.tearDown()
    }

    // MARK: - Read

    func testReadMetadataFromEmptySpec() throws {
        let metadata = try editor.readMetadata(specData: Data())
        XCTAssertNil(metadata.author)
        XCTAssertNil(metadata.description)
        XCTAssertNil(metadata.license)
        XCTAssertNil(metadata.version)
        XCTAssertTrue(metadata.additionalInfo.isEmpty)
    }

    func testReadMetadataWithAllStandardFields() throws {
        let spec = makeSpec(
            specificationVersion: 5,
            metadataFields: [
                .shortDescription: "Detects objects",
                .versionString: "1.0",
                .author: "Jane",
                .license: "MIT",
            ]
        )

        let metadata = try editor.readMetadata(specData: spec)
        XCTAssertEqual(metadata.description, "Detects objects")
        XCTAssertEqual(metadata.version, "1.0")
        XCTAssertEqual(metadata.author, "Jane")
        XCTAssertEqual(metadata.license, "MIT")
        XCTAssertTrue(metadata.additionalInfo.isEmpty)
    }

    func testReadMetadataWithUserDefinedEntries() throws {
        let metadataBytes = makeMetadata(
            stringFields: [.author: "Jane"],
            userDefined: ["framework": "PyTorch", "release": "2024-05"]
        )
        let descriptionBytes = makeModelDescription(metadata: metadataBytes)
        let spec = makeModel(specificationVersion: 5, description: descriptionBytes)

        let metadata = try editor.readMetadata(specData: spec)
        XCTAssertEqual(metadata.author, "Jane")
        XCTAssertEqual(metadata.additionalInfo["framework"], "PyTorch")
        XCTAssertEqual(metadata.additionalInfo["release"], "2024-05")
    }

    func testReadMetadataIgnoresUnknownFields() throws {
        // Add a few unknown fields at different levels and verify they don't interfere.
        var metadataWriter = ProtobufWriter()
        metadataWriter.writeString(fieldNumber: ModelSpecEditor.MetadataFieldTag.author.rawValue, value: "Jane")
        // Unknown varint field inside Metadata.
        metadataWriter.writeTag(fieldNumber: 50, wireType: .varint)
        metadataWriter.writeVarint(7)

        var descriptionWriter = ProtobufWriter()
        // Unknown string field for the description (e.g., predictedFeatureName at tag 11).
        descriptionWriter.writeString(fieldNumber: 11, value: "classLabel")
        descriptionWriter.writeLengthDelimited(
            fieldNumber: 100,
            payload: metadataWriter.data
        )

        var modelWriter = ProtobufWriter()
        modelWriter.writeTag(fieldNumber: 1, wireType: .varint)
        modelWriter.writeVarint(5)
        modelWriter.writeLengthDelimited(fieldNumber: 2, payload: descriptionWriter.data)
        // Unknown fixed32 field at the model level.
        modelWriter.writeTag(fieldNumber: 17, wireType: .fixed32)
        modelWriter.append(rawBytes: Data([0xAA, 0xBB, 0xCC, 0xDD]))

        let metadata = try editor.readMetadata(specData: modelWriter.data)
        XCTAssertEqual(metadata.author, "Jane")
    }

    // MARK: - Write: round-trip

    func testSetAuthorOnEmptySpec() throws {
        let updated = try editor.setMetadataField(specData: Data(), field: .author, value: "Jane")
        let metadata = try editor.readMetadata(specData: updated)
        XCTAssertEqual(metadata.author, "Jane")
    }

    func testSetEachStandardFieldRoundTrips() throws {
        for field in MetadataField.allCases {
            let updated = try editor.setMetadataField(
                specData: Data(),
                field: field,
                value: "value-for-\(field.rawValue)"
            )
            let metadata = try editor.readMetadata(specData: updated)
            switch field {
            case .author:      XCTAssertEqual(metadata.author, "value-for-author")
            case .description: XCTAssertEqual(metadata.description, "value-for-description")
            case .license:     XCTAssertEqual(metadata.license, "value-for-license")
            case .version:     XCTAssertEqual(metadata.version, "value-for-version")
            }
        }
    }

    func testSetOverwritesExistingField() throws {
        let initial = makeSpec(
            specificationVersion: 5,
            metadataFields: [.author: "Original"]
        )

        let updated = try editor.setMetadataField(specData: initial, field: .author, value: "Replacement")
        let metadata = try editor.readMetadata(specData: updated)
        XCTAssertEqual(metadata.author, "Replacement")

        // And there's exactly one author entry in the underlying bytes.
        let authorEntries = countOccurrences(of: ModelSpecEditor.MetadataFieldTag.author.rawValue, in: updated)
        XCTAssertEqual(authorEntries, 1, "duplicate author fields would survive round-trip but mean we leaked the old value")
    }

    func testSetEmptyValueClearsField() throws {
        let initial = makeSpec(
            specificationVersion: 5,
            metadataFields: [.author: "Original", .license: "MIT"]
        )

        let updated = try editor.setMetadataField(specData: initial, field: .author, value: "")
        let metadata = try editor.readMetadata(specData: updated)
        XCTAssertNil(metadata.author)
        XCTAssertEqual(metadata.license, "MIT")
    }

    // MARK: - Preservation

    func testSetPreservesOtherMetadataFields() throws {
        let initial = makeSpec(
            specificationVersion: 5,
            metadataFields: [
                .author: "Jane",
                .license: "MIT",
                .versionString: "1.0",
                .shortDescription: "Original description",
            ]
        )

        let updated = try editor.setMetadataField(
            specData: initial,
            field: .description,
            value: "Updated description"
        )
        let metadata = try editor.readMetadata(specData: updated)
        XCTAssertEqual(metadata.description, "Updated description")
        XCTAssertEqual(metadata.author, "Jane")
        XCTAssertEqual(metadata.license, "MIT")
        XCTAssertEqual(metadata.version, "1.0")
    }

    func testSetPreservesUserDefinedMap() throws {
        let metadataBytes = makeMetadata(
            stringFields: [.author: "Jane"],
            userDefined: ["framework": "PyTorch", "task": "classification"]
        )
        let descriptionBytes = makeModelDescription(metadata: metadataBytes)
        let initial = makeModel(specificationVersion: 5, description: descriptionBytes)

        let updated = try editor.setMetadataField(specData: initial, field: .license, value: "Apache-2.0")
        let metadata = try editor.readMetadata(specData: updated)
        XCTAssertEqual(metadata.author, "Jane")
        XCTAssertEqual(metadata.license, "Apache-2.0")
        XCTAssertEqual(metadata.additionalInfo["framework"], "PyTorch")
        XCTAssertEqual(metadata.additionalInfo["task"], "classification")
    }

    func testSetPreservesUnknownTopLevelFields() throws {
        // Build a spec with a "neural network" sentinel at field 500 (length-delimited).
        let payload = Data(repeating: 0x42, count: 32)
        var modelWriter = ProtobufWriter()
        modelWriter.writeTag(fieldNumber: 1, wireType: .varint)
        modelWriter.writeVarint(5)
        modelWriter.writeLengthDelimited(fieldNumber: 500, payload: payload)
        let initial = modelWriter.data

        let updated = try editor.setMetadataField(specData: initial, field: .author, value: "Jane")

        // The neural-network field must still be there with the exact same payload.
        let extracted = try extractLengthDelimitedField(from: updated, fieldNumber: 500)
        XCTAssertEqual(extracted, payload)

        // And the specVersion is still 5.
        let specVersion = try extractVarintField(from: updated, fieldNumber: 1)
        XCTAssertEqual(specVersion, 5)

        // And metadata is set.
        let metadata = try editor.readMetadata(specData: updated)
        XCTAssertEqual(metadata.author, "Jane")
    }

    func testSetPreservesUnknownDescriptionFields() throws {
        // ModelDescription with an unknown "input" field (tag 1) AND metadata.
        let inputPayload = Data(repeating: 0x33, count: 12)
        let metadataPayload = makeMetadata(stringFields: [.author: "Old"], userDefined: [:])

        var descriptionWriter = ProtobufWriter()
        descriptionWriter.writeLengthDelimited(fieldNumber: 1, payload: inputPayload)
        descriptionWriter.writeLengthDelimited(fieldNumber: 100, payload: metadataPayload)

        let initial = makeModel(specificationVersion: 5, description: descriptionWriter.data)

        let updated = try editor.setMetadataField(specData: initial, field: .author, value: "New")

        // Locate the description payload in the updated spec, then check the input field survived.
        let updatedDescription = try extractLengthDelimitedField(from: updated, fieldNumber: 2)
        let updatedInput = try extractLengthDelimitedField(from: updatedDescription, fieldNumber: 1)
        XCTAssertEqual(updatedInput, inputPayload)

        let metadata = try editor.readMetadata(specData: updated)
        XCTAssertEqual(metadata.author, "New")
    }

    func testSetOnSpecWithoutMetadataBlockCreatesOne() throws {
        // ModelDescription exists but has no metadata.
        var descriptionWriter = ProtobufWriter()
        descriptionWriter.writeString(fieldNumber: 11, value: "classLabel") // predictedFeatureName

        let initial = makeModel(specificationVersion: 5, description: descriptionWriter.data)

        let updated = try editor.setMetadataField(specData: initial, field: .author, value: "Jane")
        let metadata = try editor.readMetadata(specData: updated)
        XCTAssertEqual(metadata.author, "Jane")

        // predictedFeatureName must still be there.
        let description = try extractLengthDelimitedField(from: updated, fieldNumber: 2)
        var reader = ProtobufReader(data: description)
        var found = false
        while !reader.isAtEnd {
            let (fn, wt) = try reader.readTag()
            if fn == 11, wt == .lengthDelimited {
                XCTAssertEqual(try reader.readString(), "classLabel")
                found = true
            } else {
                try reader.skipValue(wireType: wt)
            }
        }
        XCTAssertTrue(found, "predictedFeatureName disappeared during metadata edit")
    }

    // MARK: - Helpers for building synthetic specs

    /// Build a complete Model spec with a specification version and a Metadata block
    /// containing the given string fields (no userDefined entries).
    private func makeSpec(specificationVersion: UInt64, metadataFields: [ModelSpecEditor.MetadataFieldTag: String]) -> Data {
        let metadataBytes = makeMetadata(stringFields: metadataFields, userDefined: [:])
        let descriptionBytes = makeModelDescription(metadata: metadataBytes)
        return makeModel(specificationVersion: specificationVersion, description: descriptionBytes)
    }

    private func makeMetadata(
        stringFields: [ModelSpecEditor.MetadataFieldTag: String],
        userDefined: [String: String]
    ) -> Data {
        var writer = ProtobufWriter()
        // Stable ordering so tests are deterministic.
        for tag in stringFields.keys.sorted(by: { $0.rawValue < $1.rawValue }) {
            writer.writeString(fieldNumber: tag.rawValue, value: stringFields[tag]!)
        }
        for key in userDefined.keys.sorted() {
            var entryWriter = ProtobufWriter()
            entryWriter.writeString(fieldNumber: 1, value: key)
            entryWriter.writeString(fieldNumber: 2, value: userDefined[key]!)
            writer.writeLengthDelimited(
                fieldNumber: ModelSpecEditor.MetadataFieldTag.userDefined.rawValue,
                payload: entryWriter.data
            )
        }
        return writer.data
    }

    private func makeModelDescription(metadata: Data) -> Data {
        var writer = ProtobufWriter()
        writer.writeLengthDelimited(fieldNumber: 100, payload: metadata)
        return writer.data
    }

    private func makeModel(specificationVersion: UInt64, description: Data) -> Data {
        var writer = ProtobufWriter()
        writer.writeTag(fieldNumber: 1, wireType: .varint)
        writer.writeVarint(specificationVersion)
        writer.writeLengthDelimited(fieldNumber: 2, payload: description)
        return writer.data
    }

    // MARK: - Inspection helpers

    private func extractLengthDelimitedField(from data: Data, fieldNumber: Int) throws -> Data {
        var reader = ProtobufReader(data: data)
        while !reader.isAtEnd {
            let (fn, wt) = try reader.readTag()
            if fn == fieldNumber, wt == .lengthDelimited {
                return try reader.readLengthDelimited()
            }
            try reader.skipValue(wireType: wt)
        }
        XCTFail("Field \(fieldNumber) not found")
        return Data()
    }

    private func extractVarintField(from data: Data, fieldNumber: Int) throws -> UInt64 {
        var reader = ProtobufReader(data: data)
        while !reader.isAtEnd {
            let (fn, wt) = try reader.readTag()
            if fn == fieldNumber, wt == .varint {
                return try reader.readVarint()
            }
            try reader.skipValue(wireType: wt)
        }
        XCTFail("Varint field \(fieldNumber) not found")
        return 0
    }

    private func countOccurrences(of fieldNumber: Int, in data: Data) -> Int {
        var reader = ProtobufReader(data: data)
        var count = 0
        // Walk the model, descend into description, descend into metadata, count occurrences.
        do {
            while !reader.isAtEnd {
                let (fn, wt) = try reader.readTag()
                if fn == 2, wt == .lengthDelimited {
                    let descBytes = try reader.readLengthDelimited()
                    var descReader = ProtobufReader(data: descBytes)
                    while !descReader.isAtEnd {
                        let (descFn, descWt) = try descReader.readTag()
                        if descFn == 100, descWt == .lengthDelimited {
                            let metaBytes = try descReader.readLengthDelimited()
                            var metaReader = ProtobufReader(data: metaBytes)
                            while !metaReader.isAtEnd {
                                let (metaFn, metaWt) = try metaReader.readTag()
                                if metaFn == fieldNumber {
                                    count += 1
                                }
                                try metaReader.skipValue(wireType: metaWt)
                            }
                        } else {
                            try descReader.skipValue(wireType: descWt)
                        }
                    }
                } else {
                    try reader.skipValue(wireType: wt)
                }
            }
        } catch {
            XCTFail("countOccurrences walk failed: \(error)")
        }
        return count
    }
}
