import Foundation

/// Edits the metadata block of a CoreML model spec (`.mlmodel` protobuf) at the wire-format
/// level, preserving every byte of every field we don't touch.
///
/// We only define the path from the root `Model` message down to the four well-known
/// `Metadata` string fields:
///
///     Model.description (tag 2) → ModelDescription.metadata (tag 100) → Metadata.{1,2,3,4,100}
///
/// Everything else — neural-network bodies, ML programs, feature descriptions, etc. —
/// is preserved as opaque raw bytes. This keeps us decoupled from the rest of Apple's
/// ~1500-line Model.proto.
public final class ModelSpecEditor {

    /// Top-level `Model` message field numbers we touch.
    private enum ModelField {
        static let description = 2
    }

    /// `ModelDescription` message field numbers we touch.
    private enum DescriptionField {
        static let metadata = 100
    }

    /// `Metadata` message field numbers, in the order Apple's proto defines them.
    enum MetadataFieldTag: Int {
        case shortDescription = 1
        case versionString = 2
        case author = 3
        case license = 4
        case userDefined = 100

        static func tag(for field: MetadataField) -> MetadataFieldTag {
            switch field {
            case .description: return .shortDescription
            case .version:     return .versionString
            case .author:      return .author
            case .license:     return .license
            }
        }
    }

    public init() {}

    // MARK: - Read

    /// Decode the four standard metadata fields plus userDefined entries from a model spec.
    public func readMetadata(specData: Data) throws -> ModelMetadata {
        guard let descriptionBytes = try extractFirstField(in: specData, fieldNumber: ModelField.description) else {
            return ModelMetadata()
        }
        guard let metadataBytes = try extractFirstField(in: descriptionBytes, fieldNumber: DescriptionField.metadata) else {
            return ModelMetadata()
        }
        return try decodeMetadata(metadataBytes)
    }

    // MARK: - Write

    /// Set a single metadata string field, returning new spec bytes with the change applied.
    /// All unknown fields are preserved verbatim. Setting an empty string clears the field
    /// (proto3 strings have empty string as their default and are not serialized).
    public func setMetadataField(specData: Data, field: MetadataField, value: String) throws -> Data {
        let tag = MetadataFieldTag.tag(for: field)

        // 1. Split the Model into (description bytes, all other field bytes).
        let (descriptionBytes, preservedModelBytes) = try splitOutField(
            in: specData, fieldNumber: ModelField.description
        )

        // 2. Split the ModelDescription into (metadata bytes, all other field bytes).
        let (metadataBytes, preservedDescriptionBytes) = try splitOutField(
            in: descriptionBytes ?? Data(), fieldNumber: DescriptionField.metadata
        )

        // 3. Replace the target field inside Metadata, preserving the other metadata fields.
        let newMetadataBytes = try replaceStringField(
            in: metadataBytes ?? Data(),
            tag: tag.rawValue,
            value: value
        )

        // 4. Rebuild the ModelDescription: preserved fields first, then the (maybe-empty) metadata.
        var descriptionWriter = ProtobufWriter()
        descriptionWriter.append(rawBytes: preservedDescriptionBytes)
        if !newMetadataBytes.isEmpty {
            descriptionWriter.writeLengthDelimited(
                fieldNumber: DescriptionField.metadata,
                payload: newMetadataBytes
            )
        }

        // 5. Rebuild the Model: preserved fields, then the description (only if it now has content
        //    or was present in the original spec — never invent an empty description block).
        var modelWriter = ProtobufWriter()
        modelWriter.append(rawBytes: preservedModelBytes)
        let hadOriginalDescription = descriptionBytes != nil
        if !descriptionWriter.data.isEmpty || hadOriginalDescription {
            modelWriter.writeLengthDelimited(
                fieldNumber: ModelField.description,
                payload: descriptionWriter.data
            )
        }

        return modelWriter.data
    }

    // MARK: - Wire-format helpers

    /// Return the payload of the first occurrence of an embedded-message field, or nil.
    private func extractFirstField(in data: Data, fieldNumber: Int) throws -> Data? {
        var reader = ProtobufReader(data: data)
        while !reader.isAtEnd {
            let (fn, wt) = try reader.readTag()
            if fn == fieldNumber && wt == .lengthDelimited {
                return try reader.readLengthDelimited()
            }
            try reader.skipValue(wireType: wt)
        }
        return nil
    }

    /// Walk a message and partition it into:
    /// - `extracted`: payload of the first occurrence of the target embedded-message field, or nil
    /// - `preserved`: every other field's raw bytes, concatenated in their original order
    ///
    /// Repeat occurrences of the target field beyond the first are treated as "other" and preserved.
    /// The CoreML spec doesn't repeat `description` or `metadata`, so this is purely defensive.
    private func splitOutField(in data: Data, fieldNumber: Int) throws -> (extracted: Data?, preserved: Data) {
        var reader = ProtobufReader(data: data)
        var extracted: Data? = nil
        var preserved = Data()

        while !reader.isAtEnd {
            let tagStart = reader.offset
            let (fn, wt) = try reader.readTag()
            if fn == fieldNumber && wt == .lengthDelimited && extracted == nil {
                extracted = try reader.readLengthDelimited()
            } else {
                let bytes = try reader.skipAndCaptureField(wireType: wt, tagStart: tagStart)
                preserved.append(bytes)
            }
        }

        return (extracted, preserved)
    }

    /// Replace the value of a string field with the given fieldNumber, preserving every other
    /// field verbatim. The new field is appended at the end (protobuf is order-insensitive).
    private func replaceStringField(in data: Data, tag fieldNumber: Int, value: String) throws -> Data {
        var reader = ProtobufReader(data: data)
        var writer = ProtobufWriter()

        while !reader.isAtEnd {
            let tagStart = reader.offset
            let (fn, wt) = try reader.readTag()
            if fn == fieldNumber {
                try reader.skipValue(wireType: wt)
            } else {
                let bytes = try reader.skipAndCaptureField(wireType: wt, tagStart: tagStart)
                writer.append(rawBytes: bytes)
            }
        }

        writer.writeString(fieldNumber: fieldNumber, value: value)
        return writer.data
    }

    // MARK: - Metadata decoding

    private func decodeMetadata(_ data: Data) throws -> ModelMetadata {
        var reader = ProtobufReader(data: data)
        var author: String? = nil
        var description: String? = nil
        var license: String? = nil
        var version: String? = nil
        var userDefined: [String: String] = [:]

        while !reader.isAtEnd {
            let (fn, wt) = try reader.readTag()
            guard wt == .lengthDelimited, let metaTag = MetadataFieldTag(rawValue: fn) else {
                try reader.skipValue(wireType: wt)
                continue
            }

            switch metaTag {
            case .shortDescription: description = try reader.readString()
            case .versionString:    version = try reader.readString()
            case .author:           author = try reader.readString()
            case .license:          license = try reader.readString()
            case .userDefined:
                let entry = try decodeMapEntry(try reader.readLengthDelimited())
                if let key = entry.key {
                    userDefined[key] = entry.value ?? ""
                }
            }
        }

        return ModelMetadata(
            author: author,
            description: description,
            license: license,
            version: version,
            additionalInfo: userDefined
        )
    }

    /// Decode one entry of a `map<string, string>` field. In protobuf, map entries are
    /// length-delimited messages with key=tag1 and value=tag2.
    private func decodeMapEntry(_ data: Data) throws -> (key: String?, value: String?) {
        var reader = ProtobufReader(data: data)
        var key: String? = nil
        var value: String? = nil
        while !reader.isAtEnd {
            let (fn, wt) = try reader.readTag()
            if wt == .lengthDelimited && fn == 1 {
                key = try reader.readString()
            } else if wt == .lengthDelimited && fn == 2 {
                value = try reader.readString()
            } else {
                try reader.skipValue(wireType: wt)
            }
        }
        return (key, value)
    }
}
