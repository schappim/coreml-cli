import Foundation

/// Wire types defined by the protobuf encoding spec.
/// We only need the four types that appear in the CoreML model spec messages we touch.
enum ProtobufWireType: Int {
    case varint = 0
    case fixed64 = 1
    case lengthDelimited = 2
    case fixed32 = 5
}

enum ProtobufCodecError: Error, LocalizedError, Equatable {
    case unexpectedEndOfData
    case invalidVarint
    case unknownWireType(Int)
    case invalidUTF8
    case lengthOverflow

    var errorDescription: String? {
        switch self {
        case .unexpectedEndOfData:
            return "Unexpected end of protobuf data"
        case .invalidVarint:
            return "Invalid varint encoding (more than 10 bytes)"
        case .unknownWireType(let t):
            return "Unknown protobuf wire type: \(t)"
        case .invalidUTF8:
            return "Length-delimited field is not valid UTF-8"
        case .lengthOverflow:
            return "Length-delimited field length exceeds available data"
        }
    }
}

/// Minimal protobuf wire-format reader.
///
/// Only handles the operations needed to walk and edit a CoreML model spec:
/// reading varints/tags/strings/embedded-message bytes, and skipping fields whose
/// values we don't care about.
struct ProtobufReader {
    let data: Data
    private(set) var offset: Int

    init(data: Data) {
        self.data = data
        self.offset = data.startIndex
    }

    var isAtEnd: Bool {
        return offset >= data.endIndex
    }

    /// Decode a base-128 varint at the current offset.
    mutating func readVarint() throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        var bytesRead = 0
        while true {
            guard offset < data.endIndex else {
                throw ProtobufCodecError.unexpectedEndOfData
            }
            let byte = data[offset]
            offset += 1
            bytesRead += 1
            result |= UInt64(byte & 0x7F) << shift
            if (byte & 0x80) == 0 {
                return result
            }
            shift += 7
            // A 64-bit varint cannot occupy more than 10 bytes.
            if bytesRead >= 10 {
                throw ProtobufCodecError.invalidVarint
            }
        }
    }

    /// Read a tag and split it into field number and wire type.
    mutating func readTag() throws -> (fieldNumber: Int, wireType: ProtobufWireType) {
        let raw = try readVarint()
        let fieldNumber = Int(raw >> 3)
        let wireTypeRaw = Int(raw & 0x07)
        guard let wireType = ProtobufWireType(rawValue: wireTypeRaw) else {
            throw ProtobufCodecError.unknownWireType(wireTypeRaw)
        }
        return (fieldNumber, wireType)
    }

    /// Read the payload of a length-delimited field (string, bytes, or embedded message).
    mutating func readLengthDelimited() throws -> Data {
        let length = try readVarint()
        guard length <= UInt64(Int.max) else {
            throw ProtobufCodecError.lengthOverflow
        }
        let lengthInt = Int(length)
        // Subtract instead of adding so a near-Int.max claimed length doesn't trap.
        let remaining = data.endIndex - offset
        guard lengthInt >= 0, lengthInt <= remaining else {
            throw ProtobufCodecError.lengthOverflow
        }
        let slice = data.subdata(in: offset..<(offset + lengthInt))
        offset += lengthInt
        return slice
    }

    /// Read a length-delimited UTF-8 string value.
    mutating func readString() throws -> String {
        let bytes = try readLengthDelimited()
        guard let string = String(data: bytes, encoding: .utf8) else {
            throw ProtobufCodecError.invalidUTF8
        }
        return string
    }

    /// Advance past the value of a field whose tag has already been consumed.
    mutating func skipValue(wireType: ProtobufWireType) throws {
        switch wireType {
        case .varint:
            _ = try readVarint()
        case .fixed64:
            guard data.endIndex - offset >= 8 else {
                throw ProtobufCodecError.unexpectedEndOfData
            }
            offset += 8
        case .lengthDelimited:
            _ = try readLengthDelimited()
        case .fixed32:
            guard data.endIndex - offset >= 4 else {
                throw ProtobufCodecError.unexpectedEndOfData
            }
            offset += 4
        }
    }

    /// Skip a field whose tag has already been consumed and return the full raw bytes
    /// (including the tag) of that field. Callers pass the offset where the tag started.
    mutating func skipAndCaptureField(wireType: ProtobufWireType, tagStart: Int) throws -> Data {
        try skipValue(wireType: wireType)
        return data.subdata(in: tagStart..<offset)
    }
}

/// Minimal protobuf wire-format writer.
struct ProtobufWriter {
    private(set) var data: Data

    init() {
        self.data = Data()
    }

    mutating func writeVarint(_ value: UInt64) {
        var remaining = value
        while remaining >= 0x80 {
            data.append(UInt8((remaining & 0x7F) | 0x80))
            remaining >>= 7
        }
        data.append(UInt8(remaining))
    }

    mutating func writeTag(fieldNumber: Int, wireType: ProtobufWireType) {
        let raw = (UInt64(fieldNumber) << 3) | UInt64(wireType.rawValue)
        writeVarint(raw)
    }

    /// Write a length-delimited field (tag + length + payload bytes).
    mutating func writeLengthDelimited(fieldNumber: Int, payload: Data) {
        writeTag(fieldNumber: fieldNumber, wireType: .lengthDelimited)
        writeVarint(UInt64(payload.count))
        data.append(payload)
    }

    /// Write a string field. Empty strings are the proto3 default and are omitted.
    mutating func writeString(fieldNumber: Int, value: String) {
        if value.isEmpty { return }
        writeLengthDelimited(fieldNumber: fieldNumber, payload: Data(value.utf8))
    }

    /// Append already-encoded bytes verbatim. Used to preserve unknown fields.
    mutating func append(rawBytes: Data) {
        data.append(rawBytes)
    }
}
