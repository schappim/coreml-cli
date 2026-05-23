import XCTest
@testable import CoreMLToolkit

final class ProtobufCodecTests: XCTestCase {

    // MARK: - Varint

    func testVarintSingleByte() throws {
        for value: UInt64 in [0, 1, 42, 127] {
            var writer = ProtobufWriter()
            writer.writeVarint(value)
            XCTAssertEqual(writer.data.count, 1, "value \(value) should fit in one byte")

            var reader = ProtobufReader(data: writer.data)
            XCTAssertEqual(try reader.readVarint(), value)
            XCTAssertTrue(reader.isAtEnd)
        }
    }

    func testVarintMultiByte() throws {
        for value: UInt64 in [128, 300, 16_384, 2_097_151, 1 << 32, UInt64.max] {
            var writer = ProtobufWriter()
            writer.writeVarint(value)

            var reader = ProtobufReader(data: writer.data)
            XCTAssertEqual(try reader.readVarint(), value, "round-trip failed for \(value)")
            XCTAssertTrue(reader.isAtEnd)
        }
    }

    func testVarintTruncated() {
        // A varint byte with the continuation bit set, then EOF.
        let data = Data([0x80])
        var reader = ProtobufReader(data: data)
        XCTAssertThrowsError(try reader.readVarint()) { error in
            XCTAssertEqual(error as? ProtobufCodecError, .unexpectedEndOfData)
        }
    }

    func testVarintTooLong() {
        // 11 continuation bytes — overflows the 10-byte max for a 64-bit varint.
        let data = Data(repeating: 0x80, count: 11)
        var reader = ProtobufReader(data: data)
        XCTAssertThrowsError(try reader.readVarint()) { error in
            XCTAssertEqual(error as? ProtobufCodecError, .invalidVarint)
        }
    }

    // MARK: - Tag

    func testTagEncodingRoundTrip() throws {
        let cases: [(Int, ProtobufWireType)] = [
            (1, .varint),
            (2, .lengthDelimited),
            (16, .fixed32),
            (100, .lengthDelimited),
            (500, .lengthDelimited),
        ]
        for (fieldNumber, wireType) in cases {
            var writer = ProtobufWriter()
            writer.writeTag(fieldNumber: fieldNumber, wireType: wireType)

            var reader = ProtobufReader(data: writer.data)
            let (fn, wt) = try reader.readTag()
            XCTAssertEqual(fn, fieldNumber)
            XCTAssertEqual(wt, wireType)
        }
    }

    func testTagUnknownWireType() {
        // Wire type 3 (group start) and 4 (group end) are deprecated; treat as unknown.
        let data = Data([0x0B]) // field 1, wire type 3
        var reader = ProtobufReader(data: data)
        XCTAssertThrowsError(try reader.readTag()) { error in
            XCTAssertEqual(error as? ProtobufCodecError, .unknownWireType(3))
        }
    }

    // MARK: - Length-delimited

    func testLengthDelimitedRoundTrip() throws {
        let payload = Data([1, 2, 3, 4, 5])
        var writer = ProtobufWriter()
        writer.writeLengthDelimited(fieldNumber: 7, payload: payload)

        var reader = ProtobufReader(data: writer.data)
        let (fn, wt) = try reader.readTag()
        XCTAssertEqual(fn, 7)
        XCTAssertEqual(wt, .lengthDelimited)
        XCTAssertEqual(try reader.readLengthDelimited(), payload)
    }

    func testLengthOverflow() {
        // tag for field 1 length-delimited, then claimed length 100, but only 2 bytes of payload.
        let data = Data([0x0A, 100, 1, 2])
        var reader = ProtobufReader(data: data)
        _ = try? reader.readTag()
        XCTAssertThrowsError(try reader.readLengthDelimited()) { error in
            XCTAssertEqual(error as? ProtobufCodecError, .lengthOverflow)
        }
    }

    // MARK: - String

    func testStringRoundTrip() throws {
        var writer = ProtobufWriter()
        writer.writeString(fieldNumber: 3, value: "hello, world")

        var reader = ProtobufReader(data: writer.data)
        let (fn, wt) = try reader.readTag()
        XCTAssertEqual(fn, 3)
        XCTAssertEqual(wt, .lengthDelimited)
        XCTAssertEqual(try reader.readString(), "hello, world")
    }

    func testEmptyStringIsOmitted() {
        // proto3 default: empty strings aren't serialized.
        var writer = ProtobufWriter()
        writer.writeString(fieldNumber: 3, value: "")
        XCTAssertTrue(writer.data.isEmpty)
    }

    func testInvalidUTF8() {
        // Manually write a tag + length + bytes that aren't valid UTF-8.
        var writer = ProtobufWriter()
        writer.writeLengthDelimited(fieldNumber: 3, payload: Data([0xFF, 0xFE]))

        var reader = ProtobufReader(data: writer.data)
        _ = try? reader.readTag()
        XCTAssertThrowsError(try reader.readString()) { error in
            XCTAssertEqual(error as? ProtobufCodecError, .invalidUTF8)
        }
    }

    // MARK: - Skip

    func testSkipVarintField() throws {
        var writer = ProtobufWriter()
        writer.writeTag(fieldNumber: 1, wireType: .varint)
        writer.writeVarint(12345)
        writer.writeString(fieldNumber: 2, value: "after")

        var reader = ProtobufReader(data: writer.data)
        let (fn1, wt1) = try reader.readTag()
        XCTAssertEqual(fn1, 1)
        try reader.skipValue(wireType: wt1)

        let (fn2, wt2) = try reader.readTag()
        XCTAssertEqual(fn2, 2)
        XCTAssertEqual(wt2, .lengthDelimited)
        XCTAssertEqual(try reader.readString(), "after")
    }

    func testSkipLengthDelimitedField() throws {
        var writer = ProtobufWriter()
        writer.writeString(fieldNumber: 1, value: "skip me")
        writer.writeString(fieldNumber: 2, value: "keep me")

        var reader = ProtobufReader(data: writer.data)
        let (_, wt1) = try reader.readTag()
        try reader.skipValue(wireType: wt1)
        let (fn2, _) = try reader.readTag()
        XCTAssertEqual(fn2, 2)
        XCTAssertEqual(try reader.readString(), "keep me")
    }

    func testSkipFixed64() throws {
        var writer = ProtobufWriter()
        writer.writeTag(fieldNumber: 1, wireType: .fixed64)
        writer.append(rawBytes: Data(repeating: 0xAB, count: 8))
        writer.writeString(fieldNumber: 2, value: "after")

        var reader = ProtobufReader(data: writer.data)
        let (_, wt1) = try reader.readTag()
        try reader.skipValue(wireType: wt1)
        let (fn2, _) = try reader.readTag()
        XCTAssertEqual(fn2, 2)
        XCTAssertEqual(try reader.readString(), "after")
    }

    func testSkipFixed32() throws {
        var writer = ProtobufWriter()
        writer.writeTag(fieldNumber: 1, wireType: .fixed32)
        writer.append(rawBytes: Data(repeating: 0xCD, count: 4))
        writer.writeString(fieldNumber: 2, value: "after")

        var reader = ProtobufReader(data: writer.data)
        let (_, wt1) = try reader.readTag()
        try reader.skipValue(wireType: wt1)
        let (fn2, _) = try reader.readTag()
        XCTAssertEqual(fn2, 2)
        XCTAssertEqual(try reader.readString(), "after")
    }

    func testSkipFixed32Truncated() {
        var writer = ProtobufWriter()
        writer.writeTag(fieldNumber: 1, wireType: .fixed32)
        writer.append(rawBytes: Data([0x01, 0x02])) // only 2 bytes, need 4

        var reader = ProtobufReader(data: writer.data)
        _ = try? reader.readTag()
        XCTAssertThrowsError(try reader.skipValue(wireType: .fixed32)) { error in
            XCTAssertEqual(error as? ProtobufCodecError, .unexpectedEndOfData)
        }
    }

    func testSkipAndCaptureFieldReturnsExactBytes() throws {
        // Build: field 1 (varint) = 5, field 2 (string) = "hi", field 3 (varint) = 999
        var writer = ProtobufWriter()
        writer.writeTag(fieldNumber: 1, wireType: .varint)
        writer.writeVarint(5)
        let middleStart = writer.data.count
        writer.writeString(fieldNumber: 2, value: "hi")
        let middleEnd = writer.data.count
        writer.writeTag(fieldNumber: 3, wireType: .varint)
        writer.writeVarint(999)

        let expectedMiddle = writer.data.subdata(in: middleStart..<middleEnd)

        var reader = ProtobufReader(data: writer.data)
        // Skip field 1.
        let (_, wt1) = try reader.readTag()
        try reader.skipValue(wireType: wt1)
        // Capture field 2's raw bytes.
        let tagStart = reader.offset
        let (_, wt2) = try reader.readTag()
        let captured = try reader.skipAndCaptureField(wireType: wt2, tagStart: tagStart)
        XCTAssertEqual(captured, expectedMiddle)
        // Field 3 still readable.
        let (fn3, _) = try reader.readTag()
        XCTAssertEqual(fn3, 3)
        XCTAssertEqual(try reader.readVarint(), 999)
    }

    // MARK: - Reader state

    func testEmptyReaderIsAtEnd() {
        let reader = ProtobufReader(data: Data())
        XCTAssertTrue(reader.isAtEnd)
    }

    func testWriterAppendRawBytes() {
        var writer = ProtobufWriter()
        writer.append(rawBytes: Data([0xDE, 0xAD]))
        writer.append(rawBytes: Data([0xBE, 0xEF]))
        XCTAssertEqual(writer.data, Data([0xDE, 0xAD, 0xBE, 0xEF]))
    }

    // MARK: - Spec compliance (exact byte encoding)

    /// The protobuf docs use 150 → 0x96 0x01 as the canonical varint example.
    func testVarintEncodingMatchesSpecExample() {
        var writer = ProtobufWriter()
        writer.writeVarint(150)
        XCTAssertEqual(writer.data, Data([0x96, 0x01]))
    }

    func testVarintEncodingKnownBytes() {
        // Cross-checked against the protobuf encoding spec.
        let cases: [(UInt64, [UInt8])] = [
            (0,         [0x00]),
            (1,         [0x01]),
            (127,       [0x7F]),                  // largest 1-byte value
            (128,       [0x80, 0x01]),            // smallest 2-byte value
            (300,       [0xAC, 0x02]),            // second spec example
            (16383,     [0xFF, 0x7F]),            // largest 2-byte value
            (16384,     [0x80, 0x80, 0x01]),      // smallest 3-byte value
            (UInt64.max, [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]),
        ]
        for (value, expected) in cases {
            var writer = ProtobufWriter()
            writer.writeVarint(value)
            XCTAssertEqual(writer.data, Data(expected), "encoding mismatch for value \(value)")
        }
    }

    func testTagEncodingKnownBytes() {
        // tag = (fieldNumber << 3) | wireType, encoded as a varint.
        let cases: [(Int, ProtobufWireType, [UInt8])] = [
            (1,   .varint,          [0x08]),       // 1*8 + 0
            (1,   .lengthDelimited, [0x0A]),       // 1*8 + 2
            (2,   .lengthDelimited, [0x12]),
            (3,   .lengthDelimited, [0x1A]),
            (15,  .varint,          [0x78]),       // largest single-byte tag
            (16,  .varint,          [0x80, 0x01]), // first multi-byte tag
            (100, .lengthDelimited, [0xA2, 0x06]),
            (500, .lengthDelimited, [0xA2, 0x1F]),
        ]
        for (fieldNumber, wireType, expected) in cases {
            var writer = ProtobufWriter()
            writer.writeTag(fieldNumber: fieldNumber, wireType: wireType)
            XCTAssertEqual(
                writer.data, Data(expected),
                "encoding mismatch for fieldNumber=\(fieldNumber) wireType=\(wireType)"
            )
        }
    }

    // MARK: - Varint boundary coverage

    func testVarintAllSingleByteValuesRoundTrip() throws {
        for value: UInt64 in 0...127 {
            var writer = ProtobufWriter()
            writer.writeVarint(value)
            XCTAssertEqual(writer.data.count, 1, "value \(value) should encode to 1 byte")

            var reader = ProtobufReader(data: writer.data)
            XCTAssertEqual(try reader.readVarint(), value)
            XCTAssertTrue(reader.isAtEnd)
        }
    }

    func testVarintEncodedLengthByBoundary() throws {
        // (value, expected encoded length). Every entry sits on a byte-boundary of varint encoding.
        let cases: [(UInt64, Int)] = [
            (127, 1),
            (128, 2),
            (16383, 2),
            (16384, 3),
            (2_097_151, 3),
            (2_097_152, 4),
            (268_435_455, 4),
            (268_435_456, 5),
            ((1 << 35) - 1, 5),
            (1 << 35, 6),
            ((1 << 56) - 1, 8),
            (1 << 56, 9),
            (UInt64.max, 10),
        ]
        for (value, expectedLength) in cases {
            var writer = ProtobufWriter()
            writer.writeVarint(value)
            XCTAssertEqual(writer.data.count, expectedLength, "encoded length wrong for value \(value)")

            var reader = ProtobufReader(data: writer.data)
            XCTAssertEqual(try reader.readVarint(), value)
            XCTAssertTrue(reader.isAtEnd)
        }
    }

    // MARK: - Reading from an exhausted reader

    func testReadVarintFromEmptyDataThrows() {
        var reader = ProtobufReader(data: Data())
        XCTAssertThrowsError(try reader.readVarint()) { error in
            XCTAssertEqual(error as? ProtobufCodecError, .unexpectedEndOfData)
        }
    }

    func testReadTagFromEmptyDataThrows() {
        var reader = ProtobufReader(data: Data())
        XCTAssertThrowsError(try reader.readTag()) { error in
            XCTAssertEqual(error as? ProtobufCodecError, .unexpectedEndOfData)
        }
    }

    func testReadLengthDelimitedFromEmptyDataThrows() {
        var reader = ProtobufReader(data: Data())
        XCTAssertThrowsError(try reader.readLengthDelimited()) { error in
            XCTAssertEqual(error as? ProtobufCodecError, .unexpectedEndOfData)
        }
    }

    func testReadStringFromEmptyDataThrows() {
        var reader = ProtobufReader(data: Data())
        XCTAssertThrowsError(try reader.readString()) { error in
            XCTAssertEqual(error as? ProtobufCodecError, .unexpectedEndOfData)
        }
    }

    // MARK: - Zero-length / empty edge cases

    func testZeroLengthPayloadRoundTrip() throws {
        var writer = ProtobufWriter()
        writer.writeLengthDelimited(fieldNumber: 1, payload: Data())
        XCTAssertEqual(writer.data, Data([0x0A, 0x00]))

        var reader = ProtobufReader(data: writer.data)
        let (fn, wt) = try reader.readTag()
        XCTAssertEqual(fn, 1)
        XCTAssertEqual(wt, .lengthDelimited)
        XCTAssertEqual(try reader.readLengthDelimited(), Data())
        XCTAssertTrue(reader.isAtEnd)
    }

    func testZeroLengthFieldReadsAsEmptyString() throws {
        var writer = ProtobufWriter()
        writer.writeLengthDelimited(fieldNumber: 3, payload: Data())

        var reader = ProtobufReader(data: writer.data)
        _ = try reader.readTag()
        XCTAssertEqual(try reader.readString(), "")
    }

    func testSkipZeroLengthDelimitedField() throws {
        var writer = ProtobufWriter()
        writer.writeLengthDelimited(fieldNumber: 1, payload: Data())
        writer.writeString(fieldNumber: 2, value: "after")

        var reader = ProtobufReader(data: writer.data)
        let (_, wt1) = try reader.readTag()
        try reader.skipValue(wireType: wt1)
        let (fn2, _) = try reader.readTag()
        XCTAssertEqual(fn2, 2)
        XCTAssertEqual(try reader.readString(), "after")
    }

    func testAppendEmptyRawBytesIsNoOp() {
        var writer = ProtobufWriter()
        writer.writeString(fieldNumber: 1, value: "x")
        let snapshot = writer.data
        writer.append(rawBytes: Data())
        XCTAssertEqual(writer.data, snapshot)
    }

    func testNewWriterDataIsEmpty() {
        let writer = ProtobufWriter()
        XCTAssertTrue(writer.data.isEmpty)
    }

    // MARK: - Data slice handling (non-zero startIndex)

    func testReaderHandlesSliceWithNonZeroStartIndex() throws {
        // Build [prefix-junk | encoded "marcus" string field] and slice off the prefix
        // using subscript-range, which yields a Data whose startIndex is non-zero.
        var writer = ProtobufWriter()
        writer.writeString(fieldNumber: 3, value: "marcus")
        var combined = Data([0xFF, 0xFF, 0xFF])
        combined.append(writer.data)

        let slice = combined[3..<combined.endIndex]
        XCTAssertEqual(slice.startIndex, 3, "subscript-range slice should preserve origin index")

        var reader = ProtobufReader(data: slice)
        let (fn, wt) = try reader.readTag()
        XCTAssertEqual(fn, 3)
        XCTAssertEqual(wt, .lengthDelimited)
        XCTAssertEqual(try reader.readString(), "marcus")
        XCTAssertTrue(reader.isAtEnd)
    }

    func testReaderOffsetStartsAtDataStartIndex() {
        let combined = Data([0xFF, 0xFF, 0x08, 0x05])
        let slice = combined[2..<combined.endIndex]
        let reader = ProtobufReader(data: slice)
        XCTAssertEqual(reader.offset, slice.startIndex)
        XCTAssertEqual(reader.offset, 2)
    }

    // MARK: - Skip-and-capture for every wire type

    func testSkipAndCaptureVarintField() throws {
        var writer = ProtobufWriter()
        writer.writeTag(fieldNumber: 1, wireType: .varint)
        writer.writeVarint(300)
        let encoded = writer.data

        var reader = ProtobufReader(data: encoded)
        let tagStart = reader.offset
        let (_, wt) = try reader.readTag()
        let captured = try reader.skipAndCaptureField(wireType: wt, tagStart: tagStart)
        XCTAssertEqual(captured, encoded)
        XCTAssertTrue(reader.isAtEnd)
    }

    func testSkipAndCaptureFixed32Field() throws {
        let payload = Data([0xAA, 0xBB, 0xCC, 0xDD])
        var writer = ProtobufWriter()
        writer.writeTag(fieldNumber: 2, wireType: .fixed32)
        writer.append(rawBytes: payload)
        let encoded = writer.data

        var reader = ProtobufReader(data: encoded)
        let tagStart = reader.offset
        let (_, wt) = try reader.readTag()
        let captured = try reader.skipAndCaptureField(wireType: wt, tagStart: tagStart)
        XCTAssertEqual(captured, encoded)
    }

    func testSkipAndCaptureFixed64Field() throws {
        let payload = Data([1, 2, 3, 4, 5, 6, 7, 8])
        var writer = ProtobufWriter()
        writer.writeTag(fieldNumber: 3, wireType: .fixed64)
        writer.append(rawBytes: payload)
        let encoded = writer.data

        var reader = ProtobufReader(data: encoded)
        let tagStart = reader.offset
        let (_, wt) = try reader.readTag()
        let captured = try reader.skipAndCaptureField(wireType: wt, tagStart: tagStart)
        XCTAssertEqual(captured, encoded)
    }

    // MARK: - Wire type rejection completeness

    func testTagWireType4IsRejected() {
        // Wire type 4 (deprecated group-end) must be rejected, like wire type 3.
        var writer = ProtobufWriter()
        writer.writeVarint(UInt64(1 << 3) | 4)
        var reader = ProtobufReader(data: writer.data)
        XCTAssertThrowsError(try reader.readTag()) { error in
            XCTAssertEqual(error as? ProtobufCodecError, .unknownWireType(4))
        }
    }

    func testTagWireTypes6And7AreRejected() {
        for wireTypeRaw in [6, 7] {
            var writer = ProtobufWriter()
            writer.writeVarint(UInt64(1 << 3) | UInt64(wireTypeRaw))
            var reader = ProtobufReader(data: writer.data)
            XCTAssertThrowsError(try reader.readTag(), "wire type \(wireTypeRaw) should be rejected") { error in
                XCTAssertEqual(error as? ProtobufCodecError, .unknownWireType(wireTypeRaw))
            }
        }
    }

    // MARK: - Length-delimited edge cases

    func testLengthDelimitedPayloadExactlyFillsRemainingBuffer() throws {
        let payload = Data([1, 2, 3])
        var writer = ProtobufWriter()
        writer.writeLengthDelimited(fieldNumber: 1, payload: payload)

        var reader = ProtobufReader(data: writer.data)
        _ = try reader.readTag()
        XCTAssertEqual(try reader.readLengthDelimited(), payload)
        XCTAssertTrue(reader.isAtEnd)
    }

    func testReadLengthDelimitedRejectsOversizedClaim() {
        // Two attack vectors:
        //   - UInt64.max: caught by the "fits in Int" guard
        //   - UInt64(Int.max): would overflow `offset + length` without the new arithmetic
        for badLength in [UInt64.max, UInt64(Int.max)] {
            var writer = ProtobufWriter()
            writer.writeTag(fieldNumber: 1, wireType: .lengthDelimited)
            writer.writeVarint(badLength)

            var reader = ProtobufReader(data: writer.data)
            _ = try? reader.readTag()
            XCTAssertThrowsError(
                try reader.readLengthDelimited(),
                "claimed length \(badLength) should be rejected"
            ) { error in
                XCTAssertEqual(error as? ProtobufCodecError, .lengthOverflow)
            }
        }
    }

    func testLengthDelimitedRoundTripWithTrickyBytes() throws {
        // Payload that contains bytes which would parse as varint continuations or tags
        // if a faulty reader scanned past the length-prefix.
        let payload = Data([0x00, 0xFF, 0x80, 0x7F, 0x08, 0x0A])
        var writer = ProtobufWriter()
        writer.writeLengthDelimited(fieldNumber: 5, payload: payload)
        writer.writeString(fieldNumber: 6, value: "sentinel")

        var reader = ProtobufReader(data: writer.data)
        let (fn1, _) = try reader.readTag()
        XCTAssertEqual(fn1, 5)
        XCTAssertEqual(try reader.readLengthDelimited(), payload)
        let (fn2, _) = try reader.readTag()
        XCTAssertEqual(fn2, 6)
        XCTAssertEqual(try reader.readString(), "sentinel")
    }

    func testSkipFixed64Truncated() {
        var writer = ProtobufWriter()
        writer.writeTag(fieldNumber: 1, wireType: .fixed64)
        writer.append(rawBytes: Data([0x01, 0x02, 0x03])) // only 3 bytes, need 8

        var reader = ProtobufReader(data: writer.data)
        _ = try? reader.readTag()
        XCTAssertThrowsError(try reader.skipValue(wireType: .fixed64)) { error in
            XCTAssertEqual(error as? ProtobufCodecError, .unexpectedEndOfData)
        }
    }

    // MARK: - UTF-8

    func testStringRoundTripWithMultibyteCharacters() throws {
        let value = "héllo 🌏 ✓ — 日本語"
        var writer = ProtobufWriter()
        writer.writeString(fieldNumber: 3, value: value)

        var reader = ProtobufReader(data: writer.data)
        _ = try reader.readTag()
        XCTAssertEqual(try reader.readString(), value)
    }

    // MARK: - Reader sequencing

    func testReaderConsumesMixedWireTypesInOrder() throws {
        var writer = ProtobufWriter()
        writer.writeTag(fieldNumber: 1, wireType: .varint)
        writer.writeVarint(42)
        writer.writeString(fieldNumber: 2, value: "hello")
        writer.writeTag(fieldNumber: 3, wireType: .fixed32)
        writer.append(rawBytes: Data([0xAA, 0xBB, 0xCC, 0xDD]))

        var reader = ProtobufReader(data: writer.data)

        var (fn, wt) = try reader.readTag()
        XCTAssertEqual(fn, 1); XCTAssertEqual(wt, .varint)
        XCTAssertEqual(try reader.readVarint(), 42)
        XCTAssertFalse(reader.isAtEnd)

        (fn, wt) = try reader.readTag()
        XCTAssertEqual(fn, 2); XCTAssertEqual(wt, .lengthDelimited)
        XCTAssertEqual(try reader.readString(), "hello")
        XCTAssertFalse(reader.isAtEnd)

        (fn, wt) = try reader.readTag()
        XCTAssertEqual(fn, 3); XCTAssertEqual(wt, .fixed32)
        try reader.skipValue(wireType: wt)
        XCTAssertTrue(reader.isAtEnd)
    }

    func testReaderOffsetAdvancesByEncodedLength() throws {
        var writer = ProtobufWriter()
        writer.writeVarint(300) // 2 bytes
        writer.writeVarint(127) // 1 byte
        writer.writeVarint(128) // 2 bytes
        XCTAssertEqual(writer.data.count, 5)

        var reader = ProtobufReader(data: writer.data)
        XCTAssertEqual(reader.offset, 0)
        _ = try reader.readVarint()
        XCTAssertEqual(reader.offset, 2)
        _ = try reader.readVarint()
        XCTAssertEqual(reader.offset, 3)
        _ = try reader.readVarint()
        XCTAssertEqual(reader.offset, 5)
        XCTAssertTrue(reader.isAtEnd)
    }

    // MARK: - Error descriptions

    func testProtobufCodecErrorDescriptions() {
        XCTAssertNotNil(ProtobufCodecError.unexpectedEndOfData.errorDescription)
        XCTAssertNotNil(ProtobufCodecError.invalidVarint.errorDescription)
        XCTAssertTrue(
            ProtobufCodecError.unknownWireType(7).errorDescription?.contains("7") ?? false,
            "error description should include the wire type number"
        )
        XCTAssertNotNil(ProtobufCodecError.invalidUTF8.errorDescription)
        XCTAssertNotNil(ProtobufCodecError.lengthOverflow.errorDescription)
    }
}
