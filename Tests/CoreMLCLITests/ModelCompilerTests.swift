import XCTest
@testable import CoreMLToolkit

final class ModelCompilerTests: XCTestCase {

    func testCompilerInitialization() {
        let compiler = ModelCompiler()
        XCTAssertNotNil(compiler)
    }

    func testCompilationResultCodable() throws {
        let result = CompilationResult(
            sourcePath: "/source/model.mlmodel",
            outputPath: "/output/model.mlmodelc",
            originalSize: 1024000,
            compiledSize: 512000,
            success: true,
            message: "Compilation successful"
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(result)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(CompilationResult.self, from: data)

        XCTAssertEqual(decoded.sourcePath, "/source/model.mlmodel")
        XCTAssertEqual(decoded.outputPath, "/output/model.mlmodelc")
        XCTAssertEqual(decoded.originalSize, 1024000)
        XCTAssertEqual(decoded.compiledSize, 512000)
        XCTAssertTrue(decoded.success)
        XCTAssertEqual(decoded.message, "Compilation successful")
    }

    func testCompileNonExistentModel() {
        let compiler = ModelCompiler()

        XCTAssertThrowsError(try compiler.compile(modelPath: "/nonexistent/model.mlmodel")) { error in
            guard let compilerError = error as? CompilerError else {
                XCTFail("Expected CompilerError")
                return
            }

            if case .modelNotFound(let path) = compilerError {
                XCTAssertEqual(path, "/nonexistent/model.mlmodel")
            } else {
                XCTFail("Expected modelNotFound error")
            }
        }
    }

    func testValidateNonExistentModel() {
        let compiler = ModelCompiler()

        XCTAssertThrowsError(try compiler.validate(modelPath: "/nonexistent/model.mlmodel")) { error in
            guard let compilerError = error as? CompilerError else {
                XCTFail("Expected CompilerError")
                return
            }

            if case .modelNotFound(let path) = compilerError {
                XCTAssertEqual(path, "/nonexistent/model.mlmodel")
            } else {
                XCTFail("Expected modelNotFound error")
            }
        }
    }

    func testCompilerErrorDescriptions() {
        XCTAssertTrue(CompilerError.modelNotFound(path: "/test").errorDescription?.contains("/test") ?? false)
        XCTAssertNotNil(CompilerError.alreadyCompiled.errorDescription)

        let nsError = NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "test error"])
        XCTAssertTrue(CompilerError.compilationFailed(nsError).errorDescription?.contains("test error") ?? false)

        XCTAssertNotNil(CompilerError.outputDirectoryCreationFailed.errorDescription)
    }
}
