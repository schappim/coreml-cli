import ArgumentParser
import Foundation
import CoreMLToolkit

struct Serve: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Serve a Core ML model as a local HTTP inference API"
    )

    @Argument(help: "Path to the Core ML model")
    var modelPath: String

    @Option(name: .shortAndLong, help: "Port to listen on (0 picks a free one)")
    var port: Int = 8080

    @Option(name: .long, help: "Interface to bind. Defaults to loopback only")
    var host: String = "127.0.0.1"

    @Option(name: .long, help: "Compute device: cpu, gpu, ane, or all")
    var device: String = "all"

    @Option(name: [.customShort("c"), .customLong("concurrency")], help: "Predictions to run at once")
    var concurrency: Int = 4

    @Option(name: .long, help: "Largest request body accepted, in megabytes")
    var maxBodyMb: Int = 32

    @Option(name: .long, help: "Require this key via 'Authorization: Bearer' or 'X-API-Key'")
    var apiKey: String?

    @Flag(name: .long, help: "Send CORS headers so browsers can call the API")
    var cors: Bool = false

    @Flag(name: .shortAndLong, help: "Do not log requests")
    var quiet: Bool = false

    func validate() throws {
        guard (0...65535).contains(port) else {
            throw ValidationError("Port must be between 0 and 65535")
        }
        guard concurrency >= 1 else {
            throw ValidationError("Concurrency must be at least 1")
        }
        // Bounded so the byte conversion below cannot overflow and trap.
        guard (1...16384).contains(maxBodyMb) else {
            throw ValidationError("Max body size must be between 1 and 16384 MB")
        }
        guard ComputeDevice(rawValue: device) != nil else {
            throw ValidationError("Invalid device '\(device)'. Use: cpu, gpu, ane, or all")
        }
        guard HTTPServer.bindAddress(for: host) != nil else {
            throw ValidationError(
                "Cannot bind '\(host)'. Use a literal address — 127.0.0.1 (this machine only), 0.0.0.0 (all interfaces), ::1, or a specific interface address."
            )
        }
    }

    func run() throws {
        guard let computeDevice = ComputeDevice(rawValue: device) else {
            throw ValidationError("Invalid device '\(device)'. Use: cpu, gpu, ane, or all")
        }

        let info = try ModelInspector().inspect(modelPath: modelPath)

        let predictor = ModelPredictor(device: computeDevice)
        try predictor.loadModel(at: modelPath)

        let service = InferenceService(
            engine: predictor,
            modelInfo: info,
            configuration: .init(apiKey: apiKey, allowCORS: cors)
        )

        let log = RequestLog(enabled: !quiet)
        let server = HTTPServer(
            configuration: .init(
                host: host,
                port: UInt16(port),
                maxBodyBytes: maxBodyMb * 1024 * 1024,
                maxConcurrentRequests: concurrency,
                idleTimeout: 60
            ),
            handler: { request in
                let start = CFAbsoluteTimeGetCurrent()
                let response = service.handle(request)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                log.record(request: request, status: response.status, elapsedMs: elapsed)
                return response
            }
        )

        try server.start()

        let boundPort = server.boundPort ?? UInt16(port)
        printBanner(info: info, port: boundPort)

        // Park the main thread until the process is asked to stop.
        let stopped = DispatchSemaphore(value: 0)
        let signalQueue = DispatchQueue(label: "coreml.serve.signals")
        var sources: [DispatchSourceSignal] = []

        for signalNumber in [SIGINT, SIGTERM] {
            signal(signalNumber, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: signalQueue)
            source.setEventHandler {
                FileHandle.standardError.write(Data("\nShutting down…\n".utf8))
                server.stop()
                stopped.signal()
            }
            source.resume()
            sources.append(source)
        }

        stopped.wait()
        sources.forEach { $0.cancel() }
    }

    private func printBanner(info: ModelInfo, port: UInt16) {
        let displayHost = (host == "0.0.0.0" || host == "::") ? "127.0.0.1" : host
        // An IPv6 literal has to be bracketed before it can go in a URL.
        let urlHost = displayHost.contains(":") ? "[\(displayHost)]" : displayHost
        let base = "http://\(urlHost):\(port)"

        print("coreml serve — \(info.name)")
        print()
        print("  Listening on \(base)")
        print("  Device: \(device) · concurrency: \(concurrency) · max body: \(maxBodyMb) MB")
        if apiKey != nil {
            print("  Auth: API key required")
        }
        if cors {
            print("  CORS: enabled")
        }
        print()
        print("  GET  \(base)/health")
        print("  GET  \(base)/v1/info")
        print("  POST \(base)/v1/predict")
        print()
        print("  \(exampleRequest(info: info, base: base))")
        print()

        if !HTTPServer.isLoopback(host: host) {
            let warning = apiKey == nil
                ? "  Warning: bound to \(host) with no --api-key. Anyone who can reach this port can run the model.\n"
                : "  Note: bound to \(host), reachable beyond this machine.\n"
            FileHandle.standardError.write(Data(warning.utf8))
        }

        print("Press Ctrl-C to stop.")
        // stdout is block-buffered when piped; the banner should appear now,
        // not whenever the buffer happens to fill.
        fflush(stdout)
    }

    /// A copy-pasteable request matching this model's actual input.
    private func exampleRequest(info: ModelInfo, base: String) -> String {
        let endpoint = "\(base)/v1/predict"
        let auth = apiKey != nil ? " -H \"X-API-Key: $COREML_API_KEY\"" : ""

        guard info.inputs.count == 1, let input = info.inputs.first else {
            let names = info.inputs.map { "\"\($0.name)\": …" }.joined(separator: ", ")
            return "curl -X POST\(auth) -H \"Content-Type: application/json\" -d '{\"inputs\": {\(names)}}' \(endpoint)"
        }

        switch input.type {
        case "image":
            return "curl -X POST\(auth) -H \"Content-Type: image/jpeg\" --data-binary @photo.jpg \(endpoint)"
        case "multiArray":
            let width = input.shape?.reduce(1, *) ?? 4
            let sample = (0..<min(width, 4)).map { _ in "0.0" }.joined(separator: ", ")
            let ellipsis = width > 4 ? ", …" : ""
            return "curl -X POST\(auth) -H \"Content-Type: application/json\" -d '[\(sample)\(ellipsis)]' \(endpoint)"
        case "string":
            return "curl -X POST\(auth) -H \"Content-Type: text/plain\" -d 'your text here' \(endpoint)"
        default:
            return "curl -X POST\(auth) -H \"Content-Type: application/json\" -d '{\"input\": …}' \(endpoint)"
        }
    }
}

/// Serialised access log, so concurrent handlers cannot interleave lines.
final class RequestLog {
    private let enabled: Bool
    private let lock = NSLock()

    init(enabled: Bool) {
        self.enabled = enabled
    }

    /// The request line is attacker-controlled and lands in an operator's terminal,
    /// so escape control characters — an unescaped ESC or newline can forge log
    /// lines or drive the terminal.
    static func escaped(_ text: String, limit: Int = 256) -> String {
        var result = ""
        for scalar in text.unicodeScalars.prefix(limit) {
            if scalar.value < 0x20 || scalar.value == 0x7F {
                result += String(format: "\\x%02x", scalar.value)
            } else {
                result.unicodeScalars.append(scalar)
            }
        }
        if text.unicodeScalars.count > limit { result += "…" }
        return result
    }

    func record(request: HTTPRequest, status: Int, elapsedMs: Double) {
        guard enabled else { return }
        let method = Self.escaped(request.method)
        let path = Self.escaped(request.path)
        let line = "\(method) \(path) \(status) \(String(format: "%.1f", elapsedMs))ms\n"
        lock.lock()
        FileHandle.standardError.write(Data(line.utf8))
        lock.unlock()
    }
}
