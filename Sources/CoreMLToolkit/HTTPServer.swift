import Foundation
import Network

/// A small HTTP/1.1 server built on Network.framework.
///
/// Requests are handled one at a time per connection, so pipelined responses
/// can never overtake each other, while a semaphore caps how many handlers run
/// across all connections at once.
public final class HTTPServer {

    public struct Configuration {
        /// Interface to bind. Defaults to loopback so a model is not exposed to
        /// the network by accident.
        public var host: String
        /// Port to bind. Zero asks the system for an unused port.
        public var port: UInt16
        /// Largest request body accepted, in bytes.
        public var maxBodyBytes: Int
        /// Largest request header block accepted, in bytes.
        public var maxHeaderBytes: Int
        /// How many handlers may run concurrently across all connections.
        public var maxConcurrentRequests: Int
        /// How many connections may be open at once.
        public var maxConnections: Int
        /// Seconds a connection may sit idle before it is closed.
        public var idleTimeout: TimeInterval
        /// Seconds a single request has to arrive in full, however slowly it trickles.
        public var requestTimeout: TimeInterval

        public init(
            host: String = "127.0.0.1",
            port: UInt16 = 8080,
            maxBodyBytes: Int = 32 * 1024 * 1024,
            maxHeaderBytes: Int = 64 * 1024,
            maxConcurrentRequests: Int = 4,
            maxConnections: Int = 128,
            idleTimeout: TimeInterval = 60,
            requestTimeout: TimeInterval = 300
        ) {
            self.host = host
            self.port = port
            self.maxBodyBytes = maxBodyBytes
            self.maxHeaderBytes = maxHeaderBytes
            self.maxConcurrentRequests = max(1, maxConcurrentRequests)
            self.maxConnections = max(1, maxConnections)
            self.idleTimeout = idleTimeout
            self.requestTimeout = requestTimeout
        }
    }

    public typealias Handler = (HTTPRequest) -> HTTPResponse

    private let configuration: Configuration
    private let handler: Handler
    private let listenerQueue = DispatchQueue(label: "coreml.http.listener")
    private let workQueue: OperationQueue
    private let lock = NSLock()

    private var listener: NWListener?
    private var activeConnections: [ObjectIdentifier: Connection] = [:]
    private var isStopped = false

    public init(configuration: Configuration = Configuration(), handler: @escaping Handler) {
        self.configuration = configuration
        self.handler = handler

        // A width-limited queue rather than a semaphore: excess requests wait as
        // queued operations instead of as threads blocked on a wait().
        let queue = OperationQueue()
        queue.name = "coreml.http.work"
        queue.maxConcurrentOperationCount = configuration.maxConcurrentRequests
        queue.underlyingQueue = DispatchQueue(label: "coreml.http.work", attributes: .concurrent)
        self.workQueue = queue
    }

    /// The port the server actually bound, available once `start()` returns.
    public private(set) var boundPort: UInt16?

    /// Bind the socket and begin accepting connections.
    ///
    /// Returns once the listener is ready, or throws if the port cannot be bound.
    public func start() throws {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true

        guard let port = NWEndpoint.Port(rawValue: configuration.port) else {
            throw HTTPServerError.invalidPort(configuration.port)
        }
        // A hostname (rather than a literal address) does not constrain the bind at
        // all: Network.framework listens on the wildcard address and picks its own
        // port. Insist on something we can actually pin down.
        guard let boundHost = Self.bindAddress(for: configuration.host) else {
            throw HTTPServerError.unresolvableHost(configuration.host)
        }
        parameters.requiredLocalEndpoint = NWEndpoint.hostPort(host: boundHost, port: port)

        let listener: NWListener
        do {
            listener = try NWListener(using: parameters)
        } catch {
            throw HTTPServerError.bindFailed(host: configuration.host, port: configuration.port, underlying: error)
        }
        self.listener = listener

        let ready = DispatchSemaphore(value: 0)
        var startupError: Error?

        listener.stateUpdateHandler = { state in
            switch state {
            case .ready:
                ready.signal()
            case .failed(let error), .waiting(let error):
                startupError = error
                ready.signal()
            default:
                break
            }
        }

        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }

        listener.start(queue: listenerQueue)

        if ready.wait(timeout: .now() + 10) == .timedOut {
            listener.cancel()
            throw HTTPServerError.bindTimedOut(host: configuration.host, port: configuration.port)
        }

        if let error = startupError {
            listener.cancel()
            throw HTTPServerError.bindFailed(host: configuration.host, port: configuration.port, underlying: error)
        }

        let actualPort = listener.port?.rawValue ?? configuration.port
        if configuration.port != 0 && actualPort != configuration.port {
            listener.cancel()
            throw HTTPServerError.portNotHonoured(requested: configuration.port, bound: actualPort)
        }
        boundPort = actualPort
    }

    /// Literal addresses only — `localhost` is accepted as a spelling of loopback.
    public static func bindAddress(for host: String) -> NWEndpoint.Host? {
        switch host.lowercased() {
        case "localhost":
            return .ipv4(.loopback)
        case "":
            return nil
        default:
            if let address = IPv4Address(host) { return .ipv4(address) }
            if let address = IPv6Address(host) { return .ipv6(address) }
            return nil
        }
    }

    /// Whether binding this host keeps the server reachable only from this machine.
    public static func isLoopback(host: String) -> Bool {
        switch bindAddress(for: host) {
        case .ipv4(let address):
            // All of 127.0.0.0/8 is loopback; isLoopback only matches 127.0.0.1.
            return address.rawValue.first == 127
        case .ipv6(let address):
            return address.isLoopback
        default:
            return false
        }
    }

    /// Stop accepting connections and close the ones still open.
    public func stop() {
        listener?.cancel()
        listener = nil

        lock.lock()
        isStopped = true
        let open = Array(activeConnections.values)
        activeConnections.removeAll()
        lock.unlock()

        for connection in open {
            connection.close()
        }
    }

    private func accept(_ nwConnection: NWConnection) {
        let connection = Connection(
            connection: nwConnection,
            configuration: configuration,
            handler: handler,
            workQueue: workQueue
        )

        connection.onClose = { [weak self] closed in
            guard let self else { return }
            self.lock.lock()
            self.activeConnections.removeValue(forKey: ObjectIdentifier(closed))
            self.lock.unlock()
        }

        // Registered either way: the table is what keeps the connection alive
        // long enough to finish writing, even when it is being turned away.
        lock.lock()
        let stopped = isStopped
        let atCapacity = activeConnections.count >= configuration.maxConnections
        if !stopped {
            activeConnections[ObjectIdentifier(connection)] = connection
        }
        lock.unlock()

        // A connection accepted while stop() was running would otherwise be
        // registered after the table was emptied, and never closed.
        if stopped {
            nwConnection.cancel()
            return
        }

        if atCapacity {
            connection.rejectAndClose(status: 503, message: "Server is at its connection limit")
            return
        }

        connection.start()
    }
}

public enum HTTPServerError: Error, LocalizedError {
    case invalidPort(UInt16)
    case unresolvableHost(String)
    case portNotHonoured(requested: UInt16, bound: UInt16)
    case bindFailed(host: String, port: UInt16, underlying: Error)
    case bindTimedOut(host: String, port: UInt16)

    public var errorDescription: String? {
        switch self {
        case .invalidPort(let port):
            return "Invalid port: \(port)"
        case .unresolvableHost(let host):
            return "Cannot bind '\(host)'. Use a literal address — 127.0.0.1 (this machine only), 0.0.0.0 (all interfaces), ::1, or a specific interface address."
        case .portNotHonoured(let requested, let bound):
            return "Asked for port \(requested) but the socket bound \(bound)"
        case .bindFailed(let host, let port, let underlying):
            return "Could not bind \(host):\(port) — \(underlying.localizedDescription)"
        case .bindTimedOut(let host, let port):
            return "Timed out binding \(host):\(port)"
        }
    }
}

// MARK: - Connection

private final class Connection {
    private let connection: NWConnection
    private let configuration: HTTPServer.Configuration
    private let handler: HTTPServer.Handler
    private let workQueue: OperationQueue
    private let queue: DispatchQueue
    private let parser: HTTPRequestParser

    private var timer: DispatchSourceTimer?
    private var lastActivity: DispatchTime = .now()
    private var requestStartedAt: DispatchTime?
    private var isHandling = false
    private var isClosed = false

    var onClose: ((Connection) -> Void)?

    init(
        connection: NWConnection,
        configuration: HTTPServer.Configuration,
        handler: @escaping HTTPServer.Handler,
        workQueue: OperationQueue
    ) {
        self.connection = connection
        self.configuration = configuration
        self.handler = handler
        self.workQueue = workQueue
        self.queue = DispatchQueue(label: "coreml.http.connection")
        self.parser = HTTPRequestParser(
            limits: .init(
                maxHeaderBytes: configuration.maxHeaderBytes,
                maxBodyBytes: configuration.maxBodyBytes
            )
        )
    }

    func start() {
        parser.onHeadReceived = { [weak self] head in
            guard let self, head.expectsContinue else { return }
            self.send(Data("HTTP/1.1 100 Continue\r\n\r\n".utf8))
        }

        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled:
                self?.close()
            default:
                break
            }
        }

        connection.start(queue: queue)
        startTimer()
        receive()
    }

    /// Answer a request we will not serve, then hang up.
    func rejectAndClose(status: Int, message: String) {
        connection.start(queue: queue)
        let response = HTTPResponse.error(status: status, message: message)
        send(response.serialize(keepAlive: false)) { [weak self] in
            self?.close()
        }
    }

    func close() {
        queue.async { [weak self] in
            guard let self, !self.isClosed else { return }
            self.isClosed = true
            self.timer?.cancel()
            self.timer = nil
            self.connection.cancel()
            self.onClose?(self)
        }
    }

    private func receive() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] data, _, isComplete, error in
            guard let self else { return }

            if error != nil {
                self.close()
                return
            }

            if let data, !data.isEmpty {
                self.noteActivity()
                self.parser.append(data)
                self.drain()
                return
            }

            if isComplete {
                self.close()
            } else {
                self.receive()
            }
        }
    }

    /// Pull requests out of the parser one at a time, answering each before
    /// looking at the next so pipelined responses stay in order.
    private func drain() {
        switch parser.next() {
        case .needMoreData:
            receive()

        case .failure(let status, let message):
            let response = HTTPResponse.error(status: status, message: message)
            send(response.serialize(keepAlive: false)) { [weak self] in
                self?.close()
            }

        case .request(let request):
            beginHandling()
            workQueue.addOperation { [weak self] in
                guard let self else { return }

                let response = self.handler(request)
                let keepAlive = request.keepAlive
                let payload = response.serialize(
                    keepAlive: keepAlive,
                    includeBody: request.method != "HEAD"
                )

                self.send(payload) { [weak self] in
                    guard let self else { return }
                    self.endHandling()
                    if keepAlive {
                        self.queue.async { self.drain() }
                    } else {
                        self.close()
                    }
                }
            }
        }
    }

    private func send(_ data: Data, completion: (() -> Void)? = nil) {
        connection.send(content: data, completion: .contentProcessed { [weak self] error in
            if error != nil {
                self?.close()
                return
            }
            completion?()
        })
    }

    /// One timer per connection, rescheduled as things happen. The previous
    /// approach queued a fresh work item on every read, each of which stayed
    /// resident until its original deadline passed.
    private func startTimer() {
        queue.async { [weak self] in
            guard let self, !self.isClosed else { return }
            let timer = DispatchSource.makeTimerSource(queue: self.queue)
            timer.setEventHandler { [weak self] in self?.handleTimeout() }
            self.timer = timer
            timer.resume()
            self.reschedule()
        }
    }

    private func noteActivity() {
        queue.async { [weak self] in
            guard let self else { return }
            self.lastActivity = .now()
            // The clock on a whole request starts at its first byte.
            if self.requestStartedAt == nil { self.requestStartedAt = .now() }
            self.reschedule()
        }
    }

    private func beginHandling() {
        queue.async { [weak self] in
            guard let self else { return }
            self.isHandling = true
            self.reschedule()
        }
    }

    private func endHandling() {
        queue.async { [weak self] in
            guard let self else { return }
            self.isHandling = false
            self.requestStartedAt = nil
            self.lastActivity = .now()
            self.reschedule()
        }
    }

    private func reschedule() {
        guard let timer, !isClosed else { return }

        // While the model is running there is nothing to time out: the client is
        // waiting on us, not the other way round. Cutting the connection here
        // would throw away a response that is still being computed.
        guard !isHandling else {
            timer.schedule(deadline: .distantFuture)
            return
        }

        let idleDeadline = lastActivity + configuration.idleTimeout
        guard let requestStartedAt else {
            timer.schedule(deadline: idleDeadline)
            return
        }

        let requestDeadline = requestStartedAt + configuration.requestTimeout
        timer.schedule(deadline: min(idleDeadline, requestDeadline))
    }

    private func handleTimeout() {
        guard !isClosed else { return }

        // A request that was still arriving gets told why it was dropped; an idle
        // connection has nothing outstanding to answer.
        if requestStartedAt != nil {
            let response = HTTPResponse.error(status: 408, message: "Request timed out")
            send(response.serialize(keepAlive: false)) { [weak self] in
                self?.close()
            }
            return
        }
        close()
    }
}
