import Foundation
import Darwin

public struct WorkerHandshakeResponse: Sendable {
    public let protocolVersion: Int
    public let sampleRate: Int
    public let channels: Int
    public let state: String
}

public final class WorkerClient: @unchecked Sendable {
    private var socketFd: Int32 = -1
    private let sendLock = NSLock()
    private var readerThread: Thread?
    private var isConnected = false

    // Callbacks mapped by requestId
    private let callbacksLock = NSLock()
    private var chunkCallbacks: [String: @Sendable (Data) -> Void] = [:]
    private var completionCallbacks: [String: @Sendable () -> Void] = [:]
    private var errorCallbacks: [String: @Sendable (String) -> Void] = [:]
    private var cancelCallbacks: [String: @Sendable () -> Void] = [:]

    // Correlated RPC response waiters mapped by correlationId
    private var pendingRpcContinuations: [String: @Sendable (Result<Data, Error>) -> Void] = [:]

    public init() {}

    deinit {
        disconnect()
    }

    /// Connects to the worker's Unix domain socket.
    public func connect(to socketPath: String) throws {
        sendLock.lock()
        defer { sendLock.unlock() }

        if socketFd >= 0 {
            close(socketFd)
        }

        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno), userInfo: nil)
        }

        #if os(macOS)
        var nosigpipe: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &nosigpipe, socklen_t(MemoryLayout<Int32>.size))
        #endif

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)

        let pathBytes = socketPath.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            close(fd)
            throw NSError(domain: "MicoraClient", code: 1, userInfo: [NSLocalizedDescriptionKey: "Socket path too long"])
        }

        withUnsafeMutablePointer(to: &addr.sun_path.0) { ptr in
            _ = pathBytes.withUnsafeBufferPointer { buf in
                memcpy(ptr, buf.baseAddress!, buf.count)
            }
        }

        let connectRes = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                Darwin.connect(fd, sa, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }

        guard connectRes == 0 else {
            let err = errno
            close(fd)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(err), userInfo: nil)
        }

        self.socketFd = fd
        self.isConnected = true

        // Launch reader thread
        let thread = Thread { [weak self] in
            self?.readLoop(fd: fd)
        }
        thread.name = "micora.worker.client.reader"
        self.readerThread = thread
        thread.start()
    }

    /// Disconnects from the worker and fails all pending continuations.
    public func disconnect() {
        sendLock.lock()
        let wasConnected = isConnected
        isConnected = false
        if socketFd >= 0 {
            close(socketFd)
            socketFd = -1
        }
        sendLock.unlock()

        guard wasConnected else { return }

        callbacksLock.lock()
        let rpcs = pendingRpcContinuations
        pendingRpcContinuations.removeAll()

        let errors = errorCallbacks
        errorCallbacks.removeAll()
        chunkCallbacks.removeAll()
        completionCallbacks.removeAll()
        cancelCallbacks.removeAll()
        callbacksLock.unlock()

        let disconnectError = NSError(
            domain: "MicoraClient",
            code: 100,
            userInfo: [NSLocalizedDescriptionKey: "Worker connection closed"]
        )

        for (_, waiter) in rpcs {
            waiter(.failure(disconnectError))
        }

        for (_, errCb) in errors {
            errCb("Worker connection closed")
        }
    }

    /// Sends raw bytes over socket under lock.
    private func sendData(_ data: Data) throws {
        sendLock.lock()
        defer { sendLock.unlock() }

        guard socketFd >= 0 else {
            throw NSError(domain: "MicoraClient", code: 2, userInfo: [NSLocalizedDescriptionKey: "Not connected"])
        }

        try data.withUnsafeBytes { rawPtr in
            guard let base = rawPtr.baseAddress else { return }
            var remaining = rawPtr.count
            var sent = 0
            while remaining > 0 {
                let bytesSent = send(socketFd, base + sent, remaining, 0)
                if bytesSent < 0 {
                    let err = errno
                    throw NSError(domain: NSPOSIXErrorDomain, code: Int(err), userInfo: nil)
                }
                sent += bytesSent
                remaining -= bytesSent
            }
        }
    }

    /// Sends a request-correlated RPC control packet and awaits the corresponding response.
    private func sendRpcRequest(command: [String: Any]) async throws -> Data {
        let corrId = "rpc_\(UUID().uuidString.prefix(8))"
        var cmd = command
        cmd["correlation_id"] = corrId

        let jsonData = try JSONSerialization.data(withJSONObject: cmd)
        let packet = WorkerProtocol.encode(type: .controlJson, payload: jsonData)

        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Data, Error>) in
            callbacksLock.lock()
            guard isConnected else {
                callbacksLock.unlock()
                continuation.resume(throwing: NSError(domain: "MicoraClient", code: 2, userInfo: [NSLocalizedDescriptionKey: "Not connected"]))
                return
            }
            pendingRpcContinuations[corrId] = { result in
                switch result {
                case .success(let data):
                    continuation.resume(returning: data)
                case .failure(let error):
                    continuation.resume(throwing: error)
                }
            }
            callbacksLock.unlock()

            do {
                try self.sendData(packet)
            } catch {
                callbacksLock.lock()
                _ = self.pendingRpcContinuations.removeValue(forKey: corrId)
                callbacksLock.unlock()
                continuation.resume(throwing: error)
            }
        }
    }

    /// Sends an arbitrary command payload and awaits the correlated response.
    /// Used for adversarial testing of unhandled/unknown commands and protocol boundaries.
    public func sendCustomCommand(_ command: [String: Any]) async throws -> Data {
        return try await sendRpcRequest(command: command)
    }

    /// Perform handshake with worker.
    public func handshake() async throws -> WorkerHandshakeResponse {
        let payload = try await sendRpcRequest(command: [
            "type": "init",
            "protocol_version": WorkerProtocol.version
        ])
        let responseMsg = try JSONDecoder().decode(ControlMessage.self, from: payload)

        guard responseMsg.type == "ready" else {
            let err = responseMsg.error ?? "Unknown error"
            throw NSError(domain: "MicoraClient", code: 3, userInfo: [NSLocalizedDescriptionKey: err])
        }

        return WorkerHandshakeResponse(
            protocolVersion: responseMsg.protocolVersion ?? 0,
            sampleRate: responseMsg.sampleRate ?? 48000,
            channels: responseMsg.channels ?? 2,
            state: responseMsg.state ?? "ready"
        )
    }

    /// Prepare a voice profile.
    public func prepareVoice(voiceId: String, audioPath: String? = nil, preset: String? = nil) async throws -> Int {
        var cmd: [String: Any] = ["type": "prepare_voice", "voice_id": voiceId]
        if let p = audioPath { cmd["audio_path"] = p }
        if let pr = preset { cmd["preset"] = pr }

        let payload = try await sendRpcRequest(command: cmd)
        let responseMsg = try JSONDecoder().decode(ControlMessage.self, from: payload)

        if responseMsg.type == "voice_ready" {
            return responseMsg.frameCount ?? 0
        } else {
            let err = responseMsg.error ?? "Voice preparation failed"
            throw NSError(domain: "MicoraClient", code: 4, userInfo: [NSLocalizedDescriptionKey: err])
        }
    }

    /// Dynamically switches synthesis backend ("moss" or "chatterbox").
    public func switchBackend(backend: String) async throws -> ControlMessage {
        let payload = try await sendRpcRequest(command: ["type": "set_backend", "backend": backend])
        let responseMsg = try JSONDecoder().decode(ControlMessage.self, from: payload)

        if responseMsg.type == "backend_switched" {
            return responseMsg
        } else {
            let err = responseMsg.error ?? "Backend switch failed"
            throw NSError(domain: "MicoraClient", code: 5, userInfo: [NSLocalizedDescriptionKey: err])
        }
    }

    /// Query current backend capabilities.
    public func getCapabilities() async throws -> ControlMessage {
        let payload = try await sendRpcRequest(command: ["type": "get_capabilities"])
        return try JSONDecoder().decode(ControlMessage.self, from: payload)
    }

    /// Scan a local folder for voice dataset reference candidates.
    public func scanDataset(folderPath: String, maxCandidates: Int = 5) async throws -> DatasetScanResult {
        let payload = try await sendRpcRequest(command: [
            "type": "scan_dataset",
            "folder_path": folderPath,
            "max_candidates": maxCandidates
        ])
        let result = try JSONDecoder().decode(DatasetScanResult.self, from: payload)
        if let err = result.error {
            throw NSError(domain: "MicoraClient", code: 6, userInfo: [NSLocalizedDescriptionKey: err])
        }
        return result
    }

    /// Import voice audio directly from a Hugging Face repository.
    public func importHuggingFaceVoice(repoId: String, maxSamples: Int = 30, maxCandidates: Int = 5) async throws -> HfImportResult {
        let payload = try await sendRpcRequest(command: [
            "type": "import_hf",
            "repo_id": repoId,
            "max_samples": maxSamples,
            "max_candidates": maxCandidates
        ])
        let result = try JSONDecoder().decode(HfImportResult.self, from: payload)
        if let err = result.error {
            throw NSError(domain: "MicoraClient", code: 7, userInfo: [NSLocalizedDescriptionKey: err])
        }
        return result
    }

    /// Acknowledges consumed audio frames to provide bounded backpressure to the worker.
    public func ackFrames(requestId: String, frames: Int) throws {
        let cmd: [String: Any] = [
            "type": "ack_frames",
            "request_id": requestId,
            "frames": frames
        ]
        let jsonData = try JSONSerialization.data(withJSONObject: cmd)
        let packet = WorkerProtocol.encode(type: .controlJson, payload: jsonData)
        try sendData(packet)
    }

    /// Start synthesis request, receiving streamed PCM chunks.
    /// `flowControl` must be explicitly specified:
    /// - `true`: Uses credit-based pacing with the worker (standard for MessageQueue and production playback).
    /// - `false`: Unpaced streaming (only for raw low-level benchmarks or standalone unit tests).
    public func synthesize(
        requestId: String,
        voiceId: String,
        text: String,
        flowControl: Bool,
        initialCreditFrames: Int = 96_000,
        onChunk: @escaping @Sendable (Data) -> Void,
        onComplete: @escaping @Sendable () -> Void,
        onError: @escaping @Sendable (String) -> Void,
        onCancel: (@Sendable () -> Void)? = nil
    ) throws {
        callbacksLock.lock()
        chunkCallbacks[requestId] = onChunk
        completionCallbacks[requestId] = onComplete
        errorCallbacks[requestId] = onError
        if let onCancel = onCancel {
            cancelCallbacks[requestId] = onCancel
        }
        callbacksLock.unlock()

        let cmd: [String: Any] = [
            "type": "synthesize",
            "request_id": requestId,
            "voice_id": voiceId,
            "text": text,
            "flow_control": flowControl,
            "initial_credit_frames": initialCreditFrames
        ]
        let jsonData = try JSONSerialization.data(withJSONObject: cmd)
        let packet = WorkerProtocol.encode(type: .controlJson, payload: jsonData)
        try sendData(packet)
    }

    /// Cancel a running synthesis request.
    public func cancel(requestId: String) throws {
        callbacksLock.lock()
        chunkCallbacks.removeValue(forKey: requestId)
        completionCallbacks.removeValue(forKey: requestId)
        errorCallbacks.removeValue(forKey: requestId)
        let cancelCb = cancelCallbacks.removeValue(forKey: requestId)
        callbacksLock.unlock()

        let cmd: [String: Any] = [
            "type": "cancel",
            "request_id": requestId
        ]
        let jsonData = try JSONSerialization.data(withJSONObject: cmd)
        let packet = WorkerProtocol.encode(type: .controlJson, payload: jsonData)
        try sendData(packet)

        cancelCb?()
    }

    /// Send shutdown signal to worker.
    public func shutdown() throws {
        let cmd: [String: Any] = ["type": "shutdown"]
        let jsonData = try JSONSerialization.data(withJSONObject: cmd)
        let packet = WorkerProtocol.encode(type: .controlJson, payload: jsonData)
        try sendData(packet)
    }

    // MARK: - Reader Loop

    private func readLoop(fd: Int32) {
        defer {
            disconnect()
        }

        var headerBuffer = [UInt8](repeating: 0, count: 5)
        let decoder = JSONDecoder()

        while isConnected {
            // Read 5-byte header
            var readCount = 0
            var headerFailed = false
            headerBuffer.withUnsafeMutableBytes { rawBuf in
                guard let base = rawBuf.baseAddress else { return }
                while readCount < 5 {
                    let res = recv(fd, base + readCount, 5 - readCount, 0)
                    if res <= 0 {
                        headerFailed = true
                        break
                    }
                    readCount += res
                }
            }
            if headerFailed { break }

            let ptypeVal = headerBuffer[0]
            guard let ptype = WorkerPacketType(rawValue: ptypeVal) else {
                print("[WorkerClient] Unknown packet type: \(ptypeVal)")
                break
            }

            let length: Int = headerBuffer.withUnsafeBytes { raw in
                let lengthBE = raw.loadUnaligned(fromByteOffset: 1, as: UInt32.self)
                return Int(UInt32(bigEndian: lengthBE))
            }

            // Guard against oversized payload (max 8 MB)
            guard length <= 8 * 1024 * 1024 else {
                print("[WorkerClient] Oversized packet payload: \(length) bytes (max 8 MB)")
                break
            }

            // Read payload
            var payloadData = Data(count: length)
            var payloadRead = 0
            var readFailed = false
            payloadData.withUnsafeMutableBytes { ptr in
                guard let base = ptr.baseAddress else { return }
                while payloadRead < length {
                    let res = recv(fd, base + payloadRead, length - payloadRead, 0)
                    if res <= 0 {
                        readFailed = true
                        break
                    }
                    payloadRead += res
                }
            }
            if readFailed {
                print("[WorkerClient] payload read failed")
                break
            }

            handlePacket(type: ptype, payload: payloadData, decoder: decoder)
        }
    }

    private func handlePacket(type: WorkerPacketType, payload: Data, decoder: JSONDecoder) {
        switch type {
        case .controlJson:
            guard let rawJson = (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any] else {
                print("[WorkerClient] Failed to deserialize JSON control packet")
                return
            }

            let rawType = rawJson["type"] as? String
            let corrId = rawJson["correlation_id"] as? String

            // Handle cancellation push event if applicable
            if rawType == "cancelled", let reqId = rawJson["request_id"] as? String {
                callbacksLock.lock()
                let cancelCb = cancelCallbacks.removeValue(forKey: reqId)
                chunkCallbacks.removeValue(forKey: reqId)
                completionCallbacks.removeValue(forKey: reqId)
                errorCallbacks.removeValue(forKey: reqId)
                callbacksLock.unlock()
                cancelCb?()
            }

            // If correlated with an in-flight RPC continuation, resolve it
            if let corrId = corrId {
                callbacksLock.lock()
                let waiter = pendingRpcContinuations.removeValue(forKey: corrId)
                callbacksLock.unlock()
                if rawType == "error" {
                    let errMsg = rawJson["error"] as? String ?? "Worker error"
                    waiter?(.failure(NSError(domain: "MicoraWorker", code: 99, userInfo: [NSLocalizedDescriptionKey: errMsg])))
                } else {
                    waiter?(.success(payload))
                }
            }

        case .audioPcm:
            if let decoded = WorkerProtocol.decodePcmPayload(payload) {
                callbacksLock.lock()
                let cb = chunkCallbacks[decoded.requestId]
                callbacksLock.unlock()
                cb?(decoded.pcmData)
            }

        case .audioEos:
            if let decoded = WorkerProtocol.decodePcmPayload(payload) {
                callbacksLock.lock()
                let cb = completionCallbacks.removeValue(forKey: decoded.requestId)
                chunkCallbacks.removeValue(forKey: decoded.requestId)
                errorCallbacks.removeValue(forKey: decoded.requestId)
                cancelCallbacks.removeValue(forKey: decoded.requestId)
                callbacksLock.unlock()
                cb?()
            }

        case .error:
            if let rawJson = (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any] {
                if let corrId = rawJson["correlation_id"] as? String {
                    callbacksLock.lock()
                    let waiter = pendingRpcContinuations.removeValue(forKey: corrId)
                    callbacksLock.unlock()
                    let errMsg = rawJson["error"] as? String ?? "Worker error"
                    waiter?(.failure(NSError(domain: "MicoraWorker", code: 99, userInfo: [NSLocalizedDescriptionKey: errMsg])))
                }
            }
            if let msg = try? decoder.decode(ControlMessage.self, from: payload),
               let reqId = msg.requestId,
               let err = msg.error {
                callbacksLock.lock()
                let cb = errorCallbacks.removeValue(forKey: reqId)
                chunkCallbacks.removeValue(forKey: reqId)
                completionCallbacks.removeValue(forKey: reqId)
                cancelCallbacks.removeValue(forKey: reqId)
                callbacksLock.unlock()
                cb?(err)
            }
        }
    }
}
