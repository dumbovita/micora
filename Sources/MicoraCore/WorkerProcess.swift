import Foundation

public final class WorkerProcess: @unchecked Sendable {
    public let socketPath: String
    public let modelDirectory: String
    public let pythonExecutable: String
    public let backend: String

    private var process: Process?
    private let stateLock = NSLock()
    private var isTerminated = false

    public var onUnexpectedExit: (@Sendable (Int32) -> Void)?

    public init(
        socketPath: String = "/tmp/micora_\(ProcessInfo.processInfo.processIdentifier).sock",
        modelDirectory: String = "models/MOSS-TTS-Nano-100M-ONNX",
        pythonExecutable: String? = nil,
        backend: String = "moss"
    ) {
        self.socketPath = socketPath
        self.modelDirectory = (modelDirectory as NSString).expandingTildeInPath
        self.backend = backend
        if let py = pythonExecutable {
            self.pythonExecutable = py
        } else {
            // Prefer project venv if present
            let localVenvPy = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".venv/bin/python").path
            if FileManager.default.fileExists(atPath: localVenvPy) {
                self.pythonExecutable = localVenvPy
            } else {
                self.pythonExecutable = "/usr/bin/python3"
            }
        }
    }

    deinit {
        stop()
    }

    /// Starts the worker process and waits until the Unix domain socket is created.
    public func start(timeoutSeconds: Double = 15.0) throws {
        stateLock.lock()
        defer { stateLock.unlock() }

        if let proc = process, proc.isRunning {
            return
        }

        // Clean up stale socket file
        if FileManager.default.fileExists(atPath: socketPath) {
            try? FileManager.default.removeItem(atPath: socketPath)
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pythonExecutable)
        proc.arguments = [
            "-m", "worker.server",
            "--socket", socketPath,
            "--model-dir", modelDirectory,
            "--backend", backend,
            "--threads", "2"
        ]
        proc.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        proc.terminationHandler = { [weak self] p in
            guard let self = self else { return }
            self.stateLock.lock()
            let deliberate = self.isTerminated
            self.stateLock.unlock()

            if !deliberate {
                self.onUnexpectedExit?(p.terminationStatus)
            }
        }

        try proc.run()
        self.process = proc
        self.isTerminated = false

        // Wait for socket
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while Date() < deadline {
            if FileManager.default.fileExists(atPath: socketPath) {
                return
            }
            if !proc.isRunning {
                throw NSError(
                    domain: "MicoraWorker",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Worker process terminated prematurely with status \(proc.terminationStatus)"]
                )
            }
            Thread.sleep(forTimeInterval: 0.1)
        }

        stop()
        throw NSError(
            domain: "MicoraWorker",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "Timed out waiting for worker socket at \(socketPath)"]
        )
    }

    /// Stops the worker process cleanly with finite timeout and SIGKILL escalation.
    public func stop() {
        stateLock.lock()
        defer { stateLock.unlock() }

        isTerminated = true
        guard let proc = process, proc.isRunning else { return }

        proc.terminate()

        // Finite wait for exit (up to 2.0s)
        let deadline = Date().addingTimeInterval(2.0)
        while proc.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }

        // Escalate to SIGKILL if process is still running
        if proc.isRunning {
            kill(proc.processIdentifier, SIGKILL)
            let killDeadline = Date().addingTimeInterval(1.0)
            while proc.isRunning && Date() < killDeadline {
                Thread.sleep(forTimeInterval: 0.05)
            }
        }

        self.process = nil

        if FileManager.default.fileExists(atPath: socketPath) {
            try? FileManager.default.removeItem(atPath: socketPath)
        }
    }

    public var isRunning: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return process?.isRunning ?? false
    }
}
