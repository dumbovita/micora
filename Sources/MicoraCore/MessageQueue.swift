import Foundation
import Synchronization

public enum SubmissionBehavior: String, Sendable, Codable, CaseIterable {
    case addToQueue = "Add to Queue"
    case interruptCurrent = "Interrupt Current Speech"
}

public enum QueueItemState: String, Sendable, Codable, Equatable {
    case queued
    case generating
    case speaking
    case completed
    case cancelled
    case failed
}

public struct QueueItem: Identifiable, Sendable, Equatable {
    public let id: String
    public let text: String
    public let voiceId: String
    public var state: QueueItemState
    public var error: String?
    public let createdAt: Date

    public init(
        id: String = "req_\(Int(Date().timeIntervalSince1970 * 1000))_\(UUID().uuidString.prefix(6))",
        text: String,
        voiceId: String,
        state: QueueItemState = .queued,
        error: String? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.text = text
        self.voiceId = voiceId
        self.state = state
        self.error = error
        self.createdAt = createdAt
    }
}

/// Actor-based message queue managing sequential synthesis, audio streaming, and speech lifecycle.
public actor MessageQueue {
    public var submissionBehavior: SubmissionBehavior = .addToQueue

    private let client: WorkerClient
    private let ringBuffer: AudioRingBuffer
    private let secondaryRingBuffer: AudioRingBuffer?
    private let isSecondaryActive: (@Sendable () -> Bool)?

    private var itemsList: [QueueItem] = []
    private var activeItemId: String? = nil
    private var isProcessing = false
    private var isCurrentCancelled = false

    public var onQueueChanged: (@Sendable ([QueueItem]) -> Void)?

    public init(
        client: WorkerClient,
        ringBuffer: AudioRingBuffer,
        secondaryRingBuffer: AudioRingBuffer? = nil,
        isSecondaryActive: (@Sendable () -> Bool)? = nil,
        submissionBehavior: SubmissionBehavior = .addToQueue
    ) {
        self.client = client
        self.ringBuffer = ringBuffer
        self.secondaryRingBuffer = secondaryRingBuffer
        self.isSecondaryActive = isSecondaryActive
        self.submissionBehavior = submissionBehavior
    }

    /// Snapshot of all items in the queue.
    public var items: [QueueItem] {
        return itemsList
    }

    /// Sets the submission behavior (Add to Queue vs Interrupt Current Speech).
    public func setSubmissionBehavior(_ behavior: SubmissionBehavior) {
        self.submissionBehavior = behavior
    }

    /// Sets the queue change notification callback.
    public func setOnQueueChanged(_ callback: (@Sendable ([QueueItem]) -> Void)?) {
        self.onQueueChanged = callback
    }

    /// Submits a new text message.
    @discardableResult
    public func submit(text: String, voiceId: String) -> String {
        let item = QueueItem(text: text, voiceId: voiceId)

        if submissionBehavior == .interruptCurrent && activeItemId != nil {
            if let currentId = activeItemId {
                try? client.cancel(requestId: currentId)
                ringBuffer.clear()
                secondaryRingBuffer?.clear()
                isCurrentCancelled = true
                if let idx = itemsList.firstIndex(where: { $0.id == currentId }) {
                    itemsList[idx].state = .cancelled
                }
            }
            itemsList.insert(item, at: 0)
        } else {
            itemsList.append(item)
        }

        notifyChange()

        if !isProcessing {
            isProcessing = true
            Task {
                await self.processLoop()
            }
        }

        return item.id
    }

    /// Stops the currently active message immediately and advances to next queued.
    public func stopCurrent() {
        guard let currentId = activeItemId else { return }

        try? client.cancel(requestId: currentId)
        ringBuffer.clear()
        secondaryRingBuffer?.clear()
        isCurrentCancelled = true

        if let idx = itemsList.firstIndex(where: { $0.id == currentId }) {
            itemsList[idx].state = .cancelled
        }
        activeItemId = nil
        notifyChange()
    }

    /// Clears the entire queue, stops active speech immediately, and empties the queue list.
    public func clearQueue() {
        if let currentId = activeItemId {
            try? client.cancel(requestId: currentId)
            ringBuffer.clear()
            secondaryRingBuffer?.clear()
            isCurrentCancelled = true
        }

        activeItemId = nil
        itemsList.removeAll()
        notifyChange()
    }

    private func notifyChange() {
        let snapshot = itemsList
        onQueueChanged?(snapshot)
    }

    private func updateItemState(id: String, state: QueueItemState, error: String? = nil) {
        if let idx = itemsList.firstIndex(where: { $0.id == id }) {
            if itemsList[idx].state != .cancelled {
                itemsList[idx].state = state
                if let err = error {
                    itemsList[idx].error = err
                }
            }
        }
        notifyChange()
    }

    // MARK: - Queue Processing Loop

    private func processLoop() async {
        while true {
            guard let idx = itemsList.firstIndex(where: { $0.state == .queued }) else {
                isProcessing = false
                activeItemId = nil
                break
            }

            itemsList[idx].state = .generating
            activeItemId = itemsList[idx].id
            isCurrentCancelled = false
            let currentItem = itemsList[idx]
            notifyChange()

            await processItem(currentItem)
        }
    }

    private func processItem(_ item: QueueItem) async {
        let itemId = item.id
        let targetRing = self.ringBuffer
        let targetSecondary = self.secondaryRingBuffer
        let isSecActive = self.isSecondaryActive
        let hasSpoken = Atomic<Bool>(false)

        // Bounded Flow-Control Pacer:
        // Monitors frames consumed by CoreAudio and sends credit updates to the worker.
        // Guarantees zero audio truncation and strictly bounded memory (<2s ahead of playback).
        let pacer = Task { [client] in
            var lastAckedR = targetRing.readSequence
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 50_000_000) // 50ms polling interval
                guard !Task.isCancelled else { break }
                let currentR = targetRing.readSequence
                let consumed = currentR - lastAckedR
                if consumed >= 24_000 { // 0.5s of audio consumed
                    do {
                        try client.ackFrames(requestId: itemId, frames: consumed)
                        lastAckedR = currentR
                    } catch {
                        break
                    }
                }
            }
        }
        defer {
            pacer.cancel()
        }

        do {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                do {
                    try self.client.synthesize(
                        requestId: itemId,
                        voiceId: item.voiceId,
                        text: item.text,
                        flowControl: true,
                        initialCreditFrames: 96_000,
                        onChunk: { [weak self] data in
                            targetRing.write(data: data)
                            if let sec = targetSecondary, isSecActive?() ?? true {
                                sec.write(data: data)
                            }

                            if !hasSpoken.exchange(true, ordering: .relaxed) {
                                Task {
                                    await self?.handleFirstChunk(itemId: itemId)
                                }
                            }
                        },
                        onComplete: {
                            continuation.resume()
                        },
                        onError: { err in
                            continuation.resume(throwing: NSError(domain: "MicoraQueue", code: 1, userInfo: [NSLocalizedDescriptionKey: err]))
                        },
                        onCancel: {
                            continuation.resume(throwing: CancellationError())
                        }
                    )
                } catch {
                    continuation.resume(throwing: error)
                }
            }

            // Wait for audio playback to drain before marking completed (with safety deadline)
            let audioDuration = Double(targetRing.availableSamplesToRead / 2) / 48000.0
            let drainDeadline = Date().addingTimeInterval(audioDuration + 0.5)
            while !isCurrentCancelled && targetRing.availableSamplesToRead > 512 && Date() < drainDeadline {
                try? await Task.sleep(nanoseconds: 50_000_000)
            }

            if !isCurrentCancelled && activeItemId == itemId {
                if targetRing.availableSamplesToRead > 0 {
                    targetRing.clear()
                    targetSecondary?.clear()
                }
                updateItemState(id: itemId, state: .completed)
                activeItemId = nil
            }

        } catch is CancellationError {
            // Cancelled cleanly — activeItemId and state were handled by cancellation trigger
        } catch {
            if !isCurrentCancelled {
                updateItemState(id: itemId, state: .failed, error: error.localizedDescription)
                activeItemId = nil
            }
        }
    }

    private func handleFirstChunk(itemId: String) {
        if activeItemId == itemId && !isCurrentCancelled {
            updateItemState(id: itemId, state: .speaking)
        }
    }
}
