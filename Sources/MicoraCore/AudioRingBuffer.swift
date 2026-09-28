import Foundation
import Synchronization

/// Single-Producer / Single-Consumer (SPSC) lock-free ring buffer for 48 kHz stereo Float32 PCM audio.
///
/// Thread safety contract:
/// - Producer thread: Calls `write(...)` when PCM chunks arrive from the worker IPC.
/// - Consumer thread: Real-time CoreAudio render callback calls `read(...)`.
/// - Neither thread locks or allocates memory.
public final class AudioRingBuffer: @unchecked Sendable {
    public let capacityInFrames: Int
    public let channels: Int
    public let capacityInSamples: Int

    private let buffer: UnsafeMutablePointer<Float>
    private let writeFrameSeq = Atomic<Int>(0)
    private let readFrameSeq = Atomic<Int>(0)
    private let discardUpToFrameSeq = Atomic<Int>(0)

    /// Initializes ring buffer for audio with specified frame capacity and channel count.
    /// Default: 240,000 frames = 5 seconds of 48 kHz stereo audio (~1.92 MB).
    public init(capacityInFrames: Int = 240_000, channels: Int = 2) {
        precondition(channels >= 1, "Channels must be at least 1")
        precondition(capacityInFrames >= 2, "Capacity in frames must be at least 2")
        self.capacityInFrames = capacityInFrames
        self.channels = channels
        self.capacityInSamples = capacityInFrames * channels
        self.buffer = UnsafeMutablePointer<Float>.allocate(capacity: self.capacityInSamples)
        self.buffer.initialize(repeating: 0.0, count: self.capacityInSamples)
    }

    deinit {
        buffer.deinitialize(count: capacityInSamples)
        buffer.deallocate()
    }

    /// Monotonic count of frames consumed by the reader. Useful for flow-control pacing.
    public var readSequence: Int {
        return readFrameSeq.load(ordering: .relaxed)
    }

    /// Monotonic count of frames written by the producer.
    public var writeSequence: Int {
        return writeFrameSeq.load(ordering: .relaxed)
    }

    /// Number of available audio frames ready to be read.
    public var availableFramesToRead: Int {
        let discardCutoff = discardUpToFrameSeq.load(ordering: .acquiring)
        let r = readFrameSeq.load(ordering: .relaxed)
        let effectiveR = max(r, discardCutoff)
        let w = writeFrameSeq.load(ordering: .acquiring)
        return max(0, w - effectiveR)
    }

    /// Number of available samples ready to be read by the audio callback.
    public var availableSamplesToRead: Int {
        return availableFramesToRead * channels
    }

    /// Number of frame slots available to write into before buffer is full.
    public var availableFramesToWrite: Int {
        return (capacityInFrames - 1) - availableFramesToRead
    }

    /// Number of sample slots available to write into before buffer is full.
    public var availableSamplesToWrite: Int {
        return availableFramesToWrite * channels
    }

    /// Writes raw interleaved Float32 audio samples into the ring buffer.
    /// Called by the IPC producer thread.
    /// Preserves strict SPSC contract: producer ALONE modifies writeFrameSeq.
    /// Producer NEVER writes readFrameSeq.
    /// Guarantees channel alignment: only complete frames are written.
    /// Returns the number of samples actually written.
    @discardableResult
    public func write(samples: UnsafePointer<Float>, count: Int) -> Int {
        guard count > 0 else { return 0 }

        let inputFrames = count / channels
        guard inputFrames > 0 else { return 0 }

        let w = writeFrameSeq.load(ordering: .relaxed)
        let r = readFrameSeq.load(ordering: .acquiring)
        let discardCutoff = discardUpToFrameSeq.load(ordering: .relaxed)
        let effectiveR = max(r, discardCutoff)

        let usedFrames = max(0, w - effectiveR)
        let freeFrames = (capacityInFrames - 1) - usedFrames
        let toWriteFrames = min(inputFrames, max(0, freeFrames))

        if toWriteFrames < inputFrames {
            let droppedFrames = inputFrames - toWriteFrames
            print("[AudioRingBuffer] Warning: buffer full, dropped \(droppedFrames * channels) samples (\(droppedFrames) frames)")
        }

        guard toWriteFrames > 0 else { return 0 }

        let wMod = w % capacityInFrames
        let firstChunkFrames = min(toWriteFrames, capacityInFrames - wMod)
        let firstChunkSamples = firstChunkFrames * channels
        buffer.advanced(by: wMod * channels).update(from: samples, count: firstChunkSamples)

        let secondChunkFrames = toWriteFrames - firstChunkFrames
        if secondChunkFrames > 0 {
            let secondChunkSamples = secondChunkFrames * channels
            buffer.update(from: samples.advanced(by: firstChunkSamples), count: secondChunkSamples)
        }

        writeFrameSeq.store(w + toWriteFrames, ordering: .releasing)
        return toWriteFrames * channels
    }

    /// Convenience write from Swift Array of Floats.
    @discardableResult
    public func write(_ samples: [Float]) -> Int {
        samples.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return 0 }
            return write(samples: base, count: ptr.count)
        }
    }

    /// Convenience write from raw Data containing Float32 samples.
    @discardableResult
    public func write(data: Data) -> Int {
        let sampleCount = data.count / MemoryLayout<Float>.size
        return data.withUnsafeBytes { rawPtr in
            guard let base = rawPtr.baseAddress?.assumingMemoryBound(to: Float.self) else { return 0 }
            return write(samples: base, count: sampleCount)
        }
    }

    /// Reads up to `count` samples into `destination`.
    /// Called on the real-time audio callback thread.
    /// Preserves strict SPSC contract: consumer ALONE modifies readFrameSeq.
    /// Discards only frames prior to the cancellation cutoff; newly written frames remain intact.
    /// If fewer samples are available than requested, the deficit is filled with silence (0.0).
    /// Guarantees channel alignment: only complete frames are read.
    /// Returns the number of actual audio samples read (excluding silence padding).
    @discardableResult
    public func read(into destination: UnsafeMutablePointer<Float>, count: Int) -> Int {
        guard count > 0 else { return 0 }

        let discardCutoff = discardUpToFrameSeq.load(ordering: .acquiring)
        var r = readFrameSeq.load(ordering: .relaxed)

        if r < discardCutoff {
            r = discardCutoff
            readFrameSeq.store(r, ordering: .releasing)
        }

        let requestedFrames = count / channels
        let w = writeFrameSeq.load(ordering: .acquiring)

        let availableFrames = max(0, w - r)
        let toReadFrames = min(requestedFrames, availableFrames)
        let toReadSamples = toReadFrames * channels

        if toReadFrames > 0 {
            let rMod = r % capacityInFrames
            let firstChunkFrames = min(toReadFrames, capacityInFrames - rMod)
            let firstChunkSamples = firstChunkFrames * channels
            destination.update(from: buffer.advanced(by: rMod * channels), count: firstChunkSamples)

            let secondChunkFrames = toReadFrames - firstChunkFrames
            if secondChunkFrames > 0 {
                let secondChunkSamples = secondChunkFrames * channels
                destination.advanced(by: firstChunkSamples).update(from: buffer, count: secondChunkSamples)
            }

            let newR = r + toReadFrames
            readFrameSeq.store(newR, ordering: .releasing)
        }

        // Fill remaining deficit with silence (0.0)
        let deficit = count - toReadSamples
        if deficit > 0 {
            destination.advanced(by: toReadSamples).initialize(repeating: 0.0, count: deficit)
        }

        return toReadSamples
    }

    /// Atomically records a cancellation discard boundary at the current write sequence.
    /// Neither modifies readFrameSeq nor writeFrameSeq.
    /// Consumer advances its read cursor only up to this exact boundary, leaving new frames intact.
    public func clear() {
        let currentW = writeFrameSeq.load(ordering: .relaxed)
        var currentDiscard = discardUpToFrameSeq.load(ordering: .relaxed)
        while currentW > currentDiscard {
            let (exchanged, original) = discardUpToFrameSeq.compareExchange(
                expected: currentDiscard,
                desired: currentW,
                successOrdering: .releasing,
                failureOrdering: .relaxed
            )
            if exchanged { break }
            currentDiscard = original
        }
    }
}
