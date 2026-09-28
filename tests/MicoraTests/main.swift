import Foundation
import Synchronization
import MicoraCore

func testBasicWriteAndRead() {
    let buffer = AudioRingBuffer(capacityInFrames: 100, channels: 2)
    assert(buffer.availableSamplesToRead == 0, "Buffer should be empty initially")

    let input: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    let written = buffer.write(input)
    assert(written == 6, "Expected 6 samples written")
    assert(buffer.availableSamplesToRead == 6, "Expected 6 available samples")

    var output = [Float](repeating: -1.0, count: 6)
    output.withUnsafeMutableBufferPointer { ptr in
        let readCount = buffer.read(into: ptr.baseAddress!, count: 6)
        assert(readCount == 6, "Expected 6 samples read")
    }

    assert(output == input, "Output should match input")
    assert(buffer.availableSamplesToRead == 0, "Buffer should be empty after reading")
    print("✓ testBasicWriteAndRead passed")
}

func testUnderflowSilencePadding() {
    let buffer = AudioRingBuffer(capacityInFrames: 10, channels: 2)
    let input: [Float] = [1.0, 2.0]
    buffer.write(input)

    var output = [Float](repeating: 99.0, count: 6)
    output.withUnsafeMutableBufferPointer { ptr in
        let readCount = buffer.read(into: ptr.baseAddress!, count: 6)
        assert(readCount == 2, "Expected 2 actual audio samples")
    }

    assert(output[0] == 1.0 && output[1] == 2.0, "First 2 samples should match input")
    for i in 2..<6 {
        assert(output[i] == 0.0, "Deficit sample \(i) should be padded with silence 0.0")
    }
    print("✓ testUnderflowSilencePadding passed")
}

func testWraparound() {
    let buffer = AudioRingBuffer(capacityInFrames: 4, channels: 2) // 8 samples
    let input1: [Float] = [1.0, 2.0, 3.0, 4.0]
    buffer.write(input1)

    var out1 = [Float](repeating: 0.0, count: 4)
    out1.withUnsafeMutableBufferPointer { ptr in
        _ = buffer.read(into: ptr.baseAddress!, count: 4)
    }
    assert(out1 == input1)

    let input2: [Float] = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    let written = buffer.write(input2)
    assert(written == 6)

    var out2 = [Float](repeating: 0.0, count: 6)
    out2.withUnsafeMutableBufferPointer { ptr in
        let count = buffer.read(into: ptr.baseAddress!, count: 6)
        assert(count == 6)
    }
    assert(out2 == input2)
    print("✓ testWraparound passed")
}

func testClearDiscardsBufferedAudio() {
    let buffer = AudioRingBuffer(capacityInFrames: 50, channels: 2)
    buffer.write([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert(buffer.availableSamplesToRead == 6)

    buffer.clear()
    assert(buffer.availableSamplesToRead == 0)

    var out = [Float](repeating: 1.0, count: 4)
    out.withUnsafeMutableBufferPointer { ptr in
        let count = buffer.read(into: ptr.baseAddress!, count: 4)
        assert(count == 0)
    }
    assert(out == [0.0, 0.0, 0.0, 0.0])
    print("✓ testClearDiscardsBufferedAudio passed")
}

final class ConcurrencyBox: @unchecked Sendable {
    var samples: [Float] = []
    let lock = NSLock()
    func append(_ newSamples: [Float]) {
        lock.lock()
        samples.append(contentsOf: newSamples)
        lock.unlock()
    }
}

func testConcurrentProducerConsumer() {
    let buffer = AudioRingBuffer(capacityInFrames: 1000, channels: 2)
    let totalSamples = 50_000
    let chunkSize = 64
    let box = ConcurrencyBox()
    box.samples.reserveCapacity(totalSamples)

    let group = DispatchGroup()

    // Producer thread
    group.enter()
    Thread.detachNewThread {
        var sent = 0
        while sent < totalSamples {
            let count = min(chunkSize, totalSamples - sent)
            let chunk = (0..<count).map { Float(sent + $0) }
            let written = buffer.write(chunk)
            sent += written
            if written == 0 {
                usleep(100)
            }
        }
        group.leave()
    }

    // Consumer thread
    group.enter()
    Thread.detachNewThread {
        var readTotal = 0
        var temp = [Float](repeating: 0.0, count: chunkSize)
        while readTotal < totalSamples {
            let actual = temp.withUnsafeMutableBufferPointer { ptr in
                buffer.read(into: ptr.baseAddress!, count: chunkSize)
            }
            if actual > 0 {
                box.append(Array(temp[0..<actual]))
                readTotal += actual
            } else {
                usleep(100)
            }
        }
        group.leave()
    }

    let waitResult = group.wait(timeout: .now() + 5.0)
    assert(waitResult == .success, "Concurrent test timed out")
    assert(box.samples.count == totalSamples, "Received sample count mismatch")
    for i in 0..<totalSamples {
        assert(box.samples[i] == Float(i), "Sample mismatch at index \(i)")
    }
    print("✓ testConcurrentProducerConsumer passed (50,000 samples transferred concurrently)")
}

func testSwiftWorkerClientIntegration() async throws {
    print("\n--- Testing Swift WorkerProcess and WorkerClient Integration ---")
    let sockPath = "/tmp/micora_swift_ipc_\(ProcessInfo.processInfo.processIdentifier).sock"
    let worker = WorkerProcess(socketPath: sockPath)

    print("Starting worker process from Swift...")
    try worker.start(timeoutSeconds: 15.0)
    assert(worker.isRunning, "Worker should be running")

    let client = WorkerClient()
    print("Connecting WorkerClient to \(sockPath)...")
    try client.connect(to: sockPath)

    print("Performing IPC handshake...")
    let handshake = try await client.handshake()
    print("Handshake response: sampleRate=\(handshake.sampleRate) channels=\(handshake.channels) state=\(handshake.state)")
    assert(handshake.sampleRate == 48000, "Sample rate should be 48000")
    assert(handshake.channels == 2, "Channels should be 2")
    assert(handshake.protocolVersion == WorkerProtocol.version)

    print("Preparing voice profile...")
    let frameCount = try await client.prepareVoice(voiceId: "voice_ava", preset: "Ava")
    print("Voice profile ready with \(frameCount) frames")
    assert(frameCount > 0)

    print("Synthesizing speech via Swift IPC into AudioRingBuffer and AudioPlayer...")
    let ringBuffer = AudioRingBuffer(capacityInFrames: 48000 * 10, channels: 2)
    let player = AudioPlayer(ringBuffer: ringBuffer)
    player.volume = 0.5 // 50% test volume
    try player.start()
    assert(player.isPlaying, "Audio player should be active")

    // Helper to synthesize and wait for playback
    func synthesizeAndWait(reqId: String, text: String) async throws -> (chunks: Int, seconds: Double) {
        let chunkCounter = Atomic<Int>(0)
        let sampleCounter = Atomic<Int>(0)
        let t0 = Date()

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            do {
                try client.synthesize(
                    requestId: reqId,
                    voiceId: "voice_ava",
                    text: text,
                    flowControl: false,
                    onChunk: { data in
                        chunkCounter.wrappingAdd(1, ordering: .relaxed)
                        sampleCounter.wrappingAdd(data.count / MemoryLayout<Float>.size, ordering: .relaxed)
                        ringBuffer.write(data: data)
                    },
                    onComplete: {
                        continuation.resume()
                    },
                    onError: { err in
                        continuation.resume(throwing: NSError(domain: "MicoraClient", code: 10, userInfo: [NSLocalizedDescriptionKey: err]))
                    }
                )
            } catch {
                continuation.resume(throwing: error)
            }
        }

        let elapsed = Date().timeIntervalSince(t0)
        let audioSec = Double(sampleCounter.load(ordering: .relaxed) / 2) / 48000.0
        print("  ↳ Generated [\(reqId)] \(audioSec)s audio in \(String(format: "%.2f", elapsed))s (chunks: \(chunkCounter.load(ordering: .relaxed)))")

        // Wait for buffer to drain to speaker
        while ringBuffer.availableSamplesToRead > 512 {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        return (chunkCounter.load(ordering: .relaxed), audioSec)
    }

    print("\n[Message 1 - Standard message]")
    let res1 = try await synthesizeAndWait(reqId: "msg_01", text: "Swift IPC üzerinden gerçek zamanlı ses sentezi testi.")
    assert(res1.chunks > 5)
    print("✓ Message 1 playback complete")

    print("\n[Message 2 - Repeated message test]")
    let res2 = try await synthesizeAndWait(reqId: "msg_02", text: "İkinci mesaj başarıyla hoparlörden çalındı.")
    assert(res2.chunks > 5)
    print("✓ Message 2 playback complete")

    print("\n[Message 3 - Long message test]")
    let longText = "Micora, Apple Silicon M2 için optimize edilmiş, düşük gecikmeli ve yüksek kaliteli yerel ses klonlama sistemidir."
    let res3 = try await synthesizeAndWait(reqId: "msg_03", text: longText)
    assert(res3.chunks > 15)
    assert(res3.seconds > 4.0)
    print("✓ Message 3 (long message) playback complete")

    print("\n[Cancellation test during physical playback]")
    let cancelReqId = "swift_cancel_01"
    let cancelChunkCount = Atomic<Int>(0)
    let hasCancelled = Atomic<Bool>(false)

    try client.synthesize(
        requestId: cancelReqId,
        voiceId: "voice_ava",
        text: "Bu uzun bir cümle olup ortasında iptal edilmesi gerekmektedir. Kesintisiz akış anında durdurulacak.",
        flowControl: false,
        onChunk: { _ in
            let c = cancelChunkCount.wrappingAdd(1, ordering: .relaxed)
            if c.newValue >= 2 && !hasCancelled.exchange(true, ordering: .relaxed) {
                print("Triggering cancellation after \(c.newValue) chunks...")
                try? client.cancel(requestId: cancelReqId)
                ringBuffer.clear() // Instant silence
            }
        },
        onComplete: {
            print("Completed callback called (unexpected if cancelled early)")
        },
        onError: { err in
            print("Error callback: \(err)")
        }
    )

    // Wait for cancellation to take effect
    try await Task.sleep(nanoseconds: 600_000_000)
    let chunksAtCancel = cancelChunkCount.load(ordering: .relaxed)
    print("Chunks received after cancel: \(chunksAtCancel)")
    try await Task.sleep(nanoseconds: 300_000_000)
    assert(cancelChunkCount.load(ordering: .relaxed) == chunksAtCancel, "No chunks should arrive after cancel")
    assert(ringBuffer.availableSamplesToRead == 0, "Ring buffer should remain empty after cancellation clear")
    print("✓ Mid-stream cancellation and instant silence verified")

    player.stop()
    assert(!player.isPlaying)

    print("\n--- Testing MessageQueue System ---")
    let queueBuffer = AudioRingBuffer(capacityInFrames: 48000 * 5, channels: 2)
    let queuePlayer = AudioPlayer(ringBuffer: queueBuffer)
    queuePlayer.volume = 0.3
    try queuePlayer.start()
    let queue = MessageQueue(client: client, ringBuffer: queueBuffer, submissionBehavior: .addToQueue)

    // Submit two messages to queue
    let idA = await queue.submit(text: "Birinci kuyruk mesajı.", voiceId: "voice_ava")
    let idB = await queue.submit(text: "İkinci kuyruk mesajı.", voiceId: "voice_ava")

    let initialQueue = await queue.items
    assert(initialQueue.count == 2, "Queue should contain 2 items")
    print("Queue initialized with 2 messages: \(idA), \(idB)")

    // Wait until queue processes both items (allowing time for real-time physical playback)
    var allDone = false
    for _ in 0..<120 {
        let currentItems = await queue.items
        let nonFinished = currentItems.filter { $0.state == .queued || $0.state == .generating || $0.state == .speaking }
        if nonFinished.isEmpty {
            allDone = true
            break
        }
        try await Task.sleep(nanoseconds: 100_000_000)
    }
    assert(allDone, "Both queued items should complete")
    let completedQueue = await queue.items
    assert(completedQueue.allSatisfy { $0.state == .completed }, "All items should be completed")
    print("✓ MessageQueue sequential execution verified")

    // Test Interrupt Mode
    print("Testing MessageQueue Interrupt Current Speech mode...")
    await queue.setSubmissionBehavior(.interruptCurrent)
    let longId = await queue.submit(text: "Bu mesaj uzun olup yarıda kesilecek ve kesinti doğrulanacak.", voiceId: "voice_ava")
    try await Task.sleep(nanoseconds: 500_000_000)
    let interId = await queue.submit(text: "Araya giren acil mesaj.", voiceId: "voice_ava")

    try await Task.sleep(nanoseconds: 600_000_000)
    let interruptItems = await queue.items
    if let first = interruptItems.first(where: { $0.id == longId }) {
        assert(first.state == .cancelled, "Previous message should be marked cancelled")
    }
    print("✓ MessageQueue interrupt behavior verified (interrupted \(longId) with \(interId))")

    // Clear queue before testing explicit Stop and subsequent submission
    await queue.clearQueue()
    try await Task.sleep(nanoseconds: 200_000_000)

    // Test Cancel then New Message
    print("Testing Cancel Active Message then Submit New Message...")
    await queue.setSubmissionBehavior(.addToQueue)
    let toCancelId = await queue.submit(text: "Bu mesaj iptal edilecek mesajdır.", voiceId: "voice_ava")
    try await Task.sleep(nanoseconds: 400_000_000)
    await queue.stopCurrent()
    let itemsAfterCancel = await queue.items
    let cancelledItem = itemsAfterCancel.first(where: { $0.id == toCancelId })
    assert(cancelledItem?.state == .cancelled, "Item should be cancelled")

    // Now submit a new message immediately after cancel
    let nextMsgId = await queue.submit(text: "İptal sonrası yeni mesaj başarıyla başladı.", voiceId: "voice_ava")
    var nextMsgDone = false
    for _ in 0..<80 {
        let items = await queue.items
        if let nextItem = items.first(where: { $0.id == nextMsgId }) {
            if nextItem.state == .completed {
                nextMsgDone = true
                break
            }
        }
        try await Task.sleep(nanoseconds: 100_000_000)
    }
    assert(nextMsgDone, "New message submitted after cancel MUST start and complete successfully")
    print("✓ Cancel then new message verified (cancelled \(toCancelId), successfully completed \(nextMsgId))")

    queuePlayer.stop()

    // Test Dynamic Backend Switching: MOSS -> Chatterbox -> MOSS
    print("\n--- Testing Dynamic Backend Switching (MOSS <-> Chatterbox) ---")
    let capMoss = try await client.getCapabilities()
    print("Initial Capabilities: backend=\(capMoss.backend ?? "nil"), isStreaming=\(capMoss.isStreaming ?? false)")
    assert(capMoss.backend == "moss", "Initial backend should be moss")
    assert(capMoss.isStreaming == true, "MOSS should report streaming")

    print("Switching backend to Chatterbox Multilingual V3...")
    let switchRes1 = try await client.switchBackend(backend: "chatterbox")
    print("Switch response: backend=\(switchRes1.backend ?? "nil"), name=\(switchRes1.name ?? "nil")")
    assert(switchRes1.backend == "chatterbox")

    let capChatter = try await client.getCapabilities()
    print("Chatterbox Capabilities: backend=\(capChatter.backend ?? "nil"), isStreaming=\(capChatter.isStreaming ?? false)")
    assert(capChatter.backend == "chatterbox")
    assert(capChatter.isStreaming == false)

    print("Preparing voice in Chatterbox...")
    let chatterFrames = try await client.prepareVoice(
        voiceId: "voice_chatter_test",
        audioPath: "models/reference_speech.wav"
    )
    print("Chatterbox voice prepared (\(chatterFrames) conditioning frames)")

    print("Synthesizing Turkish speech with Chatterbox through Swift IPC...")
    let chatterChunks = Atomic<Int>(0)
    let chatterSamples = Atomic<Int>(0)
    let tChatter = Date()

    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
        do {
            try client.synthesize(
                requestId: "chatter_01",
                voiceId: "voice_chatter_test",
                text: "Chatterbox Türkçe ses sentezi Swift istemcisinden başarıyla çalışıyor.",
                flowControl: false,
                onChunk: { data in
                    chatterChunks.wrappingAdd(1, ordering: .relaxed)
                    chatterSamples.wrappingAdd(data.count / MemoryLayout<Float>.size, ordering: .relaxed)
                },
                onComplete: {
                    continuation.resume()
                },
                onError: { err in
                    continuation.resume(throwing: NSError(domain: "MicoraClient", code: 20, userInfo: [NSLocalizedDescriptionKey: err]))
                }
            )
        } catch {
            continuation.resume(throwing: error)
        }
    }
    let chatterElapsed = Date().timeIntervalSince(tChatter)
    let chatterSec = Double(chatterSamples.load(ordering: .relaxed) / 2) / 48000.0
    print("✓ Chatterbox generated \(chatterSec)s audio in \(String(format: "%.2f", chatterElapsed))s (chunks: \(chatterChunks.load(ordering: .relaxed)))")
    assert(chatterChunks.load(ordering: .relaxed) > 0, "Chatterbox must deliver PCM chunks")
    assert(chatterSec > 1.0, "Chatterbox must generate valid length audio")

    print("Switching backend back to MOSS...")
    let switchRes2 = try await client.switchBackend(backend: "moss")
    assert(switchRes2.backend == "moss")
    print("Switch response: backend=\(switchRes2.backend ?? "nil"), name=\(switchRes2.name ?? "nil")")

    print("Re-preparing voice in MOSS to verify regression-free state...")
    let mossFrames = try await client.prepareVoice(voiceId: "voice_ava", preset: "Ava")
    assert(mossFrames > 0)

    print("Synthesizing speech with MOSS after backend return...")
    let mossChunksAfter = Atomic<Int>(0)
    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
        do {
            try client.synthesize(
                requestId: "moss_after_return",
                voiceId: "voice_ava",
                text: "MOSS motoruna geri dönüş testi başarılı.",
                flowControl: false,
                onChunk: { _ in
                    mossChunksAfter.wrappingAdd(1, ordering: .relaxed)
                },
                onComplete: {
                    continuation.resume()
                },
                onError: { err in
                    continuation.resume(throwing: NSError(domain: "MicoraClient", code: 21, userInfo: [NSLocalizedDescriptionKey: err]))
                }
            )
        } catch {
            continuation.resume(throwing: error)
        }
    }
    assert(mossChunksAfter.load(ordering: .relaxed) > 0, "MOSS must resume normal generation")
    print("✓ MOSS synthesis after returning from Chatterbox verified")

    // Test Voice Dataset Scanning via Swift IPC
    print("\n--- Testing Voice Dataset Folder Scanning via Swift IPC ---")
    let candidateFolders = [
        "datasets/tts_mazlum_kiper_tur",
        ".reference/MOSS-TTS-Nano/assets/audio",
        "benchmarks/audio_comparison/moss"
    ]
    let scanFolder = candidateFolders.first(where: { FileManager.default.fileExists(atPath: $0) })

    if let folder = scanFolder {
        let datasetScan = try await client.scanDataset(folderPath: folder, maxCandidates: 3)
        print("Dataset scan result for \(folder): total=\(datasetScan.total_files ?? 0), usable=\(datasetScan.usable_count ?? 0), candidates=\(datasetScan.candidates?.count ?? 0)")
        assert(datasetScan.total_files ?? 0 > 0, "Scan must find audio files")
        assert(datasetScan.usable_count ?? 0 > 0, "Scan must identify usable audio files")
        guard let firstCandidate = datasetScan.candidates?.first else {
            fatalError("Expected at least one candidate from dataset scan")
        }
        print("Top candidate: \(firstCandidate.filename) (duration: \(firstCandidate.duration_sec)s, speech: \(Int(firstCandidate.speech_ratio * 100))%, RMS: \(firstCandidate.rms_db) dBFS)")
        assert(firstCandidate.is_usable, "Candidate must be marked usable")
        assert(!firstCandidate.effectiveReferencePath.isEmpty)

        // Prepare voice with scanned candidate
        print("Preparing voice from scanned candidate...")
        let candidateVoiceFrames = try await client.prepareVoice(
            voiceId: "voice_scanned_candidate",
            audioPath: firstCandidate.effectiveReferencePath
        )
        print("Scanned voice ready with \(candidateVoiceFrames) frames")
        assert(candidateVoiceFrames > 0)
        print("✓ Dataset folder scanning and candidate voice preparation verified")
    } else {
        print("Skipping dataset scan integration: no sample audio fixtures directory found")
    }

    print("Shutting down worker from Swift...")
    try client.shutdown()
    client.disconnect()
    worker.stop()
    print("✓ testSwiftWorkerClientIntegration passed!")
}

func testAudioDeviceDiscovery() {
    print("\n--- Testing CoreAudio Device Discovery ---")
    guard let defaultID = AudioDeviceManager.defaultOutputDeviceID() else {
        fatalError("No default output device found")
    }
    print("Default output device ID: \(defaultID)")

    let devices = AudioDeviceManager.allDevices()
    assert(!devices.isEmpty, "Should discover at least 1 audio device")

    for dev in devices {
        print("Device [\(dev.id)]: \"\(dev.name)\" (UID: \(dev.uid)) - In: \(dev.inputChannels)ch, Out: \(dev.outputChannels)ch, Rate: \(dev.nominalSampleRate)Hz")
    }

    guard let defaultDev = AudioDeviceManager.getDeviceInfo(deviceID: defaultID) else {
        fatalError("Failed to get info for default output device")
    }
    assert(defaultDev.isOutput, "Default output device should be an output")
    assert(defaultDev.outputChannels >= 2, "Default output device should have at least 2 channels")
    print("✓ testAudioDeviceDiscovery passed (\(devices.count) devices found)")
}

func testAudioPlayerLifecycle() throws {
    print("\n--- Testing CoreAudio AudioPlayer Lifecycle ---")
    let ringBuffer = AudioRingBuffer(capacityInFrames: 48000, channels: 2)
    let player = AudioPlayer(ringBuffer: ringBuffer)

    assert(!player.isPlaying)
    try player.start()
    assert(player.isPlaying)

    // Write 0.1s of audio frames (4800 frames = 9600 Float32 samples)
    let sampleCount = 9600
    var samples = [Float](repeating: 0, count: sampleCount)
    for i in 0..<(sampleCount / 2) {
        let val = sin(Float(i) * 2.0 * .pi * 440.0 / 48000.0) * 0.05
        samples[i * 2] = val
        samples[i * 2 + 1] = val
    }
    let written = ringBuffer.write(samples)
    assert(written == sampleCount)

    // Let audio player render callback drain the buffer
    Thread.sleep(forTimeInterval: 0.15)

    player.stop()
    assert(!player.isPlaying)
    print("✓ testAudioPlayerLifecycle passed")
}

func testHfImportAndDatasetScanDecoding() {
    let decoder = JSONDecoder()

    // 1. DatasetScanResult without 'type' field (as scanner originally returned)
    let jsonWithoutType = """
    {
        "folder_path": "/path/to/dataset",
        "total_files": 10,
        "usable_count": 8,
        "rejected_count": 2,
        "candidates": [
            {
                "file_path": "/path/to/dataset/audio1.wav",
                "filename": "audio1.wav",
                "duration_sec": 7.5,
                "sample_rate": 48000,
                "channels": 1,
                "peak_db": -12.0,
                "rms_db": -22.0,
                "clipping_ratio": 0.0,
                "silence_ratio": 0.1,
                "speech_ratio": 0.9,
                "is_usable": true,
                "extracted_segment_path": "/path/to/dataset/.micora_candidates/audio1_seg.wav",
                "score": 45.0
            }
        ]
    }
    """.data(using: .utf8)!

    guard let scanWithoutType = try? decoder.decode(DatasetScanResult.self, from: jsonWithoutType) else {
        fatalError("Failed to decode DatasetScanResult without 'type' key")
    }
    assert(scanWithoutType.total_files == 10)
    assert(scanWithoutType.usable_count == 8)
    assert(scanWithoutType.candidates?.count == 1)
    assert(scanWithoutType.candidates?.first?.effectiveReferencePath == "/path/to/dataset/.micora_candidates/audio1_seg.wav")

    // 2. HfImportResult with nested scan
    let hfJson = """
    {
        "type": "hf_imported",
        "repo_id": "https://www.youtube.com/watch?v=kEM4q3TfHoY",
        "repo_type": "url",
        "voice_name": "Küfürbaz Haydo Padişahın Aşçısı",
        "folder_path": "/path/to/datasets/kufurbaz_haydo",
        "extracted_count": 1,
        "scan": {
            "type": "dataset_scanned",
            "folder_path": "/path/to/datasets/kufurbaz_haydo",
            "total_files": 1,
            "usable_count": 1,
            "rejected_count": 0,
            "candidates": [
                {
                    "file_path": "/path/to/datasets/kufurbaz_haydo/source.wav",
                    "filename": "source.wav",
                    "duration_sec": 307.71,
                    "sample_rate": 48000,
                    "channels": 2,
                    "peak_db": -3.8,
                    "rms_db": -25.3,
                    "clipping_ratio": 0.0,
                    "silence_ratio": 0.213,
                    "speech_ratio": 0.787,
                    "is_usable": true,
                    "extracted_segment_path": "/path/to/datasets/kufurbaz_haydo/.micora_candidates/source_seg_39s.wav",
                    "score": 31.45
                }
            ]
        }
    }
    """.data(using: .utf8)!

    guard let hfResult = try? decoder.decode(HfImportResult.self, from: hfJson) else {
        fatalError("Failed to decode HfImportResult with nested scan")
    }
    assert(hfResult.type == "hf_imported")
    assert(hfResult.voice_name == "Küfürbaz Haydo Padişahın Aşçısı")
    assert(hfResult.scan?.candidates?.first?.score == 31.45)
    assert(hfResult.scan?.candidates?.first?.effectiveReferencePath == "/path/to/datasets/kufurbaz_haydo/.micora_candidates/source_seg_39s.wav")

    print("✓ testHfImportAndDatasetScanDecoding passed")
}

func main() async {
    print("Running Micora Unit & Integration Tests...")
    testBasicWriteAndRead()
    testUnderflowSilencePadding()
    testWraparound()
    testClearDiscardsBufferedAudio()
    testConcurrentProducerConsumer()
    print("\nALL AUDIO RING BUFFER TESTS PASSED!")

    testHfImportAndDatasetScanDecoding()

    testAudioDeviceDiscovery()

    do {
        try testAudioPlayerLifecycle()
    } catch {
        print("Audio player test failed: \(error)")
        exit(1)
    }

    do {
        try await testSwiftWorkerClientIntegration()
    } catch {
        print("Integration test failed with error: \(error)")
        exit(1)
    }
}

await main()
