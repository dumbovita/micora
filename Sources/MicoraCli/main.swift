import Foundation
import CoreAudio
import Synchronization
import MicoraCore



func printHelp() {
    print("""
    Micora CLI — Local-First Voice Synthesis & Physical Playback

    USAGE:
        micora-cli [OPTIONS]

    OPTIONS:
        -t, --text <TEXT>       Synthesize and speak the given text immediately
        -v, --voice <PRESET>    Voice preset (default: "Ava")
        -r, --ref <PATH>        Zero-shot voice reference audio file (.wav)
        -d, --device <NAME/ID>  Target audio output device (default: system default output)
        --vol <0-100>           Volume percentage (default: 100)
        --list-devices          List available CoreAudio devices and exit
        -h, --help              Show this help information

    INTERACTIVE MODE:
        Run without -t to enter interactive REPL mode.
        Type any Turkish or English sentence and press Enter to hear it spoken.
        Type 'exit' or 'quit' to exit.
    """)
}

func main() async {
    let args = CommandLine.arguments

    if args.contains("-h") || args.contains("--help") {
        printHelp()
        return
    }

    if args.contains("--list-devices") {
        print("Available CoreAudio Devices:")
        let devices = AudioDeviceManager.allDevices()
        for dev in devices {
            let role = dev.isOutput ? "Output (\(dev.outputChannels)ch)" : "Input (\(dev.inputChannels)ch)"
            print("  [\(dev.id)] \"\(dev.name)\" - \(role), SampleRate: \(Int(dev.nominalSampleRate)) Hz")
        }
        return
    }

    var targetText: String? = nil
    var voicePreset: String = "Ava"
    var refAudioPath: String? = nil
    var deviceQuery: String? = nil
    var volumeLevel: Float = 1.0

    var i = 1
    while i < args.count {
        switch args[i] {
        case "-t", "--text":
            if i + 1 < args.count { targetText = args[i + 1]; i += 1 }
        case "-v", "--voice":
            if i + 1 < args.count { voicePreset = args[i + 1]; i += 1 }
        case "-r", "--ref":
            if i + 1 < args.count { refAudioPath = args[i + 1]; i += 1 }
        case "-d", "--device":
            if i + 1 < args.count { deviceQuery = args[i + 1]; i += 1 }
        case "--vol":
            if i + 1 < args.count, let v = Float(args[i + 1]) { volumeLevel = v / 100.0; i += 1 }
        default:
            break
        }
        i += 1
    }

    // Resolve target audio device
    var targetDeviceID: AudioDeviceID? = nil
    if let query = deviceQuery {
        if let numericID = UInt32(query) {
            targetDeviceID = numericID
        } else if let dev = AudioDeviceManager.findDevice(nameContaining: query) {
            targetDeviceID = dev.id
            print("Selected device: \"\(dev.name)\" [\(dev.id)]")
        } else {
            print("Warning: Device matching '\(query)' not found. Using system default.")
        }
    }

    let defaultDevID = targetDeviceID ?? AudioDeviceManager.defaultOutputDeviceID()
    if let devID = defaultDevID, let dev = AudioDeviceManager.getDeviceInfo(deviceID: devID) {
        print("Audio Output: \"\(dev.name)\" (48 kHz stereo)")
    }

    // Initialize Audio Ring Buffer & Player
    let ringBuffer = AudioRingBuffer(capacityInFrames: 48000 * 10, channels: 2) // 10 seconds capacity
    let player = AudioPlayer(ringBuffer: ringBuffer)
    player.volume = volumeLevel

    do {
        try player.start(targetDeviceID: defaultDevID)
    } catch {
        print("Error initializing audio output: \(error)")
        return
    }
    defer {
        player.stop()
    }

    // Start Worker Process
    let socketPath = "/tmp/micora_cli_\(ProcessInfo.processInfo.processIdentifier).sock"
    let worker = WorkerProcess(socketPath: socketPath)
    print("Launching Micora Python inference worker...")
    do {
        try worker.start(timeoutSeconds: 20.0)
    } catch {
        print("Failed to start worker: \(error)")
        return
    }
    defer {
        worker.stop()
    }

    // Connect Client & Handshake
    let client = WorkerClient()
    do {
        try client.connect(to: socketPath)
        let handshake = try await client.handshake()
        print("Connected to worker (protocol v\(handshake.protocolVersion), \(handshake.sampleRate)Hz, state: \(handshake.state))")
    } catch {
        print("Handshake failed: \(error)")
        return
    }
    defer {
        try? client.shutdown()
        client.disconnect()
    }

    // Prepare Voice Profile
    let voiceId = refAudioPath != nil ? "custom_ref" : "preset_\(voicePreset)"
    print("Preparing voice profile (\(refAudioPath ?? voicePreset))...")
    do {
        let frameCount = try await client.prepareVoice(voiceId: voiceId, audioPath: refAudioPath, preset: refAudioPath == nil ? voicePreset : nil)
        print("Voice ready (\(frameCount) conditioning frames).")
    } catch {
        print("Failed to prepare voice: \(error)")
        return
    }

    let queue = MessageQueue(client: client, ringBuffer: ringBuffer, submissionBehavior: .addToQueue)

    // Speak helper function
    let speak = { (text: String) async in
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        print("Synthesizing: \"\(trimmed)\"")
        let reqId = await queue.submit(text: trimmed, voiceId: voiceId)

        while true {
            let items = await queue.items
            if let item = items.first(where: { $0.id == reqId }) {
                if item.state == .completed {
                    print("  ↳ Playback complete.")
                    break
                }
                if item.state == .cancelled {
                    print("  ↳ Playback cancelled.")
                    break
                }
                if item.state == .failed {
                    print("Synthesis error: \(item.error ?? "unknown")")
                    break
                }
            }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    if let text = targetText {
        await speak(text)
        return
    }

    // Interactive REPL Mode
    print("\n=======================================================")
    print("Micora Interactive Voice Console")
    print("Type a phrase and hit Enter to speak. (Type 'exit' to quit)")
    print("=======================================================\n")

    while true {
        print("micora> ", terminator: "")
        fflush(stdout)
        guard let line = readLine() else { break }
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed == "exit" || trimmed == "quit" {
            break
        }
        if trimmed == "/cancel" {
            await queue.stopCurrent()
            print("Cancelled active speech.")
            continue
        }

        await speak(trimmed)
    }

    print("\nExiting Micora CLI.")
}

await main()
