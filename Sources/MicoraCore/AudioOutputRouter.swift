import Foundation
import CoreAudio
import Synchronization

/// Routes synthesized audio to virtual microphone (BlackHole) and local monitor (Speakers).
public final class AudioOutputRouter: @unchecked Sendable {
    public let primaryRingBuffer: AudioRingBuffer
    public let monitorRingBuffer: AudioRingBuffer

    public let primaryPlayer: AudioPlayer
    public let monitorPlayer: AudioPlayer

    private let lock = NSLock()
    private var isStarted = false

    public init(bufferCapacityInFrames: Int = 48000 * 5) {
        self.primaryRingBuffer = AudioRingBuffer(capacityInFrames: bufferCapacityInFrames, channels: 2)
        self.monitorRingBuffer = AudioRingBuffer(capacityInFrames: bufferCapacityInFrames, channels: 2)

        self.primaryPlayer = AudioPlayer(ringBuffer: self.primaryRingBuffer)
        self.monitorPlayer = AudioPlayer(ringBuffer: self.monitorRingBuffer)

        self.primaryPlayer.volume = 1.0
        self.monitorPlayer.volume = 0.2 // Default 20% local monitoring volume
    }

    /// Whether local audio monitoring is currently active.
    public var isMonitoringActive: Bool {
        return monitorPlayer.isPlaying
    }

    deinit {
        stop()
    }

    private var customDeviceID: AudioDeviceID? = nil

    /// All available output devices.
    public var availableOutputDevices: [AudioDeviceInfo] {
        return AudioDeviceManager.allDevices().filter { $0.isOutput }
    }

    /// Whether BlackHole virtual microphone is detected on the system.
    public var isBlackHoleAvailable: Bool {
        return blackHoleDevice != nil
    }

    /// The detected BlackHole audio device, if any.
    public var blackHoleDevice: AudioDeviceInfo? {
        return AudioDeviceManager.findDevice(nameContaining: "BlackHole")
    }

    /// The default system output device (speakers or headphones).
    public var defaultOutputDevice: AudioDeviceInfo? {
        guard let id = AudioDeviceManager.defaultOutputDeviceID() else { return nil }
        return AudioDeviceManager.getDeviceInfo(deviceID: id)
    }

    /// Current primary device name.
    public var primaryDeviceName: String {
        if let customId = customDeviceID, let dev = AudioDeviceManager.getDeviceInfo(deviceID: customId) {
            let role = dev.name.contains("BlackHole") ? "Virtual Mic" : "Output"
            return "\(dev.name) (\(role))"
        }
        if let dev = blackHoleDevice {
            return "\(dev.name) (Virtual Mic)"
        }
        if let dev = defaultOutputDevice {
            return "\(dev.name) (Physical Output)"
        }
        return "Default Audio Output"
    }

    /// Starts audio routing.
    public func start() throws {
        try refreshDevices()
    }

    /// Re-evaluates and switches output devices dynamically.
    public func refreshDevices(preferredDeviceID: AudioDeviceID? = nil) throws {
        lock.lock()
        defer { lock.unlock() }

        if let preferred = preferredDeviceID, preferred != 0 {
            customDeviceID = preferred
        } else if preferredDeviceID == 0 {
            customDeviceID = nil
        }

        let targetID: AudioDeviceID
        let isVirtual: Bool

        if let customId = customDeviceID, customId != 0 {
            targetID = customId
            let name = AudioDeviceManager.getDeviceInfo(deviceID: customId)?.name ?? ""
            isVirtual = name.contains("BlackHole")
        } else if let bh = blackHoleDevice {
            targetID = bh.id
            isVirtual = true
        } else {
            targetID = AudioDeviceManager.defaultOutputDeviceID() ?? 0
            isVirtual = false
        }

        if isStarted {
            try primaryPlayer.switchDevice(to: targetID)
        } else {
            try primaryPlayer.start(targetDeviceID: targetID)
            isStarted = true
        }

        if isVirtual, let defOut = AudioDeviceManager.defaultOutputDeviceID() {
            if monitorPlayer.isPlaying {
                try monitorPlayer.switchDevice(to: defOut)
            } else {
                monitorRingBuffer.clear()
                try monitorPlayer.start(targetDeviceID: defOut)
            }
        } else {
            monitorPlayer.stop()
            monitorRingBuffer.clear()
        }
    }

    /// Stops audio routing.
    public func stop() {
        lock.lock()
        defer { lock.unlock() }

        primaryPlayer.stop()
        monitorPlayer.stop()
        primaryRingBuffer.clear()
        monitorRingBuffer.clear()
        isStarted = false
    }

    /// Clear all pending audio buffers immediately.
    public func clearBuffers() {
        primaryRingBuffer.clear()
        monitorRingBuffer.clear()
    }
}
