import Foundation
import CoreAudio
import AudioToolbox
import Synchronization

/// Low-latency real-time audio player using CoreAudio AudioUnit (HALOutput).
/// Streams 48 kHz stereo Float32 audio directly from an AudioRingBuffer to an audio device.
public final class AudioPlayer: @unchecked Sendable {
    public let ringBuffer: AudioRingBuffer
    private var audioUnit: AudioUnit?
    private let lock = NSLock()
    private let running = Atomic<Bool>(false)

    // Volume stored as percentage 0..100 for lock-free atomic access in render callback
    private let volumePercent = Atomic<Int>(100)

    // Current device ID
    private var currentDeviceID: AudioDeviceID = 0

    public init(ringBuffer: AudioRingBuffer) {
        self.ringBuffer = ringBuffer
    }

    deinit {
        stop()
    }

    /// Current volume level between 0.0 and 1.0.
    public var volume: Float {
        get { Float(volumePercent.load(ordering: .relaxed)) / 100.0 }
        set {
            let clamped = max(0.0, min(1.0, newValue))
            volumePercent.store(Int(clamped * 100.0), ordering: .relaxed)
        }
    }

    /// Whether playback is currently active.
    public var isPlaying: Bool {
        return running.load(ordering: .relaxed)
    }

    /// The AudioDeviceID currently in use.
    public var deviceID: AudioDeviceID {
        lock.lock()
        defer { lock.unlock() }
        return currentDeviceID
    }

    /// Starts audio playback using the specified device (or system default if nil/0).
    public func start(targetDeviceID: AudioDeviceID? = nil) throws {
        lock.lock()
        defer { lock.unlock() }

        if running.load(ordering: .relaxed) { return }

        let devID: AudioDeviceID
        if let target = targetDeviceID, target != 0 {
            devID = target
        } else if let defaultID = AudioDeviceManager.defaultOutputDeviceID() {
            devID = defaultID
        } else {
            throw NSError(
                domain: "MicoraAudioPlayer",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No default audio output device available"]
            )
        }

        try setupAudioUnit(deviceID: devID)

        guard let unit = audioUnit else {
            throw NSError(
                domain: "MicoraAudioPlayer",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "AudioUnit failed to instantiate"]
            )
        }

        let initStatus = AudioUnitInitialize(unit)
        guard initStatus == noErr else {
            tearDownAudioUnit()
            throw NSError(
                domain: NSOSStatusErrorDomain,
                code: Int(initStatus),
                userInfo: [NSLocalizedDescriptionKey: "AudioUnitInitialize failed (\(initStatus))"]
            )
        }

        let startStatus = AudioOutputUnitStart(unit)
        guard startStatus == noErr else {
            tearDownAudioUnit()
            throw NSError(
                domain: NSOSStatusErrorDomain,
                code: Int(startStatus),
                userInfo: [NSLocalizedDescriptionKey: "AudioOutputUnitStart failed (\(startStatus))"]
            )
        }

        currentDeviceID = devID
        running.store(true, ordering: .relaxed)
    }

    /// Stops audio playback and uninitializes the AudioUnit.
    public func stop() {
        lock.lock()
        defer { lock.unlock() }

        guard running.load(ordering: .relaxed), let unit = audioUnit else { return }

        AudioOutputUnitStop(unit)
        AudioUnitUninitialize(unit)
        tearDownAudioUnit()
        running.store(false, ordering: .relaxed)
    }

    /// Switches playback to another AudioDeviceID dynamically.
    public func switchDevice(to newDeviceID: AudioDeviceID) throws {
        lock.lock()
        defer { lock.unlock() }

        guard running.load(ordering: .relaxed) else {
            currentDeviceID = newDeviceID
            return
        }

        if currentDeviceID == newDeviceID {
            return
        }

        guard let unit = audioUnit else { return }

        // Stop unit before switching device
        AudioOutputUnitStop(unit)
        AudioUnitUninitialize(unit)
        tearDownAudioUnit()

        try setupAudioUnit(deviceID: newDeviceID)
        guard let newUnit = audioUnit else { return }

        let initStatus = AudioUnitInitialize(newUnit)
        guard initStatus == noErr else {
            tearDownAudioUnit()
            throw NSError(domain: NSOSStatusErrorDomain, code: Int(initStatus), userInfo: nil)
        }

        let startStatus = AudioOutputUnitStart(newUnit)
        guard startStatus == noErr else {
            tearDownAudioUnit()
            throw NSError(domain: NSOSStatusErrorDomain, code: Int(startStatus), userInfo: nil)
        }

        currentDeviceID = newDeviceID
    }

    // MARK: - Private Setup

    private func setupAudioUnit(deviceID: AudioDeviceID) throws {
        var desc = AudioComponentDescription(
            componentType: kAudioUnitType_Output,
            componentSubType: kAudioUnitSubType_HALOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        guard let comp = AudioComponentFindNext(nil, &desc) else {
            throw NSError(
                domain: "MicoraAudioPlayer",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "HALOutput AudioComponent not found"]
            )
        }

        var unit: AudioUnit?
        let instStatus = AudioComponentInstanceNew(comp, &unit)
        guard instStatus == noErr, let audioUnit = unit else {
            throw NSError(
                domain: NSOSStatusErrorDomain,
                code: Int(instStatus),
                userInfo: [NSLocalizedDescriptionKey: "AudioComponentInstanceNew failed (\(instStatus))"]
            )
        }

        // Enable output on bus 0
        var enableOutput: UInt32 = 1
        AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Output,
            0,
            &enableOutput,
            UInt32(MemoryLayout<UInt32>.size)
        )

        // Disable input on bus 1
        var disableInput: UInt32 = 0
        AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Input,
            1,
            &disableInput,
            UInt32(MemoryLayout<UInt32>.size)
        )

        // Set device
        var devID = deviceID
        let devStatus = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &devID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        guard devStatus == noErr else {
            AudioComponentInstanceDispose(audioUnit)
            throw NSError(
                domain: NSOSStatusErrorDomain,
                code: Int(devStatus),
                userInfo: [NSLocalizedDescriptionKey: "Failed to set audio device (\(devStatus))"]
            )
        }

        // Set stream format: 48 kHz stereo Float32 interleaved
        var streamDesc = AudioStreamBasicDescription(
            mSampleRate: 48000.0,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 8,   // 2 channels * 4 bytes
            mFramesPerPacket: 1,
            mBytesPerFrame: 8,    // 2 channels * 4 bytes
            mChannelsPerFrame: 2, // Stereo
            mBitsPerChannel: 32,  // 32-bit Float
            mReserved: 0
        )

        let fmtStatus = AudioUnitSetProperty(
            audioUnit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Input,
            0,
            &streamDesc,
            UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        )
        guard fmtStatus == noErr else {
            AudioComponentInstanceDispose(audioUnit)
            throw NSError(
                domain: NSOSStatusErrorDomain,
                code: Int(fmtStatus),
                userInfo: [NSLocalizedDescriptionKey: "Failed to set stream format (\(fmtStatus))"]
            )
        }

        // Set render callback on bus 0 input
        var callbackStruct = AURenderCallbackStruct(
            inputProc: { (inRefCon, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, ioData) -> OSStatus in
                guard let ioData = ioData else { return noErr }
                let player = Unmanaged<AudioPlayer>.fromOpaque(inRefCon).takeUnretainedValue()
                return player.render(inNumberFrames: inNumberFrames, ioData: ioData)
            },
            inputProcRefCon: Unmanaged.passUnretained(self).toOpaque()
        )

        let cbStatus = AudioUnitSetProperty(
            audioUnit,
            kAudioUnitProperty_SetRenderCallback,
            kAudioUnitScope_Input,
            0,
            &callbackStruct,
            UInt32(MemoryLayout<AURenderCallbackStruct>.size)
        )
        guard cbStatus == noErr else {
            AudioComponentInstanceDispose(audioUnit)
            throw NSError(
                domain: NSOSStatusErrorDomain,
                code: Int(cbStatus),
                userInfo: [NSLocalizedDescriptionKey: "Failed to set render callback (\(cbStatus))"]
            )
        }

        self.audioUnit = audioUnit
    }

    private func tearDownAudioUnit() {
        if let unit = audioUnit {
            AudioComponentInstanceDispose(unit)
            audioUnit = nil
        }
    }

    // MARK: - Real-Time Audio Callback

    /// High-priority real-time audio callback called by CoreAudio HAL.
    /// Strictly lock-free, allocation-free, and I/O-free.
    private func render(inNumberFrames: UInt32, ioData: UnsafeMutablePointer<AudioBufferList>) -> OSStatus {
        let bufferList = UnsafeMutableAudioBufferListPointer(ioData)
        guard bufferList.count > 0, let mData = bufferList[0].mData else {
            return noErr
        }

        let floatPtr = mData.assumingMemoryBound(to: Float.self)
        let samplesNeeded = Int(inNumberFrames * 2)

        // Read from ring buffer. Deficit is automatically padded with zero (silence).
        _ = ringBuffer.read(into: floatPtr, count: samplesNeeded)

        // Apply volume attenuation if less than 100%
        let vol = Float(volumePercent.load(ordering: .relaxed)) / 100.0
        if vol < 0.999 {
            for i in 0..<samplesNeeded {
                floatPtr[i] *= vol
            }
        }

        bufferList[0].mDataByteSize = UInt32(samplesNeeded * MemoryLayout<Float>.size)
        return noErr
    }
}
