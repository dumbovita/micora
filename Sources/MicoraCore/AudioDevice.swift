import Foundation
import CoreAudio
import AudioToolbox

/// Represents an audio device recognized by CoreAudio.
public struct AudioDeviceInfo: Identifiable, Sendable, Equatable {
    public let id: AudioDeviceID
    public let uid: String
    public let name: String
    public let outputChannels: Int
    public let inputChannels: Int
    public let nominalSampleRate: Double

    public var isOutput: Bool { outputChannels > 0 }
    public var isInput: Bool { inputChannels > 0 }
}

public enum AudioDeviceManager {
    /// Returns the default system output AudioDeviceID.
    public static func defaultOutputDeviceID() -> AudioDeviceID? {
        var deviceID = AudioDeviceID(0)
        var propSize = UInt32(MemoryLayout<AudioDeviceID>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &propSize,
            &deviceID
        )

        guard status == noErr, deviceID != 0 else {
            return nil
        }
        return deviceID
    }

    /// Enumerate all audio devices on the system.
    public static func allDevices() -> [AudioDeviceInfo] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var dataSize: UInt32 = 0
        var status = AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &dataSize
        )
        guard status == noErr, dataSize > 0 else { return [] }

        let deviceCount = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
        var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)
        status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &dataSize,
            &deviceIDs
        )
        guard status == noErr else { return [] }

        var result: [AudioDeviceInfo] = []
        for devId in deviceIDs {
            if let info = getDeviceInfo(deviceID: devId) {
                result.append(info)
            }
        }
        return result
    }

    /// Finds a device matching a case-insensitive substring in its name.
    public static func findDevice(nameContaining query: String) -> AudioDeviceInfo? {
        let q = query.lowercased()
        return allDevices().first { $0.name.lowercased().contains(q) }
    }

    /// Finds a device matching UID exactly.
    public static func findDevice(uid: String) -> AudioDeviceInfo? {
        return allDevices().first { $0.uid == uid }
    }

    /// Retrieves detailed info for an AudioDeviceID.
    public static func getDeviceInfo(deviceID: AudioDeviceID) -> AudioDeviceInfo? {
        // Name
        var nameAddress = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var nameCF: CFString?
        var propSize = UInt32(MemoryLayout<CFString?>.size)
        var status = withUnsafeMutablePointer(to: &nameCF) { ptr in
            AudioObjectGetPropertyData(
                deviceID,
                &nameAddress,
                0,
                nil,
                &propSize,
                ptr
            )
        }
        let name = (status == noErr && nameCF != nil) ? (nameCF! as String) : "Unknown Device"

        // UID
        var uidAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var uidCF: CFString?
        propSize = UInt32(MemoryLayout<CFString?>.size)
        status = withUnsafeMutablePointer(to: &uidCF) { ptr in
            AudioObjectGetPropertyData(
                deviceID,
                &uidAddress,
                0,
                nil,
                &propSize,
                ptr
            )
        }
        let uid = (status == noErr && uidCF != nil) ? (uidCF! as String) : "\(deviceID)"

        // Output channel count
        let outputChannels = getChannelCount(deviceID: deviceID, scope: kAudioObjectPropertyScopeOutput)

        // Input channel count
        let inputChannels = getChannelCount(deviceID: deviceID, scope: kAudioObjectPropertyScopeInput)

        // Nominal Sample Rate
        var rateAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyNominalSampleRate,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var sampleRate: Float64 = 48000.0
        propSize = UInt32(MemoryLayout<Float64>.size)
        _ = AudioObjectGetPropertyData(
            deviceID,
            &rateAddress,
            0,
            nil,
            &propSize,
            &sampleRate
        )

        return AudioDeviceInfo(
            id: deviceID,
            uid: uid,
            name: name,
            outputChannels: outputChannels,
            inputChannels: inputChannels,
            nominalSampleRate: sampleRate
        )
    }

    private static func getChannelCount(deviceID: AudioDeviceID, scope: AudioObjectPropertyScope) -> Int {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: scope,
            mElement: kAudioObjectPropertyElementMain
        )
        var propSize: UInt32 = 0
        var status = AudioObjectGetPropertyDataSize(deviceID, &address, 0, nil, &propSize)
        guard status == noErr, propSize > 0 else { return 0 }

        let bufferListRaw = UnsafeMutableRawPointer.allocate(byteCount: Int(propSize), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { bufferListRaw.deallocate() }

        status = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &propSize, bufferListRaw)
        guard status == noErr else { return 0 }

        let bufferList = bufferListRaw.bindMemory(to: AudioBufferList.self, capacity: 1)
        let buffers = UnsafeMutableAudioBufferListPointer(bufferList)
        var totalChannels = 0
        for buf in buffers {
            totalChannels += Int(buf.mNumberChannels)
        }
        return totalChannels
    }
}
