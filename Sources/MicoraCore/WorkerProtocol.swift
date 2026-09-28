import Foundation

public enum WorkerPacketType: UInt8, Sendable {
    case controlJson = 0x01
    case audioPcm = 0x02
    case audioEos = 0x03
    case error = 0x04
}

public struct ControlMessage: Sendable, Codable {
    public let type: String
    public let protocolVersion: Int?
    public let sampleRate: Int?
    public let channels: Int?
    public let state: String?
    public let voiceId: String?
    public let frameCount: Int?
    public let requestId: String?
    public let error: String?
    public let backend: String?
    public let name: String?
    public let isStreaming: Bool?

    public init(
        type: String,
        protocolVersion: Int? = nil,
        sampleRate: Int? = nil,
        channels: Int? = nil,
        state: String? = nil,
        voiceId: String? = nil,
        frameCount: Int? = nil,
        requestId: String? = nil,
        error: String? = nil,
        backend: String? = nil,
        name: String? = nil,
        isStreaming: Bool? = nil
    ) {
        self.type = type
        self.protocolVersion = protocolVersion
        self.sampleRate = sampleRate
        self.channels = channels
        self.state = state
        self.voiceId = voiceId
        self.frameCount = frameCount
        self.requestId = requestId
        self.error = error
        self.backend = backend
        self.name = name
        self.isStreaming = isStreaming
    }

    enum CodingKeys: String, CodingKey {
        case type
        case protocolVersion = "protocol_version"
        case sampleRate = "sample_rate"
        case channels
        case state
        case voiceId = "voice_id"
        case frameCount = "frame_count"
        case requestId = "request_id"
        case error
        case backend
        case name
        case isStreaming = "is_streaming"
    }
}

public struct DatasetCandidate: Sendable, Codable, Identifiable {
    public var id: String { file_path }
    public let file_path: String
    public let filename: String
    public let duration_sec: Double
    public let sample_rate: Int
    public let channels: Int
    public let peak_db: Double
    public let rms_db: Double
    public let clipping_ratio: Double
    public let silence_ratio: Double
    public let speech_ratio: Double
    public let is_usable: Bool
    public let rejection_reason: String?
    public let extracted_segment_path: String?
    public let score: Double

    public var effectiveReferencePath: String {
        return extracted_segment_path ?? file_path
    }
}

public struct DatasetScanResult: Sendable, Codable {
    public let type: String?
    public let folder_path: String?
    public let total_files: Int?
    public let usable_count: Int?
    public let rejected_count: Int?
    public let candidates: [DatasetCandidate]?
    public let error: String?

    public init(
        type: String? = "dataset_scanned",
        folder_path: String? = nil,
        total_files: Int? = nil,
        usable_count: Int? = nil,
        rejected_count: Int? = nil,
        candidates: [DatasetCandidate]? = nil,
        error: String? = nil
    ) {
        self.type = type
        self.folder_path = folder_path
        self.total_files = total_files
        self.usable_count = usable_count
        self.rejected_count = rejected_count
        self.candidates = candidates
        self.error = error
    }
}

public struct HfImportResult: Sendable, Codable {
    public let type: String?
    public let repo_id: String?
    public let repo_type: String?
    public let voice_name: String?
    public let folder_path: String?
    public let extracted_count: Int?
    public let scan: DatasetScanResult?
    public let error: String?

    public init(
        type: String? = "hf_imported",
        repo_id: String? = nil,
        repo_type: String? = nil,
        voice_name: String? = nil,
        folder_path: String? = nil,
        extracted_count: Int? = nil,
        scan: DatasetScanResult? = nil,
        error: String? = nil
    ) {
        self.type = type
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.voice_name = voice_name
        self.folder_path = folder_path
        self.extracted_count = extracted_count
        self.scan = scan
        self.error = error
    }
}

public enum WorkerProtocol {
    public static let version = 1

    /// Encodes a typed packet with a 5-byte header: [1 byte type][4 bytes Big-Endian length] + payload.
    public static func encode(type: WorkerPacketType, payload: Data) -> Data {
        var data = Data(capacity: 5 + payload.count)
        data.append(type.rawValue)
        var lengthBigEndian = UInt32(payload.count).bigEndian
        withUnsafeBytes(of: &lengthBigEndian) { data.append(contentsOf: $0) }
        data.append(payload)
        return data
    }

    /// Encodes a JSON control message.
    public static func encodeControl(_ message: ControlMessage) throws -> Data {
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(message)
        return encode(type: .controlJson, payload: jsonData)
    }

    /// Encodes arbitrary Codable object into control packet.
    public static func encodeControlPayload<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(value)
        return encode(type: .controlJson, payload: jsonData)
    }

    /// Decodes an AUDIO_PCM or AUDIO_EOS payload into (requestId, pcmData).
    public static func decodePcmPayload(_ payload: Data) -> (requestId: String, pcmData: Data)? {
        guard !payload.isEmpty else { return nil }
        let idLen = Int(payload[0])
        guard payload.count >= 1 + idLen else { return nil }
        guard let reqId = String(data: payload.subdata(in: 1..<(1 + idLen)), encoding: .utf8) else { return nil }
        let pcm = payload.subdata(in: (1 + idLen)..<payload.count)
        return (reqId, pcm)
    }
}
