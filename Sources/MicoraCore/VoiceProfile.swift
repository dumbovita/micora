import Foundation

public struct VoiceProfile: Identifiable, Sendable, Codable, Equatable {
    public let id: String
    public var name: String
    public var referenceAudioPath: String?
    public var presetName: String?
    public var language: String
    public var selectedReferences: [String: String]?
    public var sourcePaths: [String]?
    public let createdAt: Date

    public init(
        id: String = UUID().uuidString,
        name: String,
        referenceAudioPath: String? = nil,
        presetName: String? = nil,
        language: String = "tr",
        selectedReferences: [String: String]? = nil,
        sourcePaths: [String]? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.referenceAudioPath = referenceAudioPath
        self.presetName = presetName
        self.language = language
        self.selectedReferences = selectedReferences
        self.sourcePaths = sourcePaths
        self.createdAt = createdAt
    }

    public func referenceAudio(for backend: String) -> String? {
        if let refs = selectedReferences, let path = refs[backend], !path.isEmpty {
            return path
        }
        return referenceAudioPath
    }

    public var isBuiltin: Bool {
        return presetName != nil && referenceAudioPath == nil && (selectedReferences == nil || selectedReferences!.isEmpty)
    }

    public static let defaultProfile = VoiceProfile(
        id: "builtin_ava",
        name: "Ava (Built-in Voice)",
        presetName: "Ava",
        language: "tr"
    )
}

public final class VoiceProfileStore: @unchecked Sendable {
    private let storageURL: URL
    private let lock = NSLock()
    private var profilesList: [VoiceProfile] = []

    public init(storageDirectory: URL? = nil) {
        let baseDir: URL
        if let dir = storageDirectory {
            baseDir = dir
        } else {
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            baseDir = appSupport.appendingPathComponent("Micora", isDirectory: true)
        }

        try? FileManager.default.createDirectory(at: baseDir, withIntermediateDirectories: true)
        self.storageURL = baseDir.appendingPathComponent("profiles.json")

        load()
    }

    public var profiles: [VoiceProfile] {
        lock.lock()
        defer { lock.unlock() }
        return profilesList
    }

    public func addProfile(_ profile: VoiceProfile) {
        lock.lock()
        profilesList.removeAll { $0.id == profile.id }
        profilesList.append(profile)
        save()
        lock.unlock()
    }

    public func removeProfile(id: String) {
        lock.lock()
        // Do not allow deleting the default builtin profile
        if id != VoiceProfile.defaultProfile.id {
            profilesList.removeAll { $0.id == id }
            save()
        }
        lock.unlock()
    }

    private func load() {
        if FileManager.default.fileExists(atPath: storageURL.path) {
            do {
                let data = try Data(contentsOf: storageURL)
                let loaded = try JSONDecoder().decode([VoiceProfile].self, from: data)
                if !loaded.isEmpty {
                    self.profilesList = loaded
                    // Ensure default profile exists
                    if !profilesList.contains(where: { $0.id == VoiceProfile.defaultProfile.id }) {
                        profilesList.insert(VoiceProfile.defaultProfile, at: 0)
                    }
                    return
                }
            } catch {
                let timestamp = Int(Date().timeIntervalSince1970)
                let backupURL = storageURL.deletingLastPathComponent().appendingPathComponent("profiles.corrupt.\(timestamp).bak")
                try? FileManager.default.copyItem(at: storageURL, to: backupURL)
                print("[VoiceProfileStore] Warning: failed to decode \(storageURL.path): \(error). Backed up corrupted file to \(backupURL.path). Using default profile.")
                self.profilesList = [VoiceProfile.defaultProfile]
                return
            }
        }
        self.profilesList = [VoiceProfile.defaultProfile]
        save()
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(profilesList) else { return }
        try? data.write(to: storageURL, options: .atomic)
    }
}
