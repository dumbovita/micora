import Foundation
import Combine
import SwiftUI
import CoreAudio
import MicoraCore

public enum SynthesisMode: String, CaseIterable, Identifiable, Sendable {
    case efficient = "Efficient"
    case natural = "Natural"

    public var id: String { rawValue }

    public var backendName: String {
        switch self {
        case .efficient: return "moss"
        case .natural: return "chatterbox"
        }
    }

    public var subtitle: String {
        switch self {
        case .efficient: return "MOSS-TTS-Nano"
        case .natural: return "Chatterbox V3"
        }
    }
}

@MainActor
public final class AppState: ObservableObject {
    @Published public var inputText: String = ""
    @Published public var selectedProfileId: String = VoiceProfile.defaultProfile.id
    @Published public var synthesisMode: SynthesisMode = .efficient
    @Published public var isSwitchingMode: Bool = false
    @Published public var submissionBehavior: SubmissionBehavior = .addToQueue
    @Published public var monitorVolume: Double = 0.20
    @Published public var isVirtualMicAvailable: Bool = false
    @Published public var primaryDeviceName: String = "Detecting..."
    @Published public var workerState: String = "Starting worker..."
    @Published public var isWorkerReady: Bool = false
    @Published public var queueItems: [QueueItem] = []
    @Published public var lastError: String? = nil

    @Published public var showingVirtualMicSetup: Bool = false
    @Published public var availableOutputDevices: [AudioDeviceInfo] = []
    @Published public var selectedOutputDeviceId: AudioDeviceID = 0
    @Published public var copiedCommand: Bool = false
    @Published public var showingFileImporter: Bool = false
    @Published public var showingFolderImporter: Bool = false
    @Published public var showingHfImporter: Bool = false
    @Published public var showingNewVoiceDialog: Bool = false
    @Published public var showingDatasetCandidateDialog: Bool = false
    @Published public var pendingVoiceURL: URL? = nil
    @Published public var newVoiceName: String = ""
    @Published public var hfRepoInput: String = ""
    @Published public var isScanningDataset: Bool = false
    @Published public var isImportingHf: Bool = false
    @Published public var datasetScanResult: DatasetScanResult? = nil
    @Published public var selectedCandidatePath: String = ""

    public var profiles: [VoiceProfile] {
        profileStore.profiles
    }

    public var isSpeakingOrGenerating: Bool {
        return queueItems.contains { $0.state == .generating || $0.state == .speaking || $0.state == .queued }
    }

    private let profileStore: VoiceProfileStore
    private var worker: WorkerProcess?
    private var client: WorkerClient?
    private var router: AudioOutputRouter?
    private var queue: MessageQueue?
    private var activeProfileLoadTask: Task<Void, Never>? = nil

    public init() {
        self.profileStore = VoiceProfileStore()
    }

    public func initialize() async {
        workerState = "Starting inference worker..."
        let socketPath = "/tmp/micora_app_\(ProcessInfo.processInfo.processIdentifier).sock"
        let w = WorkerProcess(socketPath: socketPath)
        w.onUnexpectedExit = { [weak self] status in
            Task { @MainActor in
                self?.isWorkerReady = false
                self?.workerState = "Worker crashed (exit status \(status))"
                self?.lastError = "Inference worker process exited unexpectedly with code \(status)."
            }
        }
        self.worker = w

        do {
            try w.start(timeoutSeconds: 20.0)
            workerState = "Connecting to worker..."

            let c = WorkerClient()
            try c.connect(to: socketPath)
            _ = try await c.handshake()
            self.client = c

            // Initialize Audio Output Router
            let r = AudioOutputRouter()
            try r.start()
            self.router = r
            self.isVirtualMicAvailable = r.isBlackHoleAvailable
            self.primaryDeviceName = r.primaryDeviceName
            self.availableOutputDevices = r.availableOutputDevices
            r.monitorPlayer.volume = Float(monitorVolume)

            // Initialize Message Queue
            let q = MessageQueue(
                client: c,
                ringBuffer: r.primaryRingBuffer,
                secondaryRingBuffer: r.monitorRingBuffer,
                isSecondaryActive: { [weak r] in r?.isMonitoringActive ?? false },
                submissionBehavior: submissionBehavior
            )
            self.queue = q

            // Pre-load active voice profile
            await loadActiveVoiceProfile()

            // Observe queue changes
            await q.setOnQueueChanged { [weak self] items in
                Task { @MainActor in
                    self?.queueItems = items
                }
            }

            workerState = "Ready"
            isWorkerReady = true
        } catch {
            lastError = error.localizedDescription
            workerState = "Worker error: \(error.localizedDescription)"
        }
    }

    public func shutdown() {
        Task {
            await queue?.clearQueue()
        }
        router?.stop()
        try? client?.shutdown()
        client?.disconnect()
        worker?.stop()
    }

    public func submit() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, let q = queue, isWorkerReady else { return }

        inputText = ""
        Task {
            await q.submit(text: text, voiceId: selectedProfileId)
        }
    }

    public func stop() {
        Task {
            await queue?.stopCurrent()
        }
        router?.clearBuffers()
    }

    public func clearQueue() {
        Task {
            await queue?.clearQueue()
        }
        router?.clearBuffers()
    }

    public func setSynthesisMode(_ mode: SynthesisMode) async {
        guard mode != synthesisMode, !isSwitchingMode, let c = client else { return }
        isSwitchingMode = true
        activeProfileLoadTask?.cancel()
        activeProfileLoadTask = nil
        let oldMode = synthesisMode
        workerState = "Loading \(mode.rawValue) engine..."
        do {
            _ = try await c.switchBackend(backend: mode.backendName)
            self.synthesisMode = mode
            await loadActiveVoiceProfile()
            workerState = "Ready"
        } catch {
            lastError = "Failed to switch engine: \(error.localizedDescription)"
            workerState = "Ready"
            self.synthesisMode = oldMode
        }
        isSwitchingMode = false
    }

    public func setMonitorVolume(_ vol: Double) {
        self.monitorVolume = vol
        router?.monitorPlayer.volume = Float(vol)
    }

    public func refreshAudioDevices() {
        guard let r = router else { return }
        do {
            try r.refreshDevices(preferredDeviceID: selectedOutputDeviceId)
            self.isVirtualMicAvailable = r.isBlackHoleAvailable
            self.primaryDeviceName = r.primaryDeviceName
            self.availableOutputDevices = r.availableOutputDevices
        } catch {
            self.lastError = "Failed to refresh audio devices: \(error.localizedDescription)"
        }
    }

    public func selectOutputDevice(id: AudioDeviceID) {
        self.selectedOutputDeviceId = id
        refreshAudioDevices()
    }

    public func setSubmissionBehavior(_ behavior: SubmissionBehavior) {
        self.submissionBehavior = behavior
        Task {
            await queue?.setSubmissionBehavior(behavior)
        }
    }

    public func selectVoiceProfile(id: String) {
        guard id != selectedProfileId || !isWorkerReady else { return }
        self.selectedProfileId = id
        activeProfileLoadTask?.cancel()
        activeProfileLoadTask = Task {
            await loadActiveVoiceProfile()
        }
    }

    public func importVoiceProfile(audioURL: URL, name: String) async {
        let profile = VoiceProfile(
            name: name,
            referenceAudioPath: audioURL.path,
            language: "tr"
        )
        profileStore.addProfile(profile)
        selectVoiceProfile(id: profile.id)
    }

    public func deleteVoiceProfile(id: String) {
        profileStore.removeProfile(id: id)
        if selectedProfileId == id {
            selectVoiceProfile(id: VoiceProfile.defaultProfile.id)
        }
    }

    public func importVoiceFolder(folderURL: URL) async {
        guard let c = client else { return }
        isScanningDataset = true
        defer { isScanningDataset = false }
        workerState = "Scanning voice dataset folder..."
        do {
            let res = try await c.scanDataset(folderPath: folderURL.path, maxCandidates: 6)
            self.datasetScanResult = res
            self.newVoiceName = folderURL.lastPathComponent
            if let first = res.candidates?.first {
                self.selectedCandidatePath = first.effectiveReferencePath
            }
            self.showingDatasetCandidateDialog = true
            workerState = "Ready"
        } catch {
            lastError = "Folder scan failed: \(error.localizedDescription)"
            workerState = "Ready"
        }
    }

    public func importHuggingFaceVoice(repoIdOrUrl: String) async {
        guard let c = client else { return }
        isImportingHf = true
        defer { isImportingHf = false }
        workerState = "Downloading & analyzing voice..."
        do {
            let res = try await c.importHuggingFaceVoice(repoId: repoIdOrUrl, maxSamples: 30, maxCandidates: 6)
            self.datasetScanResult = res.scan
            self.newVoiceName = res.voice_name ?? "Imported Voice"
            if let first = res.scan?.candidates?.first {
                self.selectedCandidatePath = first.effectiveReferencePath
            }
            self.showingHfImporter = false
            self.showingDatasetCandidateDialog = true
            workerState = "Ready"
        } catch {
            lastError = "Voice import failed: \(error.localizedDescription)"
            workerState = "Ready"
        }
    }

    public func confirmDatasetProfile(name: String, candidatePath: String) async {
        let profile = VoiceProfile(
            name: name,
            referenceAudioPath: candidatePath,
            language: "tr",
            sourcePaths: [datasetScanResult?.folder_path ?? ""]
        )
        profileStore.addProfile(profile)
        selectVoiceProfile(id: profile.id)
        showingDatasetCandidateDialog = false
        datasetScanResult = nil
    }

    private func loadActiveVoiceProfile() async {
        guard let c = client else { return }
        guard let profile = profiles.first(where: { $0.id == selectedProfileId }) else { return }

        let targetAudio = profile.referenceAudio(for: synthesisMode.backendName)
        workerState = "Preparing voice (\(profile.name))..."
        do {
            _ = try await c.prepareVoice(
                voiceId: profile.id,
                audioPath: targetAudio,
                preset: profile.presetName
            )
            workerState = "Ready"
        } catch {
            lastError = "Failed to load voice: \(error.localizedDescription)"
            workerState = "Ready"
        }
    }
}
