import SwiftUI
import UniformTypeIdentifiers
import CoreAudio
import AppKit
import MicoraCore

struct ContentView: View {
    @ObservedObject var appState: AppState
    @FocusState private var isFieldFocused: Bool

    var body: some View {
        VStack(spacing: 16) {
            // Header Bar
            headerBar

            if !appState.isVirtualMicAvailable {
                virtualMicBanner
            }

            Divider()

            // Main Message Input Card
            inputCard

            // Secondary Controls Bar (Voice, Queue Mode, Monitor)
            controlsBar

            Divider()

            // Real-Time Message Queue
            queueSection
        }
        .padding(18)
        .frame(minWidth: 720, minHeight: 520)
        .onAppear {
            NSApp.activate(ignoringOtherApps: true)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                isFieldFocused = true
            }
        }
        .task {
            await appState.initialize()
        }
        .fileImporter(
            isPresented: $appState.showingFileImporter,
            allowedContentTypes: [.wav, .audio],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                if url.startAccessingSecurityScopedResource() {
                    appState.pendingVoiceURL = url
                    appState.newVoiceName = url.deletingPathExtension().lastPathComponent
                    appState.showingNewVoiceDialog = true
                }
            case .failure(let error):
                appState.lastError = "Failed to select file: \(error.localizedDescription)"
            }
        }
        .sheet(isPresented: $appState.showingNewVoiceDialog) {
            newVoiceSheet
        }
        .fileImporter(
            isPresented: $appState.showingFolderImporter,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                if url.startAccessingSecurityScopedResource() {
                    Task {
                        await appState.importVoiceFolder(folderURL: url)
                    }
                }
            case .failure(let error):
                appState.lastError = "Failed to select folder: \(error.localizedDescription)"
            }
        }
        .sheet(isPresented: $appState.showingDatasetCandidateDialog) {
            datasetCandidatesSheet
        }
        .sheet(isPresented: $appState.showingHfImporter) {
            hfImportSheet
        }
        .sheet(isPresented: $appState.showingVirtualMicSetup) {
            virtualMicSetupSheet
        }
    }

    // MARK: - Header Bar

    private var headerBar: some View {
        HStack(alignment: .center) {
            HStack(spacing: 8) {
                Image(systemName: "waveform.circle.fill")
                    .font(.system(size: 28))
                    .foregroundStyle(.blue)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Micora")
                        .font(.title3)
                        .fontWeight(.bold)
                    Text("Local-First Virtual Microphone TTS")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            // Output Target Badge
            Button(action: {
                appState.showingVirtualMicSetup = true
            }) {
                HStack(spacing: 6) {
                    Image(systemName: appState.isVirtualMicAvailable ? "mic.fill" : "speaker.wave.2.fill")
                        .font(.caption)
                        .foregroundStyle(appState.isVirtualMicAvailable ? .green : .orange)
                    Text(appState.primaryDeviceName)
                        .font(.caption)
                        .fontWeight(.medium)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(appState.isVirtualMicAvailable ? Color(nsColor: .separatorColor) : Color.orange.opacity(0.6), lineWidth: 0.5)
                )
            }
            .buttonStyle(.plain)
            .help("Click to configure audio output or view virtual microphone setup")

            // Worker State Indicator
            HStack(spacing: 6) {
                Circle()
                    .fill(appState.isWorkerReady ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)
                Text(appState.workerState)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color(nsColor: .separatorColor), lineWidth: 0.5)
            )
        }
    }

    // MARK: - Input Card

    private var inputCard: some View {
        HStack(alignment: .center, spacing: 10) {
            TextField(
                "Type a message in Turkish or English and press Enter to speak...",
                text: $appState.inputText
            )
            .textFieldStyle(.roundedBorder)
            .controlSize(.large)
            .font(.system(size: 14))
            .focused($isFieldFocused)
            .onSubmit {
                appState.submit()
            }

            // Action Buttons
            HStack(spacing: 8) {
                Button(action: {
                    appState.submit()
                }) {
                    Label("Speak", systemImage: "arrow.up.circle.fill")
                        .font(.headline)
                        .frame(width: 85, height: 28)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!appState.isWorkerReady || appState.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                Button(action: {
                    appState.stop()
                }) {
                    Label("Stop", systemImage: "stop.fill")
                        .font(.headline)
                        .frame(width: 75, height: 28)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .tint(.red)
                .keyboardShortcut(".", modifiers: [.command])
                .disabled(!appState.isSpeakingOrGenerating)
            }
        }
        .padding(.vertical, 4)
    }

    // MARK: - Controls Bar

    private var controlsBar: some View {
        HStack(spacing: 16) {
            // Voice Profile Selector
            HStack(spacing: 6) {
                Text("Voice:")
                    .font(.caption)
                    .fontWeight(.medium)
                Picker("", selection: Binding(
                    get: { appState.selectedProfileId },
                    set: { appState.selectVoiceProfile(id: $0) }
                )) {
                    ForEach(appState.profiles) { profile in
                        Text(profile.name).tag(profile.id)
                    }
                }
                .labelsHidden()
                .frame(width: 170)

                Menu {
                    Button("Import Audio File (.wav)...") {
                        appState.showingFileImporter = true
                    }
                    Button("Import Voice Dataset Folder...") {
                        appState.showingFolderImporter = true
                    }
                    Button("Import from YouTube or Hugging Face...") {
                        appState.hfRepoInput = ""
                        appState.showingHfImporter = true
                    }
                } label: {
                    Image(systemName: "plus")
                        .font(.caption)
                }
                .menuStyle(.borderlessButton)
                .frame(width: 20)
                .help("Add voice profile from single audio file or folder dataset")
            }

            // Mode Selector (Efficient vs Natural)
            HStack(spacing: 6) {
                Text("Mode:")
                    .font(.caption)
                    .fontWeight(.medium)
                Picker("", selection: Binding(
                    get: { appState.synthesisMode },
                    set: { mode in
                        Task {
                            await appState.setSynthesisMode(mode)
                        }
                    }
                )) {
                    ForEach(SynthesisMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 135)
                .disabled(appState.isSwitchingMode || !appState.isWorkerReady)
            }

            // Submission Behavior Selector
            HStack(spacing: 6) {
                Text("When Speaking:")
                    .font(.caption)
                    .fontWeight(.medium)
                Picker("", selection: Binding(
                    get: { appState.submissionBehavior },
                    set: { appState.setSubmissionBehavior($0) }
                )) {
                    ForEach(SubmissionBehavior.allCases, id: \.self) { behavior in
                        Text(behavior.rawValue).tag(behavior)
                    }
                }
                .labelsHidden()
                .frame(width: 170)
            }

            Spacer()

            // Local Monitor Volume Slider
            HStack(spacing: 6) {
                Image(systemName: appState.monitorVolume > 0 ? "speaker.wave.1.fill" : "speaker.slash.fill")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("Monitor:")
                    .font(.caption)
                    .fontWeight(.medium)
                Slider(
                    value: Binding(
                        get: { appState.monitorVolume },
                        set: { appState.setMonitorVolume($0) }
                    ),
                    in: 0.0...1.0
                )
                .frame(width: 75)
                Text("\(Int(appState.monitorVolume * 100))%")
                    .font(.caption2)
                    .monospacedDigit()
                    .frame(width: 32, alignment: .trailing)
            }
        }
    }

    // MARK: - Queue Section

    private var queueSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Speech Queue")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                if !appState.queueItems.isEmpty {
                    Text("(\(appState.queueItems.count))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if !appState.queueItems.isEmpty {
                    Button("Clear All") {
                        appState.clearQueue()
                    }
                    .font(.caption)
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                }
            }

            if appState.queueItems.isEmpty {
                VStack(spacing: 4) {
                    Spacer()
                    Image(systemName: "bubble.left.and.bubble.right")
                        .font(.system(size: 24))
                        .foregroundStyle(.quaternary)
                    Text("No messages in queue")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                List {
                    ForEach(appState.queueItems) { item in
                        queueItemRow(item)
                    }
                }
                .listStyle(.inset(alternatesRowBackgrounds: true))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private func queueItemRow(_ item: QueueItem) -> some View {
        HStack(alignment: .center, spacing: 10) {
            switch item.state {
            case .queued:
                Image(systemName: "clock")
                    .foregroundStyle(.secondary)
                    .frame(width: 16)
            case .generating:
                ProgressView()
                    .controlSize(.mini)
                    .frame(width: 16)
            case .speaking:
                Image(systemName: "waveform")
                    .foregroundStyle(.green)
                    .frame(width: 16)
            case .completed:
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.secondary)
                    .frame(width: 16)
            case .cancelled:
                Image(systemName: "minus.circle")
                    .foregroundStyle(.red)
                    .frame(width: 16)
            case .failed:
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.yellow)
                    .frame(width: 16)
            }

            Text(item.text)
                .font(.body)
                .lineLimit(1)
                .foregroundStyle(item.state == .cancelled ? .secondary : .primary)

            Spacer()

            Text(item.state.rawValue.capitalized)
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(badgeColor(for: item.state).opacity(0.15))
                .foregroundStyle(badgeColor(for: item.state))
                .clipShape(Capsule())
        }
        .padding(.vertical, 3)
    }

    private func badgeColor(for state: QueueItemState) -> Color {
        switch state {
        case .queued: return .gray
        case .generating: return .blue
        case .speaking: return .green
        case .completed: return .secondary
        case .cancelled: return .red
        case .failed: return .orange
        }
    }

    // MARK: - New Voice Sheet

    private var newVoiceSheet: some View {
        VStack(spacing: 16) {
            Text("Add Voice Profile")
                .font(.headline)

            VStack(alignment: .leading, spacing: 6) {
                Text("Voice Name:")
                    .font(.caption)
                TextField("e.g. My Voice", text: $appState.newVoiceName)
                    .textFieldStyle(.roundedBorder)

                if let url = appState.pendingVoiceURL {
                    Text("Audio File: \(url.lastPathComponent)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            HStack {
                Button("Cancel") {
                    appState.showingNewVoiceDialog = false
                    appState.pendingVoiceURL = nil
                }
                .keyboardShortcut(.cancelAction)

                Button("Add Voice") {
                    if let url = appState.pendingVoiceURL {
                        Task {
                            await appState.importVoiceProfile(audioURL: url, name: appState.newVoiceName)
                            appState.showingNewVoiceDialog = false
                            appState.pendingVoiceURL = nil
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(appState.newVoiceName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(24)
        .frame(width: 320)
    }

    // MARK: - Dataset Candidates Sheet

    private var datasetCandidatesSheet: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Import Voice Dataset")
                .font(.headline)

            if let res = appState.datasetScanResult {
                HStack {
                    Text("Scanned \(res.total_files ?? 0) files: \(res.usable_count ?? 0) usable, \(res.rejected_count ?? 0) rejected")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("Voice Profile Name:")
                        .font(.caption)
                        .fontWeight(.medium)
                    TextField("e.g. Narrator", text: $appState.newVoiceName)
                        .textFieldStyle(.roundedBorder)
                }

                Text("Select Reference Candidate (Top Clean Speech Segments):")
                    .font(.caption)
                    .fontWeight(.medium)

                if let candidates = res.candidates, !candidates.isEmpty {
                    ScrollView {
                        VStack(spacing: 6) {
                            ForEach(candidates) { candidate in
                                candidateRow(candidate)
                            }
                        }
                    }
                    .frame(maxHeight: 220)
                } else {
                    Text("No usable speech candidates found in folder.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            HStack {
                Button("Cancel") {
                    appState.showingDatasetCandidateDialog = false
                    appState.datasetScanResult = nil
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Create Profile") {
                    Task {
                        await appState.confirmDatasetProfile(
                            name: appState.newVoiceName,
                            candidatePath: appState.selectedCandidatePath
                        )
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(appState.newVoiceName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || appState.selectedCandidatePath.isEmpty)
            }
            .padding(.top, 4)
        }
        .padding(20)
        .frame(width: 500)
    }

    private func candidateRow(_ candidate: DatasetCandidate) -> some View {
        let isSelected = appState.selectedCandidatePath == candidate.effectiveReferencePath
        return Button(action: {
            appState.selectedCandidatePath = candidate.effectiveReferencePath
        }) {
            HStack {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isSelected ? .blue : .secondary)

                VStack(alignment: .leading, spacing: 2) {
                    Text(candidate.filename)
                        .font(.caption)
                        .fontWeight(.medium)
                        .lineLimit(1)
                    HStack(spacing: 8) {
                        Text("\(String(format: "%.1f", candidate.duration_sec))s")
                        Text("Speech: \(Int(candidate.speech_ratio * 100))%")
                        Text("Peak: \(String(format: "%.1f", candidate.peak_db)) dBFS")
                        Text("RMS: \(String(format: "%.1f", candidate.rms_db)) dBFS")
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(8)
            .background(isSelected ? Color.blue.opacity(0.1) : Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(isSelected ? Color.blue : Color(nsColor: .separatorColor), lineWidth: isSelected ? 1.5 : 0.5)
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Hugging Face Import Sheet

    private var hfImportSheet: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Import Voice from YouTube or Hugging Face")
                .font(.headline)

            Text("Paste a YouTube video link (e.g. https://youtu.be/...) or a Hugging Face dataset. Micora will automatically download the audio, isolate the cleanest speech segment, and create your voice profile.")
                .font(.caption)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 4) {
                Text("YouTube URL or Hugging Face Dataset:")
                    .font(.caption)
                    .fontWeight(.medium)
                TextField("e.g. https://www.youtube.com/watch?v=... or ailabturkiye/cemyilmaz", text: $appState.hfRepoInput)
                    .textFieldStyle(.roundedBorder)
            }

            if appState.isImportingHf {
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Downloading & analyzing audio stream...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }

            HStack {
                Button("Cancel") {
                    appState.showingHfImporter = false
                }
                .keyboardShortcut(.cancelAction)
                .disabled(appState.isImportingHf)

                Spacer()

                Button("Import Voice") {
                    Task {
                        await appState.importHuggingFaceVoice(repoIdOrUrl: appState.hfRepoInput)
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(appState.hfRepoInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || appState.isImportingHf)
            }
            .padding(.top, 4)
        }
        .padding(20)
        .frame(width: 480)
    }

    // MARK: - Virtual Microphone Banner & Setup Sheet

    private var virtualMicBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 16))
                .foregroundStyle(.orange)

            VStack(alignment: .leading, spacing: 2) {
                Text("Virtual Microphone (BlackHole) Not Detected")
                    .font(.caption)
                    .fontWeight(.bold)
                Text("Speech is currently playing to physical speakers/headphones. To use Micora in Discord, Zoom, or games, install BlackHole 2ch.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button("Setup Guide") {
                appState.showingVirtualMicSetup = true
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .tint(.orange)

            Button(action: {
                appState.refreshAudioDevices()
            }) {
                Image(systemName: "arrow.clockwise")
            }
            .controlSize(.small)
            .help("Check for BlackHole again")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.orange.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.orange.opacity(0.35), lineWidth: 1)
        )
    }

    private var virtualMicSetupSheet: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: appState.isVirtualMicAvailable ? "mic.fill" : "mic.badge.xmark")
                    .font(.title2)
                    .foregroundStyle(appState.isVirtualMicAvailable ? .green : .orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Virtual Microphone & Audio Routing")
                        .font(.headline)
                    Text(appState.isVirtualMicAvailable ? "BlackHole 2ch is active and ready." : "BlackHole 2ch driver is required to act as a virtual microphone.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            Divider()

            if !appState.isVirtualMicAvailable {
                VStack(alignment: .leading, spacing: 10) {
                    Text("How to install BlackHole 2ch:")
                        .font(.subheadline)
                        .fontWeight(.semibold)

                    Text("1. Open Terminal and run this command:")
                        .font(.caption)

                    HStack {
                        Text("brew install blackhole-2ch")
                            .font(.system(.caption, design: .monospaced))
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color(nsColor: .textBackgroundColor))
                            .clipShape(RoundedRectangle(cornerRadius: 6))

                        Button(appState.copiedCommand ? "Copied!" : "Copy") {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString("brew install blackhole-2ch", forType: .string)
                            appState.copiedCommand = true
                            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                appState.copiedCommand = false
                            }
                        }
                        .controlSize(.small)
                    }

                    Text("2. After installing, run `sudo killall coreaudiod` in Terminal (or restart your Mac).")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text("3. In Discord, Zoom, OBS, or games:")
                        .font(.caption)
                        .fontWeight(.medium)
                    Text("Set your **Input Device (Microphone)** to **BlackHole 2ch**.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    Label("BlackHole 2ch is detected and active!", systemImage: "checkmark.circle.fill")
                        .font(.subheadline)
                        .foregroundStyle(.green)
                    Text("In your voice chat app (Discord, Zoom, etc.), select **BlackHole 2ch** as your Input Device (Microphone). Anything spoken in Micora will stream directly into the call.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Divider()

            // Custom Device Selection
            VStack(alignment: .leading, spacing: 6) {
                Text("Select Output Device:")
                    .font(.caption)
                    .fontWeight(.medium)

                Picker("", selection: Binding(
                    get: { appState.selectedOutputDeviceId },
                    set: { appState.selectOutputDevice(id: $0) }
                )) {
                    Text("Auto-Detect (BlackHole if available, else Default)").tag(AudioDeviceID(0))
                    ForEach(appState.availableOutputDevices) { dev in
                        let role = dev.name.contains("BlackHole") ? " [Virtual Mic]" : ""
                        Text("\(dev.name)\(role)").tag(dev.id)
                    }
                }
                .labelsHidden()
            }

            HStack {
                Button(action: {
                    appState.refreshAudioDevices()
                }) {
                    Label("Check Devices", systemImage: "arrow.clockwise")
                }
                .controlSize(.small)

                Spacer()

                Button("Done") {
                    appState.showingVirtualMicSetup = false
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
            .padding(.top, 4)
        }
        .padding(22)
        .frame(width: 500)
    }
}

