import SwiftUI
import UniformTypeIdentifiers
import CoreAudio
import AppKit
import MicoraCore

struct ContentView: View {
    @ObservedObject var appState: AppState
    @FocusState private var isFieldFocused: Bool

    private var currentVoiceName: String {
        appState.profiles.first(where: { $0.id == appState.selectedProfileId })?.name ?? "Default Voice"
    }

    var body: some View {
        VStack(spacing: 14) {
            // Header Bar
            headerBar

            // Virtual Microphone Warning Banner (if BlackHole not active)
            if !appState.isVirtualMicAvailable {
                virtualMicBanner
            }

            // Error Banner (if any)
            if let error = appState.lastError {
                errorBanner(error)
            }

            // Main Message Composer (Hero Card)
            composerCard

            // Secondary Controls Deck (Voice, Engine, Queue Behavior, Monitor)
            controlsDeck

            // Real-Time Speech Queue
            queueSection
        }
        .padding(18)
        .frame(minWidth: 720, minHeight: 540)
        .background(Color(nsColor: .windowBackgroundColor))
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
        HStack(alignment: .center, spacing: 12) {
            // Clean Typography Title
            VStack(alignment: .leading, spacing: 1) {
                Text("Micora")
                    .font(.system(size: 16, weight: .bold))
                Text("Local-First Virtual Microphone TTS")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            // Output Target Badge Button (Clickable -> Opens Audio Setup Sheet)
            Button(action: {
                appState.showingVirtualMicSetup = true
            }) {
                HStack(spacing: 7) {
                    Circle()
                        .fill(appState.isVirtualMicAvailable ? Color.green : Color.orange)
                        .frame(width: 7, height: 7)
                    Text(appState.isVirtualMicAvailable ? "Virtual Mic: BlackHole" : appState.primaryDeviceName)
                        .font(.system(size: 12, weight: .medium))
                    Image(systemName: "chevron.right")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundStyle(.tertiary)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(Capsule())
                .overlay(
                    Capsule()
                        .stroke(appState.isVirtualMicAvailable ? Color(nsColor: .separatorColor).opacity(0.6) : Color.orange.opacity(0.5), lineWidth: 0.8)
                )
            }
            .buttonStyle(.plain)
            .help("Configure audio output or view virtual microphone setup")

            // Worker State Badge
            HStack(spacing: 6) {
                if !appState.isWorkerReady || appState.isSwitchingMode || appState.isScanningDataset || appState.isImportingHf {
                    ProgressView()
                        .controlSize(.mini)
                } else {
                    Circle()
                        .fill(Color.green)
                        .frame(width: 7, height: 7)
                }
                Text(appState.workerState)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(Capsule())
            .overlay(
                Capsule()
                    .stroke(Color(nsColor: .separatorColor).opacity(0.6), lineWidth: 0.8)
            )
        }
    }

    // MARK: - Virtual Microphone Banner

    private var virtualMicBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 13))
                .foregroundStyle(.orange)

            VStack(alignment: .leading, spacing: 1) {
                Text("Virtual Microphone (BlackHole 2ch) Not Detected")
                    .font(.system(size: 12, weight: .semibold))
                Text("Audio is playing to physical speakers. To stream into Discord, Zoom, or games, install BlackHole 2ch.")
                    .font(.system(size: 11))
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
                    .font(.caption)
            }
            .controlSize(.small)
            .help("Check for BlackHole again")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.orange.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.orange.opacity(0.3), lineWidth: 0.8)
        )
    }

    // MARK: - Error Banner

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 12))
                .foregroundStyle(.red)

            Text(message)
                .font(.system(size: 12))
                .foregroundStyle(.red)
                .lineLimit(2)

            Spacer()

            Button(action: {
                appState.lastError = nil
            }) {
                Image(systemName: "xmark")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.red.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.red.opacity(0.3), lineWidth: 0.8)
        )
    }

    // MARK: - Main Message Composer (Hero Card)

    private var composerCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Clean Text Input
            TextField(
                "Type a message in Turkish or English and press Enter to speak...",
                text: $appState.inputText,
                axis: .vertical
            )
            .lineLimit(1...4)
            .textFieldStyle(.plain)
            .font(.system(size: 14))
            .focused($isFieldFocused)
            .onSubmit {
                appState.submit()
            }
            .padding(.horizontal, 14)
            .padding(.top, 12)

            Divider()
                .opacity(0.4)

            // Bottom Action Bar within Composer Card
            HStack(alignment: .center) {
                // Voice and Mode Context Tags
                HStack(spacing: 6) {
                    Text(currentVoiceName)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color.primary.opacity(0.05))
                        .clipShape(Capsule())

                    Text(appState.synthesisMode == .efficient ? "MOSS" : "Chatterbox")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(appState.synthesisMode == .efficient ? Color.blue : Color.purple)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(
                            (appState.synthesisMode == .efficient ? Color.blue : Color.purple).opacity(0.12)
                        )
                        .clipShape(Capsule())
                }

                Spacer()

                // Keyboard Shortcut Hints
                Text("↵ to Speak")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .padding(.trailing, 6)

                // Action Buttons
                HStack(spacing: 8) {
                    if appState.isSpeakingOrGenerating {
                        Button(action: {
                            appState.stop()
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: "stop.fill")
                                    .font(.system(size: 10, weight: .bold))
                                Text("Stop")
                                    .font(.system(size: 12, weight: .semibold))
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(Color.red.opacity(0.15))
                            .foregroundStyle(.red)
                            .clipShape(Capsule())
                        }
                        .buttonStyle(.plain)
                        .keyboardShortcut(".", modifiers: [.command])
                        .help("Stop active speech")
                    }

                    Button(action: {
                        appState.submit()
                    }) {
                        HStack(spacing: 5) {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.system(size: 13, weight: .semibold))
                            Text("Speak")
                                .font(.system(size: 13, weight: .semibold))
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 5)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.regular)
                    .clipShape(Capsule())
                    .disabled(!appState.isWorkerReady || appState.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            .padding(.horizontal, 14)
            .padding(.bottom, 10)
        }
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(isFieldFocused ? Color.accentColor.opacity(0.6) : Color(nsColor: .separatorColor).opacity(0.6), lineWidth: isFieldFocused ? 1.5 : 0.8)
        )
        .shadow(color: Color.black.opacity(0.03), radius: 3, x: 0, y: 1)
    }

    // MARK: - Secondary Controls Deck (2 Clean Cards)

    private var controlsDeck: some View {
        HStack(spacing: 12) {
            // Card 1: Voice & Model
            HStack(spacing: 10) {
                // Voice Profile Selector + Add Button
                VStack(alignment: .leading, spacing: 4) {
                    Text("VOICE PROFILE")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.secondary)

                    HStack(spacing: 6) {
                        Picker("", selection: Binding(
                            get: { appState.selectedProfileId },
                            set: { appState.selectVoiceProfile(id: $0) }
                        )) {
                            ForEach(appState.profiles) { profile in
                                Text(profile.name).tag(profile.id)
                            }
                        }
                        .labelsHidden()
                        .frame(minWidth: 140, maxWidth: .infinity)

                        Menu {
                            Button {
                                appState.showingFileImporter = true
                            } label: {
                                Label("Import Audio File (.wav)...", systemImage: "doc.badge.plus")
                            }
                            Button {
                                appState.showingFolderImporter = true
                            } label: {
                                Label("Import Voice Dataset Folder...", systemImage: "folder.badge.plus")
                            }
                            Button {
                                appState.hfRepoInput = ""
                                appState.showingHfImporter = true
                            } label: {
                                Label("Import from YouTube or Hugging Face...", systemImage: "globe")
                            }
                        } label: {
                            Image(systemName: "plus")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundStyle(Color.primary.opacity(0.85))
                                .frame(width: 22, height: 22)
                                .background(Color(nsColor: .controlBackgroundColor))
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 6)
                                        .stroke(Color(nsColor: .separatorColor), lineWidth: 0.6)
                                )
                        }
                        .menuStyle(.borderlessButton)
                        .menuIndicator(.hidden)
                        .frame(width: 24, height: 24)
                        .help("Add voice profile from audio file, folder dataset, or online link")
                    }
                }
                .frame(maxWidth: .infinity)

                Divider()
                    .frame(height: 32)
                    .opacity(0.5)

                // Synthesis Engine Mode Selector
                VStack(alignment: .leading, spacing: 4) {
                    Text("SYNTHESIS ENGINE")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.secondary)

                    Picker("", selection: Binding(
                        get: { appState.synthesisMode },
                        set: { mode in
                            Task {
                                await appState.setSynthesisMode(mode)
                            }
                        }
                    )) {
                        Text("Efficient").tag(SynthesisMode.efficient)
                        Text("Natural").tag(SynthesisMode.natural)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 150)
                    .disabled(appState.isSwitchingMode || !appState.isWorkerReady)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.7))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.5), lineWidth: 0.8)
            )

            // Card 2: When Speaking Behavior & Local Monitor
            HStack(spacing: 12) {
                // When Speaking Behavior
                VStack(alignment: .leading, spacing: 4) {
                    Text("WHEN SPEAKING")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.secondary)

                    Picker("", selection: Binding(
                        get: { appState.submissionBehavior },
                        set: { appState.setSubmissionBehavior($0) }
                    )) {
                        ForEach(SubmissionBehavior.allCases, id: \.self) { behavior in
                            Text(behavior.rawValue).tag(behavior)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 145)
                }

                Divider()
                    .frame(height: 32)
                    .opacity(0.5)

                // Local Monitor Volume
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 4) {
                        Text("LOCAL MONITOR")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text("\(Int(appState.monitorVolume * 100))%")
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }

                    HStack(spacing: 6) {
                        Button(action: {
                            if appState.monitorVolume > 0 {
                                appState.setMonitorVolume(0)
                            } else {
                                appState.setMonitorVolume(0.2)
                            }
                        }) {
                            Image(systemName: appState.monitorVolume > 0 ? "speaker.wave.2.fill" : "speaker.slash.fill")
                                .font(.system(size: 12))
                                .foregroundStyle(appState.monitorVolume > 0 ? Color.primary : Color.secondary)
                        }
                        .buttonStyle(.plain)
                        .help(appState.monitorVolume > 0 ? "Mute local monitor" : "Unmute local monitor")

                        Slider(
                            value: Binding(
                                get: { appState.monitorVolume },
                                set: { appState.setMonitorVolume($0) }
                            ),
                            in: 0.0...1.0
                        )
                        .frame(width: 75)
                    }
                }
                .frame(minWidth: 125)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.7))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.5), lineWidth: 0.8)
            )
        }
    }

    // MARK: - Queue Section

    private var queueSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Queue Section Header
            HStack {
                HStack(spacing: 6) {
                    Text("Speech Queue")
                        .font(.system(size: 13, weight: .semibold))

                    if !appState.queueItems.isEmpty {
                        Text("\(appState.queueItems.count)")
                            .font(.system(size: 11, weight: .bold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 1)
                            .background(Color.accentColor.opacity(0.15))
                            .foregroundStyle(Color.accentColor)
                            .clipShape(Capsule())
                    }
                }

                Spacer()

                if !appState.queueItems.isEmpty {
                    Button(action: {
                        appState.clearQueue()
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "trash")
                                .font(.caption2)
                            Text("Clear All")
                                .font(.caption)
                        }
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("Clear completed and queued messages")
                }
            }

            // Queue Content
            if appState.queueItems.isEmpty {
                VStack(spacing: 4) {
                    Spacer()
                    Text("No messages in queue")
                        .font(.system(size: 13))
                        .foregroundStyle(.tertiary)
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color(nsColor: .separatorColor).opacity(0.3), lineWidth: 0.6)
                )
            } else {
                ScrollView {
                    LazyVStack(spacing: 6) {
                        ForEach(appState.queueItems) { item in
                            queueItemCard(item)
                        }
                    }
                    .padding(4)
                }
                .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color(nsColor: .separatorColor).opacity(0.3), lineWidth: 0.6)
                )
            }
        }
    }

    private func queueItemCard(_ item: QueueItem) -> some View {
        HStack(alignment: .center, spacing: 10) {
            // Status Icon / Visual Indicator
            ZStack {
                Circle()
                    .fill(badgeColor(for: item.state).opacity(0.12))
                    .frame(width: 26, height: 26)

                switch item.state {
                case .queued:
                    Image(systemName: "clock")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                case .generating:
                    ProgressView()
                        .controlSize(.mini)
                case .speaking:
                    Image(systemName: "waveform")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.green)
                case .completed:
                    Image(systemName: "checkmark")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(.secondary)
                case .cancelled:
                    Image(systemName: "minus.circle")
                        .font(.system(size: 12))
                        .foregroundStyle(.red)
                case .failed:
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(.orange)
                }
            }

            // Message Text
            Text(item.text)
                .font(.system(size: 13))
                .lineLimit(2)
                .foregroundStyle(item.state == .cancelled ? .secondary : .primary)

            Spacer()

            // State Badge Capsule
            HStack(spacing: 4) {
                if item.state == .speaking {
                    Circle()
                        .fill(Color.green)
                        .frame(width: 5, height: 5)
                }
                Text(item.state.rawValue.capitalized)
                    .font(.system(size: 11, weight: .medium))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(badgeColor(for: item.state).opacity(0.12))
            .foregroundStyle(badgeColor(for: item.state))
            .clipShape(Capsule())
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(item.state == .speaking ? Color.green.opacity(0.4) : Color(nsColor: .separatorColor).opacity(0.3), lineWidth: item.state == .speaking ? 1.2 : 0.6)
        )
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
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Add Voice Profile")
                        .font(.headline)
                    Text("Create a new cloned voice from an audio reference file")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Voice Name:")
                    .font(.caption)
                    .fontWeight(.medium)
                TextField("e.g. My Voice", text: $appState.newVoiceName)
                    .textFieldStyle(.roundedBorder)

                if let url = appState.pendingVoiceURL {
                    HStack(spacing: 6) {
                        Image(systemName: "waveform")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Text("Source: \(url.lastPathComponent)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 2)
                }
            }
            .padding(12)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            HStack {
                Button("Cancel") {
                    appState.showingNewVoiceDialog = false
                    appState.pendingVoiceURL = nil
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

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
        .padding(22)
        .frame(width: 360)
    }

    // MARK: - Dataset Candidates Sheet

    private var datasetCandidatesSheet: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Import Voice Dataset")
                        .font(.headline)
                    Text("Select reference audio candidate from your folder")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            if let res = appState.datasetScanResult {
                HStack(spacing: 12) {
                    Text("\(res.total_files ?? 0) Files Scanned")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("•").foregroundStyle(.tertiary)
                    Text("\(res.usable_count ?? 0) Usable")
                        .font(.caption)
                        .foregroundStyle(.green)
                    Text("•").foregroundStyle(.tertiary)
                    Text("\(res.rejected_count ?? 0) Filtered")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(8)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))

                VStack(alignment: .leading, spacing: 4) {
                    Text("Voice Profile Name:")
                        .font(.caption)
                        .fontWeight(.medium)
                    TextField("e.g. Narrator", text: $appState.newVoiceName)
                        .textFieldStyle(.roundedBorder)
                }

                Text("Select Reference Candidate:")
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
        .frame(width: 520)
    }

    private func candidateRow(_ candidate: DatasetCandidate) -> some View {
        let isSelected = appState.selectedCandidatePath == candidate.effectiveReferencePath
        return Button(action: {
            appState.selectedCandidatePath = candidate.effectiveReferencePath
        }) {
            HStack(spacing: 10) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 14))
                    .foregroundStyle(isSelected ? .blue : .secondary)

                VStack(alignment: .leading, spacing: 3) {
                    Text(candidate.filename)
                        .font(.system(size: 12, weight: .medium))
                        .lineLimit(1)
                    HStack(spacing: 8) {
                        Text("\(String(format: "%.1f", candidate.duration_sec))s")
                        Text("Speech: \(Int(candidate.speech_ratio * 100))%")
                        Text("Peak: \(String(format: "%.1f", candidate.peak_db)) dB")
                        Text("RMS: \(String(format: "%.1f", candidate.rms_db)) dB")
                    }
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(8)
            .background(isSelected ? Color.blue.opacity(0.1) : Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(isSelected ? Color.blue : Color(nsColor: .separatorColor).opacity(0.5), lineWidth: isSelected ? 1.5 : 0.6)
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Hugging Face / YouTube Import Sheet

    private var hfImportSheet: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Import from YouTube or Hugging Face")
                        .font(.headline)
                    Text("Downloads, extracts, and isolates clean speech")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("URL or Dataset Identifier:")
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
        .frame(width: 500)
    }

    // MARK: - Virtual Microphone & Audio Routing Setup Sheet

    private var virtualMicSetupSheet: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Virtual Microphone & Audio Routing")
                        .font(.headline)
                    Text(appState.isVirtualMicAvailable ? "BlackHole 2ch is active and ready." : "BlackHole 2ch is required to act as a virtual microphone.")
                        .font(.caption2)
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
                    Text("In your voice chat app (Discord, Zoom, etc.), select **BlackHole 2ch** as your Input Device (Microphone). Anything spoken in Micora will stream directly into your call.")
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
        .frame(width: 520)
    }
}
