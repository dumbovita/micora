import SwiftUI
import AppKit

@main
struct MicoraApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView(appState: appState)
                .onDisappear {
                    appState.shutdown()
                }
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified)
        .defaultSize(width: 760, height: 600)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Micora") {
                    NSApp.orderFrontStandardAboutPanel(
                        options: [
                            .applicationName: "Micora",
                            .version: "1.0.0",
                            .credits: NSAttributedString(string: "Local-First Apple Silicon Voice Cloning TTS")
                        ]
                    )
                }
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}
