import SwiftUI

@main
struct TranscribeEvalApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 720, height: 520)
        .commands {
            // Remove default commands we don't need
            CommandGroup(replacing: .newItem) {}
        }
    }
}
