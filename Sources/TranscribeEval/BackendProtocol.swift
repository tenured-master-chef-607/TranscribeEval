import Foundation
import Observation

// MARK: - TurnEntry

struct TurnEntry: Identifiable {
    let id = UUID()
    let text: String
    let ttftMs: Int?       // streaming: speech onset → first partial
    let transcribeMs: Int? // batch: bridge-reported processing time
    let totalMs: Int?      // speech onset → final received (always measured by Swift)
}

// MARK: - BackendState

@Observable
final class BackendState {
    var name: String
    var isRunning: Bool = false
    var isSpeechActive: Bool = false
    var partial: String = ""
    var turns: [TurnEntry] = []
    var errorMessage: String? = nil
    var lastDebug: String? = nil

    // Per-turn timing
    var speechStartedAt: Date? = nil
    var firstPartialAt: Date? = nil
    var lastFinalAt: Date? = nil    // set when transcript_final received; nil resets rank
    var totalScore: Int = 0         // cumulative across all turns this session
    var lastTurnRank: Int? = nil    // rank of the most recently completed sentence
    var sessionAccuracy: Double? = nil  // set after user submits reference text

    init(name: String) { self.name = name }
}

// MARK: - TranscriptionBackend

protocol TranscriptionBackend: AnyObject {
    var state: BackendState { get }
    func start()
    func stop()
    func processAudio(pcmFloat32 data: Data, sampleRate: Int)
}
