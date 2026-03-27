import Foundation

final class ProcessBackend: TranscriptionBackend {
    let state: BackendState

    private let scriptPath: String
    private let args: [String]
    private let env: [String: String]
    private let pythonPath: String?

    private var process: Process?
    private var stdinPipe: Pipe?

    init(
        name: String,
        scriptPath: String,
        args: [String] = [],
        env: [String: String] = [:],
        pythonPath: String? = nil
    ) {
        self.state = BackendState(name: name)
        self.scriptPath = scriptPath
        self.args = args
        self.env = env
        self.pythonPath = pythonPath
    }

    // MARK: - Start / Stop

    func start() {
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            DispatchQueue.main.async {
                self.state.errorMessage = "Script not found: \(self.scriptPath)"
            }
            return
        }
        spawnProcess()
    }

    func stop() {
        sendJSON(["type": "disconnect"])
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.3) {
            self.process?.terminate()
            self.process = nil
            self.stdinPipe = nil
        }
        DispatchQueue.main.async {
            self.state.isRunning = false
            self.state.isSpeechActive = false
            self.state.partial = ""
            self.state.speechStartedAt = nil
            self.state.firstPartialAt = nil
            self.state.totalScore = 0
            self.state.lastTurnRank = nil
            self.state.sessionAccuracy = nil
        }
    }

    // MARK: - Audio

    func processAudio(pcmFloat32 data: Data, sampleRate: Int) {
        guard state.isRunning else { return }
        sendJSON([
            "type": "audio",
            "audio_base64": data.base64EncodedString(),
            "sample_rate": sampleRate
        ])
    }

    // MARK: - Private

    private func spawnProcess() {
        let python: String
        if let p = pythonPath, FileManager.default.fileExists(atPath: p) {
            python = p
        } else if FileManager.default.fileExists(atPath: "/opt/homebrew/bin/python3") {
            python = "/opt/homebrew/bin/python3"
        } else {
            python = "/usr/bin/python3"
        }

        let proc = Process()
        let inPipe = Pipe()
        let outPipe = Pipe()
        let errPipe = Pipe()

        proc.executableURL = URL(fileURLWithPath: python)
        proc.arguments = ["-u", scriptPath] + args
        proc.standardInput = inPipe
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        var environment = ProcessInfo.processInfo.environment
        for (k, v) in env { environment[k] = v }
        proc.environment = environment

        proc.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                self?.state.isRunning = false
                self?.state.isSpeechActive = false
            }
        }

        do { try proc.run() } catch {
            DispatchQueue.main.async {
                self.state.errorMessage = "Launch failed: \(error.localizedDescription)"
            }
            return
        }

        process = proc
        stdinPipe = inPipe

        sendJSON(["type": "connect"])

        // stdout reader
        Thread.detachNewThread { [weak self] in self?.readLoop(pipe: outPipe) }
        // stderr sink
        Thread.detachNewThread { _ = errPipe.fileHandleForReading.readDataToEndOfFile() }

        DispatchQueue.main.async {
            self.state.isRunning = true
            self.state.errorMessage = nil
        }
    }

    private func sendJSON(_ dict: [String: Any]) {
        guard let pipe = stdinPipe,
              let data = try? JSONSerialization.data(withJSONObject: dict),
              let line = String(data: data, encoding: .utf8) else { return }
        let lineData = Data((line + "\n").utf8)
        try? pipe.fileHandleForWriting.write(contentsOf: lineData)
    }

    private func readLoop(pipe: Pipe) {
        let handle = pipe.fileHandleForReading
        var buf = Data()
        while true {
            let chunk = handle.availableData
            if chunk.isEmpty { break }
            buf.append(chunk)
            while let nl = buf.range(of: Data([0x0A])) {
                let lineData = buf[buf.startIndex..<nl.lowerBound]
                buf.removeSubrange(buf.startIndex...nl.lowerBound)
                if let s = String(data: lineData, encoding: .utf8), !s.isEmpty {
                    handleLine(s)
                }
            }
        }
    }

    private func handleLine(_ line: String) {
        guard let data = line.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type_ = json["type"] as? String else { return }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            switch type_ {

            case "vad_start":
                state.isSpeechActive = true
                state.speechStartedAt = Date()
                state.firstPartialAt = nil
                state.lastFinalAt = nil

            case "transcript_partial":
                let text = json["text"] as? String ?? ""
                guard !text.isEmpty else { return }
                if state.firstPartialAt == nil {
                    state.firstPartialAt = Date()
                }
                state.partial = text

            case "transcript_final":
                let text = json["text"] as? String ?? ""
                guard !text.isEmpty else { return }
                let now = Date()
                let transcribeMs = json["transcribe_ms"] as? Int
                let ttftMs: Int? = self.state.speechStartedAt.flatMap { start in
                    self.state.firstPartialAt.map { Int($0.timeIntervalSince(start) * 1000) }
                }
                let totalMs: Int? = state.speechStartedAt.map { Int(now.timeIntervalSince($0) * 1000) }
                let entry = TurnEntry(
                    text: text,
                    ttftMs: ttftMs,
                    transcribeMs: transcribeMs,
                    totalMs: totalMs
                )
                state.turns.insert(entry, at: 0)
                state.partial = ""
                state.isSpeechActive = false
                state.speechStartedAt = nil
                state.firstPartialAt = nil
                state.lastFinalAt = now

            case "error":
                state.errorMessage = json["text"] as? String ?? "Unknown error"

            case "debug":
                if let text = json["text"] as? String, !text.isEmpty {
                    state.lastDebug = text
                }

            default:
                break
            }
        }
    }
}
