import SwiftUI

struct ContentView: View {
    @State private var audioCapture = AudioCapture()
    @State private var referenceText: String = ""

    private static let scriptsDir: String = {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return "\(home)/Desktop/TranscribeEval/Scripts"
    }()

    private let backends: [any TranscriptionBackend] = {
        let dir = ContentView.scriptsDir
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let venv = "\(home)/Desktop/TranscribeEval/.venv/bin/python3"
        let py = FileManager.default.fileExists(atPath: venv) ? venv : nil
        return [
            SFSpeechBackend(),
            ProcessBackend(name: "Gemini Live",
                           scriptPath: "\(dir)/gemini_transcribe_bridge.py",
                           pythonPath: py),
            ProcessBackend(name: "mlx-whisper tiny.en",
                           scriptPath: "\(dir)/vad_whisper_bridge.py",
                           args: ["--model", "tiny.en", "--language", "en"],
                           pythonPath: py),
            ProcessBackend(name: "faster-whisper sliding",
                           scriptPath: "\(dir)/sliding_faster_whisper_bridge.py",
                           args: ["--model", "tiny.en", "--language", "en"],
                           pythonPath: py),
            ProcessBackend(name: "whisperx",
                           scriptPath: "\(dir)/vad_whisperx_bridge.py",
                           args: ["--model", "tiny.en", "--language", "en"],
                           pythonPath: py),
            ProcessBackend(name: "whisper.cpp",
                           scriptPath: "\(dir)/vad_whispercpp_bridge.py",
                           args: ["--model", "tiny.en", "--language", "en", "--threads", "8"],
                           pythonPath: py),
        ]
    }()

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(alignment: .top, spacing: 1) {
                    ForEach(backends.indices, id: \.self) { i in
                        BackendColumn(state: backends[i].state, rank: rank(for: backends[i].state))
                            .frame(width: 260)
                            .onChange(of: backends[i].state.speechStartedAt) { _, newVal in
                                // Clear last-turn rank badge while speech is in progress
                                if newVal != nil { backends[i].state.lastTurnRank = nil }
                            }
                            .onChange(of: backends[i].state.lastFinalAt) { _, newVal in
                                guard newVal != nil else { return }
                                // Rank for sentence N = how many backends already have >= N sentences
                                let n = backends[i].state.turns.count
                                var r = 1
                                for j in backends.indices where j != i {
                                    if backends[j].state.turns.count >= n { r += 1 }
                                }
                                backends[i].state.lastTurnRank = r
                                let words = backends[i].state.turns.first
                                    .map { AccuracyEngine.words($0.text).count } ?? 0
                                backends[i].state.totalScore += points(for: r) * words
                            }
                        Divider()
                    }
                    ReferenceColumn(
                        referenceText: $referenceText,
                        onCalculate: calculateAccuracy
                    )
                    .frame(width: 280)
                }
            }
            .frame(maxHeight: .infinity)
        }
        .frame(minWidth: 900, minHeight: 500)
        .task {
            for b in backends { audioCapture.register(b) }
        }
    }

    @ViewBuilder
    private var toolbar: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(audioCapture.isCapturing ? Color.red : Color.secondary.opacity(0.4))
                .frame(width: 8, height: 8)
                .animation(audioCapture.isCapturing
                    ? .easeInOut(duration: 0.6).repeatForever(autoreverses: true) : .default,
                           value: audioCapture.isCapturing)

            Text(audioCapture.isCapturing ? "Capturing system audio" : "Stopped")
                .font(.system(size: 12))
                .foregroundStyle(audioCapture.isCapturing ? .primary : .secondary)

            Spacer()

            if let err = audioCapture.errorMessage {
                Text(err).font(.caption).foregroundStyle(.red).lineLimit(1)
            }

            Button("Export") { exportStats() }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(backends.allSatisfy { $0.state.turns.isEmpty })

            Button(audioCapture.isCapturing ? "Stop" : "Record") {
                audioCapture.isCapturing ? stopAll() : startAll()
            }
            .buttonStyle(.borderedProminent)
            .tint(audioCapture.isCapturing ? .red : .accentColor)
            .controlSize(.small)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.bar)
    }

    private func startAll() {
        for b in backends { b.start() }
        audioCapture.start()
    }

    private func stopAll() {
        audioCapture.stop()
        for b in backends { b.stop() }
    }

    /// Concatenate all turns (oldest first) into one string and compute WER against reference.
    private func calculateAccuracy() {
        let ref = referenceText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !ref.isEmpty else { return }
        for backend in backends {
            let hypothesis = backend.state.turns
                .reversed()                      // turns stored newest-first; reverse for reading order
                .map(\.text)
                .joined(separator: " ")
            backend.state.sessionAccuracy = AccuracyEngine.accuracy(hypothesis: hypothesis, reference: ref)
        }
    }

    private func exportStats() {
        let rows: [[String: Any]] = backends.map { b in
            let s = b.state
            let xscribs  = s.turns.compactMap(\.transcribeMs).filter { $0 > 0 }
            let ttfts    = s.turns.compactMap(\.ttftMs).filter { $0 > 0 }
            // Use inference latency for batch, TTFT for streaming
            let latencies = xscribs.isEmpty ? ttfts : xscribs
            let latencyKey = xscribs.isEmpty ? "ttft" : "transcribe"
            let wordCounts = s.turns.map { AccuracyEngine.words($0.text).count }
            let avgLatency = latencies.isEmpty ? nil : latencies.reduce(0, +) / latencies.count
            let avgWords   = wordCounts.isEmpty ? nil : Double(wordCounts.reduce(0, +)) / Double(wordCounts.count)
            var row: [String: Any] = [
                "backend": s.name,
                "turns": s.turns.count,
                "total_score": s.totalScore,
                "turns_data": s.turns.reversed().map { t -> [String: Any] in
                    var d: [String: Any] = ["text": t.text]
                    if let v = t.transcribeMs, v > 0 {
                        d["transcribe_ms"] = v
                    } else if let v = t.ttftMs, v > 0 {
                        d["ttft_ms"] = v
                    }
                    return d
                }
            ]
            if let v = avgLatency  { row["avg_\(latencyKey)_ms"] = v }
            if let v = latencies.min() { row["min_\(latencyKey)_ms"] = v }
            if let v = latencies.max() { row["max_\(latencyKey)_ms"] = v }
            if let v = avgWords    { row["avg_words_per_turn"] = Double(String(format: "%.1f", v))! }
            if let v = s.sessionAccuracy { row["session_accuracy"] = Double(String(format: "%.4f", v))! }
            return row
        }
        let payload: [String: Any] = [
            "exported_at": ISO8601DateFormatter().string(from: Date()),
            "reference_text": referenceText.isEmpty ? NSNull() : referenceText,
            "backends": rows
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]),
              let json = String(data: data, encoding: .utf8) else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(json, forType: .string)
    }

    private func rank(for state: BackendState) -> Int? {
        state.lastTurnRank
    }

    private func points(for rank: Int) -> Int {
        switch rank {
        case 1: return 5
        case 2: return 3
        case 3: return 2
        default: return 1
        }
    }
}

// MARK: - BackendColumn

struct BackendColumn: View {
    let state: BackendState
    let rank: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(headerBackground)
                .animation(.easeInOut(duration: 0.25), value: rank)
                .animation(.easeInOut(duration: 0.15), value: state.partial.isEmpty)

            Divider()

            liveArea
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .frame(minHeight: 56, alignment: .topLeading)

            Divider()

            ScrollView(.vertical, showsIndicators: true) {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(state.turns) { turn in
                        TurnRow(entry: turn)
                    }
                }
                .padding(8)
            }
            .frame(maxHeight: .infinity)

            if !state.turns.isEmpty {
                Divider()
                statsFooter
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(Color.secondary.opacity(0.04))
            }
        }
    }

    // MARK: Stats footer

    @ViewBuilder
    private var statsFooter: some View {
        let xscribs = state.turns.compactMap(\.transcribeMs).filter { $0 > 0 }
        let ttfts   = state.turns.compactMap(\.ttftMs).filter { $0 > 0 }
        // Use inference time for batch backends, TTFT for streaming (SFSpeech, Gemini)
        let latencies = xscribs.isEmpty ? ttfts : xscribs
        let latencyLabel = xscribs.isEmpty ? "TTFT" : "inference"
        let words = state.turns.map { AccuracyEngine.words($0.text).count }
        let avgWords = words.isEmpty ? 0.0 : Double(words.reduce(0, +)) / Double(words.count)

        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 0) {
                Text("\(state.turns.count) turns")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                if let acc = state.sessionAccuracy {
                    Text("accuracy \(Int(acc * 100))%")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(accuracyColor(acc))
                }
            }
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 3) {
                if !latencies.isEmpty {
                    statCell(label: "avg \(latencyLabel)", value: "\(latencies.reduce(0,+)/latencies.count)ms")
                    statCell(label: "avg words/turn", value: String(format: "%.1f", avgWords))
                    statCell(label: "min \(latencyLabel)", value: "\(latencies.min()!)ms")
                    statCell(label: "max \(latencyLabel)", value: "\(latencies.max()!)ms")
                }
            }
        }
    }

    @ViewBuilder
    private func statCell(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(.system(size: 8))
                .foregroundStyle(Color.secondary.opacity(0.7))
            Text(value)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(.primary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: Header

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 7, height: 7)

            Text(state.name)
                .font(.system(size: 12, weight: .semibold))
                .lineLimit(1)

            Text("\(state.totalScore)")
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .foregroundStyle(state.totalScore > 0 ? .primary : Color.secondary.opacity(0.25))
                .monospacedDigit()
                .animation(.spring(duration: 0.2), value: state.totalScore)

            Spacer()

            // Session accuracy badge — shown after user submits reference
            if let acc = state.sessionAccuracy {
                let pct = Int(acc * 100)
                Text("\(pct)%")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(accuracyColor(acc))
                    .clipShape(Capsule())
                    .transition(.scale(scale: 0.7).combined(with: .opacity))
                    .animation(.spring(duration: 0.3), value: acc)
            }

            if let r = rank {
                Text(ordinal(r))
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(rankColor(r))
                    .clipShape(Capsule())
                    .transition(.scale(scale: 0.6).combined(with: .opacity))
            } else if !state.partial.isEmpty {
                HStack(spacing: 3) {
                    ForEach(0..<3, id: \.self) { _ in
                        Circle().fill(Color.blue).frame(width: 3, height: 3)
                    }
                }
                .transition(.opacity)
            } else if state.isSpeechActive {
                Image(systemName: "waveform")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.blue)
                    .transition(.opacity)
            }
        }
        .animation(.spring(duration: 0.25), value: rank)
        .animation(.easeInOut(duration: 0.15), value: state.partial.isEmpty)
        .animation(.easeInOut(duration: 0.15), value: state.isSpeechActive)
        .animation(.spring(duration: 0.3), value: state.sessionAccuracy)
    }

    private var headerBackground: Color {
        if let r = rank { return rankColor(r).opacity(0.12) }
        if !state.partial.isEmpty { return .blue.opacity(0.08) }
        return .secondary.opacity(0.06)
    }

    private var statusColor: Color {
        if state.errorMessage != nil { return .red }
        if state.isRunning { return state.isSpeechActive ? .blue : .green }
        return .secondary.opacity(0.5)
    }

    private func accuracyColor(_ acc: Double) -> Color {
        acc >= 0.9 ? .green : acc >= 0.7 ? .orange : .red
    }

    private func rankColor(_ r: Int) -> Color {
        switch r {
        case 1: return .green
        case 2: return .orange
        case 3: return .red
        default: return .secondary
        }
    }

    private func ordinal(_ n: Int) -> String {
        switch n {
        case 1: return "1st"
        case 2: return "2nd"
        case 3: return "3rd"
        default: return "\(n)th"
        }
    }

    // MARK: Live area

    @ViewBuilder
    private var liveArea: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let err = state.errorMessage {
                Text(err)
                    .font(.system(size: 11))
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            } else if !state.partial.isEmpty {
                Text(state.partial)
                    .font(.system(size: 12))
                    .italic()
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .animation(.easeInOut(duration: 0.1), value: state.partial)
            } else if state.isRunning {
                Text(state.isSpeechActive ? "Processing…" : "Listening…")
                    .font(.system(size: 11))
                    .foregroundStyle(Color.secondary.opacity(0.5))
            }

            if let dbg = state.lastDebug {
                Text(dbg)
                    .font(.system(size: 9))
                    .foregroundStyle(Color.secondary.opacity(0.5))
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
    }
}

// MARK: - ReferenceColumn

struct ReferenceColumn: View {
    @Binding var referenceText: String
    let onCalculate: () -> Void

    var wordCount: Int { AccuracyEngine.words(referenceText).count }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack(spacing: 6) {
                Image(systemName: "checkmark.seal.fill")
                    .font(.system(size: 9))
                    .foregroundStyle(.purple)
                Text("Reference")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                if wordCount > 0 {
                    Text("\(wordCount) words")
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(Color.purple.opacity(0.08))

            Divider()

            // Input + button
            VStack(alignment: .leading, spacing: 8) {
                Text("Paste or type the full spoken text, then calculate accuracy across all backends.")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                ZStack(alignment: .topLeading) {
                    if referenceText.isEmpty {
                        Text("e.g. \"The quick brown fox jumped over the lazy dog.\"")
                            .font(.system(size: 12))
                            .foregroundStyle(Color.secondary.opacity(0.4))
                            .padding(.top, 2)
                            .padding(.leading, 4)
                            .allowsHitTesting(false)
                    }
                    TextEditor(text: $referenceText)
                        .font(.system(size: 12))
                        .frame(maxHeight: .infinity)
                        .scrollContentBackground(.hidden)
                }
                .frame(maxHeight: .infinity)

                Button("Calculate Accuracy") {
                    onCalculate()
                }
                .buttonStyle(.borderedProminent)
                .tint(.purple)
                .controlSize(.small)
                .disabled(wordCount == 0)
                .keyboardShortcut(.return, modifiers: .command)
            }
            .padding(10)
            .frame(maxHeight: .infinity)
        }
    }
}

// MARK: - TurnRow

struct TurnRow: View {
    let entry: TurnEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(entry.text)
                .font(.system(size: 12))
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 6) {
                if let ms = entry.transcribeMs, ms > 0 {
                    badge("⟳ \(ms)ms", color: badgeColor(ms, fast: 300, slow: 800))
                } else if let ms = entry.ttftMs, ms > 0 {
                    badge("\(ms)ms", color: badgeColor(ms, fast: 300, slow: 800))
                }
            }
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 6)
        .background(Color.secondary.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    @ViewBuilder
    private func badge(_ label: String, color: Color) -> some View {
        Text(label)
            .font(.system(size: 9, weight: .medium))
            .foregroundStyle(color)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(color.opacity(0.1))
            .clipShape(Capsule())
    }

    private func badgeColor(_ ms: Int, fast: Int, slow: Int) -> Color {
        ms < fast ? .green : ms < slow ? .orange : .red
    }
}

#Preview {
    ContentView()
}
