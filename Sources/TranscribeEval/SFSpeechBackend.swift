import Foundation
import Speech
import AVFoundation

final class SFSpeechBackend: TranscriptionBackend {
    let state = BackendState(name: "SFSpeech (en-US)")

    private var recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    private let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                    sampleRate: 16000, channels: 1, interleaved: false)

    // MARK: - VAD
    private let vadEnergyThreshold: Float = 0.015
    private let vadSilenceTimeout: TimeInterval = 0.5
    private var vadIsSpeech = false
    private var vadLastVoiceTime: Date? = nil
    private var vadSilenceTimer: Timer? = nil

    func start() {
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            DispatchQueue.main.async {
                guard let self else { return }
                switch status {
                case .authorized: self.startSession()
                default: self.state.errorMessage = "Speech recognition permission denied."
                }
            }
        }
    }

    func stop() {
        vadSilenceTimer?.invalidate()
        vadSilenceTimer = nil
        vadIsSpeech = false
        state.isRunning = false
        state.isSpeechActive = false
        state.partial = ""
        state.totalScore = 0
        state.lastTurnRank = nil
        state.sessionAccuracy = nil
        request?.endAudio()
        task?.cancel()
        request = nil
        task = nil
    }

    func processAudio(pcmFloat32 data: Data, sampleRate: Int) {
        guard state.isRunning, let req = request, let fmt else { return }

        let n = data.count / MemoryLayout<Float>.size
        guard n > 0, let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(n)) else { return }
        buf.frameLength = AVAudioFrameCount(n)
        data.withUnsafeBytes { raw in
            guard let fp = raw.baseAddress?.assumingMemoryBound(to: Float.self),
                  let ch = buf.floatChannelData else { return }
            ch[0].update(from: fp, count: n)
        }
        req.append(buf)

        // VAD: compute RMS and track speech/silence
        let rms = computeRMS(data: data, count: n)
        let isVoice = rms >= vadEnergyThreshold

        DispatchQueue.main.async { [weak self] in
            self?.updateVAD(isVoice: isVoice)
        }
    }

    // MARK: - Private

    private func computeRMS(data: Data, count: Int) -> Float {
        var sum: Float = 0
        data.withUnsafeBytes { raw in
            guard let fp = raw.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
            for i in 0..<count { sum += fp[i] * fp[i] }
        }
        return count > 0 ? sqrt(sum / Float(count)) : 0
    }

    private func updateVAD(isVoice: Bool) {
        if isVoice {
            vadLastVoiceTime = Date()
            if !vadIsSpeech {
                vadIsSpeech = true
                state.isSpeechActive = true
                if state.speechStartedAt == nil {
                    state.speechStartedAt = Date()
                    state.firstPartialAt = nil
                    state.lastFinalAt = nil
                }
            }
            // Reset silence timer on each voice chunk
            vadSilenceTimer?.invalidate()
            vadSilenceTimer = Timer.scheduledTimer(withTimeInterval: vadSilenceTimeout, repeats: false) { [weak self] _ in
                self?.onSilenceDetected()
            }
        }
    }

    private func onSilenceDetected() {
        guard vadIsSpeech else { return }
        vadIsSpeech = false
        state.isSpeechActive = false
        // Signal end of audio — SFSpeech will finalize and emit isFinal=true
        request?.endAudio()
    }

    private func startSession() {
        guard let rec = recognizer, rec.isAvailable else {
            state.errorMessage = "SFSpeechRecognizer unavailable."
            return
        }

        vadIsSpeech = false
        vadSilenceTimer?.invalidate()
        vadSilenceTimer = nil

        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        req.requiresOnDeviceRecognition = false
        self.request = req

        task = rec.recognitionTask(with: req) { [weak self] result, error in
            guard let self else { return }
            DispatchQueue.main.async {
                if let error {
                    let ns = error as NSError
                    // 1110 = no speech, 301 = cancelled — both are normal restarts
                    if ns.domain == "kAFAssistantErrorDomain" && (ns.code == 1110 || ns.code == 301) {
                        self.restart()
                        return
                    }
                    self.state.errorMessage = error.localizedDescription
                    self.restart()
                    return
                }
                guard let result else { return }

                let text = result.bestTranscription.formattedString
                guard !text.isEmpty else { return }

                if self.state.firstPartialAt == nil {
                    self.state.firstPartialAt = Date()
                }

                if result.isFinal {
                    let ttftMs: Int? = self.state.speechStartedAt.flatMap { s in
                        self.state.firstPartialAt.map { Int($0.timeIntervalSince(s) * 1000) }
                    }
                    let totalMs: Int? = self.state.speechStartedAt.map {
                        Int(Date().timeIntervalSince($0) * 1000)
                    }
                    let entry = TurnEntry(text: text, ttftMs: ttftMs, transcribeMs: nil, totalMs: totalMs)
                    self.state.turns.insert(entry, at: 0)
                    self.state.partial = ""
                    self.state.isSpeechActive = false
                    self.state.speechStartedAt = nil
                    self.state.firstPartialAt = nil
                    self.state.lastFinalAt = Date()
                    self.restart()
                } else {
                    self.state.partial = text
                }
            }
        }

        state.isRunning = true
        state.errorMessage = nil
    }

    private func restart() {
        task?.cancel()
        task = nil
        request = nil
        vadIsSpeech = false
        vadSilenceTimer?.invalidate()
        vadSilenceTimer = nil
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { [weak self] in
            guard let self, self.state.isRunning else { return }
            self.startSession()
        }
    }
}
