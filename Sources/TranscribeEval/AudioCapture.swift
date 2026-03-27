import ScreenCaptureKit
import AVFoundation
import Observation
import CoreAudio

@Observable
final class AudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    var isCapturing: Bool = false
    var errorMessage: String? = nil

    private var stream: SCStream? = nil
    private var backends: [any TranscriptionBackend] = []

    // Target: 16kHz mono float32
    private let targetSampleRate: Int = 16000
    // ~30ms at 16kHz
    private let chunkSamples: Int = 480

    private var sampleAccumulator: [Float] = []
    private let audioQueue = DispatchQueue(label: "AudioCapture.audio", qos: .userInteractive)

    func register(_ backend: any TranscriptionBackend) {
        backends.append(backend)
    }

    func start() {
        guard !isCapturing else { return }

        // Request screen recording permission and get the content list.
        // If permission is not granted, this call fails with an error.
        SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: false) { [weak self] content, error in
            guard let self else { return }
            if let error {
                DispatchQueue.main.async {
                    self.errorMessage = "Screen Recording permission required. Grant it in System Settings › Privacy & Security › Screen Recording, then restart. (\(error.localizedDescription))"
                }
                return
            }
            guard let display = content?.displays.first else {
                DispatchQueue.main.async {
                    self.errorMessage = "No display found for system audio capture."
                }
                return
            }
            self.startStream(display: display)
        }
    }

    private func startStream(display: SCDisplay) {
        let filter = SCContentFilter(display: display, excludingWindows: [])

        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = false
        config.sampleRate = targetSampleRate
        config.channelCount = 1
        // Minimize video overhead — we only want audio
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.width = 2
        config.height = 2

        let scStream = SCStream(filter: filter, configuration: config, delegate: self)
        do {
            // Only add audio output — no video output added
            try scStream.addStreamOutput(self, type: .audio, sampleHandlerQueue: audioQueue)
        } catch {
            DispatchQueue.main.async { [weak self] in
                self?.errorMessage = "Failed to add audio output: \(error.localizedDescription)"
            }
            return
        }

        scStream.startCapture { [weak self] error in
            DispatchQueue.main.async {
                if let error {
                    self?.errorMessage = "Capture failed: \(error.localizedDescription)"
                } else {
                    self?.stream = scStream
                    self?.isCapturing = true
                    self?.errorMessage = nil
                }
            }
        }
    }

    func stop() {
        guard isCapturing else { return }
        let s = stream
        stream = nil
        sampleAccumulator = []
        DispatchQueue.main.async { [weak self] in
            self?.isCapturing = false
        }
        s?.stopCapture { _ in }
    }

    // MARK: - SCStreamOutput

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .audio else { return }

        // Extract AudioBufferList from CMSampleBuffer.
        // With channelCount=1 the list always has exactly 1 AudioBuffer.
        var audioBufferList = AudioBufferList()
        var retainedBlockBuffer: CMBlockBuffer?

        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: &audioBufferList,
            bufferListSize: MemoryLayout<AudioBufferList>.size,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: &retainedBlockBuffer
        )
        guard status == noErr else { return }

        let buf = audioBufferList.mBuffers
        guard let dataPtr = buf.mData, buf.mDataByteSize > 0 else { return }

        let floatCount = Int(buf.mDataByteSize) / MemoryLayout<Float>.size
        let samples = Array(
            UnsafeBufferPointer(
                start: dataPtr.assumingMemoryBound(to: Float.self),
                count: floatCount
            )
        )

        sampleAccumulator.append(contentsOf: samples)

        while sampleAccumulator.count >= chunkSamples {
            let chunk = Array(sampleAccumulator.prefix(chunkSamples))
            sampleAccumulator.removeFirst(chunkSamples)
            let data = chunk.withUnsafeBytes { Data($0) }
            for backend in backends {
                backend.processAudio(pcmFloat32: data, sampleRate: targetSampleRate)
            }
        }
    }

    // MARK: - SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        DispatchQueue.main.async { [weak self] in
            self?.isCapturing = false
            self?.stream = nil
            self?.errorMessage = "Stream stopped: \(error.localizedDescription)"
        }
    }
}

// MARK: - Errors (kept for protocol compatibility)

enum AudioCaptureError: Error, LocalizedError {
    case permissionDenied

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "Screen Recording permission denied."
        }
    }
}
