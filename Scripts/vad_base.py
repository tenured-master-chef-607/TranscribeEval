"""
Shared VAD + I/O base for TranscribeEval whisper bridges.
All bridges import this and provide a transcribe_audio(samples) -> str function.
"""

import sys
import json
import base64
import struct
import time
import threading
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor

SAMPLE_RATE = 16000
ENERGY_THRESHOLD = 0.015
SILENCE_TIMEOUT = 0.45    # seconds of silence to end a speech segment
RING_SECONDS = 0.5        # pre-speech ring buffer duration
MIN_SPEECH_SECONDS = 0.3  # ignore segments shorter than this

_stdout_lock = threading.Lock()


def emit_json(payload: dict) -> None:
    with _stdout_lock:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def emit_debug(text: str) -> None:
    emit_json({"type": "debug", "text": text})


def emit_error(text: str) -> None:
    emit_json({"type": "error", "text": text})


def _rms(samples: list) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def run_bridge(name: str, load_fn, transcribe_fn, partial_interval: float = 1.5):
    """
    Main entry point for a VAD+Whisper bridge.

    load_fn()                          — called once to load the model
    transcribe_fn(samples: list) -> str — called per segment / partial
    partial_interval                   — seconds between interim partials
    """
    emit_debug(f"{name} bridge started")
    load_fn()

    # VAD state
    vad_state = "silence"
    speech_samples: list = []
    speech_start_time = 0.0
    speech_end_time = 0.0
    last_voice_time = 0.0
    last_partial_time = 0.0
    ring_buffer: deque = deque(maxlen=int(SAMPLE_RATE * RING_SECONDS))
    lock = threading.Lock()

    # Separate executors so finals never queue behind partials.
    # model_lock ensures only one thread calls transcribe_fn at a time
    # (whisper.cpp and most backends are not thread-safe).
    model_lock = threading.Lock()
    final_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{name}-final")
    partial_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{name}-partial")
    pending_partial = [None]  # track latest partial future to skip if busy

    def on_final_done(future, seg_start, seg_end):
        try:
            text, transcribe_ms = future.result()
        except Exception as exc:
            emit_error(f"{name} transcription error: {exc}")
            return
        if not text:
            return
        speech_ms = int((seg_end - seg_start) * 1000)
        emit_json({
            "type": "transcript_final",
            "text": text,
            "transcribe_ms": transcribe_ms,
            "speech_ms": speech_ms,
        })

    def on_partial_done(future):
        try:
            text, _ = future.result()
        except Exception as exc:
            emit_error(f"{name} partial error: {exc}")
            return
        if text:
            emit_json({"type": "transcript_partial", "text": text})

    def _timed(fn, s):
        with model_lock:
            t0 = time.monotonic()
            text = fn(s)
            return text, int((time.monotonic() - t0) * 1000)

    def process_chunk(samples: list):
        nonlocal vad_state, speech_samples, speech_start_time, speech_end_time
        nonlocal last_voice_time, last_partial_time

        now = time.monotonic()
        rms = _rms(samples)
        is_voice = rms >= ENERGY_THRESHOLD

        with lock:
            ring_buffer.extend(samples)

            if vad_state == "silence":
                if is_voice:
                    vad_state = "speech"
                    pre = list(ring_buffer)[:-len(samples)] if len(samples) < len(ring_buffer) else []
                    speech_samples = pre + list(samples)
                    speech_start_time = now
                    last_voice_time = now
                    last_partial_time = now
                    emit_json({"type": "vad_start"})
                    emit_debug(f"VAD speech start (rms={rms:.4f})")

            elif vad_state == "speech":
                speech_samples.extend(samples)
                if is_voice:
                    last_voice_time = now

                silence_s = now - last_voice_time
                if silence_s >= SILENCE_TIMEOUT:
                    vad_state = "silence"
                    seg = list(speech_samples)
                    speech_samples = []
                    speech_end_time = now
                    dur = len(seg) / SAMPLE_RATE
                    emit_debug(f"VAD speech end ({dur:.2f}s)")
                    if dur >= MIN_SPEECH_SECONDS:
                        seg_start = speech_start_time
                        seg_end = speech_end_time
                        f = final_executor.submit(_timed, transcribe_fn, seg)
                        f.add_done_callback(lambda fut: on_final_done(fut, seg_start, seg_end))
                else:
                    if now - last_partial_time >= partial_interval:
                        # Skip if previous partial is still running to avoid backlog
                        prev = pending_partial[0]
                        if prev is not None and not prev.done():
                            return
                        last_partial_time = now
                        seg = list(speech_samples)
                        emit_debug(f"VAD partial ({len(seg)/SAMPLE_RATE:.2f}s)")
                        f = partial_executor.submit(_timed, transcribe_fn, seg)
                        f.add_done_callback(on_partial_done)
                        pending_partial[0] = f

    # stdin loop
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        t = msg.get("type", "")
        if t == "connect":
            emit_debug(f"{name} connected")
        elif t == "disconnect":
            break
        elif t == "audio":
            b64 = msg.get("audio_base64", "")
            if not b64:
                continue
            try:
                raw = base64.b64decode(b64)
            except Exception:
                continue
            n = len(raw) // 4
            if n == 0:
                continue
            samples = list(struct.unpack(f"<{n}f", raw[:n * 4]))
            process_chunk(samples)

    emit_debug(f"{name} bridge exiting")
    final_executor.shutdown(wait=False)
    partial_executor.shutdown(wait=False)
