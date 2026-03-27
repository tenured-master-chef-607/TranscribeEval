#!/usr/bin/env python3
"""
Gemini Live transcription bridge for TranscribeEval.
Same connection flow as Wingman. Only emits input transcription — no response output.

Requires: pip install google-genai
          GOOGLE_API_KEY in environment or TranscribeEval/.env
"""
import asyncio, base64, contextlib, json, math, os, struct, sys

try:
    from google import genai
except (ImportError, ModuleNotFoundError):
    genai = None

MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

# Client-side VAD threshold — same as vad_base.py so speech onset is measured consistently
_ENERGY_THRESHOLD = 0.015
_VAD_SILENCE_TIMEOUT = 0.7  # seconds (matches Gemini's silence_duration_ms)

LIVE_CONFIG = {
    "response_modalities": ["AUDIO"],  # must be AUDIO for speech pipeline to activate
    "input_audio_transcription": {},
    "output_audio_transcription": {},
    "generation_config": {
        "thinking_config": {
            "thinking_budget": 0,
        }
    },
    "realtime_input_config": {
        "automatic_activity_detection": {
            "disabled": False,
            "start_of_speech_sensitivity": "START_SENSITIVITY_HIGH",
            "end_of_speech_sensitivity": "END_SENSITIVITY_LOW",
            "prefix_padding_ms": 80,
            "silence_duration_ms": 700,
        }
    },
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


async def emit(payload: dict) -> None:
    _emit(payload)


async def emit_debug(msg: str) -> None:
    _emit({"type": "debug", "text": msg})


async def open_stdin_reader():
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    return reader, transport


# ---------------------------------------------------------------------------
# Audio conversion: float32 LE → int16 LE  (Gemini expects int16 PCM)
# ---------------------------------------------------------------------------

def _to_int16(data: bytes) -> bytes:
    n = len(data) // 4
    if n == 0:
        return b""
    floats = struct.unpack_from(f"<{n}f", data)
    out = bytearray(n * 2)
    for i, s in enumerate(floats):
        v = int(max(-32768.0, min(32767.0, round(max(-1.0, min(1.0, s)) * 32767.0))))
        struct.pack_into("<h", out, i * 2, v)
    return bytes(out)


def _rms_f32(data: bytes) -> float:
    n = len(data) // 4
    if n == 0:
        return 0.0
    floats = struct.unpack_from(f"<{n}f", data)
    return math.sqrt(sum(s * s for s in floats) / n)


# ---------------------------------------------------------------------------
# merge_stream_text (same as Wingman)
# ---------------------------------------------------------------------------

def merge_stream_text(acc, last, incoming):
    if not isinstance(incoming, str) or not incoming.strip():
        return acc, last, False
    if incoming == last:
        return acc, last, False
    if last and incoming.startswith(last):
        merged = acc + incoming[len(last):]
    elif acc and incoming.startswith(acc):
        merged = incoming
    elif acc.endswith(incoming):
        merged = acc
    else:
        merged = acc + incoming
    return merged, incoming, merged != acc


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

async def read_commands(reader, queue, stop):
    try:
        while not stop.is_set():
            raw = await reader.readline()
            if not raw:
                stop.set()
                return
            try:
                queue.put_nowait(json.loads(raw.decode()))
            except Exception:
                pass
    except asyncio.CancelledError:
        pass


async def forward_audio(session, queue, stop, vad_shared: dict):
    # 480 silent int16 samples = ~30ms at 16kHz — sent when queue is idle
    # to prevent Gemini from closing the WebSocket with a keepalive ping timeout.
    SILENCE = bytes(480 * 2)
    KEEPALIVE_INTERVAL = 0.2  # seconds between silence packets when idle

    count = 0
    last_voice_time: float = 0.0

    try:
        while not stop.is_set():
            now = asyncio.get_event_loop().time()
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                # No audio from Swift — check silence timeout and send keepalive
                if vad_shared["active"] and last_voice_time > 0:
                    if (now - last_voice_time) >= _VAD_SILENCE_TIMEOUT:
                        vad_shared["active"] = False
                await session.send_realtime_input(
                    audio={"data": SILENCE, "mime_type": "audio/pcm;rate=16000"}
                )
                continue

            t = msg.get("type")
            if t == "disconnect":
                stop.set()
                return
            if t != "audio":
                continue
            b64 = msg.get("audio_base64", "")
            sr = int(msg.get("sample_rate", 16000))
            if not b64:
                continue
            try:
                pcm_f32 = base64.b64decode(b64)
            except Exception:
                continue

            # Client-side VAD: emit vad_start at actual speech onset so that
            # speechStartedAt in Swift is anchored the same way as all other backends.
            rms = _rms_f32(pcm_f32)
            if rms >= _ENERGY_THRESHOLD:
                last_voice_time = now
                if not vad_shared["active"]:
                    vad_shared["active"] = True
                    await emit({"type": "vad_start"})

            pcm_i16 = _to_int16(pcm_f32)
            if not pcm_i16:
                continue
            await session.send_realtime_input(
                audio={"data": pcm_i16, "mime_type": f"audio/pcm;rate={sr}"}
            )
            count += 1
            if count == 1:
                await emit_debug(f"First audio packet sent ({len(pcm_i16)} bytes @ {sr} Hz, int16).")
            elif count % 600 == 0:
                await emit_debug(f"Forwarded {count} audio packets.")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        await emit({"type": "error", "text": f"Audio forwarding error: {e}"})
        stop.set()


async def receive_loop(session, stop, vad_shared: dict):
    """Single-pass receive loop over the entire session lifetime.

    Never breaks from session.receive() — breaking causes aclose() on the
    async generator which can tear down the underlying WebSocket, causing all
    subsequent session.receive() calls to hang indefinitely.
    Per-turn state is reset inline when turn_complete is detected.
    """
    total_msgs = 0
    input_text = ""
    last_server = ""
    last_partial = ""
    finalized = ""
    try:
        await emit_debug("Receive loop started.")
        async for msg in session.receive():
            if stop.is_set():
                return

            sc = getattr(msg, "server_content", None)
            itx_fb = getattr(msg, "input_transcription", None)
            tc_fb = bool(getattr(msg, "turn_complete", False))

            itx = getattr(sc, "input_transcription", None) or itx_fb
            raw_text = getattr(itx, "text", None) if itx else None
            finished = bool(getattr(itx, "finished", False)) if itx else False

            input_text, last_server, updated = merge_stream_text(input_text, last_server, raw_text)
            display = input_text.strip()

            total_msgs += 1
            if total_msgs <= 5 or total_msgs % 100 == 0:
                await emit_debug(
                    f"msg#{total_msgs} itx={'y' if itx else 'n'} "
                    f"finished={finished} len={len(display)}"
                )

            if updated and display and display != last_partial:
                last_partial = display
                await emit({"type": "transcript_partial", "text": display})

            if finished and display and display != finalized:
                finalized = display
                await emit({"type": "transcript_final", "text": display, "transcribe_ms": 0})

            tc = bool(getattr(sc, "turn_complete", False)) or tc_fb
            if tc:
                if display and display != finalized:
                    finalized = display
                    await emit({"type": "transcript_final", "text": display, "transcribe_ms": 0})
                vad_shared["active"] = False
                await emit_debug(f"Turn complete (total_msgs={total_msgs}).")
                # Reset per-turn state; do NOT break — keep the generator alive for next turn
                input_text = ""
                last_server = ""
                last_partial = ""
                finalized = ""

    except asyncio.CancelledError:
        pass
    except Exception as e:
        await emit({"type": "error", "text": f"Receive loop error: {e}"})
        stop.set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run():
    if genai is None:
        raise RuntimeError("google-genai not installed. pip install google-genai")

    reader, transport = await open_stdin_reader()

    raw = await reader.readline()
    if not raw:
        return
    try:
        msg = json.loads(raw.decode())
    except Exception as e:
        raise RuntimeError(f"Bad connect message: {e}")
    if msg.get("type") != "connect":
        raise RuntimeError(f"Expected connect, got {msg.get('type')}")

    # Resolve GOOGLE_API_KEY — env first, then .env file
    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        for d in [os.path.dirname(__file__), os.path.dirname(os.path.dirname(__file__))]:
            p = os.path.join(d, ".env")
            if os.path.isfile(p):
                for line in open(p):
                    line = line.strip()
                    if line.startswith("GOOGLE_API_KEY="):
                        os.environ["GOOGLE_API_KEY"] = line.split("=", 1)[1].strip()
                        break
            if os.environ.get("GOOGLE_API_KEY", "").strip():
                break

    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        raise RuntimeError("GOOGLE_API_KEY not set. Add it to TranscribeEval/.env")

    client = genai.Client()  # reads GOOGLE_API_KEY automatically
    await emit_debug(f"Connecting to Gemini Live (model={MODEL})…")

    queue: asyncio.Queue = asyncio.Queue()
    stop = asyncio.Event()
    vad_shared: dict = {"active": False}  # shared VAD state between forward_audio and receive_loop
    tasks = []

    try:
        async with client.aio.live.connect(model=MODEL, config=LIVE_CONFIG) as session:
            await emit_debug("Connected.")
            tasks = [
                asyncio.create_task(read_commands(reader, queue, stop)),
                asyncio.create_task(forward_audio(session, queue, stop, vad_shared)),
                asyncio.create_task(receive_loop(session, stop, vad_shared)),
            ]
            await stop.wait()
    finally:
        for t in tasks:
            t.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(*tasks, return_exceptions=True)
        transport.close()


def main():
    try:
        asyncio.run(run())
    except Exception as e:
        _emit({"type": "error", "text": str(e)})


if __name__ == "__main__":
    main()
