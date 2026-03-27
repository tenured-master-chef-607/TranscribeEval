#!/usr/bin/env python3
"""VAD + mlx-whisper / faster-whisper / openai-whisper bridge."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
from vad_base import run_bridge, emit_debug, emit_error

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="small")
parser.add_argument("--language", default="en")
args = parser.parse_args()

MODEL = args.model
LANGUAGE = args.language
_backend = None
_model = None


def load():
    global _backend, _model
    try:
        import mlx_whisper  # noqa
        _backend = "mlx_whisper"
        emit_debug(f"mlx-whisper ready (model={MODEL})")
        return
    except ImportError:
        pass
    try:
        from faster_whisper import WhisperModel
        emit_debug(f"Loading faster-whisper {MODEL}…")
        _model = WhisperModel(MODEL, device="cpu", compute_type="int8")
        _backend = "faster_whisper"
        emit_debug("faster-whisper ready")
        return
    except ImportError:
        pass
    try:
        import whisper
        emit_debug(f"Loading openai-whisper {MODEL}…")
        _model = whisper.load_model(MODEL)
        _backend = "openai_whisper"
        emit_debug("openai-whisper ready")
        return
    except ImportError:
        pass
    emit_error("No whisper backend found. pip install mlx-whisper OR faster-whisper OR openai-whisper")


def transcribe(samples):
    if _backend is None:
        return ""
    import numpy as np
    audio = np.array(samples, dtype=np.float32)
    lang = LANGUAGE if LANGUAGE != "auto" else None
    if _backend == "mlx_whisper":
        import mlx_whisper
        # tiny.en → mlx-community/whisper-tiny.en-mlx, tiny → mlx-community/whisper-tiny-mlx
        repo = f"mlx-community/whisper-{MODEL}-mlx"
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=repo,
                                   language=lang, verbose=False)
        return r.get("text", "").strip()
    if _backend == "faster_whisper":
        segs, _ = _model.transcribe(audio, language=lang, beam_size=5)
        return " ".join(s.text for s in segs).strip()
    if _backend == "openai_whisper":
        r = _model.transcribe(audio, language=lang, fp16=False)
        return r.get("text", "").strip()
    return ""


if __name__ == "__main__":
    run_bridge("mlx-whisper", load, transcribe, partial_interval=1.0)
