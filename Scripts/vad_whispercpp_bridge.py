#!/usr/bin/env python3
"""VAD + whisper.cpp (pywhispercpp) bridge."""
import re, sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
from vad_base import run_bridge, emit_debug, emit_error

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="small")
parser.add_argument("--language", default="en")
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()

MODEL = args.model
LANGUAGE = args.language
THREADS = args.threads
_model = None


def load():
    global _model
    try:
        from pywhispercpp.model import Model
        emit_debug(f"Loading whisper.cpp {MODEL} threads={THREADS}…")
        _model = Model(MODEL, n_threads=THREADS, language=LANGUAGE,
                       print_realtime=False, print_progress=False)
        emit_debug("whisper.cpp ready")
    except ImportError:
        emit_error("pywhispercpp not installed. pip install pywhispercpp")
    except Exception as e:
        emit_error(f"whisper.cpp load failed: {e}")


_HALLUCINATION = re.compile(r'\[.*?\]|\(.*?\)')


def _clean(text: str) -> str:
    """Strip whisper.cpp hallucination tokens like [BLANK_AUDIO] or (dramatic music)."""
    return " ".join(_HALLUCINATION.sub("", text).split())


def transcribe(samples):
    if _model is None:
        return ""
    try:
        import numpy as np
        audio = np.array(samples, dtype=np.float32)
        segs = _model.transcribe(audio)
        return _clean(" ".join(s.text.strip() for s in segs))
    except Exception as e:
        emit_error(f"whisper.cpp error: {e}")
        return ""


if __name__ == "__main__":
    run_bridge("whisper.cpp", load, transcribe, partial_interval=2.0)
