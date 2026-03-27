#!/usr/bin/env python3
"""VAD + WhisperX bridge."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
from vad_base import run_bridge, emit_debug, emit_error

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="small")
parser.add_argument("--language", default="en")
parser.add_argument("--device", default="cpu")
parser.add_argument("--compute-type", default="int8")
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()

MODEL = args.model
LANGUAGE = args.language
DEVICE = args.device
COMPUTE_TYPE = args.compute_type
BATCH_SIZE = args.batch_size
_model = None


def load():
    global _model
    try:
        import whisperx
        emit_debug(f"Loading whisperx {MODEL} device={DEVICE}…")
        _model = whisperx.load_model(MODEL, device=DEVICE, compute_type=COMPUTE_TYPE,
                                     language=LANGUAGE if LANGUAGE != "auto" else None)
        emit_debug("whisperx ready")
    except ImportError:
        emit_error("whisperx not installed. pip install whisperx")
    except Exception as e:
        emit_error(f"whisperx load failed: {e}")


def transcribe(samples):
    if _model is None:
        return ""
    try:
        import numpy as np, whisperx
        audio = np.array(samples, dtype=np.float32)
        r = _model.transcribe(audio, batch_size=BATCH_SIZE,
                              language=LANGUAGE if LANGUAGE != "auto" else None)
        return " ".join(s.get("text", "").strip() for s in r.get("segments", [])).strip()
    except Exception as e:
        emit_error(f"whisperx error: {e}")
        return ""


if __name__ == "__main__":
    run_bridge("whisperx", load, transcribe, partial_interval=2.0)
