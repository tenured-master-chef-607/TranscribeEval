#!/usr/bin/env python3
"""
faster-whisper with sliding window partials.

Unlike VAD+batch, this runs inference every 0.5s on accumulated speech audio,
giving partial results while the person is still talking.
Uses beam_size=1 (greedy) for speed.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
from vad_base import run_bridge, emit_debug, emit_error

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="tiny.en")
parser.add_argument("--language", default="en")
args = parser.parse_args()

_model = None


def load():
    global _model
    try:
        from faster_whisper import WhisperModel
        emit_debug(f"Loading faster-whisper {args.model}…")
        _model = WhisperModel(args.model, device="cpu", compute_type="int8")
        emit_debug("faster-whisper ready")
    except ImportError:
        emit_error("faster-whisper not installed. pip install faster-whisper")
    except Exception as e:
        emit_error(f"Load failed: {e}")


def transcribe(samples):
    if _model is None:
        return ""
    try:
        import numpy as np
        audio = np.array(samples, dtype=np.float32)
        lang = args.language if args.language != "auto" else None
        # beam_size=1 = greedy decoding, fastest
        segs, _ = _model.transcribe(audio, language=lang, beam_size=1,
                                    vad_filter=False, word_timestamps=False)
        return " ".join(s.text for s in segs).strip()
    except Exception as e:
        emit_error(f"Transcription error: {e}")
        return ""


if __name__ == "__main__":
    # partial_interval=0.5 → inference runs every 0.5s during speech
    run_bridge("faster-whisper", load, transcribe, partial_interval=0.5)
