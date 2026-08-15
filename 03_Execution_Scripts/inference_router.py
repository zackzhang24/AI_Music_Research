#!/usr/bin/env python3
"""
inference_router.py
───────────────────
Three-zone confidence router for YouTube-sourced tracks.

Routing rule (HARDCODED)
────────────────────────
  A file flagged as a YouTube download is routed EXCLUSIVELY through Lcrosvila
  (LAION-CLAP embedding -> production RBF-SVM).  Mippia is intentionally not
  consulted on this path — its MERT front-end is unreliable on YouTube's lossy
  AAC/Opus re-encode, so the calibrated Lcrosvila score is authoritative.

Decision matrix (calibrated thresholds)
───────────────────────────────────────
       P(AI) < 0.525          -> CONFIRMED_HUMAN     (High Confidence)
  0.525 <= P(AI) <= 0.851     -> AMBIGUOUS_OVERLAP   (Triggers Late-Fusion Flag)
       P(AI) > 0.851          -> CONFIRMED_AI        (Absolute Confidence)

Scoring
───────
  The MP3 is streamed with pedalboard.io.AudioFile in 10-second chunks; each
  chunk is downmixed to mono, resampled to CLAP's 48 kHz, embedded + scored, and
  the per-chunk P(AI) values are AVERAGED into the final track probability.

Output
──────
  A single clean JSON payload is written to STDOUT (all model-loading / progress
  noise is redirected to STDERR) so a downstream Late-Fusion text-parser can
  ingest stdout verbatim.  Exact schema:

    {
      "file_path": "path/to/file.mp3",
      "routing_path": "YouTube (100% Lcrosvila)",
      "raw_probability": 0.724,
      "acoustic_verdict": "AMBIGUOUS_OVERLAP",
      "requires_fusion_processing": true
    }

Usage:
  python3 inference_router.py <audio.mp3> [--source youtube] [--model <pkl>]
"""

import os, sys, json, pickle, argparse, contextlib
import numpy as np
import soxr
from pedalboard.io import AudioFile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ── Calibrated three-zone thresholds ───────────────────────────────────────────
HUMAN_MAX = 0.525      # P(AI) strictly below this  -> CONFIRMED_HUMAN
AI_MIN    = 0.851      # P(AI) strictly above this  -> CONFIRMED_AI
# (the inclusive band [HUMAN_MAX, AI_MIN] is the AMBIGUOUS_OVERLAP zone)

CLAP_SR        = 48000
BLOCK_SECS     = 10
MIN_BLOCK_SECS = 0.5

MODEL_AUG  = os.path.join(HERE, ".clap_svm_model_augmented.pkl")
MODEL_BASE = os.path.join(HERE, ".clap_svm_model.pkl")


# ── Lcrosvila inference engine (block -> P(AI)) ────────────────────────────────
class LcrosvilaEngine:
    def __init__(self, model_path=None):
        import run_streaming_evaluation as rse       # load_clap()
        self.clap = rse.load_clap()
        path = model_path or (MODEL_AUG if os.path.exists(MODEL_AUG) else MODEL_BASE)
        with open(path, "rb") as fh:
            saved = pickle.load(fh)
        self.scaler = saved["scaler"]
        self.svm    = saved["svm"]
        self.ai_idx = list(self.svm.classes_).index(1)
        self.model_name = os.path.basename(path)

    def score_block(self, mono_48k: np.ndarray) -> float:
        emb = self.clap.model.get_audio_embedding_from_data(
            x=[mono_48k.astype(np.float32)], use_tensor=False)
        emb = np.asarray(emb, dtype=np.float32)
        return float(self.svm.predict_proba(self.scaler.transform(emb))[0, self.ai_idx])


def score_track(engine: LcrosvilaEngine, path: str):
    """Stream `path` in 10 s chunks (mono @48 kHz), score each, average -> P(AI)."""
    scores = []
    with AudioFile(path) as af:
        sr = int(af.samplerate)
        block_frames = int(sr * BLOCK_SECS)
        min_frames   = int(sr * MIN_BLOCK_SECS)
        while af.tell() < af.frames:
            block = af.read(block_frames)                     # (channels, frames)
            if block.shape[-1] < min_frames:
                break
            mono = block.mean(axis=0) if block.ndim == 2 else block   # downmix
            if sr != CLAP_SR:
                mono = soxr.resample(mono, sr, CLAP_SR).astype(np.float32)  # -> 48 kHz
            scores.append(engine.score_block(mono))
    if not scores:
        return None, 0
    return float(np.mean(scores)), len(scores)


# ── Three-zone classification ──────────────────────────────────────────────────
def classify(p_ai: float):
    """Return (acoustic_verdict, requires_fusion_processing)."""
    if p_ai < HUMAN_MAX:                         # < 0.525  -> High Confidence human
        return "CONFIRMED_HUMAN", False
    elif p_ai <= AI_MIN:                         # 0.525..0.851 -> ambiguous overlap
        return "AMBIGUOUS_OVERLAP", True         #   -> triggers Late-Fusion
    else:                                        # > 0.851  -> Absolute Confidence AI
        return "CONFIRMED_AI", False


def main():
    ap = argparse.ArgumentParser(description="Three-zone YouTube-route confidence router (Lcrosvila).")
    ap.add_argument("audio", help="path to the target MP3")
    ap.add_argument("--source", default="youtube",
                    help="acoustic source flag; 'youtube' routes exclusively through Lcrosvila")
    ap.add_argument("--model", default=None, help="override SVM pickle path")
    args = ap.parse_args()

    real_stdout = sys.stdout            # keep a handle to the *real* stdout for the JSON

    # HARDCODED routing rule: a YouTube download goes 100% through Lcrosvila.
    if args.source.lower() == "youtube":
        routing_path = "YouTube (100% Lcrosvila)"
    else:
        routing_path = f"{args.source} (router only implements the YouTube->Lcrosvila path)"

    # Payload shaped EXACTLY to the Late-Fusion parser's schema.
    payload = {
        "file_path": args.audio,
        "routing_path": routing_path,
        "raw_probability": None,
        "acoustic_verdict": None,
        "requires_fusion_processing": False,
    }

    try:
        if not os.path.exists(args.audio):
            raise FileNotFoundError(args.audio)

        # All model-loading / progress noise -> STDERR so STDOUT stays pure JSON.
        with contextlib.redirect_stdout(sys.stderr):
            print("[router] loading Lcrosvila engine…", flush=True)
            engine = LcrosvilaEngine(args.model)
            print(f"[router] streaming + scoring {os.path.basename(args.audio)}…", flush=True)
            p_ai, n_chunks = score_track(engine, args.audio)

        if p_ai is None:
            raise RuntimeError("no audio chunks could be scored")

        verdict, requires_fusion = classify(p_ai)
        payload["raw_probability"] = round(p_ai, 3)
        payload["acoustic_verdict"] = verdict
        payload["requires_fusion_processing"] = requires_fusion

    except Exception as exc:
        payload["acoustic_verdict"] = "ERROR"
        payload["error"] = f"{type(exc).__name__}: {exc}"

    # THE ONLY thing written to real stdout: the clean JSON payload.
    print(json.dumps(payload, indent=2), file=real_stdout)


if __name__ == "__main__":
    main()
