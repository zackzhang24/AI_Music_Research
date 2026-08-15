#!/usr/bin/env python3
"""
calculate_threshold.py
──────────────────────
Calibrate the optimal Lcrosvila classification threshold on the ground-truth
datasets.

Pipeline
────────
  1. Instantiate the local Lcrosvila inference engine (LAION-CLAP embedding ->
     production RBF-SVM, .clap_svm_model_augmented.pkl).
  2. Memory-safe ingestion: stream each MP3 with pedalboard.io.AudioFile in
     10-second blocks (never loading a whole track into RAM):
         data/ground_truth_ai/     -> True_Label 1
         data/ground_truth_human/  -> True_Label 0
  3. Chunked inference: each 10 s block is resampled to CLAP's 48 kHz mono,
     scored by the engine, and the per-block P(AI) scores are AVERAGED into one
     track-level probability.
  4. Log to a pandas DataFrame (Filename, True_Label, Predicted_Score) and save
     results.csv (project root) incrementally so nothing is lost on a crash.
  5. roc_curve -> TPR/FPR -> Youden's J (TPR-FPR) -> optimal threshold.
  6. Print the optimal threshold + accuracy at that cutoff.
  7. Histogram of Human vs AI score distributions with the threshold line ->
     threshold_distribution.png.
"""

import os, sys, glob, pickle, warnings
import numpy as np
import pandas as pd
import soxr
from pedalboard.io import AudioFile

import matplotlib
matplotlib.use("Agg")                    # headless: write PNG, no display
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_streaming_evaluation as rse    # load_clap()

AI_DIR    = os.path.join(HERE, "data", "ground_truth_ai")
HUMAN_DIR = os.path.join(HERE, "data", "ground_truth_human")
MODEL_AUG = os.path.join(HERE, ".clap_svm_model_augmented.pkl")
MODEL_BASE = os.path.join(HERE, ".clap_svm_model.pkl")

RESULTS_CSV = os.path.join(ROOT, "results.csv")
PLOT_PNG    = os.path.join(ROOT, "threshold_distribution.png")

CLAP_SR    = 48000
BLOCK_SECS = 10
MIN_BLOCK_SECS = 0.5                      # ignore a tiny trailing remainder


# ── Lcrosvila inference engine ─────────────────────────────────────────────────
class LcrosvilaEngine:
    """CLAP embedding -> trained RBF-SVM; scores a single audio block -> P(AI)."""

    def __init__(self):
        self.clap = rse.load_clap()
        model_path = MODEL_AUG if os.path.exists(MODEL_AUG) else MODEL_BASE
        with open(model_path, "rb") as fh:
            saved = pickle.load(fh)
        self.scaler = saved["scaler"]
        self.svm    = saved["svm"]
        self.ai_idx = list(self.svm.classes_).index(1)
        print(f"Lcrosvila engine ready (model: {os.path.basename(model_path)}, "
              f"kernel={self.svm.kernel}, C={self.svm.C})")

    def score_block(self, mono_48k: np.ndarray) -> float:
        """mono_48k: 1-D float32 @48 kHz (<=10 s). Returns P(AI) in [0,1]."""
        emb = self.clap.model.get_audio_embedding_from_data(
            x=[mono_48k.astype(np.float32)], use_tensor=False)
        emb = np.asarray(emb, dtype=np.float32)
        return float(self.svm.predict_proba(self.scaler.transform(emb))[0, self.ai_idx])


def score_track(engine: LcrosvilaEngine, path: str) -> float:
    """Stream `path` in 10 s blocks, score each, average -> track P(AI)."""
    block_scores = []
    with AudioFile(path) as af:
        sr = int(af.samplerate)
        block_frames = int(sr * BLOCK_SECS)
        min_frames   = int(sr * MIN_BLOCK_SECS)
        while af.tell() < af.frames:
            block = af.read(block_frames)          # (channels, frames) float32
            if block.shape[-1] < min_frames:
                break
            mono = block.mean(axis=0) if block.ndim == 2 else block   # downmix -> mono
            if sr != CLAP_SR:                                          # -> CLAP's 48 kHz
                mono = soxr.resample(mono, sr, CLAP_SR).astype(np.float32)
            block_scores.append(engine.score_block(mono))
    return float(np.mean(block_scores)) if block_scores else float("nan")


def main():
    # ── 1. Model init ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("Lcrosvila threshold calibration")
    print("=" * 70)
    engine = LcrosvilaEngine()

    # ── 2. Gather ground-truth files ───────────────────────────────────────────
    ai_files    = sorted(glob.glob(os.path.join(AI_DIR, "*.mp3")))
    human_files = sorted(glob.glob(os.path.join(HUMAN_DIR, "*.mp3")))
    dataset = [(f, 1) for f in ai_files] + [(f, 0) for f in human_files]
    total = len(dataset)
    print(f"\nGround truth: {len(ai_files)} AI (label 1) + {len(human_files)} Human (label 0) "
          f"= {total} tracks\n")

    # ── 3. Chunked inference + incremental logging ─────────────────────────────
    rows = []
    for i, (path, label) in enumerate(dataset, 1):
        fname = os.path.basename(path)
        print(f"Processing {i}/{total}: {fname}", flush=True)
        try:
            score = score_track(engine, path)
        except Exception as exc:
            print(f"   [error] {fname}: {exc}", file=sys.stderr)
            score = float("nan")
        rows.append({"Filename": fname, "True_Label": label, "Predicted_Score": score})
        # Save immediately so no data is lost mid-run.
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(rows)
    df_valid = df.dropna(subset=["Predicted_Score"]).reset_index(drop=True)
    print(f"\nSaved {len(df)} rows -> {RESULTS_CSV}  ({len(df)-len(df_valid)} failed)")

    # ── 4/5. ROC -> Youden's J -> optimal threshold ───────────────────────────
    y_true = df_valid["True_Label"].values
    scores = df_valid["Predicted_Score"].values
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    opt_threshold = float(thresholds[best_idx])
    # roc_curve sets thresholds[0] above every score (an artificial "predict all
    # negative" point); guard so the reported cutoff stays in a sane [0,1] range.
    if not np.isfinite(opt_threshold) or opt_threshold > 1.0:
        opt_threshold = min(1.0, float(np.max(scores)))

    preds = (scores >= opt_threshold).astype(int)
    accuracy = float((preds == y_true).mean())

    # ── 6. Console summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLD (Youden's J = max(TPR - FPR))")
    print("=" * 70)
    print(f"  Optimal probability threshold : {opt_threshold:.4f}")
    print(f"  Youden's J at that point      : {youden_j[best_idx]:.4f} "
          f"(TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})")
    print(f"  Overall accuracy at cutoff    : {accuracy*100:.2f}%  "
          f"({int((preds==y_true).sum())}/{len(y_true)})")
    hum = scores[y_true == 0]; ai = scores[y_true == 1]
    print(f"  Human score mean/median       : {hum.mean():.3f} / {np.median(hum):.3f}")
    print(f"  AI    score mean/median       : {ai.mean():.3f} / {np.median(ai):.3f}")

    # ── 7. Histogram with threshold line ───────────────────────────────────────
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 26)
    plt.hist(hum, bins=bins, alpha=0.6, color="#2ca02c", edgecolor="white", label=f"Human (n={len(hum)})")
    plt.hist(ai,  bins=bins, alpha=0.6, color="#d62728", edgecolor="white", label=f"AI (n={len(ai)})")
    plt.axvline(opt_threshold, color="black", linestyle="--", linewidth=2.5,
                label=f"Optimal threshold = {opt_threshold:.3f}")
    plt.xlabel("Predicted P(AI)  (Lcrosvila)")
    plt.ylabel("Number of tracks")
    plt.title(f"Lcrosvila score distribution — Human vs AI\n"
              f"Youden-J optimal cutoff {opt_threshold:.3f}  |  accuracy {accuracy*100:.1f}%")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=120)
    print(f"\nSaved distribution plot -> {PLOT_PNG}")
    print("=" * 70)


if __name__ == "__main__":
    main()
