#!/usr/bin/env python3
"""
train_augmented_classifier.py
──────────────────────────────
Retrain the Lcrosvila (CLAP + SVM) classifier on the augmented mixed
distribution (Clean + AAC + Opus), using GridSearchCV to pick the best
kernel/C, and report a final validation accuracy matrix broken down by codec
for BOTH detectors.

Why Lcrosvila and not "Mippia"
──────────────────────────────
Mippia (MERT-AudioCAT -> FusionSegmentTransformer) is a frozen, pretrained
deep model loaded from two .ckpt checkpoints — it has no SVM and this repo
has no training pipeline for it (no loss/optimizer/label infrastructure,
only inference via load_from_checkpoint).  The classifier we actually own,
train, and can run GridSearchCV over is Lcrosvila's CLAP-embedding SVM. This
script retrains THAT classifier on the augmented data.  Mippia is evaluated
(frozen, unmodified) on the same held-out validation set purely to populate
its row of the final Clean-vs-Compressed accuracy matrix.

Methodology
───────────
  1. Read train_augmented_manifest.csv (1200 rows: 400 tracks x {Clean,AAC,Opus}).
  2. Track-level 80/20 train/val split — all 3 codec copies of a given
     original track go entirely into ONE split, so the SVM is never
     validated on a codec-variant of a track it trained on (no leakage).
  3. Compute CLAP embeddings for every file (cached) using the existing
     2-minute chunked + mean-pooled, deterministic clap_embed().
  4. GridSearchCV over kernel in {linear, rbf} and C in {0.1, 1, 10, 100}
     (StandardScaler + SVC, 5-fold CV on the training embeddings).
  5. Evaluate the best estimator AND frozen Mippia on the held-out
     validation tracks, broken down by codec (Clean / AAC / Opus).
  6. Print the final accuracy matrix.

Output artifacts (kept separate from the production app.py cache so this
experiment never silently swaps the live model):
  .clap_train_embeddings_augmented.npz   — cached embeddings for all 1200 files
  .clap_svm_model_augmented.pkl          — best GridSearchCV (scaler, svm)
  augmented_validation_results.csv       — per-file predictions for the val set
"""

import os, sys, csv, glob, gc, pickle, tempfile, random, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_streaming_evaluation as rse           # to_wav, clap_embed, mippia_pai, load_*

MANIFEST   = os.path.join(HERE, "train_augmented_manifest.csv")
DATA_DIR   = os.path.join(HERE, "data", "train_augmented")
EMB_CACHE  = os.path.join(HERE, ".clap_train_embeddings_augmented.npz")
SVM_CACHE  = os.path.join(HERE, ".clap_svm_model_augmented.pkl")
VAL_CSV    = os.path.join(HERE, "augmented_validation_results.csv")

VAL_FRACTION = 0.20
RANDOM_SEED  = 42
CODECS       = ["Clean", "AAC", "Opus"]


def log(msg): print(msg, flush=True)
def err(msg): print(msg, file=sys.stderr, flush=True)


def track_level_split(manifest: pd.DataFrame):
    """Split by Track_ID so all codec variants of a track stay in one split."""
    track_ids = sorted(manifest["Track_ID"].unique())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(track_ids)
    n_val = max(1, int(len(track_ids) * VAL_FRACTION))
    val_ids = set(track_ids[:n_val])
    train_ids = set(track_ids[n_val:])
    return train_ids, val_ids


def compute_embeddings(manifest: pd.DataFrame, clap):
    """Compute (or load cached) CLAP embeddings for every manifest row."""
    if os.path.exists(EMB_CACHE):
        log(f"  Loading cached embeddings: {EMB_CACHE}")
        z = np.load(EMB_CACHE, allow_pickle=True)
        return z["X"], z["y"], list(z["filenames"])

    log(f"  Computing CLAP embeddings for {len(manifest)} augmented files (one-time)…")
    X_list, y_list, fname_list = [], [], []
    tmp = tempfile.mkdtemp(prefix="aug_embed_")
    for i, row in enumerate(manifest.itertuples(index=False), 1):
        src = os.path.join(DATA_DIR, row.Label, row.Filename)
        w = os.path.join(tmp, "n.wav")
        try:
            rse.to_wav(src, w)
            X_list.append(rse.clap_embed(w, clap))
            y_list.append(1 if row.Label == "AI" else 0)
            fname_list.append(row.Filename)
        except Exception as exc:
            err(f"   [embed skip] {row.Filename}: {exc}")
        finally:
            if os.path.exists(w):
                os.remove(w)
        if i % 100 == 0 or i == len(manifest):
            log(f"    embeddings: {i}/{len(manifest)}")
    X = np.stack(X_list); y = np.array(y_list)
    np.savez_compressed(EMB_CACHE, X=X, y=y, filenames=np.array(fname_list, dtype=object))
    log(f"  Cached {len(y)} embeddings -> {EMB_CACHE}")
    return X, y, fname_list


def main():
    if not os.path.exists(MANIFEST):
        err(f"Manifest not found: {MANIFEST}. Run augment_training_data.py first.")
        sys.exit(1)

    manifest = pd.read_csv(MANIFEST)
    log("=" * 70)
    log(f"Augmented dataset: {len(manifest)} files "
        f"({manifest['Track_ID'].nunique()} tracks x {len(CODECS)} codecs)")
    log(f"  Label distribution : {dict(manifest['Label'].value_counts())}")
    log(f"  Codec distribution : {dict(manifest['Codec'].value_counts())}")
    log("=" * 70)

    train_ids, val_ids = track_level_split(manifest)
    log(f"\nTrack-level split: {len(train_ids)} train tracks / {len(val_ids)} val tracks "
        f"(no codec-variant leakage across the split)")

    # ── Load models ────────────────────────────────────────────────────────────
    log("\nLoading detectors…")
    clap   = rse.load_clap()
    s1, s2 = rse.load_mippia()
    log("Detectors ready.\n")

    # ── Embeddings (cached) ────────────────────────────────────────────────────
    X, y, filenames = compute_embeddings(manifest, clap)
    fname_to_idx = {f: i for i, f in enumerate(filenames)}

    train_mask = manifest["Track_ID"].isin(train_ids).values
    val_mask   = manifest["Track_ID"].isin(val_ids).values
    # Align manifest rows to the embedding array via filename (embeddings may have
    # skipped a few rows on extraction failure, so don't assume positional alignment)
    man_idx_by_fname = {f: i for i, f in manifest["Filename"].items()}

    train_rows = manifest[train_mask]
    val_rows   = manifest[val_mask]

    def gather(rows):
        idxs, labels = [], []
        for r in rows.itertuples(index=False):
            if r.Filename in fname_to_idx:
                idxs.append(fname_to_idx[r.Filename])
                labels.append(1 if r.Label == "AI" else 0)
        return idxs, labels

    train_idx, train_y = gather(train_rows)
    val_idx, val_y     = gather(val_rows)
    X_train, y_train = X[train_idx], np.array(train_y)
    X_val,   y_val   = X[val_idx],   np.array(val_y)
    log(f"Training embeddings: {len(X_train)}  |  Validation embeddings: {len(X_val)}")

    # ── GridSearchCV: kernel x C ──────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)

    param_grid = {
        "kernel": ["linear", "rbf"],
        "C": [0.1, 1, 10, 100],
    }
    log("\nRunning GridSearchCV (kernel x C, 5-fold CV)…")
    grid = GridSearchCV(
        SVC(probability=True, random_state=RANDOM_SEED),
        param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1,
    )
    grid.fit(X_train_sc, y_train)
    best_svm = grid.best_estimator_
    log(f"Best params: {grid.best_params_}  (CV accuracy={grid.best_score_:.4f})")

    with open(SVM_CACHE, "wb") as fh:
        pickle.dump({"scaler": scaler, "svm": best_svm, "grid_results": grid.cv_results_,
                    "best_params": grid.best_params_}, fh)
    log(f"Saved best (scaler, svm) -> {SVM_CACHE}")

    # ── Lcrosvila predictions on held-out validation set ──────────────────────
    ai_idx = list(best_svm.classes_).index(1)
    X_val_sc = scaler.transform(X_val)
    lc_proba = best_svm.predict_proba(X_val_sc)[:, ai_idx]
    lc_pred  = (lc_proba >= 0.5).astype(int)

    # ── Mippia predictions on the SAME held-out validation files (frozen, no retrain) ──
    log("\nEvaluating frozen Mippia on the held-out validation set (no retraining)…")
    mp_proba = np.full(len(val_rows), np.nan)
    val_rows_list = list(val_rows.itertuples(index=False))
    tmp = tempfile.mkdtemp(prefix="aug_mippia_")
    for i, r in enumerate(val_rows_list):
        src = os.path.join(DATA_DIR, r.Label, r.Filename)
        w = os.path.join(tmp, "n.wav")
        try:
            rse.to_wav(src, w)
            mp_proba[i] = rse.mippia_pai(w, s1, s2)
        except Exception as exc:
            err(f"   [Mippia skip] {r.Filename}: {exc}")
        finally:
            if os.path.exists(w):
                os.remove(w)
        gc.collect()
        if rse.DEVICE == "mps":
            torch.mps.empty_cache()
        if (i + 1) % 25 == 0 or (i + 1) == len(val_rows_list):
            log(f"    Mippia val progress: {i+1}/{len(val_rows_list)}")

    # ── Assemble per-file validation results ──────────────────────────────────
    val_out = val_rows.reset_index(drop=True).copy()
    val_out["Lcrosvila_P_AI"]      = [lc_proba[i] for i in range(len(val_out))]
    val_out["Lcrosvila_Pred"]      = ["AI" if p >= 0.5 else "Human" for p in val_out["Lcrosvila_P_AI"]]
    val_out["Lcrosvila_Correct"]   = val_out["Lcrosvila_Pred"] == val_out["Label"]
    val_out["Mippia_P_AI"]         = mp_proba
    val_out["Mippia_Pred"]         = ["AI" if p >= 0.5 else ("Human" if not np.isnan(p) else "ERROR")
                                      for p in val_out["Mippia_P_AI"]]
    val_out["Mippia_Correct"]      = val_out["Mippia_Pred"] == val_out["Label"]
    val_out.to_csv(VAL_CSV, index=False)
    log(f"\nSaved per-file validation results -> {VAL_CSV}")

    # ── Final accuracy matrix: rows=detector, cols=codec ───────────────────────
    log("\n" + "=" * 70)
    log("FINAL VALIDATION ACCURACY MATRIX  (held-out tracks, never seen in training)")
    log("=" * 70)
    header = f"{'Detector':<12s}" + "".join(f"{c:>12s}" for c in CODECS) + f"{'Overall':>12s}"
    log(header)
    for det, col in [("Lcrosvila", "Lcrosvila_Correct"), ("Mippia", "Mippia_Correct")]:
        cells = []
        for codec in CODECS:
            sub = val_out[val_out["Codec"] == codec]
            sub = sub[sub[col].notna()] if det == "Mippia" else sub
            acc = sub[col].mean() if len(sub) else float("nan")
            cells.append(f"{acc*100:>11.1f}%" if not np.isnan(acc) else f"{'n/a':>12s}")
        overall = val_out[col].mean()
        log(f"{det:<12s}" + "".join(cells) + f"{overall*100:>11.1f}%")

    log("\nNote: Lcrosvila was GridSearch-retrained on the mixed Clean+AAC+Opus")
    log("training set above. Mippia is the frozen pretrained model (unchanged) —")
    log("evaluated, not retrained, since this repo has no Mippia training pipeline.")
    log("=" * 70)


if __name__ == "__main__":
    main()
