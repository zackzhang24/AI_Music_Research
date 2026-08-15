#!/usr/bin/env python3
"""
run_streaming_evaluation.py
───────────────────────────
Run the two benchmark detectors over the audio ingested by
universal_audio_ingestion.py (the .wav files in url_test_downloads/).

Pipeline
────────
  1. Read ingestion_test_log.csv, keep rows whose local .wav actually exists.
  2. Load the SAME two detectors as run_benchmark_evaluation.py:
        • Mippia    — MERT-AudioCAT (Stage-1) → FusionSegmentTransformer
                      (Stage-2), pretrained checkpoints → per-track P(AI).
        • Lcrosvila — LAION-CLAP embedding → RBF-SVM.
     Because the streaming tracks are UNLABELED (so the benchmark's
     leave-one-out scheme cannot apply), the SVM is trained once on the 400
     labeled benchmark tracks in 01_Final_Datasets/ (Human vs AI) and then used
     to predict the streaming tracks.  Training embeddings are cached.
  3. Resample every file to each model's expected rate (CLAP 48 kHz; Mippia
     resamples to 24 kHz internally) via an ffmpeg transcode.
  4. Emit streaming_evaluation_results.csv merging the original metadata with
     soft probabilities + a 0.5-threshold binary Prediction for both models.

Robustness
──────────
Each model's inference is wrapped in its own try/except: an OOM
(RuntimeError) or tensor-size mismatch on one file/model is logged into the
CSV and the pipeline moves on.
"""

import os, sys, gc, glob, csv, json, subprocess, tempfile, warnings, traceback
from urllib.parse import urlparse
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Paths (post-reorganization) ────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))                 # 03_Execution_Scripts
ROOT      = os.path.dirname(HERE)
ARCHIVE   = os.path.join(ROOT, "04_Archive_Phase_1")
FST_DIR   = os.path.join(ARCHIVE, "FST-AI-Music-Detection")
CLAP_DIR  = os.path.join(ARCHIVE, "ai-music-detection")
CKPT_S1   = os.path.join(ARCHIVE, "Misc", "checkpoints", "mippia", "stage1_mert_audiocat.ckpt")
CKPT_S2   = os.path.join(ARCHIVE, "Misc", "checkpoints", "mippia", "stage2_fusion.ckpt")
CKPT_CLAP = os.path.join(ARCHIVE, "Misc", "checkpoints", "clap", "music_audioset_epoch_15_esc_90.14.pt")

DATASETS  = os.path.join(ROOT, "01_Final_Datasets")
TRAIN_DIRS = [                                                          # (dir, label)
    ("human_tracks_120s", "Human"), ("human_tracks_10s", "Human"),
    ("ai_tracks_120s", "AI"),       ("new_ai_tracks_10s", "AI"),
]

DL_DIR    = os.path.join(HERE, "url_test_downloads")
LOG_CSV   = os.path.join(HERE, "ingestion_test_log.csv")
OUT_CSV   = os.path.join(HERE, "streaming_evaluation_results.csv")
# v2 = deterministic 45-55 s CLAP embeddings; training & inference share the
# identical deterministic feature path (rebuilt whenever the chunking changes).
TRAIN_CACHE = os.path.join(HERE, ".clap_train_embeddings_det_v2.npz")
# Persisted (StandardScaler, SVM) so inference applies the EXACT fitted scaler.
SVM_MODEL   = os.path.join(HERE, ".clap_svm_model.pkl")

sys.path.insert(0, FST_DIR)
sys.path.insert(0, CLAP_DIR)
sys.path.insert(0, os.path.join(CLAP_DIR, "utils"))

import static_ffmpeg; static_ffmpeg.add_paths()

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
BEAT_DEVICE   = "cpu"
ANALYZE_CAP_S = 300
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

# ── Per-model sample-rate routing (the two detectors need DIFFERENT rates) ──────
#   CLAP (Lcrosvila)  → 48 kHz  (CLAP_SR, set in the LCROSVILA section below)
#   MERT (Mippia S-1) → 24 kHz  — m-a-p/MERT-v1-95M is a 24 kHz model; 48 kHz audio
#   MUST be downsampled before it ever reaches MERT, or the spectrogram/positional
#   encodings desync and the embeddings are garbage.
MERT_SR = 24000


def log(msg): print(msg, flush=True)
def err(msg): print(msg, file=sys.stderr, flush=True)


# ── Robust ingestion → temp wav (48 kHz stereo) ───────────────────────────────
def to_wav(src, dst, cap_s=ANALYZE_CAP_S):
    subprocess.check_call(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-t", str(cap_s),
         "-i", src, "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", dst])


# ═══════════════════════════ MIPPIA (identical to benchmark) ══════════════════
def _load_audio_beats_patched(audio_path, sr=MERT_SR):
    """Load `audio_path` and return MERT-ready segments at MERT's rate (`sr`=24 kHz).
    The source wav is 48 kHz (CLAP's rate); it is EXPLICITLY downsampled to `sr`
    here so MERT never sees 48 kHz."""
    import soundfile as sf, soxr
    from preprocess import get_segments_from_wav, find_optimal_segment_length
    _, downbeats = get_segments_from_wav(audio_path, device=BEAT_DEVICE)
    _, cleaned = find_optimal_segment_length(downbeats)
    wav, srate = sf.read(audio_path, always_2d=True)
    wav = wav.T.astype(np.float32)
    if srate != sr:                                    # 48 kHz (or any rate) → 24 kHz for MERT
        wav = soxr.resample(wav.T, srate, sr).T.astype(np.float32)
    # Invariant: from here on every sample index uses `sr` (= MERT_SR = 24 kHz).
    waveform = torch.from_numpy(wav).to(torch.float32)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    fixed = 240_000
    real_len = waveform.shape[1]                        # length of REAL audio (before any pad)
    if real_len < fixed:                               # pad ONLY enough for a single window
        waveform = torch.cat(
            [waveform, torch.zeros(1, fixed - real_len, dtype=torch.float32)], dim=1)
    max_start = max(0, real_len - fixed)               # latest start that stays inside real audio

    # BUGFIX: clamp every beat-anchored window so the 10 s slice never runs into the
    # trailing zero-padding.  Previously a late downbeat (common on short clips) placed
    # the window in silence → MERT saw zeros → logit floored → static P(AI)=0.011.
    starts = []
    for st in cleaned:
        s = min(max(0, int(st * sr)), max_start)
        if s not in starts:
            starts.append(s)
        if len(starts) >= 48:
            break
    # Fallback: no usable downbeats → evenly-spaced REAL-audio windows (never silence).
    if not starts:
        s = 0
        while s <= max_start and len(starts) < 48:
            starts.append(s); s += fixed
        if not starts:
            starts = [0]

    segs = [torch.tensor(waveform[:, s:s + fixed].squeeze().numpy(),
                         dtype=torch.float32).unsqueeze(0) for s in starts]
    stacked = torch.stack(segs); n = stacked.shape[0]
    mask = torch.zeros(48, dtype=torch.bool)
    if n < 48:
        stacked = torch.cat([stacked, torch.zeros((48 - n, 1, fixed), dtype=torch.float32)], dim=0)
        mask[n:] = True
    return stacked, mask


def _scaled_sigmoid(x):
    return torch.clamp(torch.sigmoid(x), min=0.011, max=0.989)


def load_mippia():
    from model import MERT_AudioCAT, MusicAudioClassifier
    s1 = MERT_AudioCAT.load_from_checkpoint(CKPT_S1).to(DEVICE); s1.eval()
    s2 = MusicAudioClassifier.load_from_checkpoint(
        checkpoint_path=CKPT_S2, input_dim=768,
        backbone="fusion_segment_transformer", is_emb=True).to(DEVICE)
    s2.eval()
    return s1, s2


def mippia_pai(wav_path, s1, s2):
    segs, mask = _load_audio_beats_patched(wav_path, sr=MERT_SR)   # 24 kHz for MERT
    segs = segs.to(DEVICE).to(torch.float32)
    mask = mask.to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        _logit, emb = s1(segs.squeeze(1))
        s2h = s2.half()
        emb_in = emb.unsqueeze(0)
        if emb_in.shape[1] == 1:
            emb_in = emb_in[:, 0, :].unsqueeze(0)
        raw = s2h(emb_in.to(DEVICE), mask.to(DEVICE))
    p_ai = float(_scaled_sigmoid(raw.squeeze().float()).item())
    del segs, mask, _logit, emb, emb_in, raw
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return p_ai


# ═══════════════════════════ LCROSVILA (CLAP + SVM) ═══════════════════════════
CLAP_SR         = 48000          # CLAP's native sample rate
CLAP_CHUNK      = 480000         # 10 s chunk CLAP operates on (48000 * 10)
CLAP_MAX_INTAKE = 120 * 48000    # ingest up to the first 2 minutes (5,760,000 samples)


def load_clap():
    from model_loader import CLAPMusic
    clap = CLAPMusic(model_file=CKPT_CLAP); clap.load_model()
    clap.model.eval()           # (1) explicit eval — Dropout / BatchNorm deactivated
    return clap


def clap_embed(wav_path, clap):
    """
    DETERMINISTIC 2-minute chunked + mean-pooled CLAP embedding.

    A single 10 s window is too easily skewed by one song section (intro vs drop),
    so we summarise up to the first 2 minutes of the track:
      • librosa.load @48 kHz mono — deterministic decode/resample.
      • take the first 120 s, slice into consecutive NON-overlapping 10 s chunks.
      • embed each full 10 s chunk independently via get_audio_embedding_from_data
        (len == chunk so CLAP never invokes np.random -> bit-for-bit reproducible).
      • mean-pool all chunk embeddings -> a single (512,) vector for the whole 2 min.

    Shorter files: chunk up to the available length; the sub-10 s trailing
    remainder is IGNORED (never zero-padded); a track < 10 s uses its single real
    chunk (CLAP's deterministic repeat-pad).  model.eval() keeps Dropout/BN off.
    """
    import librosa
    y, _ = librosa.load(wav_path, sr=CLAP_SR, mono=True)         # 48 kHz mono — enforced
    y = np.asarray(y, dtype=np.float32)[:CLAP_MAX_INTAKE]        # first 2 minutes only
    n_full = y.shape[0] // CLAP_CHUNK
    if n_full >= 1:
        chunks = [y[i * CLAP_CHUNK:(i + 1) * CLAP_CHUNK] for i in range(n_full)]   # full 10 s chunks
    else:
        chunks = [y]                                            # < 10 s track -> single real chunk
    clap.model.eval()                                           # Dropout / BatchNorm off
    embs = clap.model.get_audio_embedding_from_data(x=chunks, use_tensor=False)    # (N, 512)
    return np.asarray(embs, dtype=np.float32).mean(axis=0)      # mean-pool -> (512,)


def build_or_load_svm(clap):
    """Train RBF-SVM on the 400 labeled benchmark CLAP embeddings (cached).
    Persists the fitted (StandardScaler, SVM) so inference reuses the EXACT same
    scaler/model — guaranteeing train/inference feature-scaling parity."""
    import pickle
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # Fast path: reuse the saved scaler+SVM verbatim (identical scaling at inference)
    if os.path.exists(SVM_MODEL) and os.path.exists(TRAIN_CACHE):
        with open(SVM_MODEL, "rb") as fh:
            d = pickle.load(fh)
        log(f"  Loaded saved scaler+SVM: {SVM_MODEL}")
        return d["scaler"], d["svm"]

    if os.path.exists(TRAIN_CACHE):
        log(f"  Loading cached training embeddings: {TRAIN_CACHE}")
        z = np.load(TRAIN_CACHE)
        X, y = z["X"], z["y"]
    else:
        log("  Computing CLAP embeddings for 400 labeled benchmark tracks (one-time)…")
        X_list, y_list = [], []
        tmp = tempfile.mkdtemp(prefix="svm_train_")
        n = 0
        for sub, label in TRAIN_DIRS:
            files = sorted(f for f in glob.glob(os.path.join(DATASETS, sub, "*"))
                           if os.path.splitext(f)[1].lower() in AUDIO_EXT)
            for f in files:
                n += 1
                w = os.path.join(tmp, "tr.wav")
                try:
                    to_wav(f, w)
                    X_list.append(clap_embed(w, clap))
                    y_list.append(1 if label == "AI" else 0)
                except Exception as exc:
                    err(f"   [train skip] {sub}/{os.path.basename(f)}: {exc}")
                finally:
                    if os.path.exists(w):
                        os.remove(w)
                if n % 50 == 0:
                    log(f"    trained-embeddings: {n}/400")
        X = np.stack(X_list); y = np.array(y_list)
        np.savez_compressed(TRAIN_CACHE, X=X, y=y)
        log(f"  Cached {len(y)} training embeddings → {TRAIN_CACHE}")

    scaler = StandardScaler().fit(X)
    svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    svm.fit(scaler.transform(X), y)
    with open(SVM_MODEL, "wb") as fh:                    # save fitted scaler + SVM
        pickle.dump({"scaler": scaler, "svm": svm}, fh)
    log(f"  SVM trained on {len(y)} tracks "
        f"({int((y==0).sum())} Human / {int((y==1).sum())} AI) — saved {os.path.basename(SVM_MODEL)}.")
    return scaler, svm


# ════════════════════════════════════ MAIN ════════════════════════════════════
def resolved_platform(row):
    if row.get("Route") == "A_Spotify" and isinstance(row.get("Matched_YouTube_URL"), str) and row["Matched_YouTube_URL"]:
        return urlparse(row["Matched_YouTube_URL"]).netloc or "youtube.com"
    dom = row.get("Domain")
    if isinstance(dom, str) and dom:
        return dom
    return urlparse(str(row.get("Source_URL", ""))).netloc


def main():
    import pandas as pd
    log(f"Device: {DEVICE}")

    # ── 1. Ingestion log → successful rows only ───────────────────────────────
    df = pd.read_csv(LOG_CSV)
    def local_path(r):
        fn = r.get("Output_Filename")
        if isinstance(fn, str) and fn:
            return os.path.join(DL_DIR, fn)
        return ""
    df["Local_File_Path"] = df.apply(local_path, axis=1)
    ok = df[df["Local_File_Path"].apply(lambda p: bool(p) and os.path.exists(p))].copy()
    log(f"Ingestion rows: {len(df)} total → {len(ok)} with an existing local .wav\n")

    # ── 2. Load models + train SVM ────────────────────────────────────────────
    log("Loading detectors…")
    s1, s2 = load_mippia()
    clap   = load_clap()
    scaler, svm = build_or_load_svm(clap)
    log("Detectors ready.\n")

    # ── 3. Inference loop ─────────────────────────────────────────────────────
    tmp = tempfile.mkdtemp(prefix="stream_eval_")
    out_rows = []
    for i, (_, r) in enumerate(ok.iterrows(), 1):
        path = r["Local_File_Path"]
        title = r.get("Track_Title", "")
        log(f"[{i}/{len(ok)}] {os.path.basename(path)}  ({str(title)[:40]})")
        rec = {
            "Original_URL":         r.get("Source_URL", ""),
            "Resolved_Platform":    resolved_platform(r),
            "Track_Title":          title,
            "Artist_Name":          r.get("Artist_Name", ""),
            "Local_File":           os.path.basename(path),
            "Lcrosvila_P_AI": "", "Lcrosvila_P_Human": "", "Lcrosvila_Prediction": "",
            "Mippia_P_AI": "",    "Mippia_P_Human": "",    "Mippia_Prediction": "",
            "Status": "OK", "Error": "",
        }
        errors = []

        # transcode once (resample to 48k; Mippia downsamples to 24k internally)
        wav = os.path.join(tmp, f"s{i:03d}.wav")
        try:
            to_wav(path, wav)
        except Exception as exc:
            rec["Status"] = "FAILED"; rec["Error"] = f"transcode: {exc}"
            rec["Lcrosvila_Prediction"] = rec["Mippia_Prediction"] = "ERROR"
            out_rows.append(rec); err(traceback.format_exc()); continue

        # ── Mippia (isolated: OOM / size-mismatch safe) ───────────────────────
        try:
            p_ai = mippia_pai(wav, s1, s2)
            rec["Mippia_P_AI"]     = round(p_ai, 4)
            rec["Mippia_P_Human"]  = round(1 - p_ai, 4)
            rec["Mippia_Prediction"] = "AI" if p_ai >= 0.5 else "Human"
        except (RuntimeError, ValueError, Exception) as exc:
            rec["Mippia_Prediction"] = "ERROR"
            errors.append(f"Mippia[{type(exc).__name__}]: {exc}")
            err(f"[Mippia FAIL] {path}: {exc}")
            if DEVICE == "mps":
                torch.mps.empty_cache()
            gc.collect()

        # ── Lcrosvila (isolated) ──────────────────────────────────────────────
        try:
            emb = clap_embed(wav, clap).reshape(1, -1)
            p_ai = float(svm.predict_proba(scaler.transform(emb))[0, list(svm.classes_).index(1)])
            rec["Lcrosvila_P_AI"]    = round(p_ai, 4)
            rec["Lcrosvila_P_Human"] = round(1 - p_ai, 4)
            rec["Lcrosvila_Prediction"] = "AI" if p_ai >= 0.5 else "Human"
        except (RuntimeError, ValueError, Exception) as exc:
            rec["Lcrosvila_Prediction"] = "ERROR"
            errors.append(f"Lcrosvila[{type(exc).__name__}]: {exc}")
            err(f"[Lcrosvila FAIL] {path}: {exc}")

        if errors:
            both_failed = (rec["Mippia_Prediction"] == "ERROR" and rec["Lcrosvila_Prediction"] == "ERROR")
            rec["Status"] = "FAILED" if both_failed else "PARTIAL"
            rec["Error"]  = " | ".join(errors)

        log(f"    Lcrosvila={rec['Lcrosvila_Prediction']}(P_AI={rec['Lcrosvila_P_AI']})  "
            f"Mippia={rec['Mippia_Prediction']}(P_AI={rec['Mippia_P_AI']})")
        out_rows.append(rec)
        try: os.remove(wav)
        except OSError: pass

    # ── 4. Write results ──────────────────────────────────────────────────────
    cols = ["Original_URL", "Resolved_Platform", "Track_Title", "Artist_Name", "Local_File",
            "Lcrosvila_P_AI", "Lcrosvila_P_Human", "Lcrosvila_Prediction",
            "Mippia_P_AI", "Mippia_P_Human", "Mippia_Prediction", "Status", "Error"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(out_rows)

    log(f"\nWrote {OUT_CSV}  ({len(out_rows)} rows)")
    from collections import Counter
    log("  Status: " + ", ".join(f"{k}={v}" for k, v in Counter(r["Status"] for r in out_rows).items()))
    log("  Mippia preds   : " + ", ".join(f"{k}={v}" for k, v in Counter(r["Mippia_Prediction"] for r in out_rows).items()))
    log("  Lcrosvila preds: " + ", ".join(f"{k}={v}" for k, v in Counter(r["Lcrosvila_Prediction"] for r in out_rows).items()))


if __name__ == "__main__":
    main()
