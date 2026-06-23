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
TRAIN_CACHE = os.path.join(HERE, ".clap_train_embeddings.npz")

sys.path.insert(0, FST_DIR)
sys.path.insert(0, CLAP_DIR)
sys.path.insert(0, os.path.join(CLAP_DIR, "utils"))

import static_ffmpeg; static_ffmpeg.add_paths()

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
BEAT_DEVICE   = "cpu"
ANALYZE_CAP_S = 300
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def log(msg): print(msg, flush=True)
def err(msg): print(msg, file=sys.stderr, flush=True)


# ── Robust ingestion → temp wav (48 kHz stereo) ───────────────────────────────
def to_wav(src, dst, cap_s=ANALYZE_CAP_S):
    subprocess.check_call(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-t", str(cap_s),
         "-i", src, "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", dst])


# ═══════════════════════════ MIPPIA (identical to benchmark) ══════════════════
def _load_audio_beats_patched(audio_path, sr=24000):
    import soundfile as sf, soxr
    from preprocess import get_segments_from_wav, find_optimal_segment_length
    _, downbeats = get_segments_from_wav(audio_path, device=BEAT_DEVICE)
    _, cleaned = find_optimal_segment_length(downbeats)
    wav, srate = sf.read(audio_path, always_2d=True)
    wav = wav.T.astype(np.float32)
    if srate != sr:
        wav = soxr.resample(wav.T, srate, sr).T.astype(np.float32)
    waveform = torch.from_numpy(wav).to(torch.float32)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    fixed = 240_000
    if waveform.shape[1] <= fixed:
        waveform = torch.cat([waveform, torch.zeros(1, fixed, dtype=torch.float32)], dim=1)
    segs = []
    for st in cleaned:
        s = int(st * sr); e = s + fixed
        if e > waveform.size(1):
            continue
        segs.append(torch.tensor(waveform[:, s:e].squeeze().numpy(), dtype=torch.float32).unsqueeze(0))
        if len(segs) >= 48:
            break
    if not segs:
        return torch.zeros((1, 1, fixed), dtype=torch.float32), torch.ones(1, dtype=torch.bool)
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
    segs, mask = _load_audio_beats_patched(wav_path)
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
def load_clap():
    from model_loader import CLAPMusic
    clap = CLAPMusic(model_file=CKPT_CLAP); clap.load_model()
    return clap


def clap_embed(wav_path, clap):
    return clap._get_embedding([wav_path]).astype(np.float32).squeeze(0)


def build_or_load_svm(clap):
    """Train RBF-SVM on the 400 labeled benchmark CLAP embeddings (cached)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

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
    log(f"  SVM trained on {len(y)} tracks "
        f"({int((y==0).sum())} Human / {int((y==1).sum())} AI).")
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
