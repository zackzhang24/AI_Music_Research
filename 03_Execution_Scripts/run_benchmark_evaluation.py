#!/usr/bin/env python3
"""
run_benchmark_evaluation.py
───────────────────────────
Zero-shot inference across the full 400-track benchmark using two open-source
AI-music detectors, with robust mixed-container ingestion and per-track error
isolation.

Dataset (400 tracks)
────────────────────
  human_tracks_120s/   Human · 120s (long-form)   95 FMA mp3 + 5 user m4a
  human_tracks_10s/    Human · 10s  (short-form)  95 FMA mp3 + 5 user m4a
  ai_tracks_120s/      AI    · 120s (long-form)   100 SONICS-Udio mp3
  new_ai_tracks_10s/   AI    · 10s  (short-form)  100 AIME wav (MusicGen/StableAudio/AudioLDM/Riffusion)

Detectors
─────────
  Tool 1 — Lcrosvila : LAION-CLAP embedding → leave-one-out RBF-SVM.
           (The paper's pretrained models_and_scaler.pkl is gated behind
            SharePoint auth, so we reproduce its CLAP+SVM methodology with a
            leave-one-out SVM over the 400-track embedding matrix.)
  Tool 2 — Mippia    : MERT-AudioCAT (Stage-1) → FusionSegmentTransformer
                       (Stage-2), genuine pretrained checkpoints.

Robust ingestion
────────────────
Every track (.mp3 / .m4a / .wav) is transcoded to a clean temporary WAV via
ffmpeg (capped at 300 s for tractability on long files).  This guarantees both
detectors receive a decodable stream regardless of source container — in
particular the AAC/.m4a user recordings that libsndfile cannot read natively.

Confidence convention
─────────────────────
*_Confidence is the soft probability of the PREDICTED class (∈ [0.5, 1.0]) —
i.e. the model's confidence in its own binary decision.  P(AI) is recorded
internally and drives the binary prediction (AI iff P(AI) ≥ 0.5).

Output
──────
  benchmark_evaluation_results.csv  (schema per task spec)
  + macro-accuracy / FPR / FNR per tool, split by temporal class (10s vs 120s).
"""

import os, sys, gc, glob, csv, json, subprocess, tempfile, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
FST_DIR  = os.path.join(BASE, "FST-AI-Music-Detection")
CLAP_DIR = os.path.join(BASE, "ai-music-detection")
OUT_CSV  = os.path.join(BASE, "benchmark_evaluation_results.csv")

sys.path.insert(0, FST_DIR)
sys.path.insert(0, CLAP_DIR)
sys.path.insert(0, os.path.join(CLAP_DIR, "utils"))

import static_ffmpeg; static_ffmpeg.add_paths()     # ffmpeg + ffprobe on PATH


def _find_ckpt(*rel):
    for c in (os.path.join(BASE, *rel),
              os.path.join(BASE, "Misc", *rel),
              os.path.join(BASE, "checkpoints", *rel[1:]),
              os.path.join(BASE, "Misc", "checkpoints", *rel[1:])):
        if os.path.exists(c):
            return c
    return os.path.join(BASE, *rel)

CKPT_S1   = _find_ckpt("checkpoints", "mippia", "stage1_mert_audiocat.ckpt")
CKPT_S2   = _find_ckpt("checkpoints", "mippia", "stage2_fusion.ckpt")
CKPT_CLAP = _find_ckpt("checkpoints", "clap", "music_audioset_epoch_15_esc_90.14.pt")

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
BEAT_DEVICE = "cpu"
ANALYZE_CAP_S = 300        # cap transcoded analysis WAV length (s)

# ── Dataset directories ───────────────────────────────────────────────────────
DIRS = [
    ("human_tracks_120s", "Human", "120s"),
    ("human_tracks_10s",  "Human", "10s"),
    ("ai_tracks_120s",    "AI",    "120s"),
    ("new_ai_tracks_10s", "AI",    "10s"),
]
AUDIO_EXT = (".mp3", ".m4a", ".wav", ".flac", ".ogg")


def log_err(msg):
    print(msg, file=sys.stderr, flush=True)


# ── Robust ingestion: any container → clean temp WAV ──────────────────────────
def to_wav(src: str, dst: str, cap_s: int = ANALYZE_CAP_S):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-t", str(cap_s),
           "-i", src, "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", dst]
    subprocess.check_call(cmd)


# ═══════════════════════════════════ MIPPIA ═══════════════════════════════════
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
    """Return P(AI) ∈ [0,1]."""
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


# ═══════════════════════════════════ LCROSVILA ════════════════════════════════
def load_clap():
    from model_loader import CLAPMusic
    clap = CLAPMusic(model_file=CKPT_CLAP)
    clap.load_model()
    return clap


def clap_embed(wav_path, clap):
    emb = clap._get_embedding([wav_path])      # (1,512) float16
    return emb.astype(np.float32).squeeze(0)


def lcrosvila_loo(embeddings, labels):
    """Leave-one-out RBF-SVM. labels: 1=AI,0=Human. Returns list of P(AI)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    X = np.stack(embeddings); y = np.array(labels); N = len(X)
    pai = []
    for i in range(N):
        Xtr = np.delete(X, i, axis=0); ytr = np.delete(y, i)
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(X[i:i+1])
        clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
        clf.fit(Xtr, ytr)
        classes = list(clf.classes_)
        proba = clf.predict_proba(Xte)[0]
        p_ai = float(proba[classes.index(1)]) if 1 in classes else 0.0
        pai.append(p_ai)
        del sc, clf, Xtr, Xte
        if (i + 1) % 50 == 0:
            print(f"    Lcrosvila LOO: {i+1}/{N}", flush=True)
        gc.collect()
    return pai


# ═══════════════════════════════════ MAIN ═════════════════════════════════════
def main():
    print(f"Device: {DEVICE} | Beat: {BEAT_DEVICE}", flush=True)
    print("Loading detectors…", flush=True)
    s1, s2 = load_mippia()
    clap   = load_clap()
    print("Detectors ready.\n", flush=True)

    # Gather all tracks
    tracks = []
    for d, gt, tclass in DIRS:
        ddir = os.path.join(BASE, d)
        files = sorted(f for f in glob.glob(os.path.join(ddir, "*"))
                       if os.path.splitext(f)[1].lower() in AUDIO_EXT)
        for f in files:
            tracks.append({"path": f, "dir": d, "gt": gt, "tclass": tclass})
    print(f"Total tracks discovered: {len(tracks)}", flush=True)

    rows = []
    clap_embs, clap_idx, clap_labels = [], [], []

    tmpdir = tempfile.mkdtemp(prefix="bench_eval_")

    for i, tk in enumerate(tracks):
        fname = os.path.basename(tk["path"])
        row = {
            "Filename": fname, "Directory": tk["dir"],
            "Temporal_Class": tk["tclass"], "Ground_Truth_Label": tk["gt"],
            "Lcrosvila_Prediction": "", "Lcrosvila_Confidence": "",
            "Mippia_Prediction": "", "Mippia_Confidence": "",
            "Lcrosvila_Correct": "", "Mippia_Correct": "", "Status": "OK",
        }
        wav = os.path.join(tmpdir, f"t{i:04d}.wav")

        # ── Robust transcode ──────────────────────────────────────────────────
        try:
            to_wav(tk["path"], wav)
        except Exception as exc:
            log_err(f"[FAILED transcode] {tk['dir']}/{fname}: {exc}")
            row["Status"] = "FAILED"
            row["Lcrosvila_Prediction"] = row["Mippia_Prediction"] = "FAILED"
            row["Lcrosvila_Correct"] = row["Mippia_Correct"] = False
            rows.append(row)
            continue

        # ── Mippia (per-track, isolated) ──────────────────────────────────────
        try:
            p_ai = mippia_pai(wav, s1, s2)
            pred = "AI" if p_ai >= 0.5 else "Human"
            row["Mippia_Prediction"] = pred
            row["Mippia_Confidence"] = round(max(p_ai, 1 - p_ai), 3)
            row["Mippia_Correct"]    = bool(pred == tk["gt"])
        except Exception as exc:
            log_err(f"[FAILED Mippia] {tk['dir']}/{fname}: {exc}")
            row["Status"] = "FAILED"
            row["Mippia_Prediction"] = "FAILED"
            row["Mippia_Correct"] = False

        # ── Lcrosvila embedding (LOO computed after the loop) ──────────────────
        try:
            emb = clap_embed(wav, clap)
            clap_embs.append(emb)
            clap_idx.append(len(rows))          # index of this row
            clap_labels.append(1 if tk["gt"] == "AI" else 0)
        except Exception as exc:
            log_err(f"[FAILED Lcrosvila-embed] {tk['dir']}/{fname}: {exc}")
            row["Status"] = "FAILED"
            row["Lcrosvila_Prediction"] = "FAILED"
            row["Lcrosvila_Correct"] = False

        rows.append(row)

        try:
            os.remove(wav)
        except OSError:
            pass

        if (i + 1) % 25 == 0 or (i + 1) == len(tracks):
            print(f"  processed {i+1}/{len(tracks)}  "
                  f"(last: {tk['dir']}/{fname}  Mippia={row['Mippia_Prediction']})", flush=True)

    # ── Lcrosvila leave-one-out over all embedded tracks ──────────────────────
    print("\nComputing Lcrosvila predictions (CLAP + leave-one-out SVM)…", flush=True)
    if len(clap_embs) >= 3:
        pai_list = lcrosvila_loo(clap_embs, clap_labels)
        for j, ridx in enumerate(clap_idx):
            p_ai = pai_list[j]
            pred = "AI" if p_ai >= 0.5 else "Human"
            gt = rows[ridx]["Ground_Truth_Label"]
            rows[ridx]["Lcrosvila_Prediction"] = pred
            rows[ridx]["Lcrosvila_Confidence"] = round(max(p_ai, 1 - p_ai), 3)
            rows[ridx]["Lcrosvila_Correct"]    = bool(pred == gt)
    else:
        log_err("Too few CLAP embeddings for LOO SVM.")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    cols = ["Filename", "Directory", "Temporal_Class", "Ground_Truth_Label",
            "Lcrosvila_Prediction", "Lcrosvila_Confidence",
            "Mippia_Prediction", "Mippia_Confidence",
            "Lcrosvila_Correct", "Mippia_Correct", "Status"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}  ({len(rows)} rows)", flush=True)

    # ── Summary metrics by temporal class ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — macro-accuracy / FPR / FNR  (positive class = AI)")
    print("=" * 70)

    def metrics(rws, tool):
        pcol, ccol = f"{tool}_Prediction", f"{tool}_Correct"
        valid = [r for r in rws if r[pcol] in ("AI", "Human")]
        TP = sum(r["Ground_Truth_Label"] == "AI"    and r[pcol] == "AI"    for r in valid)
        FN = sum(r["Ground_Truth_Label"] == "AI"    and r[pcol] == "Human" for r in valid)
        TN = sum(r["Ground_Truth_Label"] == "Human" and r[pcol] == "Human" for r in valid)
        FP = sum(r["Ground_Truth_Label"] == "Human" and r[pcol] == "AI"    for r in valid)
        n = len(valid)
        acc      = (TP + TN) / n if n else float("nan")
        ai_rec   = TP / (TP + FN) if (TP + FN) else float("nan")   # recall AI
        hu_rec   = TN / (TN + FP) if (TN + FP) else float("nan")   # recall Human
        macro    = (ai_rec + hu_rec) / 2 if n else float("nan")
        fpr      = FP / (FP + TN) if (FP + TN) else float("nan")
        fnr      = FN / (FN + TP) if (FN + TP) else float("nan")
        return n, acc, macro, fpr, fnr

    for tclass in ["10s", "120s"]:
        sub = [r for r in rows if r["Temporal_Class"] == tclass]
        print(f"\n── Temporal class: {tclass}  (n={len(sub)}: "
              f"{sum(r['Ground_Truth_Label']=='Human' for r in sub)} Human / "
              f"{sum(r['Ground_Truth_Label']=='AI' for r in sub)} AI) ──")
        for tool in ["Lcrosvila", "Mippia"]:
            n, acc, macro, fpr, fnr = metrics(sub, tool)
            print(f"  {tool:10s} | n={n:3d} | macro-acc={macro:6.3f} | "
                  f"accuracy={acc:6.3f} | FPR={fpr:6.3f} | FNR={fnr:6.3f}")

    # Overall + failure count
    failed = sum(r["Status"] == "FAILED" for r in rows)
    print(f"\n  Rows marked FAILED (≥1 detector error): {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
