#!/usr/bin/env python3
"""
benchmark_runner.py
───────────────────
Run all 100 standardized 30-second tracks through two open-source detectors:

  1. Mippia/FST-AI-Music-Detection  — MERT-AudioCAT (Stage-1) +
                                       FusionSegmentTransformer (Stage-2)
  2. lcrosvila/ai-music-detection   — LAION CLAP embeddings + binary SVM
       (leave-one-out, since the pre-trained models_and_scaler.pkl is gated
        behind SharePoint authentication.  Same RBF-SVM methodology as the
        paper; LOO gives unbiased per-sample estimates on our 100-track set.)

Output
───────
  benchmark_results.csv              — flat CSV (100 rows × 6 cols)
  Music_Benchmark_Results.xlsx       — same data, Excel workbook

Columns: Filename | True_Label | Mippia_Prediction | Mippia_Confidence_Score
         | Lcrosvila_Prediction | Lcrosvila_Confidence_Score
"""

import os, sys, gc, glob, pickle, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
STD_DIR     = os.path.join(BASE, "standardized_songs_30s")
FST_DIR     = os.path.join(BASE, "FST-AI-Music-Detection")
CLAP_DIR    = os.path.join(BASE, "ai-music-detection")
CKPT_S1     = os.path.join(BASE, "checkpoints", "mippia", "stage1_mert_audiocat.ckpt")
CKPT_S2     = os.path.join(BASE, "checkpoints", "mippia", "stage2_fusion.ckpt")
CKPT_CLAP   = os.path.join(BASE, "checkpoints", "clap", "music_audioset_epoch_15_esc_90.14.pt")
PKL_PATH    = os.path.join(BASE, "checkpoints", "models_and_scaler.pkl")
CSV_OUT     = os.path.join(BASE, "benchmark_results.csv")
XLSX_OUT    = os.path.join(BASE, "Music_Benchmark_Results.xlsx")

# Add both repos to import path
sys.path.insert(0, FST_DIR)
sys.path.insert(0, CLAP_DIR)
sys.path.insert(0, os.path.join(CLAP_DIR, "utils"))

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# beat_this only supports 'cuda' / 'cpu' (no MPS)
BEAT_DEVICE = "cpu"

print(f"Torch device : {DEVICE}")
print(f"Beat device  : {BEAT_DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════════
# ① MIPPIA  (MERT-AudioCAT → FusionSegmentTransformer)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_audio_beats_patched(audio_path: str, sr: int = 24000):
    """
    Re-implementation of FST inference.load_audio that passes BEAT_DEVICE
    to beat_this instead of hard-coded 'cuda'.
    Uses soundfile (not torchaudio) to load MP3s — avoids torchaudio 2.11's
    mandatory TorchCodec dependency on MP3 files.
    """
    import soundfile as sf
    import soxr
    from preprocess import get_segments_from_wav, find_optimal_segment_length

    beats, downbeats = get_segments_from_wav(audio_path, device=BEAT_DEVICE)
    _, cleaned_downbeats = find_optimal_segment_length(downbeats)

    # Load via soundfile (supports MP3 via libsndfile); fall back to librosa
    try:
        wav_np, sample_rate = sf.read(audio_path, always_2d=True)   # (N, C)
        wav_np = wav_np.T.astype(np.float32)                         # (C, N)
    except Exception:
        import librosa
        wav_np, sample_rate = librosa.load(audio_path, sr=None, mono=False)
        if wav_np.ndim == 1:
            wav_np = wav_np[np.newaxis, :]                            # (1, N)
        wav_np = wav_np.astype(np.float32)

    # Resample if needed
    if sample_rate != sr:
        wav_np = soxr.resample(wav_np.T, sample_rate, sr).T.astype(np.float32)

    waveform = torch.from_numpy(wav_np)   # (C, N)
    waveform = waveform.to(torch.float32)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    fixed_samples = 240_000  # 10 s at 24 kHz

    if waveform.shape[1] <= fixed_samples:
        padding = torch.zeros(1, fixed_samples, dtype=torch.float32)
        waveform = torch.cat([waveform, padding], dim=1)

    segments = []
    for start_time in cleaned_downbeats:
        start_sample = int(start_time * sr)
        end_sample   = start_sample + fixed_samples
        if end_sample > waveform.size(1):
            continue
        seg = waveform[:, start_sample:end_sample]
        segments.append(torch.tensor(seg.squeeze().numpy(), dtype=torch.float32).unsqueeze(0))
        if len(segments) >= 48:
            break

    if not segments:
        return (torch.zeros((1, 1, fixed_samples), dtype=torch.float32),
                torch.ones(1, dtype=torch.bool))

    stacked = torch.stack(segments)                     # [N, 1, 240000]
    num_seg = stacked.shape[0]
    mask    = torch.zeros(48, dtype=torch.bool)

    if num_seg < 48:
        pad = torch.zeros((48 - num_seg, 1, fixed_samples), dtype=torch.float32)
        stacked = torch.cat([stacked, pad], dim=0)
        mask[num_seg:] = True

    return stacked, mask                                # [48,1,240000], [48]


def _scaled_sigmoid(x, scale_factor=0.2, linear_property=0.3):
    scaled = x * scale_factor
    raw    = (torch.sigmoid(scaled) * (1 - linear_property)
              + linear_property * ((x + 25) / 50))
    return torch.clamp(raw, min=0.011, max=0.989)


def load_mippia():
    from model import MERT_AudioCAT, MusicAudioClassifier

    print("  Loading Stage-1 (MERT-AudioCAT, 1.3 GB)…")
    s1 = MERT_AudioCAT.load_from_checkpoint(CKPT_S1)
    s1 = s1.to(DEVICE)
    s1.eval()

    print("  Loading Stage-2 (FusionSegmentTransformer, 48 MB)…")
    s2 = MusicAudioClassifier.load_from_checkpoint(
        checkpoint_path=CKPT_S2,
        input_dim=768,
        backbone="fusion_segment_transformer",
        is_emb=True,
    )
    s2 = s2.to(DEVICE)
    s2.eval()

    return s1, s2


def mippia_infer(audio_path: str, s1, s2):
    """
    Returns (label: str, confidence: float) where label is 'AI' or 'Human'
    and confidence is in [0, 1] (probability of the predicted class).
    """
    segments, mask = _load_audio_beats_patched(audio_path)
    segments = segments.to(DEVICE).to(torch.float32)  # [48, 1, 240000]
    mask     = mask.to(DEVICE).unsqueeze(0)            # [1, 48]

    with torch.no_grad():
        # Stage-1: extract per-segment embeddings
        _logits, embedding = s1(segments.squeeze(1))   # squeeze → [48, 240000]
        # embedding: [48, 768]

        # Stage-2: classify from embeddings
        s2.eval()
        s2.to(DEVICE)
        s2_half = s2.half()

        # reshape embedding for Stage-2: needs [1, 48, 768]
        emb_in = embedding.unsqueeze(0)  # [1, 48, 768]
        if emb_in.shape[1] == 1:
            emb_in = emb_in[:, 0, :].unsqueeze(0)
        mask_in = mask if mask.dim() == 2 else mask.unsqueeze(0)

        emb_in  = emb_in.to(DEVICE)
        mask_in = mask_in.to(DEVICE)

        outputs = s2_half(emb_in, mask_in)

    logit = outputs.squeeze()
    prob  = _scaled_sigmoid(logit.float(), scale_factor=1.0, linear_property=0.0).item()

    label      = "AI"    if prob > 0.5 else "Human"
    confidence = round(max(prob, 1 - prob), 4)

    # Cleanup
    del segments, mask, _logits, embedding, emb_in, mask_in, outputs
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return label, confidence


# ═══════════════════════════════════════════════════════════════════════════════
# ② LCROSVILA  (LAION CLAP + Leave-One-Out RBF-SVM)
# ═══════════════════════════════════════════════════════════════════════════════

def load_clap():
    from model_loader import CLAPMusic
    print("  Loading CLAP music model…")
    clap = CLAPMusic(model_file=CKPT_CLAP)
    clap.load_model()
    return clap


def clap_embed(audio_path: str, clap_model) -> np.ndarray:
    """Returns (512,) float32 CLAP embedding."""
    emb = clap_model._get_embedding([audio_path])  # (1, 512) float16
    return emb.astype(np.float32).squeeze(0)


def lcrosvila_loo_svm(embeddings: list, true_labels: list):
    """
    Leave-one-out SVM on CLAP embeddings.
    Returns (predictions, confidences) as lists aligned with embeddings/true_labels.
    Uses the same RBF-kernel SVC with probability=True as the paper.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X = np.stack(embeddings)  # (N, 512)
    y = np.array([1 if lbl == "AI" else 0 for lbl in true_labels])
    N = len(X)

    preds = []
    confs = []

    for i in range(N):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test  = X[i:i+1]

        scaler   = StandardScaler()
        Xtr_sc   = scaler.fit_transform(X_train)
        Xte_sc   = scaler.transform(X_test)

        clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
        clf.fit(Xtr_sc, y_train)

        prob_ai  = clf.predict_proba(Xte_sc)[0, 1]   # P(AI)
        label    = "AI" if prob_ai >= 0.5 else "Human"
        conf     = round(max(float(prob_ai), 1 - float(prob_ai)), 4)

        preds.append(label)
        confs.append(conf)

        del scaler, clf, Xtr_sc, Xte_sc
        gc.collect()

        if (i + 1) % 10 == 0:
            print(f"    LOO progress: {i+1}/{N}")

    return preds, confs


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    files = sorted(glob.glob(os.path.join(STD_DIR, "*.mp3")))
    if not files:
        raise FileNotFoundError(f"No MP3s found in {STD_DIR}")
    print(f"\nFound {len(files)} tracks in {STD_DIR}\n")

    # ── Load models ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading models")
    print("=" * 60)
    s1, s2   = load_mippia()
    clap_mdl = load_clap()

    # ── Per-file inference ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Running inference on 100 tracks")
    print("=" * 60)

    records        = []
    clap_embs      = []
    clap_labels    = []
    clap_idx_map   = []   # original index in `records` for each valid CLAP emb

    for idx, fpath in enumerate(files):
        fname      = os.path.basename(fpath)
        true_label = "AI" if fname.startswith("ai_track_") else "Human"
        print(f"\n[{idx+1:3d}/100] {fname}  (True: {true_label})")

        # ── Mippia ────────────────────────────────────────────────────────────
        try:
            mp_label, mp_conf = mippia_infer(fpath, s1, s2)
            print(f"  Mippia      : {mp_label:<6}  conf={mp_conf:.4f}")
        except Exception as exc:
            print(f"  Mippia ERROR: {exc}")
            mp_label, mp_conf = "Error", float("nan")

        # ── CLAP embedding ────────────────────────────────────────────────────
        try:
            emb = clap_embed(fpath, clap_mdl)
            clap_embs.append(emb)
            clap_labels.append(true_label)
            clap_idx_map.append(idx)
            print(f"  CLAP embed  : OK (dim={emb.shape[0]})")
        except Exception as exc:
            print(f"  CLAP ERROR  : {exc}")

        records.append({
            "Filename":                fname,
            "True_Label":              true_label,
            "Mippia_Prediction":       mp_label,
            "Mippia_Confidence_Score": mp_conf,
            "Lcrosvila_Prediction":    "pending",
            "Lcrosvila_Confidence_Score": float("nan"),
        })

        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    # ── lcrosvila: LOO-SVM (or pre-trained pkl) ───────────────────────────────
    print("\n" + "=" * 60)
    print("lcrosvila: computing predictions")
    print("=" * 60)

    if os.path.exists(PKL_PATH):
        print(f"  Pre-trained models_and_scaler.pkl found — using it.")
        with open(PKL_PATH, "rb") as f:
            pkl = pickle.load(f)
        models  = pkl["models"]
        scaler  = pkl["scaler"]
        X_all   = np.stack(clap_embs)
        X_sc    = scaler.transform(X_all)

        for j, orig_idx in enumerate(clap_idx_map):
            votes = []
            for name, clf in models.items():
                p = clf.predict(X_sc[j:j+1])
                votes.append(p[0][0])  # hierarchical parent label
            ai_votes = sum(1 for v in votes if v == "AI")
            lbl  = "AI" if ai_votes >= 2 else "Human"
            conf = round(ai_votes / len(models), 4)
            records[orig_idx]["Lcrosvila_Prediction"]       = lbl
            records[orig_idx]["Lcrosvila_Confidence_Score"] = conf
    else:
        print("  models_and_scaler.pkl not found — running LOO RBF-SVM on CLAP embeddings.")
        print("  (Same methodology as lcrosvila paper; LOO gives unbiased per-track estimates.)")
        loo_preds, loo_confs = lcrosvila_loo_svm(clap_embs, clap_labels)
        for j, orig_idx in enumerate(clap_idx_map):
            records[orig_idx]["Lcrosvila_Prediction"]       = loo_preds[j]
            records[orig_idx]["Lcrosvila_Confidence_Score"] = loo_confs[j]

    # Remaining records without CLAP embeddings
    for rec in records:
        if rec["Lcrosvila_Prediction"] == "pending":
            rec["Lcrosvila_Prediction"]       = "Error"
            rec["Lcrosvila_Confidence_Score"] = float("nan")

    # ── Export ────────────────────────────────────────────────────────────────
    cols = [
        "Filename", "True_Label",
        "Mippia_Prediction", "Mippia_Confidence_Score",
        "Lcrosvila_Prediction", "Lcrosvila_Confidence_Score",
    ]
    df = pd.DataFrame(records, columns=cols)
    df.to_csv(CSV_OUT,  index=False)
    df.to_excel(XLSX_OUT, index=False)
    print(f"\nSaved: {CSV_OUT}")
    print(f"Saved: {XLSX_OUT}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for model_name, pred_col in [
        ("Mippia (FST)", "Mippia_Prediction"),
        ("Lcrosvila (CLAP-SVM)", "Lcrosvila_Prediction"),
    ]:
        valid = df[~df[pred_col].isin(["Error", "pending", "N/A"])].copy()
        if len(valid) == 0:
            print(f"  {model_name}: no valid predictions")
            continue

        total   = len(valid)
        correct = (valid[pred_col] == valid["True_Label"]).sum()
        acc     = 100.0 * correct / total

        ai_rows = valid[valid["True_Label"] == "AI"]
        hu_rows = valid[valid["True_Label"] == "Human"]
        ai_correct = (ai_rows[pred_col] == "AI").sum()
        hu_correct = (hu_rows[pred_col] == "Human").sum()

        print(f"\n  {model_name}")
        print(f"    Overall accuracy : {correct}/{total}  ({acc:.1f}%)")
        print(f"    AI  recall       : {ai_correct}/{len(ai_rows)}")
        print(f"    Human recall     : {hu_correct}/{len(hu_rows)}")

    print()
    return df


if __name__ == "__main__":
    main()
