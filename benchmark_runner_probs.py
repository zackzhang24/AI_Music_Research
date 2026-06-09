#!/usr/bin/env python3
"""
benchmark_runner_probs.py
─────────────────────────
Re-run the benchmark on existing 100 standardized tracks, exposing
the full raw probability distributions instead of a single confidence score.

Changes vs. benchmark_runner.py
────────────────────────────────
  Mippia   : raw P(AI) and P(Human) extracted directly from the
             scaled-sigmoid output — before argmax.
  Lcrosvila: P(AI) and P(Human) taken straight from clf.predict_proba()
             at each leave-one-out fold.

Output
───────
  benchmark_results_with_probs.csv
  Music_Benchmark_Results_with_probs.xlsx

Columns
────────
  Filename | True_Label
  Mippia_Prediction | Mippia_Probability_AI | Mippia_Probability_Human
  Lcrosvila_Prediction | Lcrosvila_Probability_AI | Lcrosvila_Probability_Human
"""

import os, sys, gc, glob, pickle, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
STD_DIR    = os.path.join(BASE, "standardized_songs_30s")
FST_DIR    = os.path.join(BASE, "FST-AI-Music-Detection")
CLAP_DIR   = os.path.join(BASE, "ai-music-detection")
def _find_ckpt(*rel_paths):
    """Search several candidate locations under BASE and return the first hit."""
    candidates = [
        os.path.join(BASE, *rel_paths),
        os.path.join(BASE, "Misc", *rel_paths),
        os.path.join(BASE, "checkpoints", *rel_paths[1:]),      # skip 'checkpoints' prefix
        os.path.join(BASE, "Misc", "checkpoints", *rel_paths[1:]),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]   # return primary path even if missing (error will surface later)

CKPT_S1    = _find_ckpt("checkpoints", "mippia", "stage1_mert_audiocat.ckpt")
CKPT_S2    = _find_ckpt("checkpoints", "mippia", "stage2_fusion.ckpt")
CKPT_CLAP  = _find_ckpt("checkpoints", "clap", "music_audioset_epoch_15_esc_90.14.pt")
PKL_PATH   = _find_ckpt("checkpoints", "models_and_scaler.pkl")
CSV_OUT    = os.path.join(BASE, "benchmark_results_with_probs.csv")
XLSX_OUT   = os.path.join(BASE, "Music_Benchmark_Results_with_probs.xlsx")

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

BEAT_DEVICE = "cpu"   # beat_this has no MPS support

print(f"Torch device : {DEVICE}")
print(f"Beat device  : {BEAT_DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════════
# ① MIPPIA  — MERT-AudioCAT (Stage-1) → FusionSegmentTransformer (Stage-2)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_audio_beats_patched(audio_path: str, sr: int = 24000):
    """
    Replica of FST inference.load_audio with two fixes:
      • passes BEAT_DEVICE to beat_this (avoids hard-coded 'cuda')
      • uses soundfile instead of torchaudio.load (torchaudio 2.11
        dropped the old backends and requires TorchCodec for MP3)
    """
    import soundfile as sf
    import soxr
    from preprocess import get_segments_from_wav, find_optimal_segment_length

    beats, downbeats = get_segments_from_wav(audio_path, device=BEAT_DEVICE)
    _, cleaned_downbeats = find_optimal_segment_length(downbeats)

    try:
        wav_np, sample_rate = sf.read(audio_path, always_2d=True)  # (N, C)
        wav_np = wav_np.T.astype(np.float32)                        # (C, N)
    except Exception:
        import librosa
        wav_np, sample_rate = librosa.load(audio_path, sr=None, mono=False)
        if wav_np.ndim == 1:
            wav_np = wav_np[np.newaxis, :]
        wav_np = wav_np.astype(np.float32)

    if sample_rate != sr:
        wav_np = soxr.resample(wav_np.T, sample_rate, sr).T.astype(np.float32)

    waveform = torch.from_numpy(wav_np).to(torch.float32)   # (C, N)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    fixed_samples = 240_000                                  # 10 s at 24 kHz
    if waveform.shape[1] <= fixed_samples:
        waveform = torch.cat(
            [waveform, torch.zeros(1, fixed_samples, dtype=torch.float32)], dim=1
        )

    segments = []
    for start_time in cleaned_downbeats:
        s = int(start_time * sr)
        e = s + fixed_samples
        if e > waveform.size(1):
            continue
        seg = waveform[:, s:e]
        segments.append(torch.tensor(seg.squeeze().numpy(), dtype=torch.float32).unsqueeze(0))
        if len(segments) >= 48:
            break

    if not segments:
        return (torch.zeros((1, 1, fixed_samples), dtype=torch.float32),
                torch.ones(1, dtype=torch.bool))

    stacked = torch.stack(segments)                          # [N, 1, 240000]
    num_seg = stacked.shape[0]
    mask    = torch.zeros(48, dtype=torch.bool)
    if num_seg < 48:
        pad  = torch.zeros((48 - num_seg, 1, fixed_samples), dtype=torch.float32)
        stacked = torch.cat([stacked, pad], dim=0)
        mask[num_seg:] = True

    return stacked, mask                                     # [48,1,240000], [48]


def _scaled_sigmoid(x, scale_factor=1.0, linear_property=0.0):
    """
    The scaled sigmoid used in FST's run_inference.
    With scale_factor=1.0, linear_property=0.0 this reduces to:
        clamp(sigmoid(x), 0.011, 0.989)
    which is the exact call made in benchmark_runner.
    """
    scaled = x * scale_factor
    raw    = (torch.sigmoid(scaled) * (1.0 - linear_property)
              + linear_property * ((x + 25.0) / 50.0))
    return torch.clamp(raw, min=0.011, max=0.989)


def load_mippia():
    from model import MERT_AudioCAT, MusicAudioClassifier

    print("  Loading Stage-1 (MERT-AudioCAT, 1.3 GB)…")
    s1 = MERT_AudioCAT.load_from_checkpoint(CKPT_S1).to(DEVICE)
    s1.eval()

    print("  Loading Stage-2 (FusionSegmentTransformer, 48 MB)…")
    s2 = MusicAudioClassifier.load_from_checkpoint(
        checkpoint_path=CKPT_S2,
        input_dim=768,
        backbone="fusion_segment_transformer",
        is_emb=True,
    ).to(DEVICE)
    s2.eval()

    return s1, s2


def mippia_infer(audio_path: str, s1, s2):
    """
    Returns (label, prob_ai, prob_human).

    prob_ai   = P(Fake/AI)  — raw scaled-sigmoid output of the model
    prob_human= 1 - prob_ai — complement (model outputs a single logit)
    Both rounded to 6 decimal places.
    """
    segments, mask = _load_audio_beats_patched(audio_path)
    segments = segments.to(DEVICE).to(torch.float32)  # [48, 1, 240000]
    mask     = mask.to(DEVICE).unsqueeze(0)            # [1, 48]

    with torch.no_grad():
        # Stage-1 → per-segment embeddings
        _logits, embedding = s1(segments.squeeze(1))   # embedding: [48, 768]

        # Stage-2 → single binary logit
        s2.eval()
        s2_half = s2.half()

        emb_in  = embedding.unsqueeze(0)               # [1, 48, 768]
        if emb_in.shape[1] == 1:
            emb_in = emb_in[:, 0, :].unsqueeze(0)
        mask_in = mask if mask.dim() == 2 else mask.unsqueeze(0)

        raw_logit = s2_half(emb_in.to(DEVICE), mask_in.to(DEVICE))

    # ── extract raw probability distribution ──────────────────────────────────
    logit    = raw_logit.squeeze().float()             # scalar tensor
    prob_ai  = round(_scaled_sigmoid(logit).item(), 6)
    prob_human = round(1.0 - prob_ai, 6)
    label    = "AI" if prob_ai > 0.5 else "Human"

    # Cleanup
    del segments, mask, _logits, embedding, emb_in, mask_in, raw_logit
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return label, prob_ai, prob_human


# ═══════════════════════════════════════════════════════════════════════════════
# ② LCROSVILA  — LAION CLAP + Leave-One-Out RBF-SVM
# ═══════════════════════════════════════════════════════════════════════════════

def load_clap():
    from model_loader import CLAPMusic
    print("  Loading CLAP music model…")
    clap = CLAPMusic(model_file=CKPT_CLAP)
    clap.load_model()
    return clap


def clap_embed(audio_path: str, clap_model) -> np.ndarray:
    """Returns (512,) float32 CLAP embedding."""
    emb = clap_model._get_embedding([audio_path])      # (1, 512) float16
    return emb.astype(np.float32).squeeze(0)


def lcrosvila_loo_svm(embeddings: list, true_labels: list):
    """
    Leave-one-out RBF-SVM on CLAP embeddings.

    Returns three parallel lists aligned with embeddings/true_labels:
      predictions   — "AI" or "Human"
      prob_ai_list  — raw P(AI)    from clf.predict_proba()[:,1]
      prob_hu_list  — raw P(Human) from clf.predict_proba()[:,0]

    Classes inside sklearn: 0 = Human, 1 = AI
    (array is sorted by class label value, so index 0 = class 0 = Human)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X = np.stack(embeddings)                            # (N, 512)
    y = np.array([1 if lbl == "AI" else 0 for lbl in true_labels])
    N = len(X)

    predictions  = []
    prob_ai_list = []
    prob_hu_list = []

    for i in range(N):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test  = X[i:i+1]

        scaler  = StandardScaler()
        Xtr_sc  = scaler.fit_transform(X_train)
        Xte_sc  = scaler.transform(X_test)

        clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
        clf.fit(Xtr_sc, y_train)

        # predict_proba returns shape (1, 2): [[P(class0), P(class1)]]
        # class order follows sorted(unique(y_train)):
        #   if both classes present → [P(Human/0), P(AI/1)]
        proba = clf.predict_proba(Xte_sc)[0]           # shape (2,) or (1,)

        if len(proba) == 2:
            # Standard case: both Human and AI seen during training
            p_human = round(float(proba[0]), 6)
            p_ai    = round(float(proba[1]), 6)
        else:
            # Degenerate fold (only one class in train — shouldn't happen with 99 samples)
            p_ai    = round(float(proba[0]), 6) if clf.classes_[0] == 1 else 0.0
            p_human = round(1.0 - p_ai, 6)

        label = "AI" if p_ai >= 0.5 else "Human"
        predictions.append(label)
        prob_ai_list.append(p_ai)
        prob_hu_list.append(p_human)

        del scaler, clf, Xtr_sc, Xte_sc
        gc.collect()

        if (i + 1) % 10 == 0:
            print(f"    LOO progress: {i+1}/{N}")

    return predictions, prob_ai_list, prob_hu_list


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

    # ── Per-file inference ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Running inference (Batch Size = 1, strict memory cleanup)")
    print("=" * 60)

    records      = []
    clap_embs    = []
    clap_labels  = []
    clap_idx_map = []

    for idx, fpath in enumerate(files):
        fname      = os.path.basename(fpath)
        true_label = "AI" if fname.startswith("ai_track_") else "Human"
        print(f"\n[{idx+1:3d}/100] {fname}  (True: {true_label})")

        # ── Mippia ────────────────────────────────────────────────────────────
        try:
            mp_label, mp_p_ai, mp_p_hu = mippia_infer(fpath, s1, s2)
            print(f"  Mippia  : {mp_label:<6}  P(AI)={mp_p_ai:.6f}  P(Human)={mp_p_hu:.6f}")
        except Exception as exc:
            print(f"  Mippia ERROR: {exc}")
            mp_label, mp_p_ai, mp_p_hu = "Error", float("nan"), float("nan")

        # ── CLAP embedding ────────────────────────────────────────────────────
        try:
            emb = clap_embed(fpath, clap_mdl)
            clap_embs.append(emb)
            clap_labels.append(true_label)
            clap_idx_map.append(idx)
            print(f"  CLAP    : OK (dim={emb.shape[0]})")
        except Exception as exc:
            print(f"  CLAP ERROR: {exc}")

        records.append({
            "Filename":              fname,
            "True_Label":            true_label,
            "Mippia_Prediction":     mp_label,
            "Mippia_Probability_AI": mp_p_ai,
            "Mippia_Probability_Human": mp_p_hu,
            "Lcrosvila_Prediction":  "pending",
            "Lcrosvila_Probability_AI":    float("nan"),
            "Lcrosvila_Probability_Human": float("nan"),
        })

        # ── Strict memory cleanup ─────────────────────────────────────────────
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    # ── lcrosvila: LOO-SVM ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("lcrosvila: computing predictions via LOO RBF-SVM")
    print("=" * 60)

    if os.path.exists(PKL_PATH):
        # ── Use pre-trained pkl if available ──────────────────────────────────
        print(f"  Pre-trained models_and_scaler.pkl found — using it.")
        with open(PKL_PATH, "rb") as f:
            pkl = pickle.load(f)
        models = pkl["models"]
        scaler = pkl["scaler"]
        X_all  = np.stack(clap_embs)
        X_sc   = scaler.transform(X_all)

        for j, orig_idx in enumerate(clap_idx_map):
            # Collect probabilities from each ensemble member
            ai_probs = []
            for name, clf in models.items():
                try:
                    proba = clf.predict_proba(X_sc[j:j+1])[0]
                    # HiClass predict_proba may return parent-node prob
                    ai_probs.append(float(proba[-1]))  # last entry = AI/positive
                except Exception:
                    pass
            p_ai    = round(float(np.mean(ai_probs)) if ai_probs else 0.5, 6)
            p_human = round(1.0 - p_ai, 6)
            lbl     = "AI" if p_ai >= 0.5 else "Human"
            records[orig_idx]["Lcrosvila_Prediction"]       = lbl
            records[orig_idx]["Lcrosvila_Probability_AI"]   = p_ai
            records[orig_idx]["Lcrosvila_Probability_Human"] = p_human
    else:
        print("  models_and_scaler.pkl not found — running LOO RBF-SVM.")
        print("  (Same methodology as lcrosvila paper; LOO is unbiased per-track.)")
        loo_preds, loo_p_ai, loo_p_hu = lcrosvila_loo_svm(clap_embs, clap_labels)
        for j, orig_idx in enumerate(clap_idx_map):
            records[orig_idx]["Lcrosvila_Prediction"]        = loo_preds[j]
            records[orig_idx]["Lcrosvila_Probability_AI"]    = loo_p_ai[j]
            records[orig_idx]["Lcrosvila_Probability_Human"] = loo_p_hu[j]

    # Fill any records that got no CLAP embedding
    for rec in records:
        if rec["Lcrosvila_Prediction"] == "pending":
            rec["Lcrosvila_Prediction"]        = "Error"
            rec["Lcrosvila_Probability_AI"]    = float("nan")
            rec["Lcrosvila_Probability_Human"] = float("nan")

    # ── Export ────────────────────────────────────────────────────────────────
    cols = [
        "Filename", "True_Label",
        "Mippia_Prediction", "Mippia_Probability_AI", "Mippia_Probability_Human",
        "Lcrosvila_Prediction", "Lcrosvila_Probability_AI", "Lcrosvila_Probability_Human",
    ]
    df = pd.DataFrame(records, columns=cols)
    df.to_csv(CSV_OUT, index=False)
    df.to_excel(XLSX_OUT, index=False)
    print(f"\nSaved: {CSV_OUT}")
    print(f"Saved: {XLSX_OUT}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for model_name, pred_col, p_ai_col in [
        ("Mippia (FST)",           "Mippia_Prediction",    "Mippia_Probability_AI"),
        ("Lcrosvila (CLAP-SVM)",   "Lcrosvila_Prediction", "Lcrosvila_Probability_AI"),
    ]:
        valid = df[~df[pred_col].isin(["Error", "pending"])].copy()
        if len(valid) == 0:
            print(f"  {model_name}: no valid predictions")
            continue

        total      = len(valid)
        correct    = (valid[pred_col] == valid["True_Label"]).sum()
        acc        = 100.0 * correct / total
        ai_rows    = valid[valid["True_Label"] == "AI"]
        hu_rows    = valid[valid["True_Label"] == "Human"]
        ai_correct = (ai_rows[pred_col]  == "AI").sum()
        hu_correct = (hu_rows[pred_col] == "Human").sum()
        mean_p_ai  = valid[p_ai_col].mean()

        print(f"\n  {model_name}")
        print(f"    Overall accuracy : {correct}/{total}  ({acc:.1f}%)")
        print(f"    AI  recall       : {ai_correct}/{len(ai_rows)}")
        print(f"    Human recall     : {hu_correct}/{len(hu_rows)}")
        print(f"    Mean P(AI)       : {mean_p_ai:.4f}")

    print()
    return df


if __name__ == "__main__":
    main()
