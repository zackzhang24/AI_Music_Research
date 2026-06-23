#!/usr/bin/env python3
"""
app.py — Streamlit front-end for the AI-vs-Human music detector.

Run with:
    streamlit run 03_Execution_Scripts/app.py

It ties together:
  • universal_audio_ingestion.process_input()  — resolve a URL or text query,
    download a 5-min-capped .wav.
  • run_streaming_evaluation                    — the two detectors
        - Mippia    (MERT-AudioCAT → FusionSegmentTransformer, pretrained)
        - Lcrosvila (LAION-CLAP embedding → RBF-SVM trained on the 400-track
                     benchmark; training embeddings are cached on disk).

Models + the cached SVM are loaded ONCE via @st.cache_resource so they are not
re-loaded into RAM on every query.
"""

import os, sys, tempfile, gc, traceback
import streamlit as st
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import universal_audio_ingestion as uai          # process_input(), router
import run_streaming_evaluation as rse           # model load + inference helpers


# ── Heavy resources: loaded once, kept in RAM ──────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load Mippia (s1,s2) + CLAP, and fit the Lcrosvila SVM on the cached
    400-track benchmark CLAP embeddings.  Cached for the life of the process."""
    s1, s2      = rse.load_mippia()
    clap        = rse.load_clap()
    scaler, svm = rse.build_or_load_svm(clap)     # uses .clap_train_embeddings.npz cache
    return {"s1": s1, "s2": s2, "clap": clap, "scaler": scaler, "svm": svm}


def run_inference(models, wav_path):
    """Transcode → 48 kHz, run both detectors. Returns dict of probabilities,
    with per-model error isolation (OOM / tensor mismatch)."""
    out = {"lc_pai": None, "mp_pai": None, "lc_err": None, "mp_err": None}
    tmpdir = tempfile.mkdtemp(prefix="app_eval_")
    norm = os.path.join(tmpdir, "norm.wav")
    rse.to_wav(wav_path, norm)                    # resample (CLAP 48k; Mippia → 24k internally)

    # Mippia
    try:
        out["mp_pai"] = rse.mippia_pai(norm, models["s1"], models["s2"])
    except (RuntimeError, ValueError, Exception) as e:
        out["mp_err"] = f"{type(e).__name__}: {e}"
        if rse.DEVICE == "mps":
            torch.mps.empty_cache()
        gc.collect()

    # Lcrosvila
    try:
        emb = rse.clap_embed(norm, models["clap"]).reshape(1, -1)
        svm, scaler = models["svm"], models["scaler"]
        p_ai = float(svm.predict_proba(scaler.transform(emb))[0, list(svm.classes_).index(1)])
        out["lc_pai"] = p_ai
    except (RuntimeError, ValueError, Exception) as e:
        out["lc_err"] = f"{type(e).__name__}: {e}"

    try:
        os.remove(norm)
    except OSError:
        pass
    return out


def verdict(p_ai):
    return "AI" if (p_ai is not None and p_ai >= 0.5) else "Human"


# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Audio Forensics: AI vs. Human", page_icon="🎧", layout="centered")
st.title("Audio Forensics: AI vs. Human Detection")
st.caption("Resolve a song from a name/URL, download a 5-minute-capped clip, and "
           "classify it with two open-source detectors (Lcrosvila & Mippia).")

# Warm the cache once (instant on later reruns thanks to @st.cache_resource)
with st.spinner("Loading detection models (first launch only)…"):
    MODELS = load_models()

query = st.text_input("Enter a Track Name and Artist (or URL):",
                      placeholder="e.g.  Daft Punk Get Lucky   —or—   https://open.spotify.com/track/…")
go = st.button("Analyze Audio", type="primary")

if go:
    if not query.strip():
        st.warning("Please enter a track name, artist, or URL first.")
        st.stop()

    with st.spinner("Downloading and processing audio…"):
        try:
            result = uai.process_input(query)
        except Exception as e:
            st.error(f"Ingestion crashed: {type(e).__name__}: {e}")
            st.stop()

        status = result.get("Status", "")
        wav    = result.get("Local_File_Path", "") or ""

        # Halt cleanly on any non-success (NOT_FOUND / FAILED / missing file)
        if status != "SUCCESS" or not wav or not os.path.exists(wav):
            detail = result.get("Error") or status or "unknown error"
            st.error(f"❌ Could not retrieve audio for **{query}** — `{status}`.\n\n{detail}")
            st.stop()

        infer = run_inference(MODELS, wav)

    # ── Resolved-source banner ────────────────────────────────────────────────
    st.success(f"✅ Resolved & analyzed: **{result.get('Track_Title') or query}**"
               + (f" — {result.get('Artist_Name')}" if result.get("Artist_Name") else ""))
    meta_cols = st.columns(3)
    meta_cols[0].caption(f"**Route:** {result.get('Route','')}")
    meta_cols[1].caption(f"**Source:** {result.get('Domain','') or result.get('Search_Stage','')}")
    meta_cols[2].caption(f"**Clip length:** {result.get('Final_Duration_S','?')} s"
                         + ("  (capped 5:00)" if result.get("Capped_To_5min") else ""))

    st.divider()
    st.subheader("Detector verdicts")
    col_lc, col_mp = st.columns(2)

    # Lcrosvila
    with col_lc:
        st.markdown("### 🟦 Lcrosvila")
        if infer["lc_pai"] is None:
            st.error(f"Inference error:\n{infer['lc_err']}")
        else:
            p = infer["lc_pai"]
            st.metric("Prediction", verdict(p))
            st.metric("P(AI)", f"{p:.3f}")
            st.metric("P(Human)", f"{1-p:.3f}")
            st.progress(min(max(p, 0.0), 1.0), text=f"AI-likelihood {p*100:.1f}%")

    # Mippia
    with col_mp:
        st.markdown("### 🟥 Mippia")
        if infer["mp_pai"] is None:
            st.error(f"Inference error:\n{infer['mp_err']}")
        else:
            p = infer["mp_pai"]
            st.metric("Prediction", verdict(p))
            st.metric("P(AI)", f"{p:.3f}")
            st.metric("P(Human)", f"{1-p:.3f}")
            st.progress(min(max(p, 0.0), 1.0), text=f"AI-likelihood {p*100:.1f}%")

    # ── Combined note ─────────────────────────────────────────────────────────
    st.divider()
    preds = [verdict(infer["lc_pai"]) if infer["lc_pai"] is not None else None,
             verdict(infer["mp_pai"]) if infer["mp_pai"] is not None else None]
    valid = [p for p in preds if p]
    if valid and all(p == valid[0] for p in valid):
        st.info(f"Both detectors agree: **{valid[0]}**.")
    elif len(valid) == 2:
        st.warning(f"Detectors disagree — Lcrosvila: **{preds[0]}**, Mippia: **{preds[1]}**.")

st.divider()
st.caption("Lcrosvila = LAION-CLAP + RBF-SVM (trained on the 400-track benchmark). "
           "Mippia = MERT-AudioCAT + FusionSegmentTransformer (pretrained). "
           "P(AI) is the raw soft score; the 0.5 threshold sets the binary label.")
