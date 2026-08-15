#!/usr/bin/env python3
"""
app.py — Streamlit front-end for the AI-vs-Human music detector.

Run with:
    streamlit run 03_Execution_Scripts/app.py

It ties together:
  • universal_audio_ingestion.process_input()  — resolve a URL or text query,
    download a 5-min-capped .wav.
  • A direct file-uploader (Option B) that bypasses yt-dlp entirely — useful for
    isolating whether YouTube's lossy re-encode is destroying artifacts MERT/CLAP
    rely on, by feeding a local .wav/.mp3 straight into the same pipeline.
  • run_streaming_evaluation                    — the two detectors
        - Mippia    (MERT-AudioCAT → FusionSegmentTransformer, pretrained)
        - Lcrosvila (LAION-CLAP embedding → RBF-SVM, GridSearchCV-retrained on
                     the 400-track benchmark AUGMENTED with AAC/Opus-compressed
                     copies — see MODEL_PATH / .clap_svm_model_augmented.pkl).
  • calculate_fusion_verdict()                  — late-fusion router that
        combines the two raw detector scores into a single system verdict,
        weighted differently depending on whether the audio came via the
        lossy yt-dlp/YouTube path or a direct (uncompressed) file upload.

Models + the cached SVM are loaded ONCE via @st.cache_resource so they are not
re-loaded into RAM on every query.
"""

import os, sys, pickle, tempfile, gc, traceback
import streamlit as st
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import universal_audio_ingestion as uai          # process_input(), router
import run_streaming_evaluation as rse           # model load + inference helpers

# Production Lcrosvila model: GridSearchCV-retrained on the Clean+AAC+Opus
# mixed-codec dataset (see train_augmented_classifier.py), so it stays
# accurate across raw files AND lossy web-compressed streams.
MODEL_PATH = os.path.join(HERE, ".clap_svm_model_augmented.pkl")


# ── Heavy resources: loaded once, kept in RAM ──────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load Mippia (s1,s2) + CLAP, and load the production Lcrosvila scaler+SVM
    from MODEL_PATH (the codec-augmented, GridSearchCV-tuned model).  Cached
    for the life of the process."""
    s1, s2 = rse.load_mippia()
    clap   = rse.load_clap()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Production Lcrosvila model not found at {MODEL_PATH}. "
            "Run augment_training_data.py then train_augmented_classifier.py first.")
    with open(MODEL_PATH, "rb") as fh:
        saved = pickle.load(fh)
    scaler, svm = saved["scaler"], saved["svm"]
    return {"s1": s1, "s2": s2, "clap": clap, "scaler": scaler, "svm": svm}


def run_inference(models, wav_path):
    """Transcode → 48 kHz, run both detectors. Returns dict of probabilities,
    with per-model error isolation (OOM / tensor mismatch)."""
    out = {"lc_pai": None, "mp_pai": None, "lc_err": None, "mp_err": None}
    tmpdir = tempfile.mkdtemp(prefix="app_eval_")
    norm = os.path.join(tmpdir, "norm.wav")
    rse.to_wav(wav_path, norm)                    # one 48 kHz source wav for both models

    # Sample-rate routing:
    #   Mippia   → mippia_pai() downsamples 48 kHz → 24 kHz (MERT_SR) internally.
    #   Lcrosvila→ clap_embed() loads the same wav at 48 kHz (CLAP's native rate).
    # MERT therefore NEVER receives 48 kHz audio.
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


# ── Late-fusion routing (acoustic-source-aware) ─────────────────────────────────
MIPPIA_INVALIDATED_WARNING = (
    "Mippia confidence invalidated due to lossy YouTube compression. "
    "System relying on Lcrosvila."
)


def calculate_fusion_verdict(lcrosvila_score, mippia_score, input_method):
    """
    Combine the two raw detector P(AI) scores into a single system verdict via
    weighted late fusion.  Deliberately kept OUT of the UI layer so the routing
    policy can be unit-tested / swapped independently of st.* calls.

    Routing policy (acoustic-source-aware):
      input_method == "youtube" -> Lcrosvila 100% / Mippia   0%
          Mippia's MERT front-end is sensitive to the lossy AAC/Opus re-encode
          YouTube applies; Lcrosvila was retrained on a Clean+AAC+Opus mixed
          distribution and stays accurate on compressed audio, so it alone
          drives the verdict and Mippia's score is surfaced for transparency
          only (flagged invalidated, not blended in).
      input_method == "upload"  -> Mippia  80% / Lcrosvila  20%
          A direct upload is raw, uncompressed audio with no re-encode in the
          pipeline, so Mippia's full-fidelity temporal features are trusted
          and heavily prioritized.
      anything else              -> 50% / 50% fallback (no warning)

    Parameters
    ----------
    lcrosvila_score : float | None   P(AI) from Lcrosvila, or None on error.
    mippia_score    : float | None   P(AI) from Mippia, or None on error.
    input_method    : str            "youtube" | "upload" | other.

    Returns
    -------
    dict with keys:
      fused_p_ai          : float | None  — final blended P(AI)
      verdict              : "AI" | "Human" | "Unknown"
      lcrosvila_weight      : float (0-1)
      mippia_weight         : float (0-1)
      mippia_invalidated    : bool
      warning               : str | None
    """
    lc_weight, mp_weight = 0.5, 0.5
    mippia_invalidated = False
    warning = None

    if input_method == "youtube":
        lc_weight, mp_weight = 1.0, 0.0
        mippia_invalidated = True
        warning = MIPPIA_INVALIDATED_WARNING
    elif input_method == "upload":
        lc_weight, mp_weight = 0.20, 0.80

    # Graceful degradation: if one detector errored, fall back to the other
    # rather than silently treating a missing score as 0.
    if lcrosvila_score is None and mippia_score is None:
        fused = None
    elif lcrosvila_score is None:
        fused = mippia_score
    elif mippia_score is None:
        fused = lcrosvila_score
    else:
        fused = lc_weight * lcrosvila_score + mp_weight * mippia_score

    return {
        "fused_p_ai": fused,
        "verdict": "Unknown" if fused is None else ("AI" if fused >= 0.5 else "Human"),
        "lcrosvila_weight": lc_weight,
        "mippia_weight": mp_weight,
        "mippia_invalidated": mippia_invalidated,
        "warning": warning,
    }


def save_uploaded_file(uploaded_file) -> str:
    """
    Option B — direct file upload.  Write the uploaded bytes to a temp file on
    disk and return its local path.  This COMPLETELY bypasses the yt-dlp
    download phase (no network call, no YouTube re-encode) so the file's
    original .wav/.mp3 bytes reach run_inference() untouched by streaming
    compression — useful for isolating whether YouTube's lossy transcode is
    what's destroying the artifacts MERT/CLAP key off of.
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".wav"
    tmpdir = tempfile.mkdtemp(prefix="app_upload_")
    local_path = os.path.join(tmpdir, f"upload{suffix}")
    with open(local_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return local_path


# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Audio Forensics: AI vs. Human", page_icon="🎧", layout="centered")
st.title("Audio Forensics: AI vs. Human Detection")
st.caption("Resolve a song from a name/URL (downloaded via yt-dlp), or upload a local "
           "audio file directly, then classify it with two open-source detectors "
           "(Lcrosvila & Mippia).")

# Warm the cache once (instant on later reruns thanks to @st.cache_resource)
with st.spinner("Loading detection models (first launch only)…"):
    MODELS = load_models()

# ── Input method: Option A (URL / text search) vs Option B (direct upload) ─────
input_method = st.radio(
    "Choose input method:",
    ["Track Name / URL (yt-dlp)", "Upload Audio File"],
    horizontal=True,
)

query = None
uploaded_file = None

if input_method == "Track Name / URL (yt-dlp)":
    query = st.text_input("Enter a Track Name and Artist (or URL):",
                          placeholder="e.g.  Daft Punk Get Lucky   —or—   https://open.spotify.com/track/…")
else:
    uploaded_file = st.file_uploader(
        "Upload a .wav or .mp3 file:", type=["wav", "mp3"],
        help="Bypasses yt-dlp entirely — useful for isolating whether YouTube's "
             "lossy compression affects detection.")

go = st.button("Analyze Audio", type="primary")

if go:
    # ── Validate the active input method ──────────────────────────────────────
    if input_method == "Track Name / URL (yt-dlp)":
        if not query or not query.strip():
            st.warning("Please enter a track name, artist, or URL first.")
            st.stop()
    else:
        if uploaded_file is None:
            st.warning("Please upload a .wav or .mp3 file first.")
            st.stop()

    with st.spinner("Downloading and processing audio…" if input_method.startswith("Track")
                    else "Processing uploaded audio…"):

        if input_method == "Track Name / URL (yt-dlp)":
            # ── Option A: existing yt-dlp ingestion pipeline ────────────────────
            fusion_input_method = "youtube"   # drives calculate_fusion_verdict()
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

        else:
            # ── Option B: direct file upload — yt-dlp is COMPLETELY bypassed ───
            fusion_input_method = "upload"    # drives calculate_fusion_verdict()
            try:
                wav = save_uploaded_file(uploaded_file)
            except Exception as e:
                st.error(f"Could not read uploaded file: {type(e).__name__}: {e}")
                st.stop()

            # Build a result dict matching the yt-dlp path's shape, so the
            # display section below works unmodified for both input methods.
            result = {
                "Track_Title": uploaded_file.name,
                "Artist_Name": "",
                "Route": "B_DirectUpload",
                "Domain": "local upload",
                "Final_Duration_S": uai.ffprobe_duration(wav),
                "Capped_To_5min": False,
            }

        # ── Inference: same code path for both Option A and Option B ───────────
        # `wav` is just a local file path here, regardless of how it was sourced.
        # It feeds straight into the SAME normalization + 48 kHz clap_embed() /
        # 24 kHz mippia_pai() pipeline — no backend model code is altered.
        infer = run_inference(MODELS, wav)

        # ── Late fusion: combine the two raw scores per the acoustic-source-
        # aware routing policy (logic lives in calculate_fusion_verdict, not here).
        fusion = calculate_fusion_verdict(infer["lc_pai"], infer["mp_pai"], fusion_input_method)

    # ── Resolved-source banner ────────────────────────────────────────────────
    st.success(f"✅ Resolved & analyzed: **{result.get('Track_Title') or query}**"
               + (f" — {result.get('Artist_Name')}" if result.get("Artist_Name") else ""))
    meta_cols = st.columns(3)
    meta_cols[0].caption(f"**Route:** {result.get('Route','')}")
    meta_cols[1].caption(f"**Source:** {result.get('Domain','') or result.get('Search_Stage','')}")
    meta_cols[2].caption(f"**Clip length:** {result.get('Final_Duration_S','?')} s"
                         + ("  (capped 5:00)" if result.get("Capped_To_5min") else ""))

    # ── System verdict (late fusion) ────────────────────────────────────────────
    st.divider()
    st.subheader("🎯 System Verdict (Late Fusion)")

    if fusion["warning"]:
        st.warning(f"⚠️ {fusion['warning']}")

    if fusion["fused_p_ai"] is None:
        st.error("Both detectors failed — no verdict available.")
    else:
        fv_cols = st.columns(3)
        fv_cols[0].metric("Final Prediction", fusion["verdict"])
        fv_cols[1].metric("Fused P(AI)", f"{fusion['fused_p_ai']:.3f}")
        fv_cols[2].metric("Weights (Lcrosvila / Mippia)",
                          f"{fusion['lcrosvila_weight']:.0%} / {fusion['mippia_weight']:.0%}")
        st.progress(min(max(fusion["fused_p_ai"], 0.0), 1.0),
                   text=f"System AI-likelihood {fusion['fused_p_ai']*100:.1f}%")

    st.divider()
    st.subheader("Detector verdicts (individual, unweighted)")
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
        if fusion["mippia_invalidated"]:
            st.caption("⚠️ Shown for reference only — excluded from the system verdict above.")
        if infer["mp_pai"] is None:
            st.error(f"Inference error:\n{infer['mp_err']}")
        else:
            p = infer["mp_pai"]
            st.metric("Prediction", verdict(p))
            st.metric("P(AI)", f"{p:.3f}")
            st.metric("P(Human)", f"{1-p:.3f}")
            st.progress(min(max(p, 0.0), 1.0), text=f"AI-likelihood {p*100:.1f}%")

    # ── Raw agreement note (unweighted; independent of the fusion routing) ─────
    st.divider()
    preds = [verdict(infer["lc_pai"]) if infer["lc_pai"] is not None else None,
             verdict(infer["mp_pai"]) if infer["mp_pai"] is not None else None]
    valid = [p for p in preds if p]
    if valid and all(p == valid[0] for p in valid):
        st.info(f"Both detectors independently agree: **{valid[0]}**.")
    elif len(valid) == 2:
        st.caption(f"Raw detectors disagree — Lcrosvila: **{preds[0]}**, Mippia: **{preds[1]}**. "
                   f"The System Verdict above resolves this via the fusion weights.")

st.divider()
st.caption("Lcrosvila = LAION-CLAP + RBF-SVM, GridSearchCV-retrained on a Clean+AAC+Opus "
           "mixed-codec dataset. Mippia = MERT-AudioCAT + FusionSegmentTransformer (pretrained). "
           "System Verdict = late fusion: 100% Lcrosvila on the YouTube route (Mippia's MERT "
           "front-end is sensitive to lossy compression), 80% Mippia / 20% Lcrosvila on direct "
           "uploads (raw, uncompressed audio). P(AI) is the raw soft score; 0.5 is the binary threshold.")
