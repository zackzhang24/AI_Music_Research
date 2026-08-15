#!/usr/bin/env python3
"""
run_fusion_benchmark.py  —  end-to-end accuracy of the full Late-Fusion system
──────────────────────────────────────────────────────────────────────────────
Measures the REAL production pipeline on the 100-track ground-truth set:

  • Acoustic layer  : Lcrosvila P(AI) already cached in results.csv.
  • Context layer   : for every track, fetch its YouTube description / tags /
                      top comments (yt-dlp) and write the exact <mp3>.meta.json
                      sidecar universal_ingestor.py produces.
  • Fusion layer    : call the ACTUAL fusion_aggregator.run_fusion() so the
                      benchmark exercises production code, not a re-implementation.

For each track we compare the acoustic-only verdict (P >= 0.525) against the
fused verdict, then report confusion matrices + accuracy / precision / recall.

Caching: sidecars are written next to each mp3 and reused on re-runs.
Note: Spotify Copyright signal returns None without SPOTIPY_* credentials, so
      this measures 3 of the 4 designed signals (acoustic + tags + sentiment).
"""

import os, sys, json, time, contextlib, io
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import yt_dlp
import fusion_aggregator as fa

AI_DIR    = os.path.join(HERE, "data", "ground_truth_ai")
HUMAN_DIR = os.path.join(HERE, "data", "ground_truth_human")
RESULTS   = os.path.join(ROOT, "results.csv")
THR       = fa.FINAL_THRESHOLD               # 0.525
HUMAN_MAX, AI_MIN = 0.525, 0.851             # zone edges (inference_router)


def path_for(fname):
    for d in (AI_DIR, HUMAN_DIR):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None


def fetch_sidecar(mp3_path):
    """Fetch YT metadata + comments, write <mp3>.meta.json (cached)."""
    sidecar = mp3_path + fa.METADATA_SIDECAR_SUFFIX
    if os.path.exists(sidecar):
        return sidecar, "cached"
    vid = os.path.splitext(os.path.basename(mp3_path))[0]
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "noplaylist": True, "getcomments": True,
            "extractor_args": {"youtube": {"max_comments": ["40"]}}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
    comments = [(c.get("text") or "").strip()
                for c in (info.get("comments") or []) if isinstance(c, dict)]
    meta = {"id": info.get("id"), "title": info.get("title"),
            "description": info.get("description") or "",
            "tags": info.get("tags") or [],
            "comments": [c for c in comments if c][:40]}
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    return sidecar, "fetched"


def verdict_str_to_int(v):
    return 1 if v == "[AI]" else 0


def metrics(y_true, y_pred, name):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = len(y_true)
    acc  = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"\n── {name} ──")
    print(f"  Accuracy  {acc:.3f}   Precision {prec:.3f}   Recall {rec:.3f}   "
          f"Specificity {spec:.3f}   F1 {f1:.3f}")
    print(f"  Confusion: TP={tp}  FP={fp}  TN={tn}  FN={fn}   "
          f"(FP = humans mislabeled AI, FN = AI missed)")
    return dict(acc=acc, prec=prec, rec=rec, spec=spec, f1=f1,
                tp=tp, fp=fp, tn=tn, fn=fn)


def main():
    df = pd.read_csv(RESULTS).dropna(subset=["Predicted_Score"]).reset_index(drop=True)
    print(f"Loaded {len(df)} acoustic-scored tracks from results.csv")

    rows, t0 = [], time.time()
    for i, r in df.iterrows():
        fname, label, p_ai = r["Filename"], int(r["True_Label"]), float(r["Predicted_Score"])
        mp3 = path_for(fname)
        if mp3 is None:
            print(f"  [skip] {fname}: mp3 not on disk"); continue

        # match inference_router zoning
        if p_ai < HUMAN_MAX:      verdict, fuse = "CONFIRMED_HUMAN", False
        elif p_ai <= AI_MIN:      verdict, fuse = "AMBIGUOUS_OVERLAP", True
        else:                     verdict, fuse = "CONFIRMED_AI", False

        # context only needed (and only used) when fusion actually runs
        note = "bypass"
        if fuse:
            try:
                _, note = fetch_sidecar(mp3)
            except Exception as exc:
                note = f"fetch-fail({type(exc).__name__})"

        payload = {"file_path": mp3, "routing_path": "benchmark",
                   "raw_probability": p_ai, "acoustic_verdict": verdict,
                   "requires_fusion_processing": fuse}
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            res = fa.run_fusion(payload)

        acoustic_pred = 1 if p_ai >= THR else 0
        fused_pred    = verdict_str_to_int(res["final_verdict"])
        rows.append(dict(Filename=fname, True_Label=label, P_acoustic=p_ai,
                         zone=verdict, acoustic_pred=acoustic_pred,
                         fused_p=res["fused_probability"], fused_pred=fused_pred,
                         changed=(acoustic_pred != fused_pred), note=note))
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(df)}  ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    bench_csv = os.path.join(ROOT, "fusion_benchmark_results.csv")
    out.to_csv(bench_csv, index=False)
    print(f"\nSaved per-track results -> {bench_csv}")

    yt = out["True_Label"].tolist()
    print("\n" + "=" * 72)
    print(f"FUSION BENCHMARK  (N={len(out)},  AI={sum(yt)},  Human={len(yt)-sum(yt)})")
    print("Context signals live: Tags (regex) + Sentiment (comments).  "
          "Copyright=None (no creds).")
    print("=" * 72)
    m_ac = metrics(yt, out["acoustic_pred"].tolist(), "ACOUSTIC ONLY (Lcrosvila @0.525)")
    m_fu = metrics(yt, out["fused_pred"].tolist(),   "FULL SYSTEM (acoustic + late fusion)")

    changed = out[out["changed"]]
    print(f"\n── Fusion impact ──")
    print(f"  Verdicts changed by fusion: {len(changed)}/{len(out)}")
    if len(changed):
        for _, c in changed.iterrows():
            truth = "AI" if c["True_Label"] == 1 else "Human"
            direction = ("Human->AI" if c["fused_pred"] == 1 else "AI->Human")
            correct = "✓ fixed" if c["fused_pred"] == c["True_Label"] else "✗ broke"
            print(f"    {c['Filename']:<20} truth={truth:<6} "
                  f"P={c['P_acoustic']:.3f}->{c['fused_p']:.3f}  {direction}  {correct}")
    print(f"\n  Accuracy delta: {m_ac['acc']:.3f} -> {m_fu['acc']:.3f}  "
          f"({(m_fu['acc']-m_ac['acc'])*100:+.1f} pts)")
    print(f"  False positives (humans flagged AI): {m_ac['fp']} -> {m_fu['fp']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
