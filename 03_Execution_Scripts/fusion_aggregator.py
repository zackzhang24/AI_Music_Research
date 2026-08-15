#!/usr/bin/env python3
"""
fusion_aggregator.py  —  Part 2: Late Fusion module
────────────────────────────────────────────────────
Ingests the acoustic-layer JSON payload emitted by inference_router.py and
executes the multi-modal scoring logic for tracks in the AMBIGUOUS_OVERLAP zone.

Pipeline
────────
  1. JSON ingestion: parse the acoustic payload
       { file_path, routing_path, raw_probability,
         acoustic_verdict, requires_fusion_processing }
  2. BYPASS logic: if requires_fusion_processing is false the acoustic verdict
     is already high-confidence (CONFIRMED_HUMAN / CONFIRMED_AI) — return the
     final verdict immediately, executing ZERO metadata extractors.
  3. Fusion (AMBIGUOUS_OVERLAP only): query the textual/metadata extractors,
     drop any that return None (missing data), dynamically re-normalize the
     FUSION_WEIGHTS over the signals that ARE present, and compute the
     weighted-average Fused Score, strictly bounded to [0.0, 1.0].
  4. Final verdict at the calibrated 0.525 threshold:
       Fused Score >= 0.525 -> [AI]     else -> [Human]

Extractor convention: each takes file_path and returns a normalized float in
[0.0 (Human) .. 1.0 (AI)], or None when the data source has nothing.
  • Copyright — LIVE: spotify_extractor.py (ID3 -> Spotify search -> album
    Composition 'C' copyright), degrading gracefully to None when SPOTIPY_*
    credentials are not exported.
  • Tags      — LIVE: youtube_nlp_extractor.scan_text_for_ai_markers over the
    <mp3>.meta.json sidecar's title/description/tags text.
  • Sentiment — LIVE: youtube_nlp_extractor.analyze_comment_threads over the
    sidecar's comment thread.
All three fail soft: missing data -> None -> weights re-normalize.

Usage
─────
  python3 inference_router.py song.mp3 2>/dev/null | python3 fusion_aggregator.py
  python3 fusion_aggregator.py payload.json
  python3 fusion_aggregator.py --selftest
"""

import os, sys, json, argparse

# ── Configuration ──────────────────────────────────────────────────────────────
# Modular fusion weights — edit here to re-balance the system without touching
# the algorithm. Weights are re-normalized at runtime over available signals,
# so they need not sum to 1.0 (though they do by convention).
FUSION_WEIGHTS = {
    "Acoustic":  0.50,   # Lcrosvila raw_probability (always present)
    "Copyright": 0.25,   # copyright-registry lookup     (0.0 = registered/human)
    "Tags":      0.15,   # platform tag scrape           (1.0 = AI tag found)
    "Sentiment": 0.10,   # comment-section NLP sentiment (0..1 continuous)
}

FINAL_THRESHOLD = 0.525          # calibrated Youden-J cutoff: >= 0.525 -> [AI]


# ── Data extractors (ALL LIVE) ─────────────────────────────────────────────────
#   Copyright — spotify_extractor.py : ID3 tags -> Spotify search -> album
#               Composition ('C') copyright — registered human songwriting
#   Tags      — youtube_nlp_extractor: regex AI-marker scan of the description/
#               tags text in the <mp3>.meta.json sidecar (universal_ingestor)
#   Sentiment — youtube_nlp_extractor: suspicion-density heuristic over the
#               sidecar's comment thread

from spotify_extractor import extract_id3_tags
from copyright_lookup import check_copyright_registry as _catalog_copyright_check
from youtube_nlp_extractor import scan_text_for_ai_markers, analyze_comment_threads

METADATA_SIDECAR_SUFFIX = ".meta.json"   # written by universal_ingestor.py


def _load_sidecar(file_path: str) -> dict:
    """Read the <mp3>.meta.json text payload; {} when absent."""
    sidecar = file_path + METADATA_SIDECAR_SUFFIX
    if not os.path.exists(sidecar):
        return {}
    with open(sidecar, encoding="utf-8") as fh:
        return json.load(fh)


def check_copyright_registry(file_path: str):
    """
    Copyright / commercial-release lookup (LIVE — copyright_lookup.py, NO keys).
    Reads the MP3's ID3 title/artist, then queries the free public iTunes +
    MusicBrainz catalogs.
      0.0  -> a confident commercial-catalog release exists (label/album/year)
              => registered human recording (Confirmed Human)
      None -> no ID3 metadata, no confident catalog match, or a network error
              (signal excluded; fusion re-normalizes remaining weights)
    """
    try:
        title, artist = extract_id3_tags(file_path)
        return _catalog_copyright_check(title, artist)
    except Exception as exc:                   # network / bad tags — never crash fusion
        print(f"[Warning] copyright lookup failed ({type(exc).__name__}: {exc}); "
              f"skipping.", file=sys.stderr)
        return None


def scrape_platform_tags(file_path: str):
    """
    Platform tag / description scan (LIVE — youtube_nlp_extractor).
      1.0  -> non-negated AI marker (suno/udio/elevenlabs/ai cover/...) in the
              sidecar's title + description + tags text (strong AI signal)
      None -> no sidecar, no text, or markers only in negated context
    """
    try:
        meta = _load_sidecar(file_path)
        blob = " ".join(filter(None, [
            meta.get("title"), meta.get("description"),
            " ".join(meta.get("tags") or [])]))
        return scan_text_for_ai_markers(blob)
    except Exception as exc:                   # missing/corrupt text data
        print(f"[Warning] platform-tag scan failed ({type(exc).__name__}: {exc}); "
              f"skipping.", file=sys.stderr)
        return None


def analyze_comment_sentiment(file_path: str):
    """
    Comment-thread suspicion heuristic (LIVE — youtube_nlp_extractor).
      float in [0,1] -> suspicion density over the sidecar's comment thread
      None           -> no sidecar / no comments (signal excluded from fusion)
    """
    try:
        return analyze_comment_threads(_load_sidecar(file_path).get("comments"))
    except Exception as exc:                   # missing/corrupt text data
        print(f"[Warning] comment analysis failed ({type(exc).__name__}: {exc}); "
              f"skipping.", file=sys.stderr)
        return None


EXTRACTORS = {
    "Copyright": check_copyright_registry,
    "Tags":      scrape_platform_tags,
    "Sentiment": analyze_comment_sentiment,
}


# ── JSON ingestion ─────────────────────────────────────────────────────────────
def parse_acoustic_payload(raw: str) -> dict:
    """Parse + validate the acoustic layer's JSON payload."""
    payload = json.loads(raw)
    required = ["file_path", "raw_probability", "acoustic_verdict",
                "requires_fusion_processing"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"acoustic payload missing key(s): {missing}")
    if payload["raw_probability"] is None:
        raise ValueError(f"acoustic layer reported no score "
                         f"(verdict={payload['acoustic_verdict']})")
    return payload


# ── The fusion algorithm ───────────────────────────────────────────────────────
def fuse(acoustic_prob: float, extractor_results: dict) -> float:
    """
    Dynamically-weighted average of the acoustic probability and every
    extractor that did NOT return None. Weights are re-normalized over the
    available signals; the result is strictly clamped to [0.0, 1.0].
    """
    signals = {"Acoustic": acoustic_prob}
    signals.update({name: val for name, val in extractor_results.items()
                    if val is not None})

    weight_sum = sum(FUSION_WEIGHTS[name] for name in signals)
    fused = sum(FUSION_WEIGHTS[name] * val for name, val in signals.items()) / weight_sum
    return max(0.0, min(1.0, fused))       # strict [0,1] bound


def final_verdict(prob: float) -> str:
    return "[AI]" if prob >= FINAL_THRESHOLD else "[Human]"


# ── Orchestration ──────────────────────────────────────────────────────────────
def run_fusion(payload: dict) -> dict:
    file_path = payload["file_path"]
    acoustic  = float(payload["raw_probability"])

    print("=" * 66)
    print("LATE FUSION AGGREGATOR — Part 2")
    print("=" * 66)
    print(f"  File             : {os.path.basename(file_path)}")
    print(f"  Routing          : {payload.get('routing_path', 'n/a')}")
    print(f"  Acoustic score   : {acoustic:.3f}  ({payload['acoustic_verdict']})")

    # ── BYPASS: high-confidence acoustic verdict, skip all metadata checks ─────
    if not payload["requires_fusion_processing"]:
        verdict = final_verdict(acoustic)
        print("-" * 66)
        print("  Fusion           : BYPASSED (high-confidence acoustic zone —")
        print("                     no metadata extractors executed)")
        print(f"  FINAL VERDICT    : {verdict}  (acoustic score {acoustic:.3f} "
              f"vs threshold {FINAL_THRESHOLD})")
        print("=" * 66)
        return {"file_path": file_path, "fused_probability": acoustic,
                "fusion_executed": False, "triggered_modifiers": [],
                "final_verdict": verdict}

    # ── AMBIGUOUS_OVERLAP: execute the textual/metadata extractors ─────────────
    print("-" * 66)
    print("  Zone             : AMBIGUOUS_OVERLAP -> executing metadata extractors")

    results, triggered = {}, []
    for name, fn in EXTRACTORS.items():
        val = fn(file_path)
        results[name] = val
        if val is None:
            print(f"    · {name:<9} : —      (no data — excluded from fusion)")
        else:
            lean = "AI" if val >= 0.5 else "Human"
            triggered.append(f"{name}={val:.3f} ({lean} lean)")
            print(f"    · {name:<9} : {val:.3f}  (weight {FUSION_WEIGHTS[name]:.2f}, {lean} lean)")

    fused = fuse(acoustic, results)
    verdict = final_verdict(fused)

    active = ["Acoustic"] + [n for n, v in results.items() if v is not None]
    w_sum  = sum(FUSION_WEIGHTS[n] for n in active)

    print("-" * 66)
    print(f"  Triggered modifiers : {triggered if triggered else 'none (acoustic-only fallback)'}")
    print(f"  Active signals      : {', '.join(active)}  "
          f"(weights re-normalized over {w_sum:.2f})")
    print(f"  FUSED SCORE         : {fused:.3f}")
    print(f"  FINAL VERDICT       : {verdict}  "
          f"({fused:.3f} {'>=' if fused >= FINAL_THRESHOLD else '<'} {FINAL_THRESHOLD})")
    print("=" * 66)

    return {"file_path": file_path, "fused_probability": round(fused, 3),
            "fusion_executed": True, "triggered_modifiers": triggered,
            "final_verdict": verdict}


# ── Self-test ──────────────────────────────────────────────────────────────────
def _selftest():
    ok = True

    # 1. Bypass path: no extractor may run.
    calls = []
    orig = dict(EXTRACTORS)
    for name in EXTRACTORS:
        EXTRACTORS[name] = (lambda n: lambda fp: calls.append(n))(name)
    run_fusion({"file_path": "x.mp3", "routing_path": "t", "raw_probability": 0.30,
                "acoustic_verdict": "CONFIRMED_HUMAN", "requires_fusion_processing": False})
    EXTRACTORS.update(orig)
    print(f"\n[{'PASS' if not calls else 'FAIL'}] bypass ran zero extractors ({calls})")
    ok &= not calls

    # 2. Weighted math: acoustic .70, copyright 0.0, tags 1.0, sentiment .5
    #    -> (.5*.7 + .25*0 + .15*1 + .10*.5) / 1.0 = 0.55
    fused = fuse(0.70, {"Copyright": 0.0, "Tags": 1.0, "Sentiment": 0.5})
    print(f"[{'PASS' if abs(fused - 0.55) < 1e-9 else 'FAIL'}] full-signal fusion = {fused:.4f} (expect 0.5500)")
    ok &= abs(fused - 0.55) < 1e-9

    # 3. Dynamic re-normalization: only sentiment present
    #    -> (.5*.8 + .10*.2) / .60 = 0.70
    fused = fuse(0.80, {"Copyright": None, "Tags": None, "Sentiment": 0.2})
    print(f"[{'PASS' if abs(fused - 0.70) < 1e-9 else 'FAIL'}] renormalized fusion = {fused:.4f} (expect 0.7000)")
    ok &= abs(fused - 0.70) < 1e-9

    # 4. All extractors missing -> acoustic passthrough, still bounded.
    fused = fuse(0.60, {"Copyright": None, "Tags": None, "Sentiment": None})
    print(f"[{'PASS' if abs(fused - 0.60) < 1e-9 else 'FAIL'}] acoustic-only fallback = {fused:.4f} (expect 0.6000)")
    ok &= abs(fused - 0.60) < 1e-9

    # 5. Strict [0,1] bounds.
    lo, hi = fuse(0.0, {"Copyright": 0.0}), fuse(1.0, {"Tags": 1.0})
    print(f"[{'PASS' if 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0 else 'FAIL'}] bounds: {lo:.3f}, {hi:.3f} in [0,1]")
    ok &= 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0

    # 6. Verdict threshold edges.
    edge = (final_verdict(0.525), final_verdict(0.5249))
    good = edge == ("[AI]", "[Human]")
    print(f"[{'PASS' if good else 'FAIL'}] threshold edges: 0.525->{edge[0]}, 0.5249->{edge[1]}")
    ok &= good

    # 7. Live NLP wrappers survive EMPTY text data: no sidecar on disk must
    #    yield None (not a crash) and full fusion must still run end-to-end.
    t = scrape_platform_tags("/nonexistent/no_sidecar.mp3")
    c = analyze_comment_sentiment("/nonexistent/no_sidecar.mp3")
    good = t is None and c is None
    print(f"[{'PASS' if good else 'FAIL'}] empty text data -> Tags={t}, Sentiment={c} (both None, no crash)")
    ok &= good
    res = run_fusion({"file_path": "/nonexistent/no_sidecar.mp3", "routing_path": "t",
                      "raw_probability": 0.60, "acoustic_verdict": "AMBIGUOUS_OVERLAP",
                      "requires_fusion_processing": True})
    good = res["final_verdict"] == "[AI]" and abs(res["fused_probability"] - 0.60) < 1e-9
    print(f"[{'PASS' if good else 'FAIL'}] fusion on zero text signals -> acoustic passthrough "
          f"{res['fused_probability']} {res['final_verdict']}")
    ok &= good

    print("\nSELF-TEST:", "ALL PASSED ✅" if ok else "FAILURES ❌")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Late Fusion aggregator (Part 2).")
    ap.add_argument("payload", nargs="?", default="-",
                    help="path to the acoustic JSON payload, or '-' / omitted for stdin")
    args = ap.parse_args()

    raw = sys.stdin.read() if args.payload == "-" else open(args.payload, encoding="utf-8").read()
    try:
        payload = parse_acoustic_payload(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[fusion] invalid acoustic payload: {exc}", file=sys.stderr)
        sys.exit(1)
    run_fusion(payload)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    main()
