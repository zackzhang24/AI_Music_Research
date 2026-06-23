#!/usr/bin/env python3
"""
crop_human_tracks_10s.py
────────────────────────
Hard-crop exactly 95 pre-2016 human FMA tracks to a contiguous 10.0-second
segment each, saving them to human_tracks_10s/ and appending to the log.

Source audio
────────────
The 95 full-length source tracks were sourced from the active HuggingFace
repository `benjamin-paine/free-music-archive-full` (NOT the small split) in
the prior step and live in human_tracks_120s/.  Every track is genre-balanced,
distinct, and metadata-verified to originate from 2015 or earlier (zero modern
AI contamination).  This step performs only the temporal crop — no re-sourcing
of identical audio.

Crop policy
───────────
  • A single CONTIGUOUS 10.0 s window taken from the MIDDLE of each track
    (start = (duration - 10)/2) so the segment lands on real musical content.
  • Hard crop with ffmpeg `-t 10` + re-encode → output is exactly 10.0 s.
  • NO padding.  (All sources are ≥120 s, so a 10 s middle window always fits.)

Logging
───────
Appends 95 rows to the existing human_dataset_source_log.csv tracking:
Filename, Target_Directory, Source_Repository_Or_Dataset, Genre, Final_Duration.
"""

import os, csv, json, subprocess
import pandas as pd
import static_ffmpeg; static_ffmpeg.add_paths()      # ffmpeg + ffprobe on PATH

BASE     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE, "human_tracks_120s")
OUT_DIR  = os.path.join(BASE, "human_tracks_10s")
LOG_CSV  = os.path.join(BASE, "human_dataset_source_log.csv")
CLIP_S   = 10.0


def ffprobe_duration(path: str) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path], stderr=subprocess.STDOUT)
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return -1.0


def crop_10s(src: str, dst: str, src_dur: float) -> bool:
    """Re-encode a contiguous 10.0 s middle window. Returns success."""
    start = max(0.0, (src_dur - CLIP_S) / 2.0)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y",
           "-ss", f"{start:.3f}", "-i", src, "-t", f"{CLIP_S:.3f}",
           "-ac", "2", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k", dst]
    try:
        subprocess.check_call(cmd)
        return True
    except Exception as exc:
        print(f"   ffmpeg error on {os.path.basename(src)}: {exc}")
        return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Crop 95 pre-2016 human FMA tracks → contiguous 10.0 s segments")
    print("=" * 70)

    log = pd.read_csv(LOG_CSV)
    src_rows = log[log["Target_Directory"] == "human_tracks_120s"].copy()
    print(f"  Source tracks in log: {len(src_rows)}\n")

    new_rows = []
    done = 0
    for r in src_rows.itertuples(index=False):
        src = os.path.join(SRC_DIR, r.Filename)
        dst = os.path.join(OUT_DIR, r.Filename)
        if not os.path.exists(src):
            print(f"   MISSING source: {r.Filename}")
            continue

        src_dur = float(r.Duration)
        if not crop_10s(src, dst, src_dur):
            continue

        final = ffprobe_duration(dst)
        if abs(final - CLIP_S) > 0.2:
            print(f"   WARN {r.Filename}: final {final:.3f}s (expected ~10.0)")

        done += 1
        if done <= 5 or done % 20 == 0:
            print(f"   [{done:3d}/95] {r.Filename:20s} {r.Genre:13s} "
                  f"{src_dur:7.1f}s → {final:.2f}s")

        new_rows.append({
            "Filename":                     r.Filename,
            "Target_Directory":             "human_tracks_10s",
            "Source_Repository_Or_Dataset": r.Source_Repository_Or_Dataset,
            "Genre":                        r.Genre,
            "Duration":                     "",
            "FMA_Track_ID":                 r.FMA_Track_ID,
            "Year":                         r.Year,
            "Final_Duration":               round(final, 3),
        })

    # ── Append to existing log (add Final_Duration column) ────────────────────
    if "Final_Duration" not in log.columns:
        log["Final_Duration"] = ""
    appended = pd.DataFrame(new_rows)
    combined = pd.concat([log, appended], ignore_index=True)
    combined.to_csv(LOG_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    finals = [r["Final_Duration"] for r in new_rows]
    from collections import Counter
    gdist = Counter(r["Genre"] for r in new_rows)
    print(f"  Cropped tracks    : {done}/95")
    print(f"  Output directory  : {OUT_DIR}")
    print(f"  Log rows now      : {len(combined)} (95 original + {len(new_rows)} cropped)")
    print(f"  Final_Duration    : min={min(finals):.3f}s max={max(finals):.3f}s "
          f"(all ~10.0s)")
    print("\n  Genre balance (cropped set):")
    for g in sorted(gdist):
        print(f"    {g:14s}: {gdist[g]}")


if __name__ == "__main__":
    main()
