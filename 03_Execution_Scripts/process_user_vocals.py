#!/usr/bin/env python3
"""
process_user_vocals.py
──────────────────────
Crop the user's 5 raw .m4a vocal recordings (at project root) to a contiguous
10.0-second middle window, rename to the dataset schema (human_track_096..100),
store in human_tracks_10s/, delete the raw root files, and append to the log.

  • Middle crop:  Start = (Duration - 10) / 2   (captures active singing)
  • Container/codec preserved: AAC inside .m4a (NOT converted to MP3)
    — re-encoded AAC→AAC with +faststart for clean, corruption-free headers.
"""

import os, csv, json, glob, subprocess, re
import pandas as pd
import static_ffmpeg; static_ffmpeg.add_paths()      # ffmpeg + ffprobe on PATH

BASE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "human_tracks_10s")
LOG_CSV = os.path.join(BASE, "human_dataset_source_log.csv")
CLIP_S  = 10.0


def ffprobe_duration(path: str) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", path], stderr=subprocess.STDOUT)
    return float(json.loads(out)["format"]["duration"])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Source discovery: raw m4a recordings at the OUTERMOST root level ───────
    raw = sorted(glob.glob(os.path.join(BASE, "*.m4a")))
    raw = [p for p in raw if "copy" in os.path.basename(p).lower()
           or re.search(r"track[_ ]?09[6-9]|track[_ ]?100", os.path.basename(p), re.I)]
    print("=" * 66)
    print("Process user vocal recordings → 10.0 s clips")
    print("=" * 66)
    print(f"  Found {len(raw)} raw recording(s) at root:")
    for p in raw:
        print(f"    {os.path.basename(p)}")
    if len(raw) != 5:
        print(f"  WARNING: expected 5 raw files, found {len(raw)}")

    # Map each raw file to its track number from the filename
    def num_of(path):
        m = re.search(r"(09[6-9]|100)", os.path.basename(path))
        return int(m.group(1)) if m else None

    raw_sorted = sorted(raw, key=num_of)

    new_rows = []
    processed = []
    for src in raw_sorted:
        n = num_of(src)
        if n is None:
            print(f"  SKIP (no track number): {src}")
            continue
        out_name = f"human_track_{n:03d}.m4a"
        dst = os.path.join(OUT_DIR, out_name)

        dur   = ffprobe_duration(src)
        start = max(0.0, (dur - CLIP_S) / 2.0)

        # Re-encode AAC→AAC, keep .m4a container, faststart for clean headers
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y",
               "-ss", f"{start:.3f}", "-i", src, "-t", f"{CLIP_S:.3f}",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", dst]
        subprocess.check_call(cmd)

        final = ffprobe_duration(dst)
        print(f"  [{n}] {os.path.basename(src):26s} {dur:7.2f}s "
              f"→ crop@{start:.2f}s → {out_name}  ({final:.2f}s)")

        if abs(final - CLIP_S) > 0.3:
            print(f"      WARN: final {final:.3f}s deviates from 10.0s")

        processed.append(src)
        new_rows.append({
            "Filename":                     out_name,
            "Target_Directory":             "human_tracks_10s",
            "Source_Repository_Or_Dataset": "User_Recording",
            "Genre":                        "Vocal",
            "Duration":                     "",
            "FMA_Track_ID":                 "",
            "Year":                         "",
            "Final_Duration":               round(final, 3),
        })

    # ── Cleanup: delete the raw root files (NOT the human_tracks_120s copies) ──
    print("\n  Cleanup — deleting raw root recordings:")
    for src in processed:
        # safety: only delete files located directly at project root
        if os.path.dirname(os.path.abspath(src)) == BASE:
            os.remove(src)
            print(f"    deleted  {os.path.basename(src)}")

    # ── Append to log ─────────────────────────────────────────────────────────
    log = pd.read_csv(LOG_CSV)
    if "Final_Duration" not in log.columns:
        log["Final_Duration"] = ""
    combined = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(LOG_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    print(f"  Processed vocal clips : {len(new_rows)}/5")
    print(f"  Output directory      : {OUT_DIR}")
    print(f"  Log rows now          : {len(combined)}")
    finals = [r["Final_Duration"] for r in new_rows]
    if finals:
        print(f"  Final_Duration        : {min(finals):.3f}s – {max(finals):.3f}s (~10.0s)")
    print(f"  Raw root files left    : {len(glob.glob(os.path.join(BASE, '*copy*.m4a')))}")


if __name__ == "__main__":
    main()
