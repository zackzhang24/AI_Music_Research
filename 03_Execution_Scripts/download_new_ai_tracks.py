#!/usr/bin/env python3
"""
download_new_ai_tracks.py
─────────────────────────
Source 100 AI-generated music tracks (≥ 120 s) from a VERIFIED repository,
using non-Suno architecture, with no local generation.

Source
──────
Dataset    : awsaf49/sonics  ("SONICS: Synthetic Or Not", ICLR 2025)
URL        : https://huggingface.co/datasets/awsaf49/sonics
Repository : HuggingFace Datasets Hub (verified, peer-reviewed, CC-licensed)
Generator  : Udio  (algorithm tag "udio-120s")  — NON-Suno
Audio      : real pre-existing files, full-length, NOT generated locally
             and NOT trimmed/truncated.

Method
──────
SONICS stores its fake songs as 10 large zip parts.  Rather than download
tens of GB, we use HTTP range requests (remotezip) to read each zip's
central directory and extract ONLY the specific Udio tracks we want.

  1. Read fake_songs.csv metadata → select algorithm == "udio-120s"
     (every such track is 131.22 s, i.e. strictly ≥ 120 s).
  2. Open fake_songs/part_01.zip remotely; intersect with our selection.
  3. Extract candidate .mp3 files via range requests.
  4. Verify EACH track's duration with ffprobe; keep only ≥ 120 s.
  5. Stop once 100 verified tracks are saved to new_ai_tracks_120s/.
  6. Write ai_dataset_source_log.csv with full provenance.

Compliance
──────────
  ✓ Verified open-source repo (HuggingFace, ICLR 2025 dataset)
  ✗ No torrents / sketchy hosts / consumer-platform scraping
  ✗ No generative model weights downloaded — pre-existing audio only
  ✓ ffprobe duration check, ≥ 120 s, files left intact (no trimming)
  ✓ Non-Suno architecture (Udio)
"""

import os, csv, json, subprocess, sys
import pandas as pd
from huggingface_hub import hf_hub_download
from remotezip import RemoteZip
import static_ffmpeg; static_ffmpeg.add_paths()   # puts ffmpeg + ffprobe on PATH

# ── Config ─────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(BASE, "new_ai_tracks_120s")
LOG_CSV     = os.path.join(BASE, "ai_dataset_source_log.csv")
META_DIR    = os.path.join(BASE, ".hf_sonics_meta")

REPO_ID     = "awsaf49/sonics"
REPO_NAME   = "awsaf49/sonics (HuggingFace Datasets — SONICS, ICLR 2025)"
ARCH_LABEL  = "Udio (udio-120s)"
ALGORITHM   = "udio-120s"
ZIP_PART    = "fake_songs/part_01.zip"
ZIP_URL     = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{ZIP_PART}"

TARGET      = 100
MIN_DUR     = 120.0          # strictly ≥ 120 s
BUFFER      = 20             # extra candidates in case any fail verification


def ffprobe_duration(path: str) -> float:
    """Return duration in seconds via ffprobe, or -1.0 on failure."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            stderr=subprocess.STDOUT,
        )
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return -1.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    # ── 1. Metadata: select udio-120s tracks ──────────────────────────────────
    print("=" * 66)
    print("Step 1 — Load SONICS metadata, select non-Suno (Udio) ≥120s tracks")
    print("=" * 66)
    csv_path = hf_hub_download(REPO_ID, "fake_songs.csv",
                               repo_type="dataset", local_dir=META_DIR)
    df = pd.read_csv(csv_path, low_memory=False)

    u120 = df[df["algorithm"] == ALGORITHM].copy()
    # one variant per distinct song id → maximise musical diversity
    u120 = u120[u120["filename"].astype(str).str.endswith("_0")]
    print(f"  udio-120s candidate songs (distinct ids): {len(u120)}")
    print(f"  metadata duration (all): {u120['duration'].min():.2f}s "
          f"– {u120['duration'].max():.2f}s")

    # map metadata filename → expected zip member + duration
    meta_by_member = {
        f"fake_songs/{row.filename}.mp3": row
        for row in u120.itertuples(index=False)
    }

    # ── 2. Locate candidates inside part_01.zip ───────────────────────────────
    print("\n" + "=" * 66)
    print("Step 2 — Open part_01.zip remotely (range requests), find candidates")
    print("=" * 66)
    print(f"  {ZIP_URL}")
    with RemoteZip(ZIP_URL) as z:
        members_in_zip = set(z.namelist())
        candidates = [m for m in meta_by_member if m in members_in_zip]
        candidates.sort()
        n_take = min(len(candidates), TARGET + BUFFER)
        candidates = candidates[:n_take]
        print(f"  udio-120s candidates available in part_01: "
              f"{len([m for m in meta_by_member if m in members_in_zip])}")
        print(f"  extracting up to {n_take} (target {TARGET} + {BUFFER} buffer)\n")

        # ── 3-4. Extract, ffprobe-verify, keep ≥120s ──────────────────────────
        print("=" * 66)
        print("Step 3/4 — Extract via range requests + ffprobe verify (≥120 s)")
        print("=" * 66)

        log_rows = []
        kept = 0
        for member in candidates:
            if kept >= TARGET:
                break
            meta = meta_by_member[member]
            out_name = f"new_ai_track_{kept+1:03d}.mp3"
            out_path = os.path.join(OUT_DIR, out_name)

            # extract bytes for this single member (HTTP range request)
            try:
                with z.open(member) as fh:
                    data = fh.read()
                with open(out_path, "wb") as out:
                    out.write(data)
            except Exception as exc:
                print(f"  SKIP {member} — extract error: {exc}")
                continue

            # verify duration with ffprobe BEFORE accepting (no trimming)
            dur = ffprobe_duration(out_path)
            if dur < MIN_DUR:
                print(f"  REJECT {out_name} ({member.split('/')[-1]}) "
                      f"— ffprobe {dur:.2f}s < {MIN_DUR}s")
                os.remove(out_path)
                continue

            kept += 1
            size_kb = os.path.getsize(out_path) // 1024
            print(f"  [{kept:3d}/{TARGET}] {out_name}  ←  {member.split('/')[-1]}  "
                  f"ffprobe={dur:.2f}s  ({size_kb} KB)")

            log_rows.append({
                "Filename":               out_name,
                "Source_URL":             ZIP_URL,
                "Repository_Name":        REPO_NAME,
                "Generator_Architecture": ARCH_LABEL,
                "Duration":               round(dur, 3),
                "Original_SONICS_File":   member.split("/")[-1],
                "SONICS_ID":              meta.id,
                "SONICS_Split":           meta.split,
            })

    # ── 5. Write provenance log ───────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("Step 5 — Write ai_dataset_source_log.csv")
    print("=" * 66)
    fieldnames = [
        "Filename", "Source_URL", "Repository_Name",
        "Generator_Architecture", "Duration",
        "Original_SONICS_File", "SONICS_ID", "SONICS_Split",
    ]
    with open(LOG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(log_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Verified tracks saved : {kept}/{TARGET}")
    print(f"  Output directory      : {OUT_DIR}")
    print(f"  Provenance log        : {LOG_CSV}")
    if log_rows:
        durs = [r["Duration"] for r in log_rows]
        print(f"  Duration (ffprobe)    : {min(durs):.2f}s – {max(durs):.2f}s "
              f"(all ≥ {MIN_DUR}s)")
    if kept < TARGET:
        print(f"\n  ⚠  Only {kept}/{TARGET} verified — re-run to extract from more zip parts.")


if __name__ == "__main__":
    main()
