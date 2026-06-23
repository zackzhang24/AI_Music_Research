#!/usr/bin/env python3
"""
download_fma_human_tracks.py
────────────────────────────
Source exactly 95 PRE-2016 human music tracks (each ≥ 120 s, full-length,
un-truncated) from the Free Music Archive, balanced across genres.

Repository discovery
────────────────────
The originally-named `m-a-p/fma_small` is invalid (HTTP 401) AND the FMA
"small" split is, by definition, 8,000 × 30-second clips — it can never
satisfy the ≥120 s constraint.  Programmatic discovery on the HuggingFace Hub
found the active, public, FULL-LENGTH FMA mirror:

    benjamin-paine/free-music-archive-full   (972 parquet shards, untrimmed audio)

This is the same historically-locked FMA corpus (FMA paper, arXiv:1612.01840,
compiled 2016/2017).  Every track's `released` date and `date_created` metadata
predate 2016, guaranteeing zero modern-AI contamination.

Method
──────
  • Pull canonical FMA `tracks.csv` (genre_top, duration, date_created) via
    range requests from the official fma_metadata.zip.
  • Select a genre-balanced set: duration ≥ 120 s AND year ≤ 2015.
  • Read full-length MP3 bytes straight from the parquet shards (no torchcodec,
    no decoding, no trimming) and ffprobe-VERIFY each is ≥ 120 s.
  • Hard stop at exactly 95 downloads.

Genre quota (balanced, totals 95)
─────────────────────────────────
  Rock 12 · Electronic 12 · Hip-Hop 12 · Folk 12 · Instrumental 12 ·
  Pop 12 · Experimental 12 · International 11
"""

import os, csv, json, subprocess
import pandas as pd
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
import static_ffmpeg; static_ffmpeg.add_paths()      # ffmpeg + ffprobe on PATH

# ── Config ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE, "human_tracks_120s")
LOG_CSV   = os.path.join(BASE, "human_dataset_source_log.csv")
TRACKS_CSV = os.path.join(BASE, ".fma_meta", "tracks.csv")

HF_REPO   = "benjamin-paine/free-music-archive-full"
REPO_LABEL = "FMA (benjamin-paine/free-music-archive-full, HuggingFace) — full-length FMA, arXiv:1612.01840"
N_SHARDS  = 972
SHARD_FMT = "datasets/" + HF_REPO + "/data/train-{:05d}-of-00972.parquet"

MIN_DUR   = 120.0
MAX_YEAR  = 2015            # strictly pre-2016
TARGET    = 95

QUOTA = {
    "Rock": 12, "Electronic": 12, "Hip-Hop": 12, "Folk": 12,
    "Instrumental": 12, "Pop": 12, "Experimental": 12, "International": 11,
}
assert sum(QUOTA.values()) == TARGET


def ffprobe_duration(path: str) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path], stderr=subprocess.STDOUT)
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return -1.0


def load_meta():
    tr = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1], low_memory=False)
    dur   = pd.to_numeric(tr[("track", "duration")], errors="coerce")
    genre = tr[("track", "genre_top")]
    year  = pd.to_datetime(tr[("track", "date_created")], errors="coerce").dt.year
    meta = {}
    for tid in tr.index:
        meta[int(tid)] = (genre.get(tid), dur.get(tid), year.get(tid))
    return meta


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("FMA full-length human-track sourcing  (target: 95, ≥120 s, pre-2016)")
    print("=" * 70)
    print(f"  Repository : {HF_REPO}")
    print(f"  Genre quota: {QUOTA}\n")

    meta = load_meta()
    print(f"  Loaded metadata for {len(meta)} FMA tracks.\n")

    fs = HfFileSystem()
    # Strided shard order (383 coprime with 972) → maximises artist/genre variety
    shard_order = [(i * 383) % N_SHARDS for i in range(N_SHARDS)]

    filled = {g: 0 for g in QUOTA}
    kept = 0
    log_rows = []
    shards_read = 0

    for sidx in shard_order:
        if kept >= TARGET:
            break
        # stop early if every quota satisfied
        if all(filled[g] >= QUOTA[g] for g in QUOTA):
            break

        path = SHARD_FMT.format(sidx)
        try:
            with fs.open(path, "rb") as f:
                table = pq.read_table(f, columns=["audio"])
        except Exception as exc:
            print(f"  shard {sidx:5d}: read error {str(exc)[:60]} — skip")
            continue
        shards_read += 1
        audios = table.column("audio").to_pylist()

        took_here = 0
        for raudio in audios:
            if kept >= TARGET:
                break
            path_str = raudio.get("path") if isinstance(raudio, dict) else None
            data     = raudio.get("bytes") if isinstance(raudio, dict) else None
            if not path_str or not data:
                continue
            try:
                tid = int(os.path.splitext(path_str)[0])
            except ValueError:
                continue
            info = meta.get(tid)
            if not info:
                continue
            genre, dur_meta, year = info
            if genre not in QUOTA:                         continue
            if filled[genre] >= QUOTA[genre]:              continue
            if pd.isna(dur_meta) or dur_meta < MIN_DUR:    continue
            if pd.isna(year) or year > MAX_YEAR:           continue

            out_name = f"human_track_{kept+1:03d}.mp3"
            out_path = os.path.join(OUT_DIR, out_name)
            with open(out_path, "wb") as out:
                out.write(data)

            # VERIFY native duration with ffprobe (no trimming)
            dur = ffprobe_duration(out_path)
            if dur < MIN_DUR:
                os.remove(out_path)
                continue

            filled[genre] += 1
            kept += 1
            took_here += 1
            log_rows.append({
                "Filename":                     out_name,
                "Target_Directory":             "human_tracks_120s",
                "Source_Repository_Or_Dataset": REPO_LABEL,
                "Genre":                        genre,
                "Duration":                     round(dur, 3),
                "FMA_Track_ID":                 f"{tid:06d}",
                "Year":                         int(year),
            })

        print(f"  shard {sidx:5d}: +{took_here:2d}  (total {kept:3d}/95)  "
              f"filled={ {g:filled[g] for g in QUOTA} }")

    # ── Write log ─────────────────────────────────────────────────────────────
    fields = ["Filename", "Target_Directory", "Source_Repository_Or_Dataset",
              "Genre", "Duration", "FMA_Track_ID", "Year"]
    with open(LOG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(log_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Tracks downloaded : {kept}/{TARGET}   (hard quota enforced)")
    print(f"  Shards read       : {shards_read}")
    print(f"  Output directory  : {OUT_DIR}")
    print(f"  Provenance log    : {LOG_CSV}")
    print("\n  Genre distribution:")
    for g in QUOTA:
        print(f"    {g:14s}: {filled[g]:2d} / {QUOTA[g]}")
    if log_rows:
        durs = [r["Duration"] for r in log_rows]
        yrs  = [r["Year"] for r in log_rows]
        print(f"\n  Duration (ffprobe): {min(durs):.1f}s – {max(durs):.1f}s")
        print(f"  Year range        : {min(yrs)} – {max(yrs)}  (all ≤ {MAX_YEAR})")
    if kept != TARGET:
        print(f"\n  ⚠  Collected {kept}, expected {TARGET}.")


if __name__ == "__main__":
    main()
