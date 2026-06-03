#!/usr/bin/env python3
"""
rebuild_and_standardize.py
──────────────────────────
Phase 1  Delete the 50 × 5-second MusicGen AI tracks.
Phase 2  Download 50 verified AI tracks (Suno, all ≥30 s) from
         Kukedlc/suno-ai-music-dataset on HuggingFace.
         Source: hf_hub_download — no API key, no browser.
Phase 3  Use pydub to cut a 30-second clip from the MIDDLE of every
         file (50 human + 50 AI) and write them to standardized_songs_30s/.

Audio source
────────────
Dataset : Kukedlc/suno-ai-music-dataset
URL     : https://huggingface.co/datasets/Kukedlc/suno-ai-music-dataset
Content : Suno v4/v5 AI-generated music (instrumental + vocal)
Licence : Public dataset on HuggingFace Hub
"""

import os, gc, glob, time, shutil
import numpy as np
import static_ffmpeg; static_ffmpeg.add_paths()   # puts ffmpeg + ffprobe on PATH

from pydub                 import AudioSegment
from datasets              import load_dataset
from huggingface_hub       import hf_hub_download

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
SONGS_DIR     = os.path.join(BASE, "songs")
STD_DIR       = os.path.join(BASE, "standardized_songs_30s")
HF_CACHE      = os.path.join(BASE, ".hf_audio_cache")

CLIP_MS       = 30_000    # 30 seconds
MIN_DURATION  = 32        # seconds — skip anything shorter than this
TARGET_AI     = 50
EXPORT_BITRATE = "128k"
REPO_ID       = "Kukedlc/suno-ai-music-dataset"


# ── Helpers ───────────────────────────────────────────────────────────────────

def middle_clip(seg: AudioSegment, clip_ms: int = CLIP_MS) -> AudioSegment:
    """Extract a clip_ms window centred on the middle of seg."""
    total = len(seg)
    start = max(0, (total - clip_ms) // 2)
    return seg[start : start + clip_ms]


def load_audio(path: str) -> AudioSegment:
    return AudioSegment.from_file(path)


def export_mp3(seg: AudioSegment, path: str):
    seg.export(path, format="mp3", bitrate=EXPORT_BITRATE)


# ── Phase 1 — delete old 5-second AI tracks ───────────────────────────────────

def phase1_delete():
    print("=" * 62)
    print("Phase 1 — Deleting 50 × 5-second MusicGen AI tracks")
    print("=" * 62)
    deleted = 0
    for f in sorted(glob.glob(os.path.join(SONGS_DIR, "ai_track_*.mp3"))):
        os.remove(f)
        print(f"  deleted  {os.path.basename(f)}")
        deleted += 1
    print(f"  → {deleted} file(s) removed\n")


# ── Phase 2 — download 50 AI tracks from HuggingFace ─────────────────────────

def phase2_download():
    print("=" * 62)
    print(f"Phase 2 — Downloading {TARGET_AI} AI tracks from HuggingFace")
    print(f"  Repo   : {REPO_ID}")
    print(f"  Method : hf_hub_download (no auth required)")
    print("=" * 62)

    os.makedirs(HF_CACHE, exist_ok=True)

    # Stream metadata to find tracks with duration ≥ MIN_DURATION
    print(f"  Streaming metadata to find ≥{TARGET_AI} tracks with duration ≥{MIN_DURATION}s…")
    ds = load_dataset(REPO_ID, split="train", streaming=True)

    downloaded = 0
    scanned    = 0

    for row in ds:
        if downloaded >= TARGET_AI:
            break

        scanned += 1
        duration = float(row.get("duration") or 0)
        fname    = row.get("file_name", "")   # e.g. "audio/uuid.mp3"

        if duration < MIN_DURATION or not fname:
            continue

        track_num  = downloaded + 1
        dest_name  = f"ai_track_{track_num:02d}.mp3"
        dest_path  = os.path.join(SONGS_DIR, dest_name)

        print(f"  [{track_num:02d}/{TARGET_AI}]  {fname.split('/')[-1][:44]}  "
              f"({duration:.0f}s)  →  {dest_name}")

        try:
            cached = hf_hub_download(
                repo_id    = REPO_ID,
                filename   = fname,
                repo_type  = "dataset",
                local_dir  = HF_CACHE,
            )
            shutil.copy2(cached, dest_path)
            actual_size = os.path.getsize(dest_path)
            print(f"          saved  {actual_size // 1024} KB")
            downloaded += 1
        except Exception as e:
            print(f"          SKIP — download error: {e}")

        gc.collect()
        time.sleep(0.2)   # polite rate-limiting

    print(f"\n  Downloaded {downloaded}/{TARGET_AI} AI tracks  "
          f"(scanned {scanned} metadata rows)\n")
    return downloaded


# ── Phase 3 — standardize all 100 tracks to 30-second clips ──────────────────

def phase3_standardize():
    print("=" * 62)
    print(f"Phase 3 — Standardizing all tracks → 30-second clips")
    print(f"  Output : {STD_DIR}")
    print("=" * 62)

    os.makedirs(STD_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(SONGS_DIR, "*.mp3")))
    human     = [f for f in all_files if os.path.basename(f).startswith("human_track_")]
    ai        = [f for f in all_files if os.path.basename(f).startswith("ai_track_")]

    print(f"  Human tracks: {len(human)}")
    print(f"  AI tracks   : {len(ai)}")
    print(f"  Total       : {len(all_files)}\n")

    ok = skipped = 0

    for fpath in sorted(human + ai):
        fname     = os.path.basename(fpath)
        dest_path = os.path.join(STD_DIR, fname)

        try:
            seg      = load_audio(fpath)
            dur_s    = len(seg) / 1000

            if dur_s < 30:
                print(f"  SKIP  {fname}  ({dur_s:.1f}s < 30s)")
                skipped += 1
                continue

            clip = middle_clip(seg)
            export_mp3(clip, dest_path)
            print(f"  ✓  {fname:<28}  {dur_s:>6.1f}s  →  30.0s  "
                  f"({os.path.getsize(dest_path)//1024} KB)")
            ok += 1

        except Exception as e:
            print(f"  ERROR  {fname}: {e}")
            skipped += 1

        # Memory cleanup every iteration
        try: del seg, clip
        except NameError: pass
        gc.collect()

    print(f"\n  Standardized : {ok}")
    print(f"  Skipped      : {skipped}")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    phase1_delete()
    n_ai = phase2_download()
    n_std = phase3_standardize()

    # Final report
    ai_in_songs = len(glob.glob(os.path.join(SONGS_DIR, "ai_track_*.mp3")))
    human_in_songs = len(glob.glob(os.path.join(SONGS_DIR, "human_track_*.mp3")))
    std_files = len(glob.glob(os.path.join(STD_DIR, "*.mp3")))

    print()
    print("=" * 62)
    print("COMPLETE")
    print("=" * 62)
    print(f"  songs/             : {human_in_songs} human + {ai_in_songs} AI = "
          f"{human_in_songs + ai_in_songs} tracks")
    print(f"  standardized_songs_30s/ : {std_files} × 30-second clips")

    if ai_in_songs < TARGET_AI:
        print(f"\n  ⚠  Only {ai_in_songs}/50 AI tracks downloaded.")
        print("     Re-run to resume — already-saved files are skipped.")
    if std_files < human_in_songs + ai_in_songs:
        missing = (human_in_songs + ai_in_songs) - std_files
        print(f"\n  ⚠  {missing} file(s) were too short (<30s) and not standardized.")


if __name__ == "__main__":
    main()
