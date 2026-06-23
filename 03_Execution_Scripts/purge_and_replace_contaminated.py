#!/usr/bin/env python3
"""
purge_and_replace_contaminated.py
─────────────────────────────────
Data-leakage prevention for the new_ai_tracks_120s benchmark set.

Why
───
The Mippia detector was trained on the SONICS dataset.  Any track that comes
from SONICS' `train` or `valid` split is therefore TRAINING DATA the model has
already seen — evaluating on it is data leakage and inflates the score.

What this does
──────────────
  1. Audit  ai_dataset_source_log.csv  →  analyse SONICS_Split.
  2. Purge  every local audio file whose split is `train` or `valid`.
  3. Keep   the already-clean `test` tracks.
  4. Replace the purged tracks with NEW Udio `udio-120s` tracks pulled
     STRICTLY from the SONICS `test` split, until the count is back to 100.
     Every replacement is ffprobe-verified ≥ 120 s (files left intact).
  5. Renumber the survivors + replacements to new_ai_track_001..100.mp3.
  6. Overwrite ai_dataset_source_log.csv with the corrected, strictly
     test-split dataset.
  7. Print how many contaminated tracks were purged.

Source repo : awsaf49/sonics (HuggingFace Datasets, ICLR 2025)  — verified.
No local generation, no trimming, no non-test tracks survive.
"""

import os, csv, json, glob, subprocess
import pandas as pd
from huggingface_hub import hf_hub_download
from remotezip import RemoteZip
import static_ffmpeg; static_ffmpeg.add_paths()      # ffmpeg + ffprobe on PATH

# ── Config ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE, "new_ai_tracks_120s")
LOG_CSV    = os.path.join(BASE, "ai_dataset_source_log.csv")
META_DIR   = os.path.join(BASE, ".hf_sonics_meta")

REPO_ID    = "awsaf49/sonics"
REPO_NAME  = "awsaf49/sonics (HuggingFace Datasets — SONICS, ICLR 2025)"
ARCH_LABEL = "Udio (udio-120s)"
ALGORITHM  = "udio-120s"
ZIP_PART   = "fake_songs/part_01.zip"
ZIP_URL    = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{ZIP_PART}"

TARGET     = 100
MIN_DUR    = 120.0
CONTAM_SPLITS = {"train", "val", "valid"}
CLEAN_SPLIT   = "test"

LOG_FIELDS = [
    "Filename", "Source_URL", "Repository_Name",
    "Generator_Architecture", "Duration",
    "Original_SONICS_File", "SONICS_ID", "SONICS_Split",
]


def ffprobe_duration(path: str) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path], stderr=subprocess.STDOUT)
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return -1.0


def main():
    # ── 1. Audit ──────────────────────────────────────────────────────────────
    print("=" * 66)
    print("Step 1 — Audit ai_dataset_source_log.csv (SONICS_Split)")
    print("=" * 66)
    df = pd.read_csv(LOG_CSV)
    print("  Split distribution BEFORE purge:")
    for split, n in df["SONICS_Split"].value_counts().items():
        flag = "  ← CONTAMINATED (leakage)" if split in CONTAM_SPLITS else "  ← clean (test)"
        print(f"    {split:6s}: {n:3d}{flag}")

    contaminated = df[df["SONICS_Split"].isin(CONTAM_SPLITS)].copy()
    clean        = df[df["SONICS_Split"] == CLEAN_SPLIT].copy()
    print(f"\n  Contaminated tracks (train/val): {len(contaminated)}")
    print(f"  Clean tracks (test)            : {len(clean)}")

    # ── 2. Purge contaminated audio ───────────────────────────────────────────
    print("\n" + "=" * 66)
    print("Step 2 — Purge contaminated audio files (train/val)")
    print("=" * 66)
    purged = 0
    for fn in contaminated["Filename"]:
        p = os.path.join(OUT_DIR, fn)
        if os.path.exists(p):
            os.remove(p)
            purged += 1
    print(f"  Deleted {purged} contaminated audio file(s).")

    # ── 3. Stage surviving clean (test) tracks to temp names ──────────────────
    print("\n" + "=" * 66)
    print("Step 3 — Preserve clean test-split survivors")
    print("=" * 66)
    survivors = []          # list of dicts: temp_path + metadata
    used_originals = set()
    for i, row in enumerate(clean.itertuples(index=False)):
        src = os.path.join(OUT_DIR, row.Filename)
        if not os.path.exists(src):
            print(f"  WARN: expected survivor missing on disk: {row.Filename}")
            continue
        tmp = os.path.join(OUT_DIR, f"_keep_{i:03d}.mp3")
        os.rename(src, tmp)
        dur = ffprobe_duration(tmp)
        survivors.append({
            "tmp": tmp,
            "Original_SONICS_File": row.Original_SONICS_File,
            "SONICS_ID": row.SONICS_ID,
            "Duration": round(dur, 3),
        })
        used_originals.add(row.Original_SONICS_File)
    print(f"  Preserved {len(survivors)} test-split survivor(s).")

    # ── 4. Pull replacements STRICTLY from test split ─────────────────────────
    need = TARGET - len(survivors)
    print("\n" + "=" * 66)
    print(f"Step 4 — Pull {need} NEW replacements strictly from SONICS test split")
    print("=" * 66)

    csv_path = hf_hub_download(REPO_ID, "fake_songs.csv",
                              repo_type="dataset", local_dir=META_DIR)
    meta = pd.read_csv(csv_path, low_memory=False)
    test_meta = meta[(meta["algorithm"] == ALGORITHM) & (meta["split"] == CLEAN_SPLIT)].copy()
    test_meta = test_meta[test_meta["filename"].astype(str).str.endswith("_0")]
    meta_by_member = {
        f"fake_songs/{r.filename}.mp3": r for r in test_meta.itertuples(index=False)
    }

    replacements = []
    with RemoteZip(ZIP_URL) as z:
        in_zip = set(z.namelist())
        candidates = [m for m in meta_by_member
                      if m in in_zip
                      and meta_by_member[m].filename + ".mp3" not in
                          {o for o in used_originals}
                      and m.split("/")[-1] not in used_originals]
        candidates.sort()
        print(f"  Test-split candidates available in part_01: {len(candidates)}")

        idx = 0
        for member in candidates:
            if len(replacements) >= need:
                break
            orig_file = member.split("/")[-1]
            if orig_file in used_originals:
                continue
            tmp = os.path.join(OUT_DIR, f"_new_{idx:03d}.mp3")
            idx += 1
            try:
                with z.open(member) as fh:
                    data = fh.read()
                with open(tmp, "wb") as out:
                    out.write(data)
            except Exception as exc:
                print(f"  SKIP {orig_file} — extract error: {exc}")
                continue
            dur = ffprobe_duration(tmp)
            if dur < MIN_DUR:
                print(f"  REJECT {orig_file} — ffprobe {dur:.2f}s < {MIN_DUR}s")
                os.remove(tmp)
                continue
            r = meta_by_member[member]
            replacements.append({
                "tmp": tmp,
                "Original_SONICS_File": orig_file,
                "SONICS_ID": r.id,
                "Duration": round(dur, 3),
            })
            used_originals.add(orig_file)
            print(f"  [{len(replacements):3d}/{need}] {orig_file}  "
                  f"ffprobe={dur:.2f}s  (test split)")

    # ── 5. Renumber survivors + replacements → 001..100 ───────────────────────
    print("\n" + "=" * 66)
    print("Step 5 — Renumber final dataset to new_ai_track_001..100.mp3")
    print("=" * 66)
    final = survivors + replacements
    if len(final) != TARGET:
        print(f"  WARN: assembled {len(final)} tracks (target {TARGET}).")

    log_rows = []
    for i, entry in enumerate(final, start=1):
        final_name = f"new_ai_track_{i:03d}.mp3"
        final_path = os.path.join(OUT_DIR, final_name)
        os.rename(entry["tmp"], final_path)
        log_rows.append({
            "Filename":               final_name,
            "Source_URL":             ZIP_URL,
            "Repository_Name":        REPO_NAME,
            "Generator_Architecture": ARCH_LABEL,
            "Duration":               entry["Duration"],
            "Original_SONICS_File":   entry["Original_SONICS_File"],
            "SONICS_ID":              entry["SONICS_ID"],
            "SONICS_Split":           CLEAN_SPLIT,      # strictly test now
        })
    print(f"  Renumbered {len(log_rows)} files.")

    # ── 6. Overwrite log ──────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("Step 6 — Overwrite ai_dataset_source_log.csv (strictly test split)")
    print("=" * 66)
    with open(LOG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(log_rows)
    print(f"  Wrote {len(log_rows)} rows.")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("CORRECTION SUMMARY")
    print("=" * 66)
    print(f"  Contaminated tracks PURGED (train/val) : {purged}")
    print(f"  Clean test-split survivors kept        : {len(survivors)}")
    print(f"  New test-split replacements pulled     : {len(replacements)}")
    print(f"  Final dataset size                     : {len(log_rows)}")
    splits_now = pd.Series([r['SONICS_Split'] for r in log_rows]).value_counts().to_dict()
    print(f"  Final split distribution               : {splits_now}")
    durs = [r['Duration'] for r in log_rows]
    if durs:
        print(f"  Duration range (ffprobe)               : {min(durs):.2f}s – {max(durs):.2f}s")
    n_unique = len({r['Original_SONICS_File'] for r in log_rows})
    print(f"  Unique source tracks (no duplicates)   : {n_unique}/{len(log_rows)}")


if __name__ == "__main__":
    main()
