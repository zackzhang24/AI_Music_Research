#!/usr/bin/env python3
"""
Audit human_track_*.mp3 ID3 metadata.
Reports Year, Artist, Album for each file and flags anything 2023+ or fully blank.
"""

import os
import glob
import re
from mutagen.id3 import ID3, ID3NoHeaderError
from mutagen.mp3 import MP3

SONGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "songs")
AI_CUTOFF_YEAR = 2023   # Suno/Udio/MusicGen went public ~2023


# ── Source provenance ────────────────────────────────────────────────────────

SOURCE_INFO = """
SOURCE REPOSITORY
=================
Platform : Internet Archive  (https://archive.org)
Collection: netlabels         (https://archive.org/details/netlabels)
License  : Creative Commons (various CC-BY, CC-BY-SA, CC-BY-NC, CC-BY-NC-SA)
API used : https://archive.org/advancedsearch.php  (no key required)
           https://archive.org/metadata/<identifier>  (no key required)
Download : https://archive.org/download/<identifier>/<filename>

The netlabels collection is a curated archive of music released by
independent internet labels under Creative Commons licences — all
human-recorded, not AI-generated. Items were sorted by download count
(most popular first) to bias toward well-known, verified releases.
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_year(raw):
    """Pull a 4-digit year from any tag string (handles 'YYYY-MM-DD', etc.)."""
    if not raw:
        return None
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", str(raw))
    return int(m.group(1)) if m else None


def read_tags(path):
    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = {}

    def first(tag_id):
        frame = tags.get(tag_id)
        if frame is None:
            return ""
        v = str(frame)
        return v.strip()

    artist = first("TPE1") or first("TPE2")
    album  = first("TALB")
    year   = first("TDRC") or first("TYER") or first("TDAT")

    # Try to get duration via MP3 header regardless of ID3 presence
    try:
        audio = MP3(path)
        duration_s = int(audio.info.length)
    except Exception:
        duration_s = 0

    return {
        "artist": artist,
        "album":  album,
        "year_raw": year,
        "year": extract_year(year),
        "duration_s": duration_s,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(SOURCE_INFO)

    files = sorted(glob.glob(os.path.join(SONGS_DIR, "human_track_*.mp3")))
    if not files:
        print(f"ERROR: no human_track_*.mp3 files found in {SONGS_DIR}")
        return

    rows = []
    for f in files:
        name = os.path.basename(f)
        size_kb = os.path.getsize(f) // 1024
        info = read_tags(f)
        rows.append({"file": name, "size_kb": size_kb, **info})

    # ── Per-file table ────────────────────────────────────────────────────────
    print("=" * 90)
    print(f"{'FILE':<24} {'YEAR':>6}  {'ARTIST':<30}  {'ALBUM'}")
    print("=" * 90)

    flagged_new   = []   # year >= AI_CUTOFF_YEAR
    flagged_blank = []   # all three fields empty

    for r in rows:
        year_str   = str(r["year"]) if r["year"] else "—"
        artist_str = r["artist"][:29] if r["artist"] else "—"
        album_str  = r["album"][:38] if r["album"]  else "—"
        print(f"{r['file']:<24} {year_str:>6}  {artist_str:<30}  {album_str}")

        if r["year"] and r["year"] >= AI_CUTOFF_YEAR:
            flagged_new.append(r)
        if not r["artist"] and not r["album"] and not r["year"]:
            flagged_blank.append(r)

    print("=" * 90)

    # ── Summary stats ─────────────────────────────────────────────────────────
    total        = len(rows)
    has_year     = [r for r in rows if r["year"]]
    confirmed_ok = [r for r in has_year if r["year"] <= 2022]
    pct_ok       = len(confirmed_ok) / total * 100

    print(f"\nSUMMARY  ({total} files)")
    print(f"  Files with any year tag         : {len(has_year)}/{total}")
    print(f"  Confirmed ≤ 2022 (pre-AI era)   : {len(confirmed_ok)}/{total}  ({pct_ok:.0f}%)")
    print(f"  Year tag missing / unknown      : {total - len(has_year)}/{total}")
    print(f"  Flagged 2023+ (possible AI era) : {len(flagged_new)}")
    print(f"  Flagged fully blank metadata    : {len(flagged_blank)}")

    # ── Flag details ─────────────────────────────────────────────────────────
    if flagged_new:
        print(f"\n[FLAG] YEAR {AI_CUTOFF_YEAR}+ — manual review or deletion recommended:")
        for r in flagged_new:
            print(f"  {r['file']}  year={r['year']}  artist={r['artist'] or '—'}  album={r['album'] or '—'}")

    if flagged_blank:
        print(f"\n[FLAG] COMPLETELY BLANK METADATA — provenance unverifiable:")
        for r in flagged_blank:
            print(f"  {r['file']}  size={r['size_kb']} KB  duration={r['duration_s']}s")

    if not flagged_new and not flagged_blank:
        print("\n[OK] No flags raised — all tracks either have a pre-2023 year tag or no year at all.")

    print()


if __name__ == "__main__":
    main()
