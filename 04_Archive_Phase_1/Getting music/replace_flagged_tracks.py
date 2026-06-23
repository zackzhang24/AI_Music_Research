#!/usr/bin/env python3
"""
Delete the 8 human_track_*.mp3 files that lack year metadata and replace them
with 8 tracks from the Free Music Archive (freemusicarchive.org) that carry
explicit pre-2022 ID3 year tags.

Falls back to archive.org (netlabels, year:[2000 TO 2022]) for any slot that
FMA cannot fill, using file-level archive.org metadata to pre-screen for year
tags before downloading.
"""

import os
import re
import io
import glob
import time
import urllib.parse
import requests
from mutagen.id3 import ID3, ID3NoHeaderError

# ── Config ────────────────────────────────────────────────────────────────────

SONGS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "songs")
MAX_YEAR   = 2022
MIN_BYTES  = 150_000
MAX_BYTES  = 35_000_000

FLAGGED = [
    "human_track_15.mp3",
    "human_track_18.mp3",
    "human_track_20.mp3",
    "human_track_32.mp3",
    "human_track_35.mp3",
    "human_track_41.mp3",
    "human_track_46.mp3",
    "human_track_47.mp3",
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AI-Music-Detector-Dataset-Builder/1.0"})


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_year(raw):
    if not raw:
        return None
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", str(raw))
    return int(m.group(1)) if m else None


def probe_id3_year(url):
    """Range-request the first 128 KB to read the ID3 header without a full download."""
    try:
        r = SESSION.get(url, headers={"Range": "bytes=0-131071"}, timeout=25)
        if r.status_code not in (200, 206):
            return None
        buf = io.BytesIO(r.content)
        try:
            tags = ID3(fileobj=buf)
        except Exception:
            return None
        for tid in ("TDRC", "TYER", "TDAT", "TRDA"):
            frame = tags.get(tid)
            if frame:
                y = extract_year(str(frame))
                if y:
                    return y
    except Exception:
        pass
    return None


def read_id3_year(path):
    try:
        tags = ID3(path)
        for tid in ("TDRC", "TYER", "TDAT", "TRDA"):
            frame = tags.get(tid)
            if frame:
                y = extract_year(str(frame))
                if y:
                    return y
    except (ID3NoHeaderError, Exception):
        pass
    return None


def download_full(url, dest):
    try:
        r = SESSION.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=16_384):
                fh.write(chunk)
        return True
    except Exception as e:
        print(f"    [dl error] {e}")
        return False


# ── Source 1: Free Music Archive API ─────────────────────────────────────────

FMA_API = "https://freemusicarchive.org/api/get/tracks.json"

def fma_candidates(page=1, limit=100):
    """
    Returns list of (url, api_year, title) from FMA API.
    Returns None if the API is unavailable (no key, network error, etc.).
    """
    try:
        r = SESSION.get(
            FMA_API,
            params={"limit": limit, "page": page},
            timeout=15,
        )
        if r.status_code in (401, 403, 404):
            return None            # API requires key or is gone
        r.raise_for_status()
        dataset = r.json().get("dataset", [])
        results = []
        for t in dataset:
            url = t.get("track_file") or t.get("track_url", "")
            if not url:
                continue
            raw_date = (
                t.get("track_date_recorded")
                or t.get("track_date_created")
                or t.get("track_year", "")
            )
            year = extract_year(raw_date)
            if not year or year > MAX_YEAR:
                continue
            title = t.get("track_title", "unknown")
            try:
                size = int(t.get("track_file_size", 0))
            except (ValueError, TypeError):
                size = 0
            results.append((url, year, title, size))
        return results
    except Exception as e:
        print(f"  [FMA API error] {e}")
        return None


# ── Source 2: archive.org netlabels, year:[2000 TO 2022] ─────────────────────

def archive_search(page=1, rows=60, collection="netlabels"):
    params = {
        "q": f"collection:{collection} AND mediatype:audio AND year:[2000 TO {MAX_YEAR}]",
        "fl[]": ["identifier", "title", "year"],
        "sort[]": "downloads desc",
        "rows": rows,
        "page": page,
        "output": "json",
    }
    try:
        r = SESSION.get(
            "https://archive.org/advancedsearch.php", params=params, timeout=30
        )
        r.raise_for_status()
        return r.json().get("response", {}).get("docs", [])
    except Exception as e:
        print(f"  [archive search error] {e}")
        return []


def best_mp3_with_year(identifier):
    """
    Inspect archive.org file-level metadata to find an MP3 whose embedded
    'year' or 'date' field confirms a pre-2023 recording.
    Returns (url, year) or (None, None).
    """
    try:
        r = SESSION.get(
            f"https://archive.org/metadata/{identifier}", timeout=30
        )
        r.raise_for_status()
        files = r.json().get("files", [])
    except Exception:
        return None, None

    candidates = []
    for f in files:
        if "MP3" not in f.get("format", "").upper():
            continue
        try:
            size = int(f.get("size", 0))
        except (ValueError, TypeError):
            size = 0
        if not (MIN_BYTES <= size <= MAX_BYTES):
            continue

        file_year = extract_year(f.get("year") or f.get("date") or "")
        name      = f.get("name", "")
        src_score = 1 if f.get("source") == "original" else 0
        yr_score  = 1 if (file_year and file_year <= MAX_YEAR) else 0

        candidates.append((yr_score, src_score, size, name, file_year))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    yr_score, _, _, name, file_year = candidates[0]
    url = f"https://archive.org/download/{identifier}/{urllib.parse.quote(name)}"
    return url, file_year


# ── Core download loop ────────────────────────────────────────────────────────

def fill_slots(slots, source_label, candidate_gen):
    """
    Drive a generator of (url, candidate_year, label) tuples and download
    until all slots are filled or the generator is exhausted.
    Returns the number of remaining unfilled slots.
    """
    filled = 0
    needed = len(slots)

    for url, cand_year, label in candidate_gen:
        if filled >= needed:
            break

        # Pre-screen via ID3 range probe
        probed = probe_id3_year(url)
        if probed is None:
            print(f"    skip (no ID3 year in header): {label[:50]}")
            continue
        if probed > MAX_YEAR:
            print(f"    skip (year={probed} > {MAX_YEAR}): {label[:50]}")
            continue

        dest_name = slots[filled]
        dest = os.path.join(SONGS_DIR, dest_name)
        print(f"  [{filled+1}/{needed}] [{source_label}] {label[:55]}")
        print(f"    year={probed}  url=...{url[-50:]}")

        if download_full(url, dest):
            actual = os.path.getsize(dest)
            if actual < MIN_BYTES:
                os.remove(dest)
                print(f"    too small ({actual} bytes), skipped")
                continue
            confirmed = read_id3_year(dest) or probed
            print(f"    saved {actual//1024} KB, confirmed year={confirmed}")
            filled += 1
        else:
            if os.path.exists(dest):
                os.remove(dest)

        time.sleep(0.4)

    return needed - filled


def fma_generator():
    """Yield (url, year, title) from FMA API pages."""
    seen = set()
    page = 1
    while True:
        results = fma_candidates(page=page)
        if results is None:
            return   # API unavailable
        if not results:
            return   # exhausted
        for url, year, title, _ in results:
            if url not in seen:
                seen.add(url)
                yield url, year, title
        page += 1


def archive_generator(already_seen=None):
    """Yield (url, year, title) from archive.org netlabels, year-filtered."""
    seen = set(already_seen or [])
    for collection in ("netlabels", "audio_music"):
        page = 1
        while page <= 20:
            items = archive_search(page=page, collection=collection)
            if not items:
                break
            for item in items:
                ident = item.get("identifier", "")
                if ident in seen:
                    continue
                seen.add(ident)

                url, file_year = best_mp3_with_year(ident)
                if not url:
                    continue

                item_year = extract_year(str(item.get("year", "")))
                year = file_year or item_year
                if not year or year > MAX_YEAR:
                    continue

                title = item.get("title", ident)
                yield url, year, title

            page += 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Step 1: delete flagged files ──────────────────────────────────────────
    print("=" * 60)
    print("Step 1 — Deleting 8 flagged tracks (no year metadata)")
    print("=" * 60)
    for fname in FLAGGED:
        path = os.path.join(SONGS_DIR, fname)
        if os.path.exists(path):
            os.remove(path)
            print(f"  deleted  {fname}")
        else:
            print(f"  missing  {fname}  (already gone)")

    remaining_slots = list(FLAGGED)   # filenames to fill, in order

    # ── Step 2a: Free Music Archive ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 2a — Free Music Archive")
    print("="*60)

    # Quick probe to see if FMA API is up
    test = fma_candidates(page=1, limit=1)
    if test is None:
        print("  FMA API unavailable (requires key or is offline) — skipping.")
        fma_filled = 0
    else:
        unfilled = fill_slots(remaining_slots, "FMA", fma_generator())
        fma_filled = len(remaining_slots) - unfilled
        remaining_slots = remaining_slots[fma_filled:]
        print(f"  FMA filled {fma_filled}/8 slot(s)")

    # ── Step 2b: archive.org fallback ─────────────────────────────────────────
    if remaining_slots:
        print(f"\n{'='*60}")
        print(f"Step 2b — archive.org fallback ({len(remaining_slots)} slot(s) remaining)")
        print("="*60)
        unfilled = fill_slots(remaining_slots, "archive.org", archive_generator())
        arc_filled = len(remaining_slots) - unfilled
        remaining_slots = remaining_slots[arc_filled:]

    # ── Final report ─────────────────────────────────────────────────────────
    all_tracks = sorted(glob.glob(os.path.join(SONGS_DIR, "human_track_*.mp3")))
    print(f"\n{'='*60}")
    print("REPLACEMENT SUMMARY")
    print("="*60)
    print(f"  Slots targeted   : 8")
    print(f"  Successfully filled: {8 - len(remaining_slots)}")
    if remaining_slots:
        print(f"  UNFILLED: {remaining_slots}")
    print(f"  Total tracks in songs/: {len(all_tracks)}")

    if remaining_slots:
        print("\n  WARNING: some slots remain empty — re-run or add another source.")
    else:
        print("\n  All 8 replacements downloaded and confirmed pre-2023.")


if __name__ == "__main__":
    main()
