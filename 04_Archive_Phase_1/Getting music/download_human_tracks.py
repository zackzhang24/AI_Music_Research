#!/usr/bin/env python3
"""
Download 50 CC-licensed human-made MP3s from Internet Archive netlabels collection.
Netlabels is a curated collection of Creative Commons music releases.
"""

import os
import time
import urllib.parse
import requests

SONGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "songs")
TARGET = 50
MIN_BYTES = 150_000    # 150 KB — skip tiny clips
MAX_BYTES = 35_000_000 # 35 MB — skip very long files

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AI-Music-Detector-Dataset-Builder/1.0"})


def search_archive(collection, page, rows=60):
    params = {
        "q": f"collection:{collection} AND mediatype:audio",
        "fl[]": ["identifier", "title"],
        "sort[]": "downloads desc",
        "rows": rows,
        "page": page,
        "output": "json",
    }
    try:
        r = SESSION.get("https://archive.org/advancedsearch.php", params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("response", {}).get("docs", [])
    except Exception as e:
        print(f"  [search error] {e}")
        return []


def get_best_mp3(identifier):
    try:
        r = SESSION.get(f"https://archive.org/metadata/{identifier}", timeout=30)
        r.raise_for_status()
        files = r.json().get("files", [])
    except Exception as e:
        return None

    candidates = []
    for f in files:
        fmt = f.get("format", "")
        if "MP3" not in fmt.upper():
            continue
        try:
            size = int(f.get("size", 0))
        except (ValueError, TypeError):
            size = 0
        if not (MIN_BYTES <= size <= MAX_BYTES):
            continue
        name = f.get("name", "")
        # Prefer original source files over derivatives
        score = 1 if f.get("source") == "original" else 0
        candidates.append((score, size, name))

    if not candidates:
        return None

    # Best: original source, then largest within range
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, name = candidates[0]
    return f"https://archive.org/download/{identifier}/{urllib.parse.quote(name)}"


def download(url, dest):
    try:
        r = SESSION.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=16_384):
                fh.write(chunk)
        return True
    except Exception as e:
        print(f"    [download error] {e}")
        return False


def main():
    os.makedirs(SONGS_DIR, exist_ok=True)
    downloaded = 0
    seen = set()

    # Two collections for breadth; netlabels is CC-only, audio_music is broader
    collections = ["netlabels", "audio_music", "georgeblood", "78rpm"]
    col_idx = 0
    page = 1
    consecutive_empty = 0

    while downloaded < TARGET:
        collection = collections[col_idx % len(collections)]
        print(f"\n[page {page}] Searching '{collection}'...")
        items = search_archive(collection, page)

        if not items:
            consecutive_empty += 1
            if consecutive_empty >= 4:
                print("Exhausted all collections.")
                break
            col_idx += 1
            page = 1
            continue

        consecutive_empty = 0

        for item in items:
            if downloaded >= TARGET:
                break

            ident = item.get("identifier", "")
            if ident in seen:
                continue
            seen.add(ident)

            mp3_url = get_best_mp3(ident)
            if not mp3_url:
                continue

            track_num = downloaded + 1
            dest = os.path.join(SONGS_DIR, f"human_track_{track_num:02d}.mp3")
            title = item.get("title", ident)

            print(f"  [{track_num:02d}/{TARGET}] {title[:55]}")
            print(f"         {mp3_url.split('/')[-1][:60]}")

            if download(mp3_url, dest):
                actual = os.path.getsize(dest)
                if actual < MIN_BYTES:
                    os.remove(dest)
                    print(f"         skipped (too small: {actual} bytes)")
                    continue
                print(f"         saved  ({actual/1024:.0f} KB)")
                downloaded += 1
            else:
                if os.path.exists(dest):
                    os.remove(dest)

            time.sleep(0.4)  # polite rate-limiting

        page += 1
        col_idx += 1

    print(f"\n{'='*55}")
    print(f"Finished: {downloaded}/{TARGET} tracks saved to:")
    print(f"  {SONGS_DIR}")
    if downloaded < TARGET:
        print(f"  WARNING: only {downloaded} tracks downloaded — check network/sources.")


if __name__ == "__main__":
    main()
