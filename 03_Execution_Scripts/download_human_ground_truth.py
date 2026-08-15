#!/usr/bin/env python3
"""
download_human_ground_truth.py
──────────────────────────────
Build the NEGATIVE (pure-human) ground-truth set: download exactly N (default 50)
studio-audio tracks from established human artists via yt-dlp search.

Pipeline
────────
  • For each artist -> ytsearch "{Artist} official audio".
  • TITLE BLACKLIST (negative regex): drop anything that looks like a video /
    live / lyric upload — "music video", "mv", "live", "performance",
    "trailer", "lyric" — so only pure studio audio survives.
  • DURATION gate: keep only songs in [MIN_DURATION, MAX_DURATION] seconds.
  • Best audio -> FFmpegExtractAudio -> 192 kbps MP3 into
    data/ground_truth_human/%(id)s.%(ext)s
  • A GLOBAL success counter enforces an exact cap: the script stops the instant
    the Nth valid MP3 finishes downloading, across all artist searches.

Blacklist safety (word boundaries):
  \bmv\b / \blive\b prevent substring false-positives — "alive", "olive",
  "delivered", "removed" etc. never trip the filter.

Console:
    "Downloaded 1/50: <Title>"
    "Skipped (Blacklist flagged): <Title>"
    "Skipped (Duration 312s): <Title>"

Usage:
    python3 download_human_ground_truth.py
    python3 download_human_ground_truth.py --cap 50 --per-artist 20
"""

import os, re, sys, argparse
import static_ffmpeg; static_ffmpeg.add_paths()      # ffmpeg + ffprobe for MP3 conversion
import yt_dlp

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data", "ground_truth_human")

# Established human artists — the negative (non-AI) class.
ARTISTS = [
    "Taylor Swift", "Coldplay", "The Beatles", "Justin Bieber",
    "Adele", "Elton John", "Bruno Mars", "Ed Sheeran",
    "Rihanna", "Beyonce",
]

# Duration acceptance window (seconds) — song-length studio tracks only.
MIN_DURATION = 90
MAX_DURATION = 300

TARGET_COUNT = 50

# ── Negative title blacklist ───────────────────────────────────────────────────
# Word-boundaried where a bare substring would be unsafe (mv/live), so e.g.
# "alive", "olive", "delivered" never match \blive\b, and only a standalone
# "MV" token matches \bmv\b.
_FLAGS = re.IGNORECASE
BLACKLIST = [
    (re.compile(r"music[\s\-_]*video", _FLAGS), "music video"),
    (re.compile(r"\bmv\b", _FLAGS),             "mv"),
    (re.compile(r"\blive\b", _FLAGS),           "live"),
    (re.compile(r"\bperformance\b", _FLAGS),    "performance"),
    (re.compile(r"\btrailer\b", _FLAGS),        "trailer"),
    (re.compile(r"\blyric", _FLAGS),            "lyric"),   # lyric / lyrics / lyrical
]


def blacklisted(title: str):
    """Return the matched blacklist term (str) if the title is disallowed, else None."""
    if not title:
        return None
    for pattern, term in BLACKLIST:
        if pattern.search(title):
            return term
    return None


def search_artist(artist: str, n: int):
    """Flat-enumerate top-n search results (id / title / duration) for an artist."""
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    query = f"{artist} official audio"
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    return [e for e in (info.get("entries") or []) if e]


def probe_duration(url: str):
    """Fallback single-video duration probe when flat mode omits it."""
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False).get("duration")
    except Exception:
        return None


def download_mp3(url: str) -> str:
    """Best audio -> 192 kbps MP3 into OUT_DIR. Returns the .mp3 path; raises on failure."""
    opts = {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(OUT_DIR, "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return os.path.join(OUT_DIR, f"{info['id']}.mp3")


def main():
    ap = argparse.ArgumentParser(description="Download N human studio tracks (negative ground truth).")
    ap.add_argument("--artists", nargs="+", default=ARTISTS)
    ap.add_argument("--per-artist", type=int, default=20, help="search results to scan per artist")
    ap.add_argument("--cap", type=int, default=TARGET_COUNT, help="exact number of MP3s to download")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("Human (negative) ground-truth downloader — blacklist + duration gated")
    print(f"  Artists  : {len(args.artists)}  x  {args.per_artist} results each")
    print(f"  Window   : {MIN_DURATION}-{MAX_DURATION}s   |   Target: {args.cap} MP3s")
    print(f"  Output   : {OUT_DIR}/%(id)s.%(ext)s @ 192 kbps")
    print("=" * 72)

    downloaded = 0
    skip_black = skip_dur = 0
    seen_ids = set()

    for artist in args.artists:
        if downloaded >= args.cap:
            break
        print(f"\n🔎 Searching: \"{artist} official audio\"")
        try:
            entries = search_artist(artist, args.per_artist)
        except Exception as exc:
            print(f"   [search error] {artist}: {exc}", file=sys.stderr)
            continue
        print(f"   {len(entries)} candidate(s).")

        for entry in entries:
            if downloaded >= args.cap:
                break

            vid   = entry.get("id")
            title = entry.get("title") or vid or "?"
            url   = entry.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else None)
            if not url or vid in seen_ids:
                continue
            seen_ids.add(vid)

            # ── Gate 1: negative title blacklist ──────────────────────────────
            if blacklisted(title):
                skip_black += 1
                print(f"   Skipped (Blacklist flagged): {title}")
                continue

            # ── Gate 2: duration window ───────────────────────────────────────
            dur = entry.get("duration")
            if dur is None:
                dur = probe_duration(url)
            if dur is None or not (MIN_DURATION <= dur <= MAX_DURATION):
                skip_dur += 1
                shown = f"{int(dur)}s" if dur else "unknown"
                print(f"   Skipped (Duration {shown}): {title}")
                continue

            # ── Download ──────────────────────────────────────────────────────
            try:
                path = download_mp3(url)
            except Exception as exc:
                print(f"   [download error] {title}: {exc}", file=sys.stderr)
                continue
            if not os.path.exists(path):
                print(f"   [download error] {title}: output MP3 not found", file=sys.stderr)
                continue

            downloaded += 1
            print(f"   Downloaded {downloaded}/{args.cap}: {title}")

            # ── EXACT global cap: terminate the instant the Nth MP3 lands ──────
            if downloaded >= args.cap:
                print(f"\n✅ Reached exact cap of {args.cap}. Terminating.")
                break

    print("\n" + "=" * 72)
    print(f"DONE — {downloaded}/{args.cap} downloaded  |  "
          f"skipped: {skip_black} blacklisted, {skip_dur} out-of-window.")
    print(f"Output: {OUT_DIR}")
    if downloaded < args.cap:
        print(f"⚠  Ran out of candidates before reaching {args.cap} — widen --artists / --per-artist.")
    print("=" * 72)


# ── Self-test: blacklist substring-safety (the tricky cases) ───────────────────
def _selftest():
    must_flag = [
        "Adele - Hello (Official Music Video)",   # music video
        "BTS - Dynamite MV",                      # mv token
        "Coldplay - Yellow (Live at Glastonbury)",# live
        "Bruno Mars - 24K Magic Live Performance",# live / performance
        "Taylor Swift - Album Trailer",           # trailer
        "Ed Sheeran - Perfect (Lyric Video)",     # lyric
        "Rihanna - Diamonds Lyrics",              # lyrics
    ]
    must_pass = [
        "Adele - Hello (Official Audio)",         # clean
        "Coldplay - Yellow (Official Audio)",     # clean
        "Justin Bieber - Stay Alive",             # 'alive' must NOT trip \blive\b
        "Elton John - Your Song (Olive remaster)",# 'olive' must NOT trip \blive\b
        "Bruno Mars - The Package Was Delivered", # 'delivered' must NOT trip \blive\b
        "The Beatles - Something",                # clean
    ]
    ok = True
    for t in must_flag:
        term = blacklisted(t)
        flag = "PASS" if term else "FAIL"
        if not term: ok = False
        print(f"  [{flag}] FLAG expected | {t!r:52s} -> {term}")
    for t in must_pass:
        term = blacklisted(t)
        flag = "PASS" if term is None else "FAIL"
        if term is not None: ok = False
        print(f"  [{flag}] PASS expected | {t!r:52s} -> {term}")
    print("\nSELF-TEST:", "ALL PASSED ✅" if ok else "FAILURES ❌")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    main()
