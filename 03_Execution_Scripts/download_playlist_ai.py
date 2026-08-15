#!/usr/bin/env python3
"""
download_playlist_ai.py
───────────────────────
Download exactly N (default 50) AI-audio tracks from a MANUALLY VERIFIED
YouTube playlist, keeping only song-length videos.

Design
──────
  • No text/regex validation — the playlist is trusted as all-AI content.
  • Duration gate only: keep videos in the [MIN_DURATION, MAX_DURATION] window
    (default 90–300 s) to drop trailers / hour-long compilations.
  • Best audio -> FFmpegExtractAudio -> 192 kbps MP3, saved to
    data/ground_truth_ai/%(id)s.%(ext)s
  • A success counter enforces an EXACT cap: the script stops the instant the
    Nth valid MP3 finishes downloading (it does not keep scanning the playlist).

Console:
    "Downloaded 1/50: <Title>"
    "Skipped (Duration 85s): <Title>"

Usage:
    python3 download_playlist_ai.py
    python3 download_playlist_ai.py --cap 50 --playlist "https://www.youtube.com/playlist?list=..."
"""

import os, sys, argparse
import static_ffmpeg; static_ffmpeg.add_paths()      # ffmpeg + ffprobe for MP3 conversion
import yt_dlp

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data", "ground_truth_ai")

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLT26wpyuKBpMf99VVaNuCmYeUXcTOQEwJ"

# Duration acceptance window (seconds) — song-length only.
MIN_DURATION = 90
MAX_DURATION = 300

TARGET_COUNT = 50


def enumerate_playlist(url: str):
    """Flat-enumerate playlist entries (id / title / duration) without downloading."""
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return [e for e in (info.get("entries") or []) if e]


def probe_duration(url: str):
    """Fallback: fetch a single video's duration when flat mode omits it."""
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return info.get("duration")
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
    ap = argparse.ArgumentParser(description="Download N duration-filtered AI tracks from a verified YouTube playlist.")
    ap.add_argument("--playlist", default=PLAYLIST_URL)
    ap.add_argument("--cap", type=int, default=TARGET_COUNT, help="exact number of MP3s to download")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("Verified-playlist AI downloader (duration-gated, hard cap)")
    print(f"  Playlist : {args.playlist}")
    print(f"  Window   : {MIN_DURATION}-{MAX_DURATION}s   |   Target: {args.cap} MP3s")
    print(f"  Output   : {OUT_DIR}/%(id)s.%(ext)s @ 192 kbps")
    print("=" * 72)

    try:
        entries = enumerate_playlist(args.playlist)
    except Exception as exc:
        print(f"[playlist error] {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Playlist contains {len(entries)} entr(y/ies). Iterating…\n")

    downloaded = 0
    skipped = 0

    for entry in entries:
        vid   = entry.get("id")
        title = entry.get("title") or vid or "?"
        url   = entry.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else None)
        if not url:
            continue

        # ── Duration gate (no regex / text checks) ────────────────────────────
        dur = entry.get("duration")
        if dur is None:                       # flat mode occasionally omits it
            dur = probe_duration(url)

        if dur is None or not (MIN_DURATION <= dur <= MAX_DURATION):
            skipped += 1
            shown = f"{int(dur)}s" if dur else "unknown"
            print(f"Skipped (Duration {shown}): {title}")
            continue

        # ── Download the MP3 ──────────────────────────────────────────────────
        try:
            path = download_mp3(url)
        except Exception as exc:
            print(f"[download error] {title}: {exc}", file=sys.stderr)
            continue
        if not os.path.exists(path):
            print(f"[download error] {title}: output MP3 not found", file=sys.stderr)
            continue

        downloaded += 1
        print(f"Downloaded {downloaded}/{args.cap}: {title}")

        # ── EXACT cap: terminate the instant the Nth MP3 lands ────────────────
        if downloaded >= args.cap:
            print(f"\n✅ Reached exact cap of {args.cap}. Terminating.")
            break

    print("\n" + "=" * 72)
    print(f"DONE — {downloaded}/{args.cap} downloaded, {skipped} skipped (outside duration window).")
    print(f"Output: {OUT_DIR}")
    if downloaded < args.cap:
        print(f"⚠  Playlist exhausted before reaching {args.cap} — only {downloaded} valid tracks found.")
    print("=" * 72)


if __name__ == "__main__":
    main()
