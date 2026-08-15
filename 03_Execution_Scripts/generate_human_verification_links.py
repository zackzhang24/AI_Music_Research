#!/usr/bin/env python3
"""
generate_human_verification_links.py
────────────────────────────────────
Produce a clickable, human-readable verification log for the 50 MP3s in
data/ground_truth_human/ so their data purity can be manually spot-checked.

Each downloaded file is named <youtube_id>.mp3 (yt-dlp `%(id)s`), so:
  • the YouTube URL is reconstructed from the id,
  • the exact clip duration is read locally with ffprobe (MM:SS),
  • the Artist - Title label is re-fetched from YouTube metadata (the MP3s were
    saved without ID3 tags), preferring the structured artist/track fields and
    falling back to the video title.

Output: human_verification_links.txt in the project ROOT, one line each:
    MM:SS | Artist - Title | https://www.youtube.com/watch?v=<id>
"""

import os, re, sys, glob, subprocess, json
import static_ffmpeg; static_ffmpeg.add_paths()
import yt_dlp

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
SRC_DIR  = os.path.join(HERE, "data", "ground_truth_human")
OUT_FILE = os.path.join(ROOT, "human_verification_links.txt")


def ffprobe_seconds(path: str) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", path], stderr=subprocess.STDOUT)
    return float(json.loads(out)["format"]["duration"])


def mmss(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def clean_title(title: str) -> str:
    """Strip trailing '(Official Audio)' / '(Audio)' / '(Remastered ...)' etc."""
    return re.sub(
        r"\s*[\(\[](?:official\s*audio|audio|official\s*music\s*video|"
        r"remaster[^\)\]]*|lyric[^\)\]]*|taylor'?s\s*version)[\)\]]\s*$",
        "", title, flags=re.I).strip()


def artist_title(meta: dict) -> str:
    artist = (meta.get("artist") or meta.get("creator") or "").strip()
    track  = (meta.get("track") or "").strip()
    title  = (meta.get("title") or "").strip()
    if artist and track:
        return f"{artist} - {clean_title(track)}"
    if " - " in title:                      # title already 'Artist - Title (...)'
        return clean_title(title)
    uploader = (meta.get("uploader") or "").replace(" - Topic", "").strip()
    name = artist or uploader
    return f"{name} - {clean_title(title)}" if name else clean_title(title)


def fetch_meta(url: str) -> dict:
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def main():
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.mp3")))
    if not files:
        print(f"No MP3s found in {SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {len(files)} tracks from {SRC_DIR} …")
    lines = []
    for i, path in enumerate(files, 1):
        vid = os.path.splitext(os.path.basename(path))[0]
        url = f"https://www.youtube.com/watch?v={vid}"
        dur = mmss(ffprobe_seconds(path))
        try:
            meta = fetch_meta(url)
            label = artist_title(meta)
        except Exception as exc:
            label = f"[metadata unavailable] ({vid})"
            print(f"  [{i}/{len(files)}] meta error for {vid}: {exc}", file=sys.stderr)
        line = f"{dur} | {label} | {url}"
        lines.append(line)
        print(f"  [{i:02d}/{len(files)}] {line}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nGenerated human_verification_links.txt with {len(lines)} clickable URLs.")
    print(f"Location: {OUT_FILE}")


if __name__ == "__main__":
    main()
