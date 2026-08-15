#!/usr/bin/env python3
"""
universal_ingestor.py  —  production universal entry point
───────────────────────────────────────────────────────────
Accept ANY media URL, resolve the audio stream + metadata for the source
platform, and hand the payload straight to the downstream classifiers.

Flow
────
  1. CLI:            python3 universal_ingestor.py --url [LINK]
  2. Interception:   parse the URL to pick the routing pathway
       • Spotify URL  -> resolve track Title + Artist (spotipy if credentials
         are exported, else a lightweight page scrape) and convert the text to
         a YouTube search:   ytsearch1:"{artist} - {title}"
       • Native URL (YouTube / SoundCloud / Bandcamp) -> raw URL passes
         directly to the download handler.
  3. Download:       yt-dlp (programmatic) -> best audio -> local MP3,
                     MONO @ 48 kHz (CLAP-native), platform metadata embedded
                     into the ID3 headers (FFmpegMetadata == --embed-metadata).
                     The video description / tags / top comments are persisted
                     next to the MP3 as <file>.meta.json for the NLP fusion slots.
  4. Pipeline:       subprocess chain, exactly the production invocation:
                       python3 inference_router.py [temp.mp3] | python3 fusion_aggregator.py
  5. Cleanup:        strict finally-block safety check wipes the temp MP3 (and
                     its session folder) the moment the fusion verdict is logged.

Usage
─────
  python3 universal_ingestor.py --url "https://www.youtube.com/watch?v=..."
  python3 universal_ingestor.py --url "https://open.spotify.com/track/..."
  python3 universal_ingestor.py --url "https://soundcloud.com/artist/track"
  python3 universal_ingestor.py --url "..." --keep      # debug: skip cleanup
"""

import os
import re
import sys
import json
import html
import shutil
import argparse
import tempfile
import subprocess
import urllib.request
from urllib.parse import urlparse

import static_ffmpeg; static_ffmpeg.add_paths()          # ffmpeg + ffprobe
import yt_dlp

HERE = os.path.dirname(os.path.abspath(__file__))

TMP_PREFIX  = "universal_ingest_"        # cleanup only ever touches dirs with this prefix
TARGET_SR   = 48000                      # CLAP-native sample rate
TARGET_CH   = 1                          # mono
MP3_KBPS    = "192"

ROUTER_PY   = os.path.join(HERE, "inference_router.py")
FUSION_PY   = os.path.join(HERE, "fusion_aggregator.py")

SIDECAR_SUFFIX = ".meta.json"            # text metadata payload for the NLP fusion slots
MAX_COMMENTS   = 40                      # cap on comments pulled per video


# ══════════════════════════════════════════════════════════════════════════════
# 2. Platform interception logic
# ══════════════════════════════════════════════════════════════════════════════
def detect_platform(url: str) -> str:
    if url.startswith("spotify:"):
        return "spotify"
    host = urlparse(url).netloc.lower()
    if "spotify.com" in host:
        return "spotify"
    if any(d in host for d in ("youtube.com", "youtu.be")):
        return "youtube"
    if "soundcloud.com" in host:
        return "soundcloud"
    if "bandcamp.com" in host:
        return "bandcamp"
    return "generic"          # unknown host: let yt-dlp's extractors try it


def _spotify_track_id(url: str):
    m = re.search(r"open\.spotify\.com/(?:intl-[a-z\-]+/)?track/([A-Za-z0-9]+)", url) \
        or re.search(r"spotify:track:([A-Za-z0-9]+)", url)
    return m.group(1) if m else None


def _spotify_meta_via_api(track_id: str):
    """Preferred path: the existing spotipy Client-Credentials client."""
    from spotify_extractor import _get_client          # raises RuntimeError w/o creds
    track = _get_client().track(track_id)
    return track["name"], track["artists"][0]["name"]


def _spotify_meta_via_scrape(url: str):
    """Lightweight no-auth fallback: read Title/Artist out of the page's meta tags."""
    import ssl, certifi                    # framework Python lacks system CA certs
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    page = urllib.request.urlopen(req, timeout=15, context=ctx).read().decode("utf-8", "ignore")

    title = artist = None
    m = re.search(r'<meta property="og:title" content="([^"]+)"', page)
    if m:
        title = html.unescape(m.group(1)).strip()
    # <title>Song - song and lyrics by Artist | Spotify</title>
    m = re.search(r"<title>(.+?) [-–] song(?: and lyrics)? by (.+?) \|", page)
    if m:
        title = title or html.unescape(m.group(1)).strip()
        artist = html.unescape(m.group(2)).strip()
    if artist is None:
        # og:description historically reads "Artist · Song · Year"
        m = re.search(r'<meta property="og:description" content="([^"]+)"', page)
        if m and "·" in m.group(1):
            artist = html.unescape(m.group(1)).split("·")[0].strip()
    return title, artist


def resolve_spotify(url: str):
    """Spotify URL -> (title, artist) -> ytsearch1 query string."""
    track_id = _spotify_track_id(url)
    if not track_id:
        raise ValueError(f"Could not parse a Spotify track id from: {url}")

    title = artist = None
    try:                                   # 1st choice: official API via spotipy
        title, artist = _spotify_meta_via_api(track_id)
        print(f"[ingestor] Spotify API resolved   : {artist} - {title}")
    except Exception as exc:               # no creds / network — fall back to scrape
        print(f"[ingestor] spotipy unavailable ({exc}); falling back to page scrape",
              file=sys.stderr)
        title, artist = _spotify_meta_via_scrape(url)
        print(f"[ingestor] Spotify scrape resolved: {artist} - {title}")

    if not title or not artist:
        raise RuntimeError(f"Failed to resolve Title/Artist for Spotify track {track_id}")

    query = f'ytsearch1:"{artist} - {title}"'
    return query, {"title": title, "artist": artist}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Stream download + tagging (yt-dlp, programmatic)
# ══════════════════════════════════════════════════════════════════════════════
def download_audio(target: str, tmpdir: str) -> str:
    """
    target: raw URL (native platforms) or a ytsearch1: query (Spotify route).
    Returns (mp3_path, info): a mono 48 kHz MP3 with embedded ID3 metadata plus
    the yt-dlp info dict (title / description / tags / comments) for the sidecar.
    """
    opts = {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio",              # best audio -> MP3
             "preferredcodec": "mp3", "preferredquality": MP3_KBPS},
            {"key": "FFmpegMetadata",                  # == --embed-metadata
             "add_metadata": True},
        ],
        # force mono + 48 kHz during the MP3 extraction pass
        "postprocessor_args": {"extractaudio": ["-ac", str(TARGET_CH),
                                                "-ar", str(TARGET_SR)]},
        # pull the text payload for the NLP fusion slots (description + comments)
        "getcomments": True,
        "extractor_args": {"youtube": {"max_comments": [str(MAX_COMMENTS)]}},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(target, download=True)
    if "entries" in info:                              # ytsearch wrapper -> first hit
        info = info["entries"][0]

    path = os.path.join(tmpdir, f"{info['id']}.mp3")
    if not os.path.exists(path):
        raise RuntimeError(f"yt-dlp reported success but {path} is missing")
    print(f"[ingestor] downloaded: {info.get('title', info['id'])}")
    return path, info


def write_metadata_sidecar(mp3_path: str, info: dict, spotify_meta: dict = None) -> str:
    """
    Persist the platform TEXT payload next to the final MP3 as
    <mp3>.meta.json so the fusion layer's NLP extractors (description scan,
    comment heuristic) can consume it via file_path.
    """
    comments = [(c.get("text") or "").strip()
                for c in (info.get("comments") or []) if isinstance(c, dict)]
    meta = {
        "id":          info.get("id"),
        "title":       info.get("title"),
        "uploader":    info.get("uploader"),
        "description": info.get("description") or "",
        "tags":        info.get("tags") or [],
        "comments":    [c for c in comments if c][:MAX_COMMENTS],
    }
    if spotify_meta:                       # authoritative catalog identity
        meta["title"], meta["artist"] = spotify_meta["title"], spotify_meta["artist"]
    sidecar = mp3_path + SIDECAR_SUFFIX
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"[ingestor] metadata sidecar written: {os.path.basename(sidecar)} "
          f"(description {len(meta['description'])} chars, "
          f"{len(meta['tags'])} tags, {len(meta['comments'])} comments)")
    return sidecar


def _probe_audio(path: str):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels,sample_rate", "-of", "json", path])
    st = json.loads(out)["streams"][0]
    return int(st["channels"]), int(st["sample_rate"])


def enforce_mono_48k(path: str) -> str:
    """Verify mono/48 kHz; if the postprocessor args didn't stick, fix with one
    explicit ffmpeg pass that preserves the embedded ID3 metadata."""
    ch, sr = _probe_audio(path)
    if ch == TARGET_CH and sr == TARGET_SR:
        print(f"[ingestor] format verified: {ch} ch @ {sr} Hz (target met)")
        return path
    print(f"[ingestor] got {ch} ch @ {sr} Hz -> converting to "
          f"{TARGET_CH} ch @ {TARGET_SR} Hz")
    fixed = path.replace(".mp3", f".{TARGET_SR}mono.mp3")
    subprocess.check_call(
        ["ffmpeg", "-v", "error", "-y", "-i", path,
         "-ac", str(TARGET_CH), "-ar", str(TARGET_SR),
         "-b:a", f"{MP3_KBPS}k", "-map_metadata", "0", fixed])
    os.remove(path)
    return fixed


def override_id3(path: str, meta: dict):
    """Spotify route: stamp the AUTHORITATIVE catalog Title/Artist over whatever
    the YouTube mirror embedded."""
    from mutagen.easyid3 import EasyID3
    from mutagen.id3 import ID3, ID3NoHeaderError
    try:
        tags = EasyID3(path)
    except ID3NoHeaderError:
        ID3().save(path)
        tags = EasyID3(path)
    tags["title"], tags["artist"] = meta["title"], meta["artist"]
    tags.save()
    print(f"[ingestor] ID3 overridden with Spotify metadata: "
          f"{meta['artist']} - {meta['title']}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Pipeline execution:  inference_router.py [mp3] | fusion_aggregator.py
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(mp3_path: str) -> int:
    print("\n[ingestor] handing off ->  "
          f"python3 inference_router.py {os.path.basename(mp3_path)} "
          "| python3 fusion_aggregator.py\n", flush=True)

    router = subprocess.Popen(
        [sys.executable, ROUTER_PY, mp3_path],
        cwd=HERE, stdout=subprocess.PIPE, stderr=sys.stderr)
    fusion = subprocess.Popen(
        [sys.executable, FUSION_PY],           # reads the JSON payload on stdin
        cwd=HERE, stdin=router.stdout)
    router.stdout.close()                      # let the pipe see router's EOF
    fusion.communicate()
    router.wait()

    if router.returncode != 0:
        print(f"[ingestor] inference_router exited {router.returncode}", file=sys.stderr)
    if fusion.returncode != 0:
        print(f"[ingestor] fusion_aggregator exited {fusion.returncode}", file=sys.stderr)
    return fusion.returncode or router.returncode or 0


# ══════════════════════════════════════════════════════════════════════════════
# 5. Session cleanup (strict safety check)
# ══════════════════════════════════════════════════════════════════════════════
def safe_cleanup(tmpdir: str):
    """Delete the session temp folder ONLY if it is genuinely ours: it must
    resolve inside the system temp root AND carry our session prefix."""
    if not tmpdir or not os.path.isdir(tmpdir):
        return
    real     = os.path.realpath(tmpdir)
    tmp_root = os.path.realpath(tempfile.gettempdir())
    if not real.startswith(tmp_root + os.sep) or \
       not os.path.basename(real).startswith(TMP_PREFIX):
        print(f"[ingestor] REFUSING to delete non-session path: {real}", file=sys.stderr)
        return
    shutil.rmtree(real, ignore_errors=True)
    print(f"[ingestor] session cleanup complete — wiped {real}")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Universal media-URL ingestor -> acoustic router -> late fusion.")
    ap.add_argument("--url", required=True, help="source media link (any platform)")
    ap.add_argument("--keep", action="store_true",
                    help="debug: keep the temp MP3 instead of wiping it")
    args = ap.parse_args()

    platform = detect_platform(args.url)
    print("=" * 66)
    print("UNIVERSAL INGESTOR")
    print("=" * 66)
    print(f"  URL      : {args.url}")
    print(f"  Platform : {platform}")

    tmpdir = tempfile.mkdtemp(prefix=TMP_PREFIX)
    exit_code = 1
    try:
        # ── interception: pick the download target ────────────────────────────
        spotify_meta = None
        if platform == "spotify":
            target, spotify_meta = resolve_spotify(args.url)
            print(f"  Route    : Spotify -> {target}")
        else:
            target = args.url                     # native: raw URL straight through
            print(f"  Route    : native ({platform}) -> yt-dlp direct")
        print("-" * 66)

        # ── download, tag, verify ─────────────────────────────────────────────
        mp3_path, info = download_audio(target, tmpdir)
        mp3_path = enforce_mono_48k(mp3_path)
        if spotify_meta:
            override_id3(mp3_path, spotify_meta)
        # sidecar AFTER the format fix so its name tracks the final MP3 path
        write_metadata_sidecar(mp3_path, info, spotify_meta)

        # ── classify ──────────────────────────────────────────────────────────
        exit_code = run_pipeline(mp3_path)

    finally:
        # strict deletion safety check — runs on success, failure, or Ctrl-C
        if args.keep:
            print(f"[ingestor] --keep set: temp session preserved at {tmpdir}")
        else:
            safe_cleanup(tmpdir)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
