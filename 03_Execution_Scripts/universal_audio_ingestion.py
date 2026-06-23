#!/usr/bin/env python3
"""
universal_audio_ingestion.py
────────────────────────────
A standalone universal audio-ingestion router built on yt-dlp.

Routing
───────
  Route A — Spotify (open.spotify.com/track/…):
      Scrape Track_Title / Artist_Name / Duration_MS from Spotify's public
      embed metadata, then resolve the audio on YouTube with the query
          ytsearch1:"<Title>" "<Artist>" "Provided to YouTube"
      (targets the auto-generated "… - Topic" uploads = clean source audio).
      The YouTube candidate's duration MUST match the Spotify duration within
      a 3.0-second margin before any download proceeds.

  Route B — Direct URL (YouTube / SoundCloud / any other domain):
      The URL is handed straight to yt-dlp's extractor.

  Route C — Plain-text query (input has no http(s):// protocol):
      Resolve the audio from a raw text string via a 3-stage fallback:
        Primary : ytsearch1:"<text> Provided to YouTube"
        Stage 1 : ytsearch1:"<text> Official Audio"     (relaxed YouTube)
        Stage 2 : scsearch1:"<text>"                     (SoundCloud cross-search)
        Stage 3 : if all yield nothing → Status=NOT_FOUND, continue.

  Input dispatch is performed by process_input(input_data): an input containing
  http:// or https:// goes to Route A/B; everything else goes to Route C.

5-Minute Cap (temporal slicing)
───────────────────────────────
  Each stream's full duration is probed first.  If it exceeds 5:00, only the
  first 0:00–5:00 is downloaded via yt-dlp's native `download_ranges`
  (+ force_keyframes_at_cuts).  Shorter tracks download in full.

Output
──────
  • All media transcoded to .wav (FFmpegExtractAudio) into
    03_Execution_Scripts/url_test_downloads/
  • A verification log written to ingestion_test_log.csv.

Every network call / download is wrapped in try/except so one bad URL never
halts the batch.
"""

import os, re, csv, json, sys, subprocess, traceback
from urllib.parse import urlparse
import requests
import static_ffmpeg; static_ffmpeg.add_paths()           # ffmpeg + ffprobe on PATH

import yt_dlp
from yt_dlp.utils import download_range_func

# ── Paths / config ─────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
DL_DIR    = os.path.join(HERE, "url_test_downloads")
URLS_TXT  = os.path.join(HERE, "target_urls.txt")
LOG_CSV   = os.path.join(HERE, "ingestion_test_log.csv")

MAX_SECONDS   = 300          # 5-minute cap
MATCH_MARGIN  = 3.0          # Route-A Spotify↔YouTube duration tolerance (s)
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

SPOTIFY_TRACK_RE = re.compile(r"open\.spotify\.com/track/([A-Za-z0-9]+)", re.I)
DOMAIN_RE        = re.compile(r"https?://(?:www\.)?([^/]+)/?", re.I)

LOG_FIELDS = [
    "Input_Source", "Route", "Search_Stage", "Domain", "Track_Title", "Artist_Name",
    "Spotify_Duration_MS", "Matched_YouTube_URL", "Source_Duration_S",
    "Duration_Match", "Capped_To_5min", "Output_Filename", "Local_File_Path",
    "Final_Duration_S", "Status", "Error",
]


# ── ffprobe helper ─────────────────────────────────────────────────────────────
def ffprobe_duration(path: str) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path], stderr=subprocess.STDOUT)
        return round(float(json.loads(out)["format"]["duration"]), 3)
    except Exception:
        return -1.0


# ── Spotify metadata (no API key — public embed JSON) ──────────────────────────
def get_spotify_metadata(url: str) -> dict:
    """Return {title, artist, duration_ms} for a Spotify track URL."""
    m = SPOTIFY_TRACK_RE.search(url)
    if not m:
        raise ValueError("Not a Spotify track URL")
    track_id = m.group(1)

    embed = f"https://open.spotify.com/embed/track/{track_id}"
    r = requests.get(embed, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()

    nm = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                   r.text, re.S)
    if not nm:
        raise ValueError("Could not locate __NEXT_DATA__ in Spotify embed page")
    data = json.loads(nm.group(1))

    entity = data["props"]["pageProps"]["state"]["data"]["entity"]
    title    = entity.get("title") or entity.get("name") or ""
    artists  = entity.get("artists") or []
    if artists:
        artist = ", ".join(a.get("name", "") for a in artists if a.get("name"))
    else:
        artist = entity.get("subtitle", "")          # embed fallback
    duration_ms = entity.get("duration") or entity.get("duration_ms") or 0

    if not title:
        raise ValueError("Spotify metadata missing title")
    return {"title": title, "artist": artist, "duration_ms": int(duration_ms)}


# ── yt-dlp helpers ─────────────────────────────────────────────────────────────
def _base_opts():
    return {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(DL_DIR, "%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "ignoreerrors": False,
    }


def ytdlp_probe(target: str) -> dict:
    """Extract info WITHOUT downloading. `target` may be a URL or ytsearchN:… query."""
    with yt_dlp.YoutubeDL({**_base_opts()}) as ydl:
        info = ydl.extract_info(target, download=False)
    if info and info.get("_type") == "playlist":      # search results / playlists
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise ValueError("No entries returned")
        info = entries[0]
    return info


def ytdlp_download_wav(url: str, full_duration_s) -> tuple:
    """Download `url` as .wav, capping at 5 min if longer. Returns (wav_path, capped)."""
    opts = {**_base_opts()}
    opts["postprocessors"] = [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}]

    capped = full_duration_s is not None and full_duration_s > MAX_SECONDS
    if capped:
        opts["download_ranges"] = download_range_func(None, [(0, MAX_SECONDS)])
        opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info and info.get("_type") == "playlist":
            info = [e for e in (info.get("entries") or []) if e][0]
        wav_path = os.path.join(DL_DIR, f"{info['id']}.wav")
    return wav_path, capped


# ── Route C — plain-text multi-stage resolver ──────────────────────────────────
def route_c_search(query_text: str):
    """
    Resolve a raw text query to a playable candidate via a 3-stage fallback.
    Returns (candidate_info, stage_name) on the first stage that yields a video,
    or (None, "NONE") if every stage yields no valid audio stream.
    """
    stages = [
        ("Primary_ProvidedToYouTube", f'ytsearch1:"{query_text} Provided to YouTube"'),
        ("Stage1_OfficialAudio",      f'ytsearch1:"{query_text} Official Audio"'),
        ("Stage2_SoundCloud",         f'scsearch1:"{query_text}"'),
    ]
    for stage_name, q in stages:
        try:
            cand = ytdlp_probe(q)                      # raises on empty / extractor error
        except Exception as exc:
            print(f"   [Route C/{stage_name}] no result: {exc}", file=sys.stderr, flush=True)
            continue
        if cand and (cand.get("id") or cand.get("webpage_url")):
            return cand, stage_name
    return None, "NONE"


# ── Core router ────────────────────────────────────────────────────────────────
def process_input(input_data: str) -> dict:
    """
    Universal dispatch for a single input (URL *or* plain-text query):
      • contains http(s)://  → Route A (Spotify) or Route B (direct URL)
      • otherwise            → Route C (text search + multi-stage fallback)
    Routes, (validates), downloads a 5-min-capped .wav, returns a metadata dict.
    """
    input_data = input_data.strip()
    row = {k: "" for k in LOG_FIELDS}
    row["Input_Source"] = input_data

    is_url = ("http://" in input_data) or ("https://" in input_data)
    if is_url:
        dom = DOMAIN_RE.match(input_data)
        row["Domain"] = dom.group(1).lower() if dom else ""

    try:
        # ───────────────────────── Route A — Spotify ─────────────────────────
        if is_url and "open.spotify.com/track/" in input_data:
            row["Route"] = "A_Spotify"
            meta = get_spotify_metadata(input_data)
            row["Track_Title"]         = meta["title"]
            row["Artist_Name"]         = meta["artist"]
            row["Spotify_Duration_MS"] = meta["duration_ms"]
            spotify_s = meta["duration_ms"] / 1000.0

            query = f'ytsearch1:"{meta["title"]}" "{meta["artist"]}" "Provided to YouTube"'
            cand = ytdlp_probe(query)
            yt_url  = cand.get("webpage_url") or cand.get("original_url") or cand.get("url")
            yt_dur  = cand.get("duration")
            row["Matched_YouTube_URL"] = yt_url
            row["Source_Duration_S"]   = yt_dur

            if yt_dur is None:
                row["Status"] = "FAILED_NO_YT_DURATION"
                return row

            delta = abs(float(yt_dur) - spotify_s)
            row["Duration_Match"] = bool(delta <= MATCH_MARGIN)
            if not row["Duration_Match"]:
                row["Status"] = f"SKIPPED_DURATION_MISMATCH(Δ={delta:.1f}s>{MATCH_MARGIN}s)"
                return row

            wav, capped = ytdlp_download_wav(yt_url, yt_dur)

        # ──────────────────────── Route B — Direct URL ───────────────────────
        elif is_url:
            row["Route"] = "B_Direct"
            info = ytdlp_probe(input_data)
            row["Track_Title"]       = info.get("title", "")
            row["Artist_Name"]       = info.get("uploader") or info.get("artist") or ""
            full_dur = info.get("duration")
            row["Source_Duration_S"] = full_dur
            row["Duration_Match"]    = "NA"
            target = info.get("webpage_url") or input_data
            wav, capped = ytdlp_download_wav(target, full_dur)

        # ──────────────────────── Route C — Text query ───────────────────────
        else:
            row["Route"] = "C_TextSearch"
            cand, stage = route_c_search(input_data)
            row["Search_Stage"] = stage
            if cand is None:                                  # Stage 3 — all failed
                row["Status"] = "NOT_FOUND"
                return row
            row["Track_Title"] = cand.get("title", "")
            row["Artist_Name"] = cand.get("uploader") or cand.get("artist") or ""
            full_dur = cand.get("duration")
            row["Source_Duration_S"] = full_dur
            row["Duration_Match"]    = "NA"
            target = cand.get("webpage_url") or cand.get("original_url") or cand.get("url")
            row["Matched_YouTube_URL"] = target
            row["Domain"] = urlparse(target).netloc if target else ""
            wav, capped = ytdlp_download_wav(target, full_dur)

        # ───────────────────────── Common post-download ──────────────────────
        row["Capped_To_5min"] = bool(capped)
        if os.path.exists(wav):
            row["Output_Filename"]  = os.path.basename(wav)
            row["Local_File_Path"]  = wav
            row["Final_Duration_S"] = ffprobe_duration(wav)
            row["Status"] = "SUCCESS"
        else:
            row["Status"] = "FAILED_NO_OUTPUT_FILE"
        return row

    except Exception as exc:
        row["Status"] = "FAILED"
        row["Error"]  = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {input_data}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        return row


# Backward-compatible alias (legacy callers used process_url)
process_url = process_input


# ── Main batch driver ──────────────────────────────────────────────────────────
def main():
    os.makedirs(DL_DIR, exist_ok=True)
    if not os.path.exists(URLS_TXT):
        print(f"No target_urls.txt at {URLS_TXT}", file=sys.stderr)
        sys.exit(1)

    with open(URLS_TXT) as f:
        inputs = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    print("=" * 68)
    print(f"Universal Audio Ingestion — {len(inputs)} input(s)  | URLs + text | 5-min cap | → .wav")
    print(f"Downloads → {DL_DIR}")
    print("=" * 68)

    rows = []
    for i, item in enumerate(inputs, 1):
        kind = "URL " if ("http://" in item or "https://" in item) else "TEXT"
        print(f"\n[{i}/{len(inputs)}] ({kind}) {item}", flush=True)
        row = process_input(item)        # already internally try/except-wrapped
        rows.append(row)
        print(f"   Route={row['Route']} Stage={row['Search_Stage']} Status={row['Status']} "
              f"Title={str(row['Track_Title'])[:46]!r} "
              f"Out={row['Output_Filename']} Final={row['Final_Duration_S']}s "
              f"Capped={row['Capped_To_5min']}", flush=True)

    with open(LOG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)

    ok = sum(r["Status"] == "SUCCESS" for r in rows)
    print("\n" + "=" * 68)
    print(f"DONE — {ok}/{len(rows)} succeeded.  Log: {LOG_CSV}")
    print("  Status breakdown:")
    from collections import Counter
    for s, n in Counter(r["Status"] for r in rows).items():
        print(f"    {s}: {n}")
    print("=" * 68)


if __name__ == "__main__":
    main()
