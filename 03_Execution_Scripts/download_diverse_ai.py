#!/usr/bin/env python3
"""
download_diverse_ai.py  —  build a SOURCE-DIVERSE AI ground-truth set
──────────────────────────────────────────────────────────────────────
The original data/ground_truth_ai/ set was 50 tracks from ONE channel
("Kelly Boesch AI Art"), which let the classifier win on a single production
fingerprint (100% in-domain CV = confound, not detection). This script rebuilds
the AI set with a strict PER-CHANNEL CAP across many creators + tools, so no
single fingerprint can dominate.

Gates (same purity standard as scrape_ai_ground_truth.py):
  • regex AI whitelist (reused) on title + description + tags
  • duration window 90–300 s
  • PER-CHANNEL CAP (default 3): at most N tracks per uploader
  • dedupe against the existing single-channel set (fresh tracks only)

Downloads best audio -> 192 kbps MP3 -> data/ground_truth_ai_diverse/%(id)s.mp3
and records id,channel,title,duration -> diverse_ai_manifest.csv (for source-aware
splits later).

Usage:
  python3 download_diverse_ai.py                 # target 50, cap 3/channel
  python3 download_diverse_ai.py --target 50 --per-channel 3
"""

import os, re, csv, sys, glob, argparse
from collections import Counter
import static_ffmpeg; static_ffmpeg.add_paths()
import yt_dlp

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
OLD_AI = os.path.join(HERE, "data", "ground_truth_ai")
OUT    = os.path.join(HERE, "data", "ground_truth_ai_diverse")
MANIFEST = os.path.join(ROOT, "diverse_ai_manifest.csv")

sys.path.insert(0, HERE)
from scrape_ai_ground_truth import find_ai_declaration      # reuse proven whitelist

MIN_DUR, MAX_DUR = 90, 300

# Reject non-music (tutorials/news/listicles/commentary) and derivatives (covers,
# versions, remakes) so only fully-AI-generated ORIGINAL songs survive — matching
# the human set of original commercial songs. A naive "AI music" search returns
# ~40% junk without this.
TITLE_BLACKLIST = re.compile(
    r"\b(tutorial|how ?to|guide|walkthrough|explained|review|reaction|react|news|"
    r"top ?\d+|top ten|best ai|comparison|trick|documentary|interview|podcast|"
    r"cover|version|reimagined|remake|rendition|tribute|parody|complete guide|"
    r"distribute|voice cloning|own your|we are the world)\b", re.I)
CHANNEL_BLACKLIST = re.compile(
    r"(news|tutorial|howto|instanthowto|tech\b|wiz|board ?cast|CBC|automation labs|"
    r"future of ai|discoverer|stratvert)", re.I)


def is_contaminated(title, channel):
    """True if the candidate is a tutorial/news/cover rather than an original AI song."""
    return bool(TITLE_BLACKLIST.search(title or "") or CHANNEL_BLACKLIST.search(channel or ""))

# Diverse queries: two tools (Suno/Udio) across several genres so the sample
# spans many creators and production styles, not one channel's aesthetic.
QUERIES = [
    "Suno AI song", "Udio AI song", "AI generated song", "made with Suno",
    "Udio full song", "AI generated pop song", "AI generated rap song",
    "AI generated rock song", "AI generated country song", "suno v4 full song",
    "AI generated song original", "Udio music track", "AI generated ballad",
    "AI generated electronic song",
    # broadened for more sources / styles
    "AI generated jazz song", "AI generated metal song", "AI generated folk song",
    "AI generated R&B song", "AI generated indie song", "AI generated lofi song",
    "AI generated blues song", "AI generated reggae song", "AI generated EDM track",
    "Suno original song", "Udio original song", "AI generated acoustic song",
    "AI generated soul song", "AI generated punk song", "AI generated funk song",
    "AI generated gospel song", "AI generated ambient track", "AI generated house music",
]


def search(query, n):
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(f"ytsearch{n}:{query}", download=False)
    return [e for e in (info.get("entries") or []) if e]


def meta(url):
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as y:
        i = y.extract_info(url, download=False)
    return {"title": i.get("title") or "", "description": i.get("description") or "",
            "tags": i.get("tags") or [], "uploader": i.get("uploader") or i.get("channel") or "?",
            "duration": i.get("duration"), "url": i.get("webpage_url") or url, "id": i.get("id")}


def download(url):
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "format": "bestaudio/best",
            "outtmpl": os.path.join(OUT, "%(id)s.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3",
                                "preferredquality": "192"}]}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=True)
    return os.path.join(OUT, f"{info['id']}.mp3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--per-channel", type=int, default=3)
    ap.add_argument("--per-query", type=int, default=25)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    old_ids = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{OLD_AI}/*.mp3")}
    have_ids = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{OUT}/*.mp3")}
    per_channel = Counter()
    rows, seen = [], set(old_ids) | have_ids

    print("=" * 72)
    print(f"DIVERSE AI downloader — target {args.target}, cap {args.per_channel}/channel")
    print(f"  excluding {len(old_ids)} old single-channel ids | queries: {len(QUERIES)}")
    print("=" * 72)

    got = len(have_ids)
    for q in QUERIES:
        if got >= args.target:
            break
        print(f"\n🔎 {q!r}")
        try:
            cands = search(q, args.per_query)
        except Exception as e:
            print(f"   [search error] {e}", file=sys.stderr); continue

        for e in cands:
            if got >= args.target:
                break
            vid = e.get("id")
            if not vid or vid in seen:
                continue
            seen.add(vid)
            url = e.get("url") or f"https://www.youtube.com/watch?v={vid}"
            try:
                m = meta(url)
            except Exception as ex:
                print(f"   [meta err] {ex}", file=sys.stderr); continue

            ch = m["uploader"]
            hay = " ".join([m["title"], m["description"], " ".join(m["tags"])])
            kw = find_ai_declaration(hay)
            if not kw:
                continue
            if is_contaminated(m["title"], ch):          # tutorial / news / cover
                print(f"   ⏭️  contaminated (tutorial/news/cover): {m['title'][:45]}")
                continue
            if not (m["duration"] and MIN_DUR <= m["duration"] <= MAX_DUR):
                continue
            if per_channel[ch] >= args.per_channel:
                print(f"   ⏭️  channel cap reached for {ch!r}: {m['title'][:45]}")
                continue

            try:
                path = download(url)
            except Exception as ex:
                print(f"   [dl err] {m['title'][:40]}: {ex}", file=sys.stderr); continue
            if not os.path.exists(path):
                continue
            per_channel[ch] += 1
            got += 1
            rows.append({"id": vid, "channel": ch, "title": m["title"],
                         "duration": int(m["duration"]), "matched": kw})
            print(f"   ✅ {got}/{args.target}  [{ch[:28]}]  {m['title'][:42]}  (‘{kw}’)")

    # write/append manifest
    write_header = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "channel", "title", "duration", "matched"])
        if write_header:
            w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 72)
    print(f"DONE — {got} diverse AI tracks in {OUT}")
    print(f"Channels represented: {len(per_channel)}")
    for ch, n in per_channel.most_common():
        print(f"   {n}x  {ch}")
    print(f"Manifest: {MANIFEST}")
    print("=" * 72)
    if got < args.target:
        print(f"⚠  only {got}/{args.target} — widen QUERIES / --per-query / --per-channel")


if __name__ == "__main__":
    main()
