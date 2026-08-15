#!/usr/bin/env python3
"""
download_human_amateur.py  —  amateur / indie HUMAN control set
────────────────────────────────────────────────────────────────
Tests the "mastering-quality confound": is the detector learning true AI
artifacts, or just "polished major-label master vs. lo-fi upload"? To find out
we need HUMAN music that is deliberately LOW-FIDELITY — bedroom pop, acoustic
covers, garage bands, open-mic / busking — the opposite production profile from
the 50 commercial major-label tracks.

Purity strategy (must be HUMAN, must be lo-fi, must be disjoint):
  • QUERIES imply a live human performer (acoustic cover, bedroom, garage, open mic).
  • EXCLUDE any AI declaration (reuse the AI whitelist as a NEGATIVE filter) —
    an amateur track that says "Suno"/"Udio"/"AI generated" is dropped.
  • EXCLUDE major-label / distributor channels (VEVO, "- Topic", Records) so the
    control set stays genuinely unpolished.
  • dedupe against the existing commercial human set (disjoint by video id).

-> data/ground_truth_human_amateur/%(id)s.mp3  + human_amateur_manifest.csv
"""

import os, re, csv, sys, glob, argparse
from collections import Counter
import static_ffmpeg; static_ffmpeg.add_paths()
import yt_dlp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COMMERCIAL = os.path.join(HERE, "data", "ground_truth_human")
OUT = os.path.join(HERE, "data", "ground_truth_human_amateur")
MANIFEST = os.path.join(ROOT, "human_amateur_manifest.csv")

sys.path.insert(0, HERE)
from scrape_ai_ground_truth import find_ai_declaration      # reuse as NEGATIVE filter

MIN_DUR, MAX_DUR = 90, 300

# Queries that imply a real, lo-fi human performance.
QUERIES = [
    "acoustic cover bedroom", "bedroom pop original song", "garage band original song",
    "singer songwriter acoustic original", "open mic live performance song",
    "busking street performance cover", "home recorded original song",
    "unsigned artist original song", "indie demo song lo-fi", "acoustic cover living room",
    "first original song teenager", "raw acoustic session original",
    "diy bedroom recording song", "amateur singer original song",
    "coffee shop live acoustic",
]

# Drop pro/label uploads so the control set stays genuinely unpolished.
LABEL_BLACKLIST = re.compile(r"(vevo|- ?topic|\brecords\b|\bofficial\b.*\bvideo\b|"
                             r"entertainment|\blabel\b)", re.I)
# Drop non-music / commentary.
TITLE_BLACKLIST = re.compile(r"\b(tutorial|how ?to|lesson|reaction|podcast|vlog|"
                             r"interview|gear review|unboxing|explained)\b", re.I)


def search(q, n):
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True,
                           "skip_download": True}) as y:
        info = y.extract_info(f"ytsearch{n}:{q}", download=False)
    return [e for e in (info.get("entries") or []) if e]


def meta(url):
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True,
                           "noplaylist": True}) as y:
        i = y.extract_info(url, download=False)
    return {"title": i.get("title") or "", "description": i.get("description") or "",
            "tags": i.get("tags") or [], "uploader": i.get("uploader") or i.get("channel") or "?",
            "duration": i.get("duration"), "id": i.get("id")}


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
    ap.add_argument("--target", type=int, default=30)
    ap.add_argument("--per-channel", type=int, default=2)
    ap.add_argument("--per-query", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    commercial_ids = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{COMMERCIAL}/*.mp3")}
    have = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{OUT}/*.mp3")}
    seen = set(commercial_ids) | have
    per_ch = Counter(); rows = []
    got = len(have)
    print("=" * 72)
    print(f"AMATEUR HUMAN control set — target {args.target}, cap {args.per_channel}/channel")
    print(f"  excluding {len(commercial_ids)} commercial human ids (disjoint)")
    print("=" * 72)

    for q in QUERIES:
        if got >= args.target:
            break
        print(f"\n🔎 {q!r}")
        try:
            cands = search(q, args.per_query)
        except Exception as e:
            print(f"   [search err] {e}", file=sys.stderr); continue
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
            except Exception:
                continue
            ch = m["uploader"]
            hay = " ".join([m["title"], m["description"], " ".join(m["tags"])])
            # NEGATIVE AI filter: an amateur track declaring AI is NOT human ground truth
            if find_ai_declaration(hay):
                print(f"   ⏭️  AI marker -> not human: {m['title'][:45]}"); continue
            if LABEL_BLACKLIST.search(ch) or LABEL_BLACKLIST.search(m["title"]):
                continue                                  # too pro / label upload
            if TITLE_BLACKLIST.search(m["title"]):
                continue
            if not (m["duration"] and MIN_DUR <= m["duration"] <= MAX_DUR):
                continue
            if per_ch[ch] >= args.per_channel:
                continue
            try:
                path = download(url)
            except Exception:
                continue
            if not os.path.exists(path):
                continue
            per_ch[ch] += 1; got += 1
            rows.append({"id": vid, "channel": ch, "title": m["title"], "duration": int(m["duration"])})
            print(f"   ✅ {got}/{args.target}  [{ch[:26]}]  {m['title'][:44]}")

    write_header = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "channel", "title", "duration"])
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print("\n" + "=" * 72)
    print(f"DONE — {got} amateur human tracks from {len(per_ch)} channels")
    print(f"Manifest: {MANIFEST}")
    print("=" * 72)


if __name__ == "__main__":
    main()
