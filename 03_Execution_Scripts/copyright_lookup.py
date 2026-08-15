#!/usr/bin/env python3
"""
copyright_lookup.py  —  credential-free commercial-release / copyright lookup
──────────────────────────────────────────────────────────────────────────────
Drop-in replacement for the Spotify copyright signal that needs NO API keys.
Uses the free public iTunes Search API (primary) + MusicBrainz (fallback) to
answer: "is this a commercially released song, by whom, on what label, and when?"

A commercial catalog release is a strong HUMAN signal (hobbyist AI uploads on
YouTube are almost never in the iTunes/MusicBrainz catalog). Fuzzy match
verification prevents an AI track titled "Yellow" from matching Coldplay.

Public API surface (fusion-compatible):
  lookup_release(title, artist)      -> dict {artist,album,label,year,source} | None
  check_copyright_registry(title, artist) -> 0.0 (registered human) | None

Return convention:
  0.0   -> confident commercial-catalog match found  => Confirmed Human
  None  -> no confident match  => Neutral / Missing Data (fusion drops the weight)
"""

import sys, time, difflib
import requests

ITUNES = "https://itunes.apple.com/search"
MB     = "https://musicbrainz.org/ws/2/recording"
UA     = "AIMusicDetector/1.0 (research benchmark)"
TIMEOUT = 15

# Match acceptance: how similar the catalog hit must be to what we searched for.
ARTIST_SIM_MIN = 0.60      # artist strings must be reasonably close
TITLE_SIM_MIN  = 0.55      # title strings must be reasonably close


import re as _re
_PARENS = _re.compile(r"[\(\[][^\)\]]*"
                      r"(remaster|official|audio|video|visuali|lyric|version|"
                      r"deluxe|mono|stereo|feat|remix|explicit|hd|4k)[^\)\]]*[\)\]]", _re.I)

def _clean(s: str) -> str:
    """Strip parenthetical/marketing junk but KEEP words intact (for search + compare)."""
    s = (s or "").lower()
    s = _PARENS.sub(" ", s)                       # drop (Remastered 2009), (Official Audio)…
    for junk in ["official audio", "official video", "official music video",
                 "lyric video", "visualizer", "remastered", "feat.", "ft.", "- topic"]:
        s = s.replace(junk, " ")
    return " ".join(s.split())                    # collapse whitespace (no char-splitting!)


def _norm(s: str) -> str:
    """Cleaned + punctuation-stripped, for fuzzy comparison."""
    s = _clean(s)
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s)
    return " ".join(s.split())


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _itunes(title: str, artist: str):
    # search on CLEANED text so "(Official Audio)"/"(Visualizer)" don't derail ranking
    term = f"{_clean(artist)} {_clean(title)}".strip()
    try:
        r = requests.get(ITUNES, params={"term": term, "entity": "song",
                                          "limit": 5}, timeout=TIMEOUT)
        results = r.json().get("results", [])
    except Exception as exc:
        print(f"[copyright] iTunes error: {exc}", file=sys.stderr)
        return None
    for x in results:
        t_sim = _sim(title, x.get("trackName", ""))
        a_sim = _sim(artist, x.get("artistName", "")) if artist else 0.0
        # accept if BOTH title & artist close, OR title near-exact (covers the
        # empty-artist case: "- Topic" uploads whose title is just the song),
        # OR artist near-exact.
        if (t_sim >= TITLE_SIM_MIN and a_sim >= ARTIST_SIM_MIN) or t_sim >= 0.85 or a_sim >= 0.9:
            return {"artist": x.get("artistName"), "album": x.get("collectionName"),
                    "label": x.get("copyright"), "year": (x.get("releaseDate") or "")[:4],
                    "genre": x.get("primaryGenreName"), "source": "iTunes",
                    "match": round((t_sim + a_sim) / 2, 2)}
    return None


def _musicbrainz(title: str, artist: str):
    """Fallback: MusicBrainz recording search (has label via release, free, no key)."""
    try:
        r = requests.get(MB, params={"query": f'recording:"{title}" AND artist:"{artist}"',
                                      "fmt": "json", "limit": 3},
                         headers={"User-Agent": UA}, timeout=TIMEOUT)
        recs = r.json().get("recordings", [])
    except Exception as exc:
        print(f"[copyright] MusicBrainz error: {exc}", file=sys.stderr)
        return None
    for rec in recs:
        a_name = (rec.get("artist-credit") or [{}])[0].get("name", "")
        if _sim(title, rec.get("title", "")) >= TITLE_SIM_MIN and _sim(artist, a_name) >= ARTIST_SIM_MIN:
            rel = (rec.get("releases") or [{}])[0]
            label = None
            for li in (rel.get("label-info") or []):
                label = (li.get("label") or {}).get("name"); break
            return {"artist": a_name, "album": rel.get("title"), "label": label,
                    "year": (rec.get("first-release-date") or "")[:4],
                    "genre": None, "source": "MusicBrainz",
                    "match": round(_sim(title, rec.get("title", "")), 2)}
    return None


def lookup_release(title: str, artist: str):
    """Return commercial-release metadata (iTunes, then MusicBrainz) or None."""
    if not title:
        return None
    hit = _itunes(title, artist or "")
    if hit:
        return hit
    if artist:
        return _musicbrainz(title, artist)
    return None


def check_copyright_registry(title: str, artist: str):
    """Fusion signal: 0.0 if a confident commercial release is found, else None."""
    return 0.0 if lookup_release(title, artist) else None


if __name__ == "__main__":
    tests = [("Yellow", "Coldplay"), ("Blank Space", "Taylor Swift"),
             ("Get You", "Daniel Caesar"),
             ("Undercooked", "Nuclear Bum Records"),        # AI hobbyist -> None
             ("Trump vs Iran AI Rap Battle", "Politi Pop"), # AI -> None
             ("Yellow", "Nuclear Bum Records")]             # title-match but wrong artist -> None
    print("=" * 70)
    print("COPYRIGHT / COMMERCIAL-RELEASE LOOKUP  (iTunes + MusicBrainz, no keys)")
    print("=" * 70)
    for t, a in tests:
        rel = lookup_release(t, a)
        score = check_copyright_registry(t, a)
        if rel:
            print(f"  {a} – {t}")
            print(f"     -> {rel['source']}: {rel['artist']} · {rel['album']} · "
                  f"{rel['year']}  (label: {rel['label'] or 'n/a'}, match {rel['match']})  => {score}")
        else:
            print(f"  {a} – {t}\n     -> no confident catalog match  => {score}")
        time.sleep(0.3)
    print("=" * 70)
