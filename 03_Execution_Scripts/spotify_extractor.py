#!/usr/bin/env python3
"""
spotify_extractor.py  —  real Copyright-registry extractor for Late Fusion
───────────────────────────────────────────────────────────────────────────
Replaces the mock check_copyright_registry() in fusion_aggregator.py with a
live Spotify-catalog lookup (standalone module for now — fusion_aggregator.py
is wired up in a later step).

Logic
─────
  1. extract_id3_tags(file_path)  — mutagen reads the MP3's ID3 header and
     returns (Title, Artist).
  2. check_spotify_registry(title, artist) — two-step DEEP PUBLISHING lookup:
       a. Search API: track:{title} artist:{artist} -> top hit -> album["id"]
       b. Album endpoint sp.album(album_id) -> `copyrights` array -> the
          type=='C' entry (Composition copyright = registered human
          SONGWRITING, distinct from 'P', the phonographic recording
          copyright that AI uploads can also carry).
     (Spotify does not expose ISWC codes directly; the album Composition
      copyright is the closest authoritative publishing signal. The earlier
      ISRC check was flawed: distributors hand ISRCs to ANY upload, including
      AI tracks, so it proved nothing about human authorship.)

Return convention (fusion-compatible)
─────────────────────────────────────
  0.0   -> a valid type=='C' Composition copyright exists  => human
           songwriting registered (Confirmed Human signal)
  None  -> no track found, no album id, copyrights missing / P-only, or any
           API error  => Neutral / Missing Data (signal excluded from fusion)

Auth
────
  Credentials come from environment variables (never hardcoded):
      export SPOTIPY_CLIENT_ID="..."
      export SPOTIPY_CLIENT_SECRET="..."

Standalone test
───────────────
  python3 spotify_extractor.py --verify        # live check: Daniel Caesar - Get You
  python3 spotify_extractor.py path/to/song.mp3
  python3 spotify_extractor.py path/to/song.mp3 --title "Yellow" --artist "Coldplay"
  (the --title/--artist overrides matter for our ground-truth MP3s, which were
   downloaded as bare <youtube_id>.mp3 files with EMPTY ID3 headers)
"""

import os
import sys
import argparse

from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# ── Spotify client (Client-Credentials flow, env-var driven) ───────────────────
_SPOTIFY = None          # lazy singleton so import never triggers a network call


def _get_client() -> spotipy.Spotify:
    """Build (once) and return an app-authenticated Spotify client."""
    global _SPOTIFY
    if _SPOTIFY is None:
        # SpotifyClientCredentials reads SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET
        # from the environment on its own; we pre-check to fail with a clear
        # message instead of a deep spotipy traceback.
        missing = [v for v in ("SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET")
                   if not os.environ.get(v)]
        if missing:
            raise RuntimeError(
                f"Missing environment variable(s): {', '.join(missing)}. "
                f"Export your Spotify app credentials before running.")
        _SPOTIFY = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(),
            requests_timeout=10, retries=2)
    return _SPOTIFY


# ── 1. Metadata parsing (mutagen) ──────────────────────────────────────────────
def extract_id3_tags(file_path: str):
    """
    Read the MP3's ID3 tags and return (title, artist) as stripped strings.
    Either element is None when the tag is absent or the file has no ID3 header.
    """
    try:
        tags = EasyID3(file_path)
    except ID3NoHeaderError:                      # MP3 with no ID3 header at all
        return None, None

    def first(key):
        vals = tags.get(key) or []
        val = (vals[0] or "").strip() if vals else ""
        return val or None

    return first("title"), first("artist")


# ── 2/3. Spotify search + Composition-copyright verification ──────────────────
def _find_composition_copyright(copyrights) -> str:
    """Return the text of the type=='C' (Composition / songwriting) entry, or None.
    'C' is the composition copyright — distinct from 'P', the phonographic
    (recording) copyright that even AI uploads can carry."""
    for entry in copyrights or []:
        try:
            if entry.get("type") == "C":
                text = entry.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        except (AttributeError, KeyError):        # malformed entry — keep scanning
            continue
    return None


def check_spotify_registry(title: str, artist: str):
    """
    Query the Spotify catalog for `title` by `artist` and verify PUBLISHING
    registration via the album's Composition copyright (deep publishing
    metadata — Spotify does not expose ISWC codes directly, so the album
    `copyrights` array is the closest authoritative signal).

      0.0   -> top track hit's album carries a valid type=='C' (Composition)
               copyright string  => human songwriting registered
      None  -> no usable metadata, no search hit, missing album id, copyrights
               absent / P-only, or any network error (fusion drops the weight)
    """
    if not title or not artist:
        return None                               # nothing to search on

    sp = _get_client()                            # RuntimeError (no creds) propagates
    try:
        # 1st call: track search — grab the album id off the top hit
        resp = sp.search(q=f"track:{title} artist:{artist}", type="track", limit=1)
        items = (resp or {}).get("tracks", {}).get("items", [])
        if not items:
            return None                           # track not in the catalog
        album_id = (items[0].get("album") or {}).get("id")
        if not album_id:
            return None                           # no album to interrogate

        # 2nd call: album endpoint — the copyrights array lives here
        album = sp.album(album_id)
        c_text = _find_composition_copyright((album or {}).get("copyrights"))
        return 0.0 if c_text else None            # C found -> Confirmed Human
    except Exception as exc:                      # timeout / KeyError / bad payload
        print(f"[spotify] registry check failed for {artist!r} - {title!r}: {exc}",
              file=sys.stderr)
        return None


# ── Fusion-ready wrapper (drop-in for fusion_aggregator's Copyright slot) ──────
def check_copyright_registry(file_path: str):
    """file_path -> ID3 (title, artist) -> Spotify ISRC check. 0.0 or None."""
    title, artist = extract_id3_tags(file_path)
    return check_spotify_registry(title, artist)


# ── Standalone test harness ────────────────────────────────────────────────────
def _live_verify(title="Get You", artist="Daniel Caesar") -> int:
    """Quick live check: a known human track must pull the 'C' copyright -> 0.0."""
    print("=" * 64)
    print(f"LIVE VERIFY — {artist} - {title} (Composition copyright)")
    print("=" * 64)
    try:
        sp = _get_client()
        items = sp.search(q=f"track:{title} artist:{artist}",
                          type="track", limit=1)["tracks"]["items"]
        album_id = items[0]["album"]["id"]
        album = sp.album(album_id)
        print(f"  Top hit    : {items[0]['artists'][0]['name']} - {items[0]['name']}")
        print(f"  Album      : {album['name']}  (id {album_id})")
        for entry in album.get("copyrights") or []:
            print(f"  Copyright  : [{entry.get('type')}] {entry.get('text')}")
        score = check_spotify_registry(title, artist)
        verdict = "Confirmed Human / Registered" if score == 0.0 else "Neutral / Missing Data"
        print(f"  Result     : {score}  ({verdict})")
        print("=" * 64)
        return 0 if score == 0.0 else 1
    except RuntimeError as exc:
        print(f"  Result     : ABORTED — {exc}")
        print("=" * 64)
        return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Spotify Composition-copyright registry check.")
    ap.add_argument("mp3", nargs="?", help="path to a local MP3 file")
    ap.add_argument("--title",  default=None, help="override the ID3 title")
    ap.add_argument("--artist", default=None, help="override the ID3 artist")
    ap.add_argument("--verify", action="store_true",
                    help="live sanity check on a known human track (Daniel Caesar - Get You)")
    args = ap.parse_args()

    if args.verify:
        sys.exit(_live_verify())
    if not args.mp3:
        ap.error("provide an MP3 path, or use --verify")
    if not os.path.exists(args.mp3):
        sys.exit(f"File not found: {args.mp3}")

    print("=" * 64)
    print("SPOTIFY COPYRIGHT-REGISTRY EXTRACTOR — standalone test")
    print("=" * 64)
    print(f"  File   : {args.mp3}")

    id3_title, id3_artist = extract_id3_tags(args.mp3)
    print(f"  ID3    : title={id3_title!r}  artist={id3_artist!r}")

    title  = args.title  or id3_title
    artist = args.artist or id3_artist
    print(f"  Query  : title={title!r}  artist={artist!r}"
          + ("  (from --title/--artist override)" if args.title or args.artist else ""))

    if not title or not artist:
        print("  Result : None  (no usable Title/Artist metadata — Neutral / Missing Data)")
        print("=" * 64)
        sys.exit(0)

    try:
        score = check_spotify_registry(title, artist)
    except RuntimeError as exc:                   # credentials not exported
        print(f"  Result : ABORTED — {exc}")
        print("=" * 64)
        sys.exit(1)
    if score == 0.0:
        # re-run verbosely to show which Composition copyright matched
        try:
            hit = _get_client().search(q=f"track:{title} artist:{artist}",
                                       type="track", limit=1)["tracks"]["items"][0]
            c_text = _find_composition_copyright(
                _get_client().album(hit["album"]["id"]).get("copyrights"))
            print(f"  Match  : {hit['artists'][0]['name']} - {hit['name']}"
                  f"  (C: {c_text})")
        except Exception:
            pass
        print("  Result : 0.0  (Composition copyright found — Confirmed Human / Registered)")
    else:
        print("  Result : None  (no hit / no 'C' copyright — Neutral / Missing Data)")
    print("=" * 64)
