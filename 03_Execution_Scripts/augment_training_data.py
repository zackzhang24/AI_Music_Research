#!/usr/bin/env python3
"""
augment_training_data.py
─────────────────────────
Multi-codec data augmentation for the 400-track training set.

For each of the 400 pristine tracks (the same TRAIN_DIRS used by
run_streaming_evaluation.build_or_load_svm), generate two web-compressed
copies — 128 kbps AAC and 128 kbps Opus, the two codecs YouTube actually
streams — and save them alongside the original into a mixed-distribution
training folder:

    data/train_augmented/
        Human/
            <id>_clean.<ext>     (pristine original, untouched bytes)
            <id>_aac.m4a         (128 kbps AAC)
            <id>_opus.opus       (128 kbps Opus)
        AI/
            ... same pattern ...

A manifest CSV (train_augmented_manifest.csv) records every output row:
Filename, Label, Codec, Source_File, Source_Directory.

Run standalone:
    python3 augment_training_data.py
"""

import os, sys, csv, glob, shutil, subprocess
import static_ffmpeg; static_ffmpeg.add_paths()    # ffmpeg + ffprobe on PATH

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
DATASETS = os.path.join(ROOT, "01_Final_Datasets")
OUT_DIR  = os.path.join(HERE, "data", "train_augmented")
MANIFEST = os.path.join(HERE, "train_augmented_manifest.csv")

# Same 400-track pristine training set used by Lcrosvila's SVM baseline.
TRAIN_DIRS = [
    ("human_tracks_120s", "Human"), ("human_tracks_10s", "Human"),
    ("ai_tracks_120s", "AI"),       ("new_ai_tracks_10s", "AI"),
]
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

AAC_BITRATE  = "128k"
OPUS_BITRATE = "128k"


def transcode(src, dst, codec):
    """codec: 'aac' or 'opus'. Raises CalledProcessError on failure."""
    if codec == "aac":
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src,
               "-c:a", "aac", "-b:a", AAC_BITRATE, dst]
    elif codec == "opus":
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src,
               "-c:a", "libopus", "-b:a", OPUS_BITRATE, dst]
    else:
        raise ValueError(codec)
    subprocess.check_call(cmd)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for label in ("Human", "AI"):
        os.makedirs(os.path.join(OUT_DIR, label), exist_ok=True)

    sources = []
    for sub, label in TRAIN_DIRS:
        files = sorted(f for f in glob.glob(os.path.join(DATASETS, sub, "*"))
                       if os.path.splitext(f)[1].lower() in AUDIO_EXT)
        for f in files:
            sources.append((f, sub, label))

    print("=" * 70)
    print(f"Multi-codec augmentation — {len(sources)} pristine tracks "
          f"-> {len(sources)*3} files (Clean + AAC + Opus)")
    print(f"Output: {OUT_DIR}")
    print("=" * 70)

    rows = []
    ok = skip = 0
    for i, (src, sub, label) in enumerate(sources, 1):
        track_id = f"{sub}_{os.path.splitext(os.path.basename(src))[0]}"
        src_ext = os.path.splitext(src)[1].lower()

        clean_dst = os.path.join(OUT_DIR, label, f"{track_id}_clean{src_ext}")
        aac_dst   = os.path.join(OUT_DIR, label, f"{track_id}_aac.m4a")
        opus_dst  = os.path.join(OUT_DIR, label, f"{track_id}_opus.opus")

        try:
            # Clean: copy pristine bytes untouched (no re-encode)
            if not os.path.exists(clean_dst):
                shutil.copyfile(src, clean_dst)
            # AAC 128k (simulates YouTube's AAC audio stream)
            if not os.path.exists(aac_dst):
                transcode(src, aac_dst, "aac")
            # Opus 128k (simulates YouTube's WebM/Opus audio stream)
            if not os.path.exists(opus_dst):
                transcode(src, opus_dst, "opus")

            for dst, codec in [(clean_dst, "Clean"), (aac_dst, "AAC"), (opus_dst, "Opus")]:
                rows.append({
                    "Filename": os.path.basename(dst), "Label": label, "Codec": codec,
                    "Source_File": os.path.basename(src), "Source_Directory": sub,
                    "Track_ID": track_id,
                })
            ok += 1
        except Exception as exc:
            print(f"  [SKIP] {sub}/{os.path.basename(src)}: {exc}", file=sys.stderr, flush=True)
            skip += 1

        if i % 25 == 0 or i == len(sources):
            print(f"  {i}/{len(sources)} tracks augmented (ok={ok}, skip={skip})", flush=True)

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Filename", "Label", "Codec",
                                          "Source_File", "Source_Directory", "Track_ID"])
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 70)
    print(f"DONE — {ok}/{len(sources)} tracks augmented, {skip} skipped.")
    print(f"Total files in {OUT_DIR}: {len(rows)}")
    print(f"Manifest: {MANIFEST}")
    print("=" * 70)


if __name__ == "__main__":
    main()
