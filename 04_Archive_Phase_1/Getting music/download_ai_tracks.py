#!/usr/bin/env python3
"""
Build the AI-generated half of the AI Music Detector benchmark dataset.

Strategy
--------
Phase 1  Try to pull pre-generated audio from a public HuggingFace dataset
         (m-a-p/MusicBench — contains AI-generated test clips from multiple models).
Phase 2  Generate locally with facebook/musicgen-small if Phase 1 falls short.

Output   : songs/ai_track_01.mp3 .. songs/ai_track_50.mp3
Model    : facebook/musicgen-small  (Meta AI Research, ICML 2023)
           https://huggingface.co/facebook/musicgen-small
           "Simple and Controllable Music Generation", Copet et al. 2023
           Licence: CC-BY-NC 4.0 (non-commercial research)
"""

import os, sys, time, subprocess, importlib
import numpy as np

SONGS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "songs")
TARGET         = 50
SAMPLE_RATE    = 32_000   # MusicGen-small native sample rate
MAX_NEW_TOKENS = 256      # ≈ 5 s audio; increase to 512 for ≈ 10 s (2× slower)
BATCH_SIZE     = 4        # tracks generated per forward pass

# 50 genre-diverse prompts – maximises stylistic variety in the AI dataset
PROMPTS = [
    "upbeat electronic dance music with synthesizer arpeggios and four-on-the-floor kick",
    "relaxing lo-fi hip hop with mellow piano chords and subtle vinyl crackle",
    "jazz quartet with walking upright bass, brushed drums and improvised saxophone",
    "heavy metal guitar riff with high-gain distortion and rapid double bass drum",
    "classical string quartet in C minor in the style of Beethoven, adagio movement",
    "reggae rhythm with offbeat guitar skank, melodica lead and deep sub bass",
    "ambient soundscape with slowly evolving atmospheric synth pads and long reverb",
    "bossa nova acoustic guitar with light shaker and soft flute melody",
    "epic orchestral battle soundtrack with brass fanfare, strings and thunderous timpani",
    "acoustic delta blues guitar with shuffling 12-bar progression and slide technique",
    "synthwave with retro arpeggio leads, gated reverb snare and analog bass synth",
    "celtic folk with fiddle melody, bodhrán and fingerpicked acoustic guitar",
    "trap music with triplet hi-hats, 808 bass slides and atmospheric piano chord",
    "classical piano etude with fast chromatic runs in the style of Chopin op. 10",
    "funk groove with slap bass, tight brass stabs and wah rhythm guitar",
    "ambient drone music with slowly drifting low-frequency sustained tones",
    "flamenco guitar with rapid picado fingerpicking, palmas handclap rhythm",
    "drum and bass with rolling amen breakbeat and powerful sub bass line",
    "bluegrass with banjo picking, fiddle countermelody and upright bass",
    "cinematic film score suspense scene with strings, brass swells and piano",
    "deep techno with industrial kick percussion, dark filtered synthesizers",
    "smooth jazz with electric piano comping, walking bass and light brushed snare",
    "folk acoustic guitar fingerstyle ballad with Travis picking pattern",
    "80s synth pop with Roland drum machine, Juno bass and bright lead synth",
    "African hand percussion ensemble with interlocking polyrhythmic patterns",
    "chamber music piano trio in romantic style, expressive cello melody",
    "indie rock with jangly Rickenbacker guitar, melodic bass and steady backbeat",
    "tropical house with marimba melody, steel drum accents and laid-back beat",
    "minimalist piano piece with sparse repeating motifs and deliberate silence",
    "Cuban son montuno with brass clave, congas and piano guajeo",
    "progressive rock instrumental with odd-time riff, mellotron and bass solo",
    "new age meditation music with crystal singing bowls and gentle water sounds",
    "bebop jazz piano trio with fast bebop line, comping chords and brushed swing",
    "K-pop style production with bright plucked synth, punchy kick and driving beat",
    "dark orchestral horror underscore with dissonant string clusters and low brass",
    "traditional Japanese melody with koto and shakuhachi bamboo flute",
    "post-rock guitar instrumental building from sparse clean guitar to loud walls",
    "gospel choir accompanied by Hammond organ, claps and tambourine",
    "Latin jazz with timbales solo, vibraphone melody and piano montuno",
    "neo-classical piano with subtle glitchy electronic texture underneath",
    "ragtime piano in the style of Scott Joplin, bright and rhythmically playful",
    "psychedelic rock with autowah guitar, phaser and swirling studio effects",
    "Nordic folk with nyckelharpa drone, hurdy-gurdy and steady frame drum",
    "deep house music with warm detuned chord stabs and chopped vocal sample",
    "marching band fanfare with trumpet, trombone, bass drum and snare cadence",
    "Japanese city pop with clean chorus guitar licks, synth bass and smooth chord",
    "orchestral symphony slow movement featuring oboe solo over pizzicato strings",
    "glitch hop with time-stretched beat stutters and heavy digital compression",
    "acoustic fingerstyle guitar in DADGAD open tuning with natural harmonics",
    "dramatic tango with bandoneon tremolo, violin counter-melody and staccato piano",
]

assert len(PROMPTS) == TARGET, f"Need exactly {TARGET} prompts"


# ── Utilities ─────────────────────────────────────────────────────────────────

def pip_install(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet"] + list(packages)
    )


def ensure(import_name, pip_name=None):
    try:
        return importlib.import_module(import_name)
    except ImportError:
        pip_install(pip_name or import_name)
        return importlib.import_module(import_name)


def save_mp3(audio_np: np.ndarray, sample_rate: int, path: str, bitrate: int = 128):
    """Encode float32 numpy audio as MP3 via lameenc (no ffmpeg required)."""
    lameenc = ensure("lameenc")
    peak = np.abs(audio_np).max()
    if peak > 0:
        audio_np = audio_np / peak * 0.95
    pcm = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(1)
    enc.silence()
    data = enc.encode(pcm.tobytes()) + enc.flush()
    with open(path, "wb") as fh:
        fh.write(data)
    return len(data)


# ── Phase 1: HuggingFace pre-generated dataset ────────────────────────────────

def phase1_hf_dataset(slots):
    """Try m-a-p/MusicBench for AI-generated test-split audio. Returns filled count."""
    print("Phase 1 — probing m-a-p/MusicBench on HuggingFace…")
    try:
        ensure("datasets", "datasets")
        from datasets import load_dataset

        ds = load_dataset(
            "m-a-p/MusicBench", split="test", streaming=True, trust_remote_code=True
        )
        first = next(iter(ds))
        cols = list(first.keys())
        print(f"  Dataset columns: {cols}")

        # Only proceed if there is actual audio in the dataset
        audio_col = next(
            (c for c in ("generated_audio", "audio") if c in first and first[c] is not None),
            None,
        )
        if audio_col is None:
            print("  No audio column found — skipping Phase 1")
            return 0

        filled = 0
        for i, sample in enumerate(ds):
            if filled >= len(slots):
                break
            clip = sample.get(audio_col)
            if clip is None:
                continue
            arr = np.asarray(clip["array"], dtype=np.float32)
            sr  = clip["sampling_rate"]
            dest = os.path.join(SONGS_DIR, slots[filled])
            nb   = save_mp3(arr, sr, dest)
            dur  = len(arr) / sr
            print(f"  [{filled+1}/{len(slots)}] MusicBench item {i}  {dur:.1f}s  {nb//1024} KB  → {slots[filled]}")
            filled += 1

        return filled

    except Exception as exc:
        print(f"  Phase 1 unavailable: {exc}")
        return 0


# ── Phase 2: facebook/musicgen-small local generation ─────────────────────────

def load_model():
    """Install dependencies, load model and processor, return (model, processor, device, sr)."""
    import torch

    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"  Device: {device.upper()}")

    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    repo = "facebook/musicgen-small"
    print(f"  Loading {repo} (≈300 MB first run)…")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(repo)
    model     = MusicgenForConditionalGeneration.from_pretrained(repo)
    model     = model.to(device)
    model.eval()
    sr = model.config.audio_encoder.sampling_rate
    print(f"  Model ready in {time.time()-t0:.1f}s  |  sample_rate={sr} Hz")
    return model, processor, device, sr


def phase2_generate(slots, prompts):
    """Generate tracks in batches; print per-batch timing and ETA."""
    import torch

    # Install deps if needed
    for pkg, pip in [("torch", "torch"), ("transformers", "transformers>=4.31.0"),
                     ("accelerate", "accelerate"), ("lameenc", "lameenc")]:
        ensure(pkg, pip)

    model, processor, device, sr = load_model()

    n        = len(slots)
    filled   = 0
    t_start  = time.time()

    while filled < n:
        end     = min(filled + BATCH_SIZE, n)
        b_slots = slots[filled:end]
        b_proms = prompts[filled:end]

        print(f"\n  Batch [{filled+1}–{end}/{n}]")
        for p in b_proms:
            print(f"    ▸ {p[:72]}")

        t0     = time.time()
        inputs = processor(text=b_proms, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            audio_values = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        elapsed = time.time() - t0
        per_track = elapsed / len(b_slots)
        done_so_far = filled + len(b_slots)
        remaining = n - done_so_far
        eta = per_track * remaining

        print(f"  Generated in {elapsed:.1f}s ({per_track:.1f}s/track)  ETA {eta/60:.1f} min")

        for j, slot in enumerate(b_slots):
            audio_np = audio_values[j, 0].cpu().float().numpy()
            dest     = os.path.join(SONGS_DIR, slot)
            nb       = save_mp3(audio_np, sr, dest)
            dur      = len(audio_np) / sr
            print(f"  Saved {slot}  {dur:.1f}s  {nb//1024} KB")
            filled += 1

    return filled


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SONGS_DIR, exist_ok=True)

    all_slots    = [f"ai_track_{i:02d}.mp3" for i in range(1, TARGET + 1)]
    all_prompts  = list(PROMPTS)

    # Resume: skip already-downloaded tracks
    missing_idx  = [i for i, s in enumerate(all_slots)
                    if not os.path.exists(os.path.join(SONGS_DIR, s))]
    slots   = [all_slots[i]   for i in missing_idx]
    prompts = [all_prompts[i] for i in missing_idx]

    if not slots:
        print("All 50 AI tracks already present — nothing to do.")
        return

    print("=" * 65)
    print(f"AI Track Dataset Builder  —  {len(slots)} track(s) to acquire")
    print("=" * 65)
    print()
    print("Source : facebook/musicgen-small")
    print("Paper  : Copet et al., 2023  'Simple and Controllable Music Generation'")
    print("Repo   : https://huggingface.co/facebook/musicgen-small")
    print("Licence: CC-BY-NC 4.0  (non-commercial research use)")
    print(f"Clip   : ≈{MAX_NEW_TOKENS/50:.0f}s per track  |  32 kHz mono  |  128 kbps MP3")
    print()

    # Phase 1 — try HuggingFace dataset (fast path)
    filled = phase1_hf_dataset(slots)
    slots   = slots[filled:]
    prompts = prompts[filled:]

    # Phase 2 — local MusicGen generation
    if slots:
        print(f"\nPhase 2 — local MusicGen generation  ({len(slots)} track(s))")
        phase2_generate(slots, prompts)

    # Final summary
    ai_tracks    = sorted(f for f in os.listdir(SONGS_DIR) if f.startswith("ai_track_"))
    human_tracks = sorted(f for f in os.listdir(SONGS_DIR) if f.startswith("human_track_"))
    total        = len(ai_tracks) + len(human_tracks)

    print(f"\n{'='*65}")
    print("DATASET SUMMARY")
    print("="*65)
    print(f"  AI tracks   : {len(ai_tracks):>3}  (ai_track_01 – ai_track_50)")
    print(f"  Human tracks: {len(human_tracks):>3}  (human_track_01 – human_track_50)")
    print(f"  Total       : {total:>3}")
    print(f"  Location    : {SONGS_DIR}")
    print()
    print("  AI-track provenance: facebook/musicgen-small, ICML 2023")
    print("  Human-track provenance: Internet Archive netlabels (CC-licensed)")
    if len(ai_tracks) < TARGET:
        print(f"\n  WARNING: only {len(ai_tracks)}/50 AI tracks present — re-run to fill gaps.")


if __name__ == "__main__":
    main()
