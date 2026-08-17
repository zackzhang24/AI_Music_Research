Context-Aware AI Music Detector

A research prototype for detecting AI-generated music using both acoustic analysis and contextual evidence.

Most AI-music detectors only analyze the audio itself. This project explores whether detection can become more reliable by also looking at the information surrounding a track — such as copyright metadata, creator descriptions, and audience discussion.

How It Works

The system first runs the audio through an acoustic detector and estimates the probability that the track is AI-generated.

Two open-source models were evaluated:

* Lcrosvila — uses LAION-CLAP audio embeddings with an SVM classifier
* Mippia / Fusion Segment Transformer — uses MERT-based features and a transformer designed to analyze musical structure across time

Testing showed that the two models behave differently depending on the input. Lcrosvila was much more stable on short clips, while Mippia performed significantly better when given longer audio.

Input Length	Lcrosvila	  Mippia
10 seconds	  96.0%	      70.0%
120 seconds	  99.0%	      95.5%

Because different models are more reliable under different conditions, the system does not treat every detector score equally.

⸻

Input Handling

The application can analyze either:

* a direct audio file
* a YouTube URL
* a track search that resolves to an online source

For online audio, the pipeline uses yt-dlp to retrieve accessible audio and metadata before running inference.

The input source matters because streaming platforms may compress or transcode uploaded audio. The system therefore routes and interprets model outputs differently depending on whether it is analyzing a direct file or platform-derived audio.

⸻

Late-Fusion Logic

The acoustic model acts as the first layer.

Audio Input
    │
    ▼
Acoustic Analysis
    │
    ▼
P(AI)

If the acoustic evidence strongly indicates either human or AI authorship, the system can return that result directly.

If the acoustic score falls into an uncertain range, the system triggers a late-fusion layer and gathers additional contextual evidence.

                Acoustic Score
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
     Human-like    Ambiguous    AI-like
                       │
                       ▼
                  Late Fusion

The late-fusion layer uses three additional signals:

© Copyright / Distribution Metadata

The system queries available Spotify metadata and copyright fields.

Existing commercial metadata can support a human-authorship verdict, although it is treated as evidence rather than definitive proof because AI-generated music can also be commercially distributed.

- Creator Description

YouTube descriptions are scanned for AI-related disclosures.

For example:

"Made with Suno"
        → AI signal
"AI-generated track"
        → AI signal
"Not made with AI"
        → should not be treated as an AI admission

The text parser also handles basic edge cases such as negation and accidental keyword matches.

- Audience Discussion

The system analyzes comment threads for the density of AI-related suspicion terms such as references to the track sounding “robotic,” “synthetic,” or generated.

Because comments are inherently noisy, this is treated as a weaker supporting signal rather than a standalone verdict.

⸻

Final Decision

The contextual signals are combined with the acoustic result using a weighted late-fusion step.

              Acoustic Evidence
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
   Copyright      Creator       Audience
    Metadata     Description     Comments
       │             │             │
       └─────────────┼─────────────┘
                     ▼
              Final AI Score

If one contextual source is unavailable, the remaining signals can still contribute to the final result.

This makes the system more flexible than relying on a single audio model and allows the final prediction to incorporate several different types of authorship evidence.

⸻

Results

In the final evaluation:

System	AI Recall	Human Specificity	Balanced Accuracy
Acoustic model only	91%	90%	90.5%
Full late-fusion system	90%	100%	95.0%

The fusion system slightly reduced AI recall while improving its ability to avoid falsely labeling human-created music as AI.

Balanced accuracy improved from 90.5% to 95.0%.

⸻

Demo

The project includes a Streamlit interface that allows users to enter a track or URL and view:

* resolved track information
* acoustic detector scores
* contextual evidence
* final classification

<img width="857" height="779" alt="Screenshot 2026-08-17 at 11 22 56 AM" src="https://github.com/user-attachments/assets/78c8467c-b397-4e81-b4f3-f631d43a5008" />

⸻

Tech Stack

Python · Streamlit · PyTorch · LAION-CLAP · MERT · scikit-learn · yt-dlp · Spotify Web API

⸻

Project Takeaway

Rather than treating AI-music detection as purely an audio-classification problem, this project approaches it as a music provenance problem.

The goal is to combine evidence from the music itself with evidence about where the track came from, how it is described, and what information exists around it to make a more informed final prediction.
