# Flippa AI · SupoClip Linux Worker (AGPL-3.0 · Render Free Tier)

Open-source Linux video clipping worker adapted from [`cspenn/supoclip`](https://github.com/cspenn/supoclip) under **GNU Affero General Public License v3.0 (AGPL-3.0)**, tuned specifically to run inside **Render Free Web Service** limits (**512 MB RAM · 0.1 CPU · $0/mo**).

## Render 512 MB Adaptations
1. **Zero-OOM Free Speech Transcription (`transcribe.py`)**: Replaces heavy local `faster-whisper` (`ctranslate2`, ~400+ MB RAM) with lightweight API-based free transcription extracting a tiny 16 kHz 32 kbps MP3:
   - **Groq Free Whisper API** (`whisper-large-v3-turbo`, native word timestamps, `<1s`, set optional `GROQ_API_KEY`), or
   - **Gemini Free Audio API** (`google-genai`, uses your existing `GEMINI_API_KEY`), or
   - Pre-analyzed subtitle cues passed directly from Flippa AI.
   *(Local `faster-whisper` remains optionally supported on >=2 GB servers if `faster-whisper` is installed and `TRANSCRIBE_PROVIDER=whisper` is set.)*
2. **Low-Memory Face Tracking (`face_detect.py`)**: Uses OpenCV built-in Haar Cascade (`<25 MB RAM`) with multi-frame median aggregation for `9:16`, `1:1`, and `16:9` crop boxes (no heavy `mediapipe`/`matplotlib` dependency bloat).
3. **Single-Thread FFmpeg (`clip.py`)**: Runs `ffmpeg` with `-threads 1 -preset superfast -rc-lookahead 10` and `gc.collect()` cleanup between clips so peak memory stays well under 512 MB.

## Required Render Environment Variables
- `SUPOCLIP_WORKER_SECRET` *(Required)*: Shared secret matching `SUPOCLIP_WORKER_SECRET` in Flippa AI.
- `GEMINI_API_KEY` *(Recommended)*: Free Gemini API key for transcript timing & viral highlight scoring.
- `GROQ_API_KEY` *(Optional Free Alternative)*: Free Groq API key for ultra-fast `whisper-large-v3-turbo` word-level timestamps.
