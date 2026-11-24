# Modifications Notice (AGPL-3.0 Section 5)

This directory contains a modified Linux worker derivative of **SupoClip**
(`https://github.com/cspenn/supoclip`), licensed under the
**GNU Affero General Public License Version 3 (19 November 2007)**.

## Summary of Modifications for Flippa AI & Render Free Tier (512 MB RAM)

Date: 2026-11-24

1. **Removed Apple Silicon (`parakeet-mlx`) dependency** (`src/pipeline/transcribe.py`, `requirements.txt`):
   Replaced macOS MLX audio transcription with a low-memory free transcription cascade suited for 512 MB Linux containers: Groq free-tier `whisper-large-v3-turbo` (`GROQ_API_KEY`, native word timestamps), Gemini free audio transcription (`GEMINI_API_KEY`), and optional local `faster-whisper` on >=2 GB hosts, preserving SupoClip's exact word-level millisecond cache schema (`[{"text", "start_ms", "end_ms"}]`).
2. **Low-Memory Face Tracking (`src/pipeline/face_detect.py`)**:
   Replaced heavy MediaPipe/TFLite runtime dependencies with built-in OpenCV Haar Cascade face tracking (`FACE_DETECTOR=opencv`, optional `mediapipe` when installed) while preserving SupoClip's multi-frame median center aggregation and `calculate_crop_box` (`9:16`, `1:1`, `16:9`).
3. **Single-Thread Render FFmpeg Pipeline (`src/pipeline/clip.py`, `Dockerfile`, `render.yaml`)**:
   Tuned `ffmpeg` (`-threads 1`, `superfast`, bounded lookahead) and added `render.yaml` + dynamic `${PORT:-10000}` startup for Render Free Web Services (`512 MB RAM`, `$0/month`), alongside optional Free-tier watermark burn-in and audio-safe mapping.
4. **Headless HTTP Worker API (`app.py`)**:
   Replaced NiceGUI desktop UI with an authenticated FastAPI service (`/health`, `/jobs`, `/jobs/{job_id}`, `/jobs/{job_id}/clips/{clip_index}`) using constant-time `X-Worker-Secret` verification (`SUPOCLIP_WORKER_SECRET`) and SSRF-guarded streaming downloads (`src/pipeline/download.py`).
5. **Subtitle Multi-Aspect Support (`src/pipeline/subtitles.py`)**:
   Preserved SupoClip's 6-word karaoke ASS generator with dynamic `PlayResX`/`PlayResY` scaling across `9:16`, `1:1`, and `16:9`.
