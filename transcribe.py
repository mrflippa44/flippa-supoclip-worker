# Adapted from https://github.com/cspenn/supoclip (AGPL-3.0)
# Modified for Render Free Tier (512 MB RAM): replaces Apple Silicon `parakeet-mlx`
# and heavy local Whisper weights with low-memory API free transcription
# (Groq free whisper-large-v3-turbo or Gemini free audio transcription), while
# optionally supporting local faster-whisper on >=2 GB hosts and preserving
# SupoClip's exact word-level millisecond schema [{"text", "start_ms", "end_ms"}].

"""Low-memory audio transcription with word-level millisecond timestamps."""

from __future__ import annotations

import gc
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import structlog

from src.config import get_config
from src.exceptions import TranscriptionError

logger = structlog.get_logger(__name__)

_CACHE_VERSION = 3
_MAX_INLINE_AUDIO_BYTES = 18 * 1024 * 1024

try:
    from faster_whisper import WhisperModel

    WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None  # type: ignore[assignment,misc]
    WHISPER_AVAILABLE = False


def active_transcription_provider() -> str:
    """Resolve the active free transcription backend for the current runtime."""
    cfg = get_config()
    pref = (os.environ.get("TRANSCRIBE_PROVIDER") or cfg.transcribe_provider or "auto").strip().lower()
    groq_key = (os.environ.get("GROQ_API_KEY") or cfg.groq_api_key or "").strip()
    gemini_key = (os.environ.get("GEMINI_API_KEY") or cfg.gemini_api_key or "").strip()

    if pref == "none":
        return "caller-segments-only"
    if pref == "groq" and groq_key:
        return "groq-whisper-free"
    if pref == "gemini" and gemini_key:
        return "gemini-audio-free"
    if pref == "whisper" and WHISPER_AVAILABLE:
        return "faster-whisper-local"

    if groq_key:
        return "groq-whisper-free"
    if gemini_key:
        return "gemini-audio-free"
    if WHISPER_AVAILABLE:
        return "faster-whisper-local"
    return "caller-segments-only"


def _cache_path_for(video_path: str | Path) -> Path:
    p = Path(video_path)
    return p.parent / f"{p.stem}.transcript_cache.json"


def load_cached_transcript(video_path: str | Path) -> list[dict] | None:
    """Load cached transcript from .transcript_cache.json (cache v3)."""
    cache_path = _cache_path_for(video_path)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        logger.warning("transcript_cache.load_failed", path=str(cache_path), error=str(exc))
        return None

    if data.get("version") != _CACHE_VERSION:
        return None
    words = data.get("words")
    return words if isinstance(words, list) else None


def save_transcript_cache(video_path: str | Path, words: list[dict]) -> None:
    """Persist word-level timestamps beside the video."""
    cache_path = _cache_path_for(video_path)
    payload = {
        "version": _CACHE_VERSION,
        "video_path": str(Path(video_path).resolve()),
        "words": words,
    }
    try:
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception as exc:
        logger.warning("transcript_cache.save_failed", path=str(cache_path), error=str(exc))


def _has_audio_stream(video_path: Path) -> bool:
    """Return True if ffprobe detects at least one audio stream."""
    try:
        res = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            check=False,
            timeout=20,
        )
        return bool(res.stdout and res.stdout.strip())
    except Exception:
        return True


def _extract_compact_audio(video_path: Path, stem_dir: Path) -> tuple[Path, str]:
    """Extract ultra-compact mono 16kHz 32kbps audio for 512MB-safe API transcription."""
    mp3_path = stem_dir / f"{video_path.stem}.audio16k.mp3"
    res = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-threads",
            "1",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-b:a",
            "32k",
            str(mp3_path),
        ],
        capture_output=True,
        check=False,
        timeout=180,
    )
    if res.returncode == 0 and mp3_path.exists() and mp3_path.stat().st_size > 0:
        return mp3_path, "audio/mp3"

    m4a_path = stem_dir / f"{video_path.stem}.audio16k.m4a"
    res2 = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-threads",
            "1",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-b:a",
            "32k",
            str(m4a_path),
        ],
        capture_output=True,
        check=False,
        timeout=180,
    )
    if res2.returncode == 0 and m4a_path.exists() and m4a_path.stat().st_size > 0:
        return m4a_path, "audio/mp4"

    stderr = (res2.stderr or res.stderr or b"").decode(errors="replace").strip()
    raise TranscriptionError(f"Audio extraction failed: {stderr}")


def _spread_phrase_to_words(text: str, start_ms: int, end_ms: int) -> list[dict]:
    tokens = [t.strip() for t in text.split() if t.strip()]
    if not tokens:
        return []
    safe_start = max(0, start_ms)
    safe_end = max(safe_start + len(tokens) * 90, end_ms)
    step = max(80, (safe_end - safe_start) // len(tokens))
    out: list[dict] = []
    for idx, tok in enumerate(tokens):
        w_start = safe_start + idx * step
        w_end = min(safe_end, w_start + step) if idx < len(tokens) - 1 else safe_end
        out.append({"text": tok, "start_ms": w_start, "end_ms": max(w_start + 80, w_end)})
    return out


def _transcribe_with_groq(audio_path: Path, mime_type: str, api_key: str) -> list[dict]:
    """Free-tier Groq Whisper (`whisper-large-v3-turbo`) with native word timestamps."""
    with audio_path.open("rb") as fh:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data=[
                ("model", "whisper-large-v3-turbo"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
                ("timestamp_granularities[]", "segment"),
            ],
            files={"file": (audio_path.name, fh, mime_type)},
            timeout=90.0,
        )
    if resp.status_code >= 400:
        raise TranscriptionError(f"Groq transcription HTTP {resp.status_code}: {resp.text[:300]}")

    payload = resp.json()
    words: list[dict] = []
    raw_words = payload.get("words") or []
    if isinstance(raw_words, list) and raw_words:
        for w in raw_words:
            text = str(w.get("word") or "").strip()
            if not text:
                continue
            start_ms = max(0, int(round(float(w.get("start", 0.0)) * 1000)))
            end_ms = max(start_ms + 80, int(round(float(w.get("end", 0.0)) * 1000)))
            words.append({"text": text, "start_ms": start_ms, "end_ms": end_ms})
        if words:
            return words

    for seg in payload.get("segments") or []:
        seg_text = str(seg.get("text") or "").strip()
        s_ms = max(0, int(round(float(seg.get("start", 0.0)) * 1000)))
        e_ms = max(s_ms + 200, int(round(float(seg.get("end", 0.0)) * 1000)))
        words.extend(_spread_phrase_to_words(seg_text, s_ms, e_ms))
    return words


def _transcribe_with_gemini(audio_path: Path, mime_type: str, api_key: str, model: str) -> list[dict]:
    """Free-tier Gemini audio transcription returning SupoClip millisecond word cues."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    audio_bytes = audio_path.read_bytes()

    audio_part: Any
    uploaded_name: str | None = None
    try:
        if len(audio_bytes) <= _MAX_INLINE_AUDIO_BYTES:
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        else:
            uploaded = client.files.upload(file=str(audio_path), config={"mime_type": mime_type})
            uploaded_name = uploaded.name
            audio_part = uploaded

        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "cues": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "startSec": types.Schema(type=types.Type.NUMBER),
                            "endSec": types.Schema(type=types.Type.NUMBER),
                            "text": types.Schema(type=types.Type.STRING),
                        },
                        required=["startSec", "endSec", "text"],
                    ),
                )
            },
            required=["cues"],
        )

        resp = client.models.generate_content(
            model=model,
            contents=[
                audio_part,
                (
                    "Transcribe all spoken dialogue accurately into short sequential cues (3 to 7 words each). "
                    "Return exact startSec and endSec timestamps in seconds matching when each phrase is spoken. "
                    "If there is no speech, return an empty cues array []."
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction="Return strictly valid JSON matching the requested schema.",
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        raw_text = (resp.text or "{}").strip()
        parsed = json.loads(raw_text)
        words: list[dict] = []
        for cue in parsed.get("cues") or []:
            text = str(cue.get("text") or "").strip()
            if not text:
                continue
            s_ms = max(0, int(round(float(cue.get("startSec", 0.0)) * 1000)))
            e_ms = max(s_ms + 200, int(round(float(cue.get("endSec", 0.0)) * 1000)))
            words.extend(_spread_phrase_to_words(text, s_ms, e_ms))
        return words
    finally:
        del audio_bytes
        if uploaded_name:
            try:
                client.files.delete(name=uploaded_name)
            except Exception:
                pass


@lru_cache(maxsize=1)
def _get_whisper_model() -> "WhisperModel":
    if not WHISPER_AVAILABLE or WhisperModel is None:
        raise TranscriptionError("faster-whisper is not installed on this low-memory worker.")
    cfg = get_config()
    return WhisperModel(
        cfg.whisper_model,
        device="cpu",
        compute_type=cfg.whisper_compute_type,
        cpu_threads=cfg.whisper_cpu_threads,
    )


def _transcribe_with_local_whisper(audio_path: Path) -> list[dict]:
    model = _get_whisper_model()
    segments, _ = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
    )
    words: list[dict] = []
    for seg in segments:
        seg_words = getattr(seg, "words", None) or []
        if seg_words:
            for w in seg_words:
                text = str(getattr(w, "word", "") or "").strip()
                if not text:
                    continue
                s_ms = max(0, int(round(float(getattr(w, "start", 0.0)) * 1000)))
                e_ms = max(s_ms + 80, int(round(float(getattr(w, "end", 0.0)) * 1000)))
                words.append({"text": text, "start_ms": s_ms, "end_ms": e_ms})
        else:
            s_ms = max(0, int(round(float(getattr(seg, "start", 0.0)) * 1000)))
            e_ms = max(s_ms + 200, int(round(float(getattr(seg, "end", 0.0)) * 1000)))
            words.extend(_spread_phrase_to_words(str(getattr(seg, "text", "") or ""), s_ms, e_ms))
    return words


def transcribe_video(video_path: str | Path) -> list[dict]:
    """Transcribe video speech using Render 512MB-safe free providers.

    Returns SupoClip-compatible word dicts:
    [{"text": str, "start_ms": int, "end_ms": int}, ...]
    """
    p = Path(video_path)
    if not p.exists():
        raise TranscriptionError(f"Video file not found: {p}")

    cached = load_cached_transcript(p)
    if cached is not None:
        return cached

    if not _has_audio_stream(p):
        logger.info("transcribe.no_audio_stream", path=str(p))
        return []

    provider = active_transcription_provider()
    if provider == "caller-segments-only":
        logger.info("transcribe.using_caller_subtitles_only")
        return []

    cfg = get_config()
    groq_key = (os.environ.get("GROQ_API_KEY") or cfg.groq_api_key or "").strip()
    gemini_key = (os.environ.get("GEMINI_API_KEY") or cfg.gemini_api_key or "").strip()

    audio_path: Path | None = None
    try:
        audio_path, mime_type = _extract_compact_audio(p, p.parent)
        words: list[dict] = []

        if provider == "groq-whisper-free" and groq_key:
            try:
                words = _transcribe_with_groq(audio_path, mime_type, groq_key)
            except Exception as exc:
                logger.warning("transcribe.groq_fallback_to_gemini", error=str(exc))
                if gemini_key:
                    words = _transcribe_with_gemini(audio_path, mime_type, gemini_key, cfg.gemini_model)
        elif provider == "gemini-audio-free" and gemini_key:
            words = _transcribe_with_gemini(audio_path, mime_type, gemini_key, cfg.gemini_model)
        elif provider == "faster-whisper-local" and WHISPER_AVAILABLE:
            words = _transcribe_with_local_whisper(audio_path)

        logger.info("transcribe.complete", provider=provider, word_count=len(words))
        if words:
            save_transcript_cache(p, words)
        return words
    except Exception as exc:
        logger.warning("transcribe.non_fatal_fallback", provider=provider, error=str(exc))
        return []
    finally:
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)
        gc.collect()


def format_transcript_text(words: list[dict]) -> str:
    """Format word list into timestamped plain text for LLM viral selection."""
    parts: list[str] = []
    for word in words:
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        start_ms = int(word.get("start_ms", 0))
        end_ms = int(word.get("end_ms", 0))
        parts.append(f"{text} [{start_ms}-{end_ms}]")
    return " ".join(parts)
