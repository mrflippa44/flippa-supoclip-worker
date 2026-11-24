# Adapted from https://github.com/cspenn/supoclip (AGPL-3.0)
# Modified for Render Free Web Service Docker deployment (512 MB RAM, dynamic $PORT).
"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """SupoClip Linux worker configuration tuned for 512 MB Render Free Tier."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=10000, validation_alias="PORT", ge=1, le=65535)
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    temp_dir: Path = Field(default=Path("/tmp/supoclip-worker"), validation_alias="TEMP_DIR")

    # Worker authentication secret (must match SUPOCLIP_WORKER_SECRET in Flippa AI)
    worker_secret: str = Field(default="", validation_alias="SUPOCLIP_WORKER_SECRET")

    # Low-memory transcription options for 512 MB containers:
    # auto (groq -> gemini -> optional faster-whisper), groq, gemini, whisper, none
    transcribe_provider: str = Field(default="auto", validation_alias="TRANSCRIBE_PROVIDER")
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    whisper_model: str = Field(default="tiny", validation_alias="WHISPER_MODEL")
    whisper_compute_type: str = Field(default="int8", validation_alias="WHISPER_COMPUTE_TYPE")
    whisper_cpu_threads: int = Field(default=1, validation_alias="WHISPER_CPU_THREADS", ge=1)

    # Lightweight face detector: opencv (default <25MB RAM), mediapipe, auto
    face_detector: str = Field(default="opencv", validation_alias="FACE_DETECTOR")

    # Video limits & single-thread low-RAM ffmpeg encoding
    max_video_duration: int = Field(default=3600, validation_alias="MAX_VIDEO_DURATION", ge=0)
    max_clips: int = Field(default=6, validation_alias="MAX_CLIPS", ge=1)
    ffmpeg_preset: str = Field(default="superfast", validation_alias="FFMPEG_PRESET")
    ffmpeg_crf: int = Field(default=24, validation_alias="FFMPEG_CRF", ge=0, le=51)
    ffmpeg_threads: int = Field(default=1, validation_alias="FFMPEG_THREADS", ge=1, le=8)

    # Gemini API (free tier supported for transcript & viral highlight scoring)
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-4-hydrogen", validation_alias="GEMINI_MODEL")

    default_min_clip_length: int = Field(default=10, validation_alias="DEFAULT_MIN_CLIP_LENGTH")
    default_max_clip_length: int = Field(default=55, validation_alias="DEFAULT_MAX_CLIP_LENGTH")

    FONTS_DIR: ClassVar[Path] = Path("fonts")

    def ensure_temp_dirs(self) -> None:
        """Create temp directory structure."""
        for subdir in ("", "uploads", "clips"):
            (self.temp_dir / subdir).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return cached worker Config instance."""
    return Config()
