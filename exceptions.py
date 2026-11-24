# Adapted from https://github.com/cspenn/supoclip (AGPL-3.0)
"""Custom exceptions for SupoClip clipping pipeline."""


class SupoClipError(Exception):
    """Base class for every error raised by SupoClip."""


class DownloadError(SupoClipError):
    """Raised when video acquisition (URL fetch / yt-dlp / upload) fails."""


class TranscriptionError(SupoClipError):
    """Raised when audio transcription fails."""


class AnalysisError(SupoClipError):
    """Raised when transcript highlight analysis fails."""


class InsufficientSegmentsError(AnalysisError):
    """Raised when analysis yields fewer than one usable clip segment."""


class ClipGenerationError(SupoClipError):
    """Raised when ffmpeg clip generation fails."""


class ConfigurationError(SupoClipError):
    """Raised when required configuration is missing or invalid."""
