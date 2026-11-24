# Adapted from https://github.com/cspenn/supoclip (AGPL-3.0)
# Extended for Render 512 MB Free Tier with single-thread bounded libx264 flags,
# 9:16/1:1/16:9 aspect ratios, multi-frame face tracking, optional Free-tier
# watermark burn-in, and explicit memory cleanup between renders.

"""FFmpeg video clip cutting, face-tracked cropping, subtitle burn-in, and MP4 encoding."""

from __future__ import annotations

import asyncio
import gc
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import structlog

from src.config import get_config
from src.exceptions import ClipGenerationError
from src.pipeline.face_detect import (
    calculate_crop_box,
    detect_face_center_multi,
)
from src.pipeline.subtitles import SubtitleStyle, write_ass_file

log = structlog.get_logger(__name__)

RESOLUTIONS: dict[tuple[str, str], tuple[int, int]] = {
    ("9:16", "720p"): (720, 1280),
    ("9:16", "1080p"): (1080, 1920),
    ("1:1", "720p"): (720, 720),
    ("1:1", "1080p"): (1080, 1080),
    ("16:9", "720p"): (1280, 720),
    ("16:9", "1080p"): (1920, 1080),
}


@dataclass(slots=True)
class TranscriptSegment:
    """Selected highlight segment."""

    start_s: float
    end_s: float
    text: str = ""
    relevance_score: float = 0.85
    reasoning: str = ""
    title: str = ""
    description: str = ""
    hashtags: tuple[str, ...] = ()
    virality_hook: str = ""
    hook_strength: int = 84
    pacing: int = 82
    retention_potential: int = 83


@dataclass(slots=True)
class ClipOptions:
    """Clip rendering parameters."""

    aspect_ratio: str = "9:16"
    output_resolution: str = "1080p"
    subtitle_style: SubtitleStyle | None = None
    watermark_text: str | None = None


def _escape_filter_path(path: str | Path) -> str:
    text = str(path)
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("[", "\\[").replace("]", "\\]")
    text = text.replace(",", "\\,").replace(";", "\\;")
    return text


def filter_words_for_segment(words: list[dict], start_s: float, end_s: float) -> list[dict]:
    """Return words inside [start_s, end_s] shifted relative to start_s."""
    start_ms = int(start_s * 1000)
    end_ms = int(end_s * 1000)
    filtered: list[dict] = []
    for word in words:
        w_start = int(word["start_ms"])
        w_end = int(word["end_ms"])
        midpoint = (w_start + w_end) // 2
        if start_ms <= midpoint <= end_ms:
            adj = dict(word)
            adj["start_ms"] = max(0, w_start - start_ms)
            adj["end_ms"] = max(adj["start_ms"] + 80, w_end - start_ms)
            filtered.append(adj)
    return filtered


def _resolve_watermark_font() -> str:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            return candidate
    return ""


def _build_main_video_filter(
    crop_box: tuple[int, int, int, int],
    out_width: int,
    out_height: int,
    ass_path: Path | None,
    fonts_dir: Path | None,
    watermark_text: str | None,
) -> str:
    x, y, crop_w, crop_h = crop_box
    parts: list[str] = [
        f"crop={crop_w}:{crop_h}:{x}:{y}",
        f"scale={out_width}:{out_height}",
    ]
    if ass_path is not None:
        ass_filter = f"ass={_escape_filter_path(ass_path)}"
        if fonts_dir is not None:
            ass_filter += f":fontsdir={_escape_filter_path(fonts_dir)}"
        parts.append(ass_filter)

    if watermark_text:
        safe_text = watermark_text.replace("'", "").replace(":", "").strip()[:32]
        if safe_text:
            font_path = _resolve_watermark_font()
            font_clause = f"fontfile={_escape_filter_path(font_path)}:" if font_path else ""
            parts.append(
                f"drawtext={font_clause}text='{safe_text}':fontcolor=white@0.8:fontsize=24:x=w-tw-24:y=24:box=1:boxcolor=black@0.4:boxborderw=8"
            )
    return ",".join(parts)


def _probe_video_info(video_path: Path) -> tuple[int, int, float, bool]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, check=False, timeout=30)
    if res.returncode != 0:
        raise ClipGenerationError(f"ffprobe failed on {video_path}")
    data = json.loads(res.stdout or b"{}")
    streams = data.get("streams") or []
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v_stream:
        raise ClipGenerationError("Source file does not contain a video stream.")
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    width = int(v_stream.get("width") or 1280)
    height = int(v_stream.get("height") or 720)
    duration = float(data.get("format", {}).get("duration") or v_stream.get("duration") or 15.0)
    return width, height, max(1.0, duration), a_stream is not None


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(cmd, capture_output=True, check=False, timeout=240)
    except OSError as exc:
        raise ClipGenerationError(f"ffmpeg could not be executed: {exc}") from exc


def _find_fonts_dir() -> Path | None:
    candidate = Path(__file__).parent
    for _ in range(5):
        fonts = candidate / "fonts"
        if fonts.is_dir():
            return fonts
        candidate = candidate.parent
    return None


async def generate_clip(
    source_video: str | Path,
    segment: TranscriptSegment,
    words: list[dict],
    output_path: str | Path,
    options: ClipOptions | None = None,
) -> tuple[Path, bool, tuple[int, int, int, int]]:
    """Generate a single face-tracked, subtitled MP4 clip within 512 MB RAM."""
    opts = options or ClipOptions()
    cfg = get_config()
    source = Path(source_video)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ratio = opts.aspect_ratio if opts.aspect_ratio in ("9:16", "1:1", "16:9") else "9:16"
    res_key = opts.output_resolution if opts.output_resolution in ("720p", "1080p") else "1080p"
    out_width, out_height = RESOLUTIONS[(ratio, res_key)]

    frame_w, frame_h, total_dur, has_audio = _probe_video_info(source)
    start_s = max(0.0, min(total_dur - 1.0, float(segment.start_s)))
    end_s = max(start_s + 1.0, min(total_dur, float(segment.end_s)))
    duration_s = max(1.0, end_s - start_s)

    face_center = detect_face_center_multi(source, start_s, end_s, samples=5)
    crop_box = calculate_crop_box(
        frame_width=frame_w,
        frame_height=frame_h,
        face_center=face_center,
        target_width=out_width,
        target_height=out_height,
    )

    ass_path: Path | None = None
    if opts.subtitle_style is not None and words:
        segment_words = filter_words_for_segment(words, start_s, end_s)
        if segment_words:
            styled = SubtitleStyle(
                font_family=opts.subtitle_style.font_family,
                font_size=max(30, round(out_height * 0.032)),
                font_color=opts.subtitle_style.font_color,
                secondary_color=opts.subtitle_style.secondary_color,
                outline_color=opts.subtitle_style.outline_color,
                outline_width=opts.subtitle_style.outline_width,
                shadow_depth=opts.subtitle_style.shadow_depth,
                position_y_pct=opts.subtitle_style.position_y_pct,
                video_width=out_width,
                video_height=out_height,
                uppercase=opts.subtitle_style.uppercase,
                bold=opts.subtitle_style.bold,
            )
            ass_path = out.parent / f"{out.stem}.ass"
            write_ass_file(segment_words, ass_path, style=styled)

    fonts_dir = _find_fonts_dir()
    vf_chain = _build_main_video_filter(
        crop_box=crop_box,
        out_width=out_width,
        out_height=out_height,
        ass_path=ass_path,
        fonts_dir=fonts_dir,
        watermark_text=opts.watermark_text,
    )

    preset = cfg.ffmpeg_preset if cfg.ffmpeg_preset in ("ultrafast", "superfast", "veryfast") else "superfast"
    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-threads",
        str(cfg.ffmpeg_threads),
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration_s:.3f}",
        "-vf",
        vf_chain,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(cfg.ffmpeg_crf),
        "-rc-lookahead",
        "10",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level",
        "4.0",
        "-max_muxing_queue_size",
        "512",
    ]
    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k", "-ac", "2"])
    else:
        cmd.append("-an")
    cmd.extend(["-movflags", "+faststart", str(out)])

    try:
        result = await asyncio.to_thread(_run_ffmpeg, cmd)
        if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            stderr = (result.stderr or b"").decode(errors="replace").strip()
            raise ClipGenerationError(f"ffmpeg failed (exit {result.returncode}): {stderr}")
        return out, face_center is not None, crop_box
    finally:
        if ass_path is not None:
            ass_path.unlink(missing_ok=True)
        gc.collect()
