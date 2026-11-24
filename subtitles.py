# Adapted from https://github.com/cspenn/supoclip (AGPL-3.0)
# Extended PlayResX calculation for 9:16, 1:1, and 16:9 aspect ratios.

"""Karaoke ASS subtitle generation via pysubs2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pysubs2
import structlog

logger = structlog.get_logger(__name__)

_WINDOW_SIZE: int = 6
_MIN_WORD_DURATION_MS: int = 40


@dataclass(slots=True)
class SubtitleStyle:
    """Configuration for burned-in ASS karaoke subtitles."""

    font_family: str = "DejaVu Sans"
    font_size: int = 56
    font_color: str = "#FFFFFF"
    secondary_color: str = "#B4B4C2"
    outline_color: str = "#09090E"
    outline_width: float = 3.0
    shadow_depth: float = 1.5
    position_y_pct: int = 82
    video_width: int = 1080
    video_height: int = 1920
    uppercase: bool = False
    bold: bool = True


def hex_to_bgr_color(hex_color: str) -> pysubs2.Color:
    cleaned = hex_color.lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Expected #RRGGBB format, got: {hex_color!r}")
    r = int(cleaned[0:2], 16)
    g = int(cleaned[2:4], 16)
    b = int(cleaned[4:6], 16)
    return pysubs2.Color(r, g, b, 0)


def calculate_margin_v(position_y_pct: int, video_height: int) -> int:
    pct_from_bottom = max(5, min(90, 100 - position_y_pct))
    return int(video_height * pct_from_bottom / 100)


def _build_style(style: SubtitleStyle) -> pysubs2.SSAStyle:
    ssa_style = pysubs2.SSAStyle()
    ssa_style.fontname = style.font_family
    ssa_style.fontsize = style.font_size
    ssa_style.primarycolor = hex_to_bgr_color(style.font_color)
    ssa_style.outlinecolor = hex_to_bgr_color(style.outline_color)
    ssa_style.backcolor = hex_to_bgr_color(style.outline_color)
    ssa_style.outline = style.outline_width
    ssa_style.shadow = style.shadow_depth
    ssa_style.bold = style.bold
    ssa_style.alignment = pysubs2.Alignment.BOTTOM_CENTER
    ssa_style.marginv = calculate_margin_v(style.position_y_pct, style.video_height)
    ssa_style.marginl = int(style.video_width * 0.06)
    ssa_style.marginr = int(style.video_width * 0.06)
    return ssa_style


def _escape_ass_text(text: str) -> str:
    text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _hex_to_ass_override(hex_color: str) -> str:
    color = hex_to_bgr_color(hex_color)
    return f"&H{color.b:02X}{color.g:02X}{color.r:02X}&"


def _collect_valid_words(words: list[dict]) -> tuple[list[dict], int]:
    valid: list[dict] = []
    skipped = 0
    for word_data in words:
        duration_ms = int(word_data["end_ms"]) - int(word_data["start_ms"])
        if duration_ms < _MIN_WORD_DURATION_MS:
            skipped += 1
            continue
        valid.append(word_data)
    return valid, skipped


def _render_word(text: str, uppercase: bool) -> str:
    if uppercase:
        text = text.upper()
    return _escape_ass_text(text)


def _build_window_text(
    window: list[dict],
    active_index: int,
    style: SubtitleStyle,
    secondary_override: str,
) -> str:
    parts: list[str] = []
    for idx, word_data in enumerate(window):
        rendered = _render_word(str(word_data["text"]), style.uppercase)
        if idx == active_index:
            parts.append(rendered)
        else:
            parts.append(f"{{\\c{secondary_override}}}{rendered}{{\\r}}")
    return " ".join(parts)


def _emit_window_events(
    window: list[dict],
    style: SubtitleStyle,
    secondary_override: str,
) -> list[pysubs2.SSAEvent]:
    events: list[pysubs2.SSAEvent] = []
    for active_index, word_data in enumerate(window):
        event = pysubs2.SSAEvent()
        event.start = int(word_data["start_ms"])
        event.end = int(word_data["end_ms"])
        event.text = _build_window_text(window, active_index, style, secondary_override)
        events.append(event)
    return events


def generate_ass_subtitles(words: list[dict], style: SubtitleStyle | None = None) -> str:
    """Generate ASS subtitle file content with SupoClip 6-word karaoke phrase windows."""
    resolved = style or SubtitleStyle()
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = str(resolved.video_width)
    subs.info["PlayResY"] = str(resolved.video_height)
    subs.styles["Default"] = _build_style(resolved)
    valid_words, skipped = _collect_valid_words(words)
    secondary_override = _hex_to_ass_override(resolved.secondary_color)
    for start in range(0, len(valid_words), _WINDOW_SIZE):
        window = valid_words[start : start + _WINDOW_SIZE]
        subs.events.extend(_emit_window_events(window, resolved, secondary_override))
    logger.info("generated_ass_subtitles", total_words=len(words), events=len(subs.events), skipped=skipped)
    return subs.to_string("ass")


def write_ass_file(
    words: list[dict],
    output_path: str | Path,
    style: SubtitleStyle | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ass_content = generate_ass_subtitles(words, style)
    path.write_text(ass_content, encoding="utf-8")
    return path
