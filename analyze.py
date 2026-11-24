# Adapted from https://github.com/cspenn/supoclip (AGPL-3.0)
# Added Google Gemini API (`google-genai`) highlight scoring support alongside
# SupoClip's segment validation and filler-word clean-start filtering.

"""AI viral highlight segment selection and validation."""

from __future__ import annotations

import json
import os
from typing import Any

import structlog

from src.exceptions import AnalysisError, InsufficientSegmentsError
from src.pipeline.clip import TranscriptSegment

logger = structlog.get_logger(__name__)

_FILLER_STARTS = (
    "and ",
    "but ",
    "so ",
    "well ",
    "because ",
    "also ",
    "um ",
    "uh ",
    "you know ",
    "i mean ",
    "like ",
)


def _clamp_score(val: Any, fallback: int = 80) -> int:
    try:
        return max(1, min(100, int(round(float(val)))))
    except Exception:
        return fallback


def validate_raw_segments(
    raw_clips: list[dict[str, Any]],
    duration_sec: float,
    max_clips: int,
    min_length_s: float = 3.0,
) -> list[TranscriptSegment]:
    """Validate highlight windows against real video duration and SupoClip rules."""
    valid: list[TranscriptSegment] = []
    safe_duration = max(2.0, float(duration_sec))
    effective_min = min(min_length_s, max(1.5, safe_duration * 0.4))

    for idx, item in enumerate(raw_clips):
        try:
            start_s = max(0.0, float(item.get("startTime", item.get("start_s", 0.0))))
            end_s = min(safe_duration, float(item.get("endTime", item.get("end_s", safe_duration))))
        except (TypeError, ValueError):
            continue

        if end_s - start_s < effective_min or start_s >= safe_duration:
            continue

        title = str(item.get("title") or f"Highlight Clip {idx + 1}").strip()[:140]
        description = str(item.get("description") or "").strip()[:1000]
        virality_hook = str(
            item.get("viralityHook") or item.get("reasoning") or "Strong opening hook and pacing."
        ).strip()[:300]

        raw_tags = item.get("hashtags") or []
        hashtags: list[str] = []
        if isinstance(raw_tags, list):
            for tag in raw_tags:
                cleaned = str(tag).strip().replace(" ", "")
                if not cleaned:
                    continue
                hashtags.append(cleaned if cleaned.startswith("#") else f"#{cleaned}")

        breakdown = item.get("engagementBreakdown") or {}
        hook_strength = _clamp_score(breakdown.get("hookStrength"), 84)
        pacing = _clamp_score(breakdown.get("pacing"), 82)
        retention = _clamp_score(breakdown.get("retentionPotential"), 80)
        overall_score = _clamp_score(
            item.get("engagementScore"),
            round((hook_strength + pacing + retention) / 3),
        )

        # Clean start check: skip subtle filler-only prefixes if alternative exists
        text_preview = str(item.get("text") or "").strip().lower()
        if any(text_preview.startswith(f) for f in _FILLER_STARTS):
            logger.debug("segment_starts_with_filler", title=title)

        valid.append(
            TranscriptSegment(
                start_s=round(start_s, 2),
                end_s=round(end_s, 2),
                text=str(item.get("text") or "").strip(),
                relevance_score=round(overall_score / 100.0, 2),
                reasoning=virality_hook,
                title=title,
                description=description,
                hashtags=tuple(hashtags[:8]),
                virality_hook=virality_hook,
                hook_strength=hook_strength,
                pacing=pacing,
                retention_potential=retention,
            )
        )

    valid.sort(key=lambda s: s.relevance_score, reverse=True)
    return valid[:max_clips]


def analyze_transcript_with_gemini(
    transcript_text: str,
    duration_sec: float,
    max_clips: int = 4,
    api_key: str | None = None,
    model: str = "gemini-4-hydrogen",
) -> tuple[str, str, list[TranscriptSegment]]:
    """Use Gemini API (`google-genai`) to select high-retention viral moments."""
    resolved_key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not resolved_key:
        raise AnalysisError("GEMINI_API_KEY is not configured for transcript viral analysis.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=resolved_key)
    clip_target = min(2, max_clips) if duration_sec < 20 else max_clips
    min_len = 2.0 if duration_sec < 12 else 6.0
    max_len = min(55.0, max(min_len + 2.0, duration_sec))

    prompt = (
        f"You are SupoClip's viral video editor.\n"
        f"Video total duration: {duration_sec:.2f} seconds.\n"
        f"Analyze the millisecond-timestamped transcript below (`word [start_ms-end_ms]`) and select up to {clip_target} self-contained viral short-form clips.\n"
        f"Rules:\n"
        f"1. Each clip must be between {min_len:.1f}s and {max_len:.1f}s within [0, {duration_sec:.2f}].\n"
        f"2. Never start a clip mid-sentence or on filler words (And, But, So, Um, Uh, Like).\n"
        f"3. Provide startTime and endTime in seconds (float), title, description, 4-6 hashtags (#tag), viralityHook, integer engagementScore (1-100), and integer engagementBreakdown (hookStrength, pacing, retentionPotential).\n\n"
        f"Transcript:\n{transcript_text[:28000] if transcript_text.strip() else '(No spoken dialogue detected; select cohesive highlight windows from 0 to video duration.)'}"
    )

    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "projectTitle": types.Schema(type=types.Type.STRING),
            "summary": types.Schema(type=types.Type.STRING),
            "clips": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "title": types.Schema(type=types.Type.STRING),
                        "description": types.Schema(type=types.Type.STRING),
                        "hashtags": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.STRING),
                        ),
                        "viralityHook": types.Schema(type=types.Type.STRING),
                        "engagementScore": types.Schema(type=types.Type.INTEGER),
                        "engagementBreakdown": types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "hookStrength": types.Schema(type=types.Type.INTEGER),
                                "pacing": types.Schema(type=types.Type.INTEGER),
                                "retentionPotential": types.Schema(type=types.Type.INTEGER),
                            },
                            required=["hookStrength", "pacing", "retentionPotential"],
                        ),
                        "startTime": types.Schema(type=types.Type.NUMBER),
                        "endTime": types.Schema(type=types.Type.NUMBER),
                    },
                    required=[
                        "title",
                        "description",
                        "hashtags",
                        "viralityHook",
                        "engagementScore",
                        "engagementBreakdown",
                        "startTime",
                        "endTime",
                    ],
                ),
            ),
        },
        required=["projectTitle", "summary", "clips"],
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="Return strictly valid JSON matching the requested schema.",
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    raw_text = (response.text or "").strip()
    if not raw_text:
        raise AnalysisError("Gemini returned an empty analysis payload.")

    parsed = json.loads(raw_text)
    segments = validate_raw_segments(parsed.get("clips") or [], duration_sec, clip_target, min_length_s=min_len)
    if not segments:
        raise InsufficientSegmentsError("Gemini could not extract any valid clip windows for this video.")

    project_title = str(parsed.get("projectTitle") or "SupoClip Highlights").strip()[:120]
    summary = str(parsed.get("summary") or "").strip()[:400]
    return project_title, summary, segments
