# Adapted from https://github.com/cspenn/supoclip (AGPL-3.0)
# Headless FastAPI Linux worker entrypoint tuned for Render Free Web Service
# (512 MB RAM, 0.1 vCPU, dynamic $PORT).

"""SupoClip Linux headless worker API (AGPL-3.0)."""

from __future__ import annotations

import asyncio
import gc
import hmac
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from src.config import get_config
from src.pipeline.analyze import (
    analyze_transcript_with_gemini,
    validate_raw_segments,
)
from src.pipeline.clip import (
    ClipOptions,
    TranscriptSegment,
    _probe_video_info,
    filter_words_for_segment,
    generate_clip,
)
from src.pipeline.download import download_remote_video
from src.pipeline.face_detect import is_face_tracker_available, is_mediapipe_available
from src.pipeline.subtitles import SubtitleStyle
from src.pipeline.transcribe import (
    active_transcription_provider,
    format_transcript_text,
    transcribe_video,
)

logger = structlog.get_logger(__name__)

_SAFE_JOB_ID_RE = re.compile(r"^[a-f0-9]{16,40}$")
_MAX_UPLOAD_BYTES = 150 * 1024 * 1024
_MAX_STORED_JOBS = 6

app = FastAPI(
    title="Flippa AI · SupoClip Linux Worker (Render Free Tier)",
    version="1.1.0",
    license_info={"name": "AGPL-3.0", "url": "https://www.gnu.org/licenses/agpl-3.0.html"},
)


@dataclass
class ClipArtifact:
    clip_index: int
    title: str
    description: str
    hashtags: list[str]
    virality_hook: str
    engagement_score: int
    engagement_breakdown: dict[str, int]
    start_time: float
    end_time: float
    subtitles: list[dict[str, Any]]
    file_path: Path
    file_name: str
    face_detected: bool
    crop_box: tuple[int, int, int, int]
    resolution: str
    aspect_ratio: str
    watermarked: bool


@dataclass
class WorkerJob:
    job_id: str
    status: str = "queued"
    stage: str = "Queued in SupoClip Linux worker..."
    progress_pct: int = 5
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    project_title: str = "SupoClip Highlights"
    summary: str = ""
    duration_sec: float = 0.0
    error: str | None = None
    work_dir: Path = field(default_factory=lambda: Path("/tmp/supoclip-worker"))
    clips: list[ClipArtifact] = field(default_factory=list)


_JOBS: dict[str, WorkerJob] = {}


def _verify_worker_secret(x_worker_secret: str | None) -> None:
    expected = (os.environ.get("SUPOCLIP_WORKER_SECRET") or get_config().worker_secret or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="SUPOCLIP_WORKER_SECRET is not configured on the SupoClip worker.",
        )
    provided = (x_worker_secret or "").strip()
    if not provided or not hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid or missing worker secret.")


def _evict_old_jobs() -> None:
    if len(_JOBS) < _MAX_STORED_JOBS:
        return
    oldest_ids = sorted(_JOBS.keys(), key=lambda jid: _JOBS[jid].created_at)
    while len(_JOBS) >= _MAX_STORED_JOBS and oldest_ids:
        victim_id = oldest_ids.pop(0)
        victim = _JOBS.pop(victim_id, None)
        if victim and victim.work_dir.exists():
            shutil.rmtree(victim.work_dir, ignore_errors=True)


def _words_from_caller_subtitles(raw_segments: list[dict[str, Any]] | None) -> list[dict]:
    """Synthesize SupoClip word dicts from pre-analyzed cues if worker speech API is omitted."""
    if not raw_segments:
        return []
    out: list[dict] = []
    for seg in raw_segments:
        for cue in seg.get("subtitles") or []:
            text = str(cue.get("text") or "").strip()
            if not text:
                continue
            tokens = [t for t in text.split() if t]
            if not tokens:
                continue
            start_ms = max(0, int(round(float(cue.get("startTime", 0.0)) * 1000)))
            end_ms = max(start_ms + len(tokens) * 90, int(round(float(cue.get("endTime", 0.0)) * 1000)))
            step = max(80, (end_ms - start_ms) // len(tokens))
            for idx, tok in enumerate(tokens):
                ws = start_ms + idx * step
                we = min(end_ms, ws + step) if idx < len(tokens) - 1 else end_ms
                out.append({"text": tok, "start_ms": ws, "end_ms": max(ws + 80, we)})
    out.sort(key=lambda w: w["start_ms"])
    return out


def _words_to_cue_list(
    words: list[dict],
    clip_start_s: float,
    clip_end_s: float,
    fallback_text: str,
) -> list[dict[str, Any]]:
    rel_words = filter_words_for_segment(words, clip_start_s, clip_end_s)
    if not rel_words:
        clean_fallback = fallback_text.strip()
        if not clean_fallback:
            return []
        return [
            {
                "startTime": round(clip_start_s, 2),
                "endTime": round(min(clip_end_s, clip_start_s + 4.0), 2),
                "text": clean_fallback[:120],
            }
        ]

    cues: list[dict[str, Any]] = []
    window_size = 5
    for idx in range(0, len(rel_words), window_size):
        chunk = rel_words[idx : idx + window_size]
        phrase = " ".join(str(w.get("text", "")).strip() for w in chunk).strip()
        if not phrase:
            continue
        abs_start = round(clip_start_s + (int(chunk[0]["start_ms"]) / 1000.0), 2)
        abs_end = round(
            min(clip_end_s, clip_start_s + (int(chunk[-1]["end_ms"]) / 1000.0)),
            2,
        )
        cues.append(
            {
                "startTime": abs_start,
                "endTime": max(round(abs_start + 0.25, 2), abs_end),
                "text": phrase,
            }
        )
    return cues


def _set_job_stage(job: WorkerJob, status: str, stage: str, progress_pct: int) -> None:
    job.status = status
    job.stage = stage
    job.progress_pct = progress_pct
    job.updated_at = time.time()


async def _run_job_pipeline(
    job: WorkerJob,
    source_path: Path | None,
    source_url: str | None,
    project_title_hint: str,
    aspect_ratio: str,
    resolution: str,
    watermark: bool,
    max_clips: int,
    pre_segments_raw: list[dict[str, Any]] | None,
) -> None:
    cfg = get_config()
    try:
        video_path = source_path
        if video_path is None:
            if not source_url:
                raise ValueError("Neither uploaded video nor source_url was provided.")
            _set_job_stage(job, "downloading", "Fetching remote media stream safely...", 15)
            video_path = await download_remote_video(source_url, job.work_dir, "source")

        _, _, duration_sec, _ = _probe_video_info(video_path)
        job.duration_sec = round(duration_sec, 2)

        provider_label = active_transcription_provider()
        _set_job_stage(
            job,
            "transcribing",
            f"Transcribing speech ({provider_label})...",
            35,
        )
        words = await asyncio.to_thread(transcribe_video, video_path)
        if not words and pre_segments_raw:
            words = _words_from_caller_subtitles(pre_segments_raw)
        transcript_text = format_transcript_text(words)

        _set_job_stage(
            job,
            "analyzing",
            "Selecting high-retention highlight timestamps...",
            55,
        )
        segments: list[TranscriptSegment]
        if pre_segments_raw:
            segments = validate_raw_segments(
                pre_segments_raw,
                duration_sec=duration_sec,
                max_clips=max_clips,
                min_length_s=1.5 if duration_sec < 12 else 3.0,
            )
            job.project_title = project_title_hint or "Flippa AI Project"
        else:
            gemini_key = (os.environ.get("GEMINI_API_KEY") or cfg.gemini_api_key or "").strip()
            title_out, summary_out, segments = await asyncio.to_thread(
                analyze_transcript_with_gemini,
                transcript_text,
                duration_sec,
                max_clips,
                gemini_key,
                cfg.gemini_model,
            )
            job.project_title = project_title_hint or title_out
            job.summary = summary_out

        if not segments:
            raise ValueError("No valid clip highlight segments could be generated from this video.")

        watermark_text = "Flippa AI" if watermark else None
        out_res = "1080p" if resolution == "1080p" else "720p"
        out_ratio = aspect_ratio if aspect_ratio in ("9:16", "1:1", "16:9") else "9:16"
        sub_style = SubtitleStyle()

        total_segs = len(segments)
        for idx, seg in enumerate(segments):
            stage_pct = 65 + round(((idx + 1) / max(1, total_segs)) * 30)
            _set_job_stage(
                job,
                "rendering",
                f"Face tracking & burning karaoke captions for clip {idx + 1}/{total_segs}...",
                min(96, stage_pct),
            )
            slug = re.sub(r"[^a-z0-9-_]+", "-", (seg.title or f"clip-{idx + 1}").lower()).strip("-")[:45] or f"clip-{idx + 1}"
            clip_file_name = f"{slug}-{out_ratio.replace(':', 'x')}-{out_res}.mp4"
            clip_out_path = job.work_dir / "clips" / f"clip_{idx}.mp4"

            _, face_detected, crop_box = await generate_clip(
                source_video=video_path,
                segment=seg,
                words=words,
                output_path=clip_out_path,
                options=ClipOptions(
                    aspect_ratio=out_ratio,
                    output_resolution=out_res,
                    subtitle_style=sub_style,
                    watermark_text=watermark_text,
                ),
            )

            cues = _words_to_cue_list(words, seg.start_s, seg.end_s, seg.text)
            overall_score = max(1, min(100, round(seg.relevance_score * 100)))
            job.clips.append(
                ClipArtifact(
                    clip_index=idx,
                    title=seg.title or f"Highlight Clip {idx + 1}",
                    description=seg.description or "Viral highlight moment rendered with SupoClip.",
                    hashtags=list(seg.hashtags),
                    virality_hook=seg.virality_hook or "Strong opening hook.",
                    engagement_score=overall_score,
                    engagement_breakdown={
                        "hookStrength": seg.hook_strength,
                        "pacing": seg.pacing,
                        "retentionPotential": seg.retention_potential,
                    },
                    start_time=seg.start_s,
                    end_time=seg.end_s,
                    subtitles=cues,
                    file_path=clip_out_path,
                    file_name=clip_file_name,
                    face_detected=face_detected,
                    crop_box=crop_box,
                    resolution=out_res,
                    aspect_ratio=out_ratio,
                    watermarked=bool(watermark_text),
                )
            )
            gc.collect()

        _set_job_stage(
            job,
            "completed",
            f"Generated {len(job.clips)} {out_ratio} MP4 clips via SupoClip Linux.",
            100,
        )
    except Exception as exc:
        logger.error("worker_job_failed", job_id=job.job_id, error=str(exc))
        job.error = str(exc)
        _set_job_stage(job, "failed", f"Processing failed: {exc}", 100)
    finally:
        if source_path and source_path.exists():
            source_path.unlink(missing_ok=True)
        dl_source = job.work_dir / "source.mp4"
        if dl_source.exists():
            dl_source.unlink(missing_ok=True)
        gc.collect()


@app.get("/health")
async def health() -> dict[str, Any]:
    ffmpeg_ok = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    secret_set = bool((os.environ.get("SUPOCLIP_WORKER_SECRET") or get_config().worker_secret or "").strip())
    provider = active_transcription_provider()
    face_ready = is_face_tracker_available()
    return {
        "ok": ffmpeg_ok and secret_set,
        "engine": "supoclip-agpl3-linux",
        "hostingProfile": "render-free-512mb",
        "license": "AGPL-3.0",
        "source": "https://github.com/cspenn/supoclip",
        "ffmpegReady": ffmpeg_ok,
        "whisperReady": provider != "caller-segments-only",
        "transcriptionProvider": provider,
        "mediapipeReady": face_ready,
        "faceTracker": "mediapipe" if is_mediapipe_available() else ("opencv-haar" if face_ready else "center-crop"),
        "secretConfigured": secret_set,
    }


@app.post("/jobs")
async def create_job(
    request: Request,
    x_worker_secret: str | None = Header(default=None, alias="X-Worker-Secret"),
    video: UploadFile | None = File(default=None),
    source_url: str | None = Form(default=None),
    project_title: str | None = Form(default=None),
    aspect_ratio: str | None = Form(default=None),
    resolution: str | None = Form(default=None),
    watermark: str | None = Form(default=None),
    max_clips: int | None = Form(default=None),
    segments_json: str | None = Form(default=None),
) -> dict[str, Any]:
    _verify_worker_secret(x_worker_secret)

    cfg = get_config()
    cfg.ensure_temp_dirs()
    _evict_old_jobs()

    body_json: dict[str, Any] = {}
    ctype = (request.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        body_json = await request.json()

    resolved_url = (source_url or body_json.get("source_url") or body_json.get("url") or "").strip() or None
    resolved_title = str(project_title or body_json.get("project_title") or "").strip()[:140]
    resolved_ratio = str(aspect_ratio or body_json.get("aspect_ratio") or "9:16")
    resolved_res = str(resolution or body_json.get("resolution") or "1080p")
    wm_raw = watermark if watermark is not None else body_json.get("watermark", False)
    resolved_watermark = str(wm_raw).lower() in ("1", "true", "yes")
    resolved_max_clips = max(1, min(6, int(max_clips or body_json.get("max_clips") or 4)))

    pre_segments_raw: list[dict[str, Any]] | None = None
    raw_seg_payload = segments_json or body_json.get("segments")
    if raw_seg_payload:
        pre_segments_raw = (
            json.loads(raw_seg_payload) if isinstance(raw_seg_payload, str) else list(raw_seg_payload)
        )

    job_id = uuid.uuid4().hex
    work_dir = cfg.temp_dir / f"job_{job_id}"
    (work_dir / "clips").mkdir(parents=True, exist_ok=True)

    saved_upload_path: Path | None = None
    if video is not None:
        ext = Path(video.filename or "upload.mp4").suffix.lower()
        if ext not in (".mp4", ".mov", ".webm", ".m4v"):
            ext = ".mp4"
        saved_upload_path = work_dir / f"upload{ext}"
        written = 0
        with saved_upload_path.open("wb") as fh:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    shutil.rmtree(work_dir, ignore_errors=True)
                    raise HTTPException(status_code=413, detail="Uploaded file exceeds 150 MB Render free limit.")
                fh.write(chunk)
        if not resolved_title and video.filename:
            resolved_title = Path(video.filename).stem[:120]

    if saved_upload_path is None and resolved_url is None:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Provide an uploaded video or HTTPS source_url.")

    job = WorkerJob(
        job_id=job_id,
        work_dir=work_dir,
        project_title=resolved_title or "SupoClip Project",
    )
    _JOBS[job_id] = job

    asyncio.create_task(
        _run_job_pipeline(
            job=job,
            source_path=saved_upload_path,
            source_url=resolved_url,
            project_title_hint=resolved_title,
            aspect_ratio=resolved_ratio,
            resolution=resolved_res,
            watermark=resolved_watermark,
            max_clips=resolved_max_clips,
            pre_segments_raw=pre_segments_raw,
        )
    )

    return {
        "jobId": job_id,
        "status": job.status,
        "stage": job.stage,
        "progressPct": job.progress_pct,
    }


@app.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    x_worker_secret: str | None = Header(default=None, alias="X-Worker-Secret"),
) -> dict[str, Any]:
    _verify_worker_secret(x_worker_secret)
    if not _SAFE_JOB_ID_RE.match(job_id) or job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="Worker job not found.")

    job = _JOBS[job_id]
    return {
        "jobId": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "progressPct": job.progress_pct,
        "projectTitle": job.project_title,
        "summary": job.summary,
        "durationSec": job.duration_sec,
        "error": job.error,
        "clips": [
            {
                "clipIndex": c.clip_index,
                "title": c.title,
                "description": c.description,
                "hashtags": c.hashtags,
                "viralityHook": c.virality_hook,
                "engagementScore": c.engagement_score,
                "engagementBreakdown": c.engagement_breakdown,
                "startTime": c.start_time,
                "endTime": c.end_time,
                "subtitles": c.subtitles,
                "fileName": c.file_name,
                "faceDetected": c.face_detected,
                "cropBox": list(c.crop_box),
                "resolution": c.resolution,
                "aspectRatio": c.aspect_ratio,
                "watermarked": c.watermarked,
            }
            for c in job.clips
        ],
    }


@app.get("/jobs/{job_id}/clips/{clip_index}")
async def download_job_clip(
    job_id: str,
    clip_index: int,
    x_worker_secret: str | None = Header(default=None, alias="X-Worker-Secret"),
) -> FileResponse:
    _verify_worker_secret(x_worker_secret)
    if not _SAFE_JOB_ID_RE.match(job_id) or job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="Worker job not found.")
    job = _JOBS[job_id]
    if clip_index < 0 or clip_index >= len(job.clips):
        raise HTTPException(status_code=404, detail="Clip index not found.")

    artifact = job.clips[clip_index]
    if not artifact.file_path.exists():
        raise HTTPException(status_code=404, detail="Rendered clip file was evicted or missing.")

    return FileResponse(
        path=str(artifact.file_path),
        media_type="video/mp4",
        filename=artifact.file_name,
    )


if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(app, host=cfg.host, port=cfg.port)
