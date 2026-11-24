# Adapted from https://github.com/cspenn/supoclip (AGPL-3.0)
# Added strict SSRF validation across redirects and direct HTTPS MP4/MOV/WebM ingestion
# alongside SupoClip's yt-dlp downloader options.

"""SSRF-guarded remote video download utilities."""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
import structlog

from src.exceptions import DownloadError

logger = structlog.get_logger(__name__)

_MAX_DOWNLOAD_BYTES = 250 * 1024 * 1024
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".m4v")


def _is_blocked_ip(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
        return (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
        )
    except ValueError:
        return True


def assert_safe_public_https_url(raw_url: str) -> str:
    """Validate that a remote URL is clean HTTPS pointing only to public IPs."""
    parsed = urlparse(raw_url.strip())
    if parsed.scheme.lower() != "https" or parsed.username or parsed.password:
        raise DownloadError("Only clean HTTPS URLs without embedded credentials are allowed.")
    if parsed.port not in (None, 443):
        raise DownloadError("Custom URL ports are not permitted.")

    host = (parsed.hostname or "").lower()
    if (
        not host
        or host == "localhost"
        or host.endswith(".local")
        or host.endswith(".internal")
        or host.endswith(".localhost")
        or host.endswith(".arpa")
    ):
        raise DownloadError("Private or local hostnames are blocked.")

    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise DownloadError(f"Could not resolve hostname '{host}': {exc}") from exc

    ips = {info[4][0] for info in infos if info and info[4]}
    if not ips or any(_is_blocked_ip(ip) for ip in ips):
        raise DownloadError("Target URL resolved to a restricted network address.")
    return parsed.geturl()


async def download_remote_video(source_url: str, output_dir: Path, stem: str) -> Path:
    """Download an accessible direct HTTPS MP4/MOV/WebM video safely."""
    current_url = assert_safe_public_https_url(source_url)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_path = output_dir / f"{stem}.mp4"

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=15.0),
        follow_redirects=False,
        headers={
            "User-Agent": "FlippaSupoClipWorker/1.0",
            "Accept": "video/*,application/mp4,application/octet-stream;q=0.9,*/*;q=0.1",
        },
    ) as client:
        response: httpx.Response | None = None
        for _ in range(5):
            assert_safe_public_https_url(current_url)
            resp = await client.send(client.build_request("GET", current_url), stream=True)
            if 300 <= resp.status_code < 400 and resp.headers.get("location"):
                next_loc = resp.headers["location"]
                await resp.aclose()
                current_url = urljoin(current_url, next_loc)
                continue
            response = resp
            break

        if response is None:
            raise DownloadError("Too many redirects while downloading video URL.")

        try:
            if response.status_code >= 400:
                raise DownloadError(f"Remote media host returned HTTP {response.status_code}.")

            ctype = (response.headers.get("content-type") or "").lower()
            path_lower = urlparse(current_url).path.lower()
            has_ext = any(path_lower.endswith(ext) for ext in _VIDEO_EXTENSIONS)
            is_video = ctype.startswith("video/") or "application/mp4" in ctype
            is_octet = "application/octet-stream" in ctype or ctype == ""

            if not is_video and not (is_octet and has_ext):
                mime_label = ctype.split(";")[0] or "unknown"
                raise DownloadError(
                    f"URL returned '{mime_label}' instead of a direct MP4/MOV/WebM stream."
                )

            written = 0
            with dest_path.open("wb") as fh:
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    written += len(chunk)
                    if written > _MAX_DOWNLOAD_BYTES:
                        raise DownloadError("Remote video exceeded 250 MB worker import limit.")
                    fh.write(chunk)
        finally:
            await response.aclose()

    if not dest_path.exists() or dest_path.stat().st_size == 0:
        raise DownloadError("Downloaded file was empty.")
    return dest_path
