"""Renders Reeldesk videos to mp4 and puts them back on the video.

    python render_video.py --video <id>   # one video (a Render button, a move to Ready for preview, a manual run)
    python render_video.py --queued       # every video still queued (the scheduled sweep)

For each video: download its transcript, screenshots and saved narration audio; render every format
the video asks for with assemble.py using only that saved audio (never a TTS call); upload each mp4
as a video_files row (kind "video"); record the render state and post a note. The connection and the
render user's sign-in come from the environment (see README). Logs name a video only by its id and
never print narration, URLs or credentials, because a public repository's Actions logs are public.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from assemble import narration_audio_hash
from reeldesk import Connection, Reeldesk, ReeldeskError, check_id, now_iso

HERE = Path(__file__).resolve().parent
ASSEMBLE = HERE / "assemble.py"

# The video's `format` field (Reeldesk's New shot dialog) → the renders it needs.
FORMATS_FOR = {"shorts": ["shorts"], "youtube": ["youtube"], "both": ["shorts", "youtube"]}
DEFAULT_FORMATS = ["youtube"]
# The sweep leaves a freshly queued video to the dispatch that queued it; it only picks up one that
# has waited this long (a dispatch that never arrived or failed to start).
SWEEP_GRACE = timedelta(minutes=10)
# A "rendering" claim older than this is treated as abandoned (a cancelled or crashed run).
STALE_CLAIM = timedelta(minutes=90)

ENV = {
    "base_url": "REELDESK_APP_URL", "org_code": "REELDESK_ORG_CODE",
    "api_key": "REELDESK_API_KEY", "client_id": "REELDESK_CLIENT_ID",
}
ENV_EMAIL, ENV_PASSWORD = "REELDESK_RENDER_EMAIL", "REELDESK_RENDER_PASSWORD"


class RenderError(Exception):
    """A render that cannot go ahead, with a message safe for public logs and the video's note."""


# ── what to render ───────────────────────────────────────────────────────────────────────────────

def formats_for(video: dict) -> list[str]:
    return FORMATS_FOR.get(video.get("format") or "", DEFAULT_FORMATS)


def narration_holders(transcript: list[dict]):
    """(beat number, holder) for every unit that speaks: each highlight with narration, or the beat
    itself when it has no highlights. Excluded beats are skipped, as assemble.py skips them."""
    for number, beat in enumerate(transcript, start=1):
        if beat.get("excluded"):
            continue
        holders = beat.get("highlights") or [beat]
        for holder in holders:
            if (holder.get("narration") or "").strip():
                yield number, holder


def upload_identity_name(narration: str, suffix: str) -> str:
    """The name assemble.py looks for an uploaded take under: it depends only on the text."""
    return f"audio-{narration_audio_hash('upload', 'manual', 'manual', narration)}{suffix}"


def place_saved_audio(transcript: list[dict], audio: dict[str, Path], project: Path) -> list[int]:
    """Puts each line's saved take where assemble.py finds it without knowing the editor's voice
    settings; returns the beat numbers that have a line with no saved take."""
    missing: list[int] = []
    for number, holder in narration_holders(transcript):
        text = holder["narration"]
        chosen = audio.get(holder.get("audio") or "")
        uploaded = next((p for name, p in audio.items() if name == upload_identity_name(text, p.suffix)), None)
        source = chosen or uploaded
        if not source:
            missing.append(number)
            continue
        target = project / upload_identity_name(text, source.suffix)
        if source != target:
            shutil.copyfile(source, target)
    return sorted(set(missing))


# ── the steps ────────────────────────────────────────────────────────────────────────────────────

def download_project(rd: Reeldesk, video_id: str, project: Path) -> tuple[list[dict], dict[str, Path]]:
    """Transcript and screenshots into the project folder, saved audio into project/saved-audio."""
    files = rd.list_files(video_id)
    transcripts = [f for f in files if f.get("kind") == "transcript"]
    if not transcripts:
        raise RenderError("The video has no transcript yet.")
    transcript_path = rd.download(transcripts[0], project / "transcript.json")
    for shot in (f for f in files if f.get("kind") == "shot"):
        rd.download(shot, project / Path(shot["name"]).name)
    audio = {Path(a["name"]).name: rd.download(a, project / "saved-audio" / Path(a["name"]).name)
             for a in files if a.get("kind") == "audio"}
    print(f"  downloaded {len(files)} files")
    return json.loads(transcript_path.read_text()), audio


def output_path(project: Path, fmt: str) -> Path:
    return project / f"{project.name}{'' if fmt == 'shorts' else '-' + fmt}.mp4"


def render_formats(project: Path, formats: list[str]) -> list[tuple[str, Path]]:
    outputs = []
    for fmt in formats:
        result = subprocess.run([sys.executable, str(ASSEMBLE), str(project), "--engine", "saved", "--format", fmt],
                                capture_output=True, text=True)
        if result.returncode != 0:
            # assemble.py's own messages name a beat or a tool, never narration text.
            last = (result.stderr or result.stdout).strip().splitlines()[-1:] or ["no output"]
            raise RenderError(f"Rendering {fmt} failed: {last[0][:300]}")
        outputs.append((fmt, output_path(project, fmt)))
        print(f"  rendered {fmt}")
    return outputs


def publish(rd: Reeldesk, video: dict, outputs: list[tuple[str, Path]]) -> str:
    """Uploads each mp4 and points its video_files row at it (one row per format, reused on a
    re-render). Returns the content id the video's video_content_id should point at."""
    video_id = video["id"]
    context = video.get("shots_context") or video_id
    existing = {f.get("name"): f for f in rd.list_files(video_id) if f.get("kind") == "video"}
    content_ids = {}
    for fmt, path in outputs:
        name = f"{video_id}-{fmt}.mp4"
        presign = rd.upload(path, name, "video/mp4", context)
        row = {"video_id": video_id, "content_id": presign["contentId"], "name": name,
               "content_type": "video/mp4", "cdn_url": presign.get("cdnUrl") or "", "kind": "video"}
        old = existing.get(name)
        if old:
            rd.update_file(old["id"], row)
            rd.delete_content(old["content_id"])
        else:
            rd.add_file(row)
        content_ids[fmt] = presign["contentId"]
        print(f"  uploaded {fmt}")
    return content_ids.get("youtube") or next(iter(content_ids.values()))


def claim(rd: Reeldesk, video: dict, force: bool) -> bool:
    """Marks the video as rendering, unless another run is already on it."""
    started = parse_time(video.get("render_started_at"))
    busy = video.get("render_status") == "rendering" and started and now() - started < STALE_CLAIM
    if busy and not force:
        print("  already rendering in another run: skipped")
        return False
    rd.update_video(video["id"], {"render_status": "rendering", "render_started_at": now_iso(), "error_message": "",
                                  "render_job_id": os.environ.get("GITHUB_RUN_ID", "local")})
    return True


def finish(rd: Reeldesk, video_id: str, primary_content_id: str, formats: list[str]) -> None:
    rd.update_video(video_id, {"render_status": "done", "rendered_at": now_iso(), "video_content_id": primary_content_id})
    rd.add_note(video_id, f"Rendered {' and '.join(formats)} with the saved narration.")


def fail(rd: Reeldesk, video_id: str, message: str) -> None:
    rd.update_video(video_id, {"render_status": "failed", "error_message": message})
    rd.add_note(video_id, f"Render failed: {message}")


# ── coordinators ─────────────────────────────────────────────────────────────────────────────────

def render_video(rd: Reeldesk, video_id: str, force: bool = False) -> bool:
    print(f"Video {video_id}:")
    video = rd.get_video(check_id(video_id))
    if not claim(rd, video, force):
        return True
    try:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / video_id
            project.mkdir()
            transcript, audio = download_project(rd, video_id, project)
            missing = place_saved_audio(transcript, audio, project)
            if missing:
                raise RenderError(f"No saved narration audio for beat(s) {', '.join(map(str, missing))}: "
                                  "preview and save those lines in the editor, then render again.")
            formats = formats_for(video)
            primary = publish(rd, video, render_formats(project, formats))
        finish(rd, video_id, primary, formats)
        print("  done")
        return True
    except (RenderError, ReeldeskError) as err:
        print(f"  failed: {err}")
        fail(rd, video_id, str(err))
        return False


def queued_video_ids(rd: Reeldesk) -> list[str]:
    """Videos still queued after the grace period, oldest request first."""
    cutoff = now() - SWEEP_GRACE
    queued = [v for v in rd.list_videos(render_status="queued")
              if (parse_time(v.get("render_requested_at")) or cutoff) <= cutoff]
    return [v["id"] for v in sorted(queued, key=lambda v: v.get("render_requested_at") or "")]


def now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


def connect_from_env() -> Reeldesk:
    missing = [name for name in [*ENV.values(), ENV_EMAIL, ENV_PASSWORD] if not os.environ.get(name)]
    if missing:
        sys.exit(f"Missing settings: {', '.join(missing)} (see README).")
    rd = Reeldesk(Connection(**{key: os.environ[name] for key, name in ENV.items()}))
    rd.sign_in(os.environ[ENV_EMAIL], os.environ[ENV_PASSWORD])
    return rd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--video", help="Render this video id.")
    target.add_argument("--queued", action="store_true", help="Render every video still queued.")
    parser.add_argument("--force", action="store_true", help="Render even if another run has claimed the video.")
    args = parser.parse_args()

    rd = connect_from_env()
    ids = [args.video] if args.video else queued_video_ids(rd)
    if not ids:
        print("Nothing queued.")
        return
    results = [render_video(rd, video_id, args.force) for video_id in ids]
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
