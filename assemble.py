#!/usr/bin/env python3
"""
Assemble a project folder's captured beats (<project_dir>/transcript.json + that same folder's own
screenshot files) into a narrated vertical (1080x1920) short/reel video, or a landscape (1920x1080)
one.

This is a standalone, self-contained tool — the whole `render/` folder is meant to be handed to a
video editor by itself (zip it up, or just copy the folder), independent of the rest of whatever
project it came from. It has no dependencies outside this folder and pip's numpy/requests/Pillow.
See README.md for the full setup + usage walkthrough.

Pipeline per beat:
  1. Synthesize narration audio — Gemini (`gemini-3.1-flash-tts-preview`, the default), OpenAI's
     TTS API (`tts-1-hd`), ElevenLabs, or macOS's built-in `say` (offline, zero-setup, noticeably
     more robotic) — and measure its duration with ffprobe. That's how long the beat holds; it has
     to be known before step 2, since planning a beat's visual timeline needs the total to split.
  2. Plan the beat's visual timeline (plan_beat_segments): usually one segment — the UI screenshot
     cropped to the neighborhood of its highlight(s) (viewport-relative bounding boxes, of the
     element(s) the narration is about — the same shape editor.html writes into transcript.json) at
     the video frame's own aspect ratio, scaled to fill the frame exactly (full-bleed, no
     letterboxing — a landscape desktop screenshot fit whole into a portrait frame would otherwise
     leave large flat gaps above/below; cropping to the relevant region first avoids that entirely).
     Beats with no highlight get a plain center crop. Two cases split it into more than one segment
     instead: highlights spread across different areas each get their own crop and dwell time, with
     a hard cut between them (group_highlights); and a highlight too wide/tall for any static crop
     to contain becomes a pan — the crop slides across it over the segment's duration instead of
     losing content off either edge (compute_focus_pan). Segments are rendered silently and
     concatenated before narration is muxed on — segment count is invisible past build_beat_clip.
     Done in Pillow; the caption (if enabled) is a *separate* static image vstacked back on per
     segment, not baked into the cropped/zoomed frame, since Homebrew's standard ffmpeg build
     doesn't compile in freetype/fontconfig support for `drawtext` anyway.
  3. Concatenate every beat's clip into <project_dir>/<project-folder-name>.mp4.

Fails soft: if `say`/ffmpeg/Pillow is missing or a step errors, stops and reports exactly which one
broke. The screenshots + transcript.json are already a usable deliverable on their own (hand off to
any video editor) even if this script never runs.

Usage:
    python3 assemble.py <project_dir> [--format shorts|youtube] [--captions] [--fps 30]   # saved audio only (default)
    python3 assemble.py <project_dir> --engine gemini|openai|elevenlabs|macos [--voice Kore]  # generate narration
    python3 assemble.py <project_dir> --no-audio   # silent preview, no TTS call, no key needed

<project_dir> is any folder containing transcript.json and the screenshot files it references,
flat, side by side — e.g. one downloaded from the Reeldesk board's "Download shots + transcript"
action.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Output format presets. WIDTH/HEIGHT/CAPTION_AREA_HEIGHT/CAPTION_FONT_SIZE below are the *active*
# format's values — main() overwrites them from FORMATS[args.format] before any rendering happens.
# A global reassignment rather than threading a format object through every function is deliberate:
# this is a single CLI invocation picking one format for its whole run, not a library serving both
# formats concurrently, so the extra parameter on every call site would be pure ceremony.
FORMATS = {
    # Portrait, shorts/reels (TikTok/YouTube Shorts/Instagram Reels) — a landscape UI screenshot is
    # always width-constrained here, so the highlight-driven crop (see compute_focus_crop) is doing
    # real work: picking which part of the screen fills the frame.
    "shorts": {"width": 1080, "height": 1920, "caption_fraction": 320 / 1920},
    # Landscape, standard YouTube — close enough to a typical desktop screenshot's own aspect that
    # the crop only needs to trim modestly, not aggressively push in the way shorts does.
    "youtube": {"width": 1920, "height": 1080, "caption_fraction": 0.12},
}
# YouTube only treats a 9:16 upload as a *Short* (Shorts shelf/feed placement) up to this length;
# past it, the same file still uploads fine but as an ordinary video. assemble() warns rather than
# blocks, since a longer shorts-format cut is still a legitimate thing to want (e.g. handing off to
# TikTok/Reels, which have their own separate limits) — this is specifically a YouTube quirk.
SHORTS_MAX_SECONDS = 3 * 60

# Basic xfade/acrossfade transition styles offered between beats — off ("none") by default, since a
# hard cut is the more common choice for this kind of UI-driven tutorial content and re-encoding
# with transitions costs real time; ffmpeg supports many more xfade names than this, these five just
# cover the common cases without the list becoming a research project.
TRANSITIONS = ["none", "fade", "dissolve", "wipeleft", "wiperight", "slideup"]

WIDTH, HEIGHT = FORMATS["shorts"]["width"], FORMATS["shorts"]["height"]
CAPTION_AREA_HEIGHT = round(HEIGHT * FORMATS["shorts"]["caption_fraction"])
# Also set by main() alongside WIDTH/HEIGHT — namespaces cached frame/clip filenames so rendering
# youtube after shorts for the same feature can't silently reuse the other format's wrongly-shaped
# cropped frames (confirmed this would otherwise collide: both used the same beat-NN-content.png).
CURRENT_FORMAT = "shorts"
FRAME_MARGIN = 40
BG_COLOR = (13, 17, 23)          # dark slate — neutral, doesn't fight screenshot content
ACCENT_COLOR = (39, 171, 57)     # nostackai brand green — thin rule between screenshot and caption
CAPTION_COLOR = (255, 255, 255)
CAPTION_FONT_SIZE = 46
CAPTION_LINE_SPACING = 14
# First one found wins — macOS is the primary target (matches the `say` fallback engine and the
# Homebrew ffmpeg this whole tool assumes), but a couple of common Linux/Windows bold-sans paths
# are included too so handing this folder to an editor on a different OS doesn't hard-fail on
# --captions specifically; _load_caption_font() falls back to PIL's built-in font if none exist.
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",              # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",           # Linux (Debian/Ubuntu, common)
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",   # Linux (RHEL/Fedora, common)
    "C:\\Windows\\Fonts\\arialbd.ttf",                                # Windows
]

# The old design fit the *whole* screenshot into the frame and relied on zoompan to crop toward a
# highlight afterward — a landscape screenshot (e.g. 1600x1000) fit into a portrait frame is always
# width-constrained, so most of the frame sat empty (confirmed live: read as "a lot of black
# space," and papering over it with a blurred backdrop was rejected too — a short should be full
# real content, not filler). The fix is to crop to the highlight's own neighborhood *before*
# composing, at the frame's exact aspect ratio, so the very first frame is already full-bleed —
# no letterbox, no blur, ever. FOCUS_FILL_FACTOR controls how tightly that crop hugs the highlight
# (lower = more surrounding context); MIN_CROP_WIDTH_FRACTION floors how narrow the crop can get
# (as a fraction of the screenshot's own width) so a tiny element (e.g. a nav icon) doesn't force
# an extreme upscale of a handful of source pixels into blur.
FOCUS_FILL_FACTOR = 0.85
MIN_CROP_WIDTH_FRACTION = 0.40

# The crop frames the right neighborhood, but doesn't by itself tell a viewer which element in
# that neighborhood the narration is actually about — draw a translucent rounded box around the
# highlight's own position in the final frame so it reads as "look here," the way a real screen-
# recording tool's cursor spotlight would. Global defaults for now; a per-beat `highlight_style`
# override would be a natural extension if a single style stops fitting every beat.
HIGHLIGHT_OVERLAY_COLOR = (39, 171, 57)     # nostackai brand green
HIGHLIGHT_OVERLAY_FILL_ALPHA = 40           # out of 255 — a faint wash, not a solid block
HIGHLIGHT_OVERLAY_BORDER_ALPHA = 235
HIGHLIGHT_OVERLAY_BORDER_FRACTION = 0.006   # border width, as a fraction of min(canvas dimensions)
HIGHLIGHT_OVERLAY_RADIUS_FRACTION = 0.018   # corner radius, same basis
HIGHLIGHT_OVERLAY_PADDING = 0.08            # extra room around the raw highlight, as a fraction of its own size

# Redaction fill — deliberately the same near-black used elsewhere for a masked field, not pure
# #000, so it doesn't read as a video/decoding glitch.
REDACTION_FILL_COLOR = (20, 20, 20)

# Minimum average per-pixel content-energy (see content_energy_map) a crop needs to count as "has
# real UI in it, not a blank panel" — below this, the crop search kicks in to nudge toward content
# instead of staying centered on empty space. Picked empirically against real console-app
# screenshots (flat panel backgrounds score well under this; any text/icon-bearing region clears
# it easily) — not a universal constant, revisit if a very different, sparser UI starts tripping it.
BLANK_CROP_THRESHOLD = 6.0

# Since composing now starts already-cropped to the highlight's neighborhood, the zoom animation on
# top is just a small amount of motion/energy, not the thing doing the framing — one constant works
# for every beat. It pushes in toward the highlight (see hold_zoom_focus), not the frame's center, so
# a highlight near an edge stays fully in frame instead of being trimmed off by a centered zoom.
PUSH_ZOOM_TARGET = 1.15

# The focus ring is not there when a shot first appears: it fades in this many seconds after the
# shot loads (capped at half the beat, see ring_reveal_seconds, so it still shows on a short beat)
# to pull the eye to the highlight. The push-in zoom starts at that same moment, so the motion and
# the ring land together.
RING_REVEAL_DELAY_SECONDS = 2.0
RING_FADE_SECONDS = 0.25

# A timed highlight's (see is_timed_highlight) own default fade-in/fade-out, used whenever its
# `fadeIn`/`fadeOut` field is left unset. Mirrors editor.html's own highlightOpacityAt defaults
# exactly (0 = appear instantly, 0.3s soft fade-out) — the two must stay in sync or the editor's
# own preview would visibly lie about the real render's pacing.
DEFAULT_TIMED_FADE_IN_SECONDS = 0.0
DEFAULT_TIMED_FADE_OUT_SECONDS = 0.3

# ffmpeg's `fade` filter needs a non-zero duration to mean anything — a `fadeIn` of 0 (appear
# instantly, the default) still runs through the same filter, just with this floor instead of 0,
# so the overlay is still correctly gated invisible before its own start time.
MIN_FFMPEG_FADE_SECONDS = 0.05

# A timed highlight with no explicit `duration` means "stays on screen" (see
# resolve_highlight_timing) — once it has faded in, it needs to actually be visible for a moment
# before its segment can safely end, not fade in and immediately get cut off by a segment sized
# only for the group's own (possibly nonexistent) spoken narration. See
# required_timed_highlight_duration.
MIN_TIMED_VISIBLE_SECONDS = 1.0

# "Comfortable" pan speed, as a fraction of the panning axis's own *starting* window dimension
# traveled per second — resolution/crop-size independent by construction (see
# cap_pan_for_duration). ~0.14 means ~7s to cross one full window-width; a highlight overflowing
# by more than that in the time its own narration takes just won't be fully revealed by the end —
# better a slow, readable partial reveal than a fast, illegible complete one. A highlight's own
# `panSpeed` (resolve_group_pan_speed) is a multiplier on this (1.0 = unscaled).
DEFAULT_PAN_SPEED_FRACTION = 0.14

# How much a pan segment's crop window grows (zooms OUT) from its first frame to its last — the
# Ken Burns half of the effect, so a viewer reads "camera pulling back while sliding across,"
# rather than a flat lateral slide at a fixed size. 1.0 would mean no growth at all.
PAN_ZOOM_OUT_FACTOR = 1.3

# A short narration line (e.g. a "pause and look" beat with no click) can produce only ~1.5-2s of
# audio — too quick for the push-in to read as a deliberate look, more like a blip, and too quick
# to actually register what's being highlighted before the beat moves on. Floor every beat's
# on-screen duration here; the audio track itself is padded with silence to match. Longer beats
# are fine (explicitly confirmed) — err toward giving the zoomed-in interaction time to be read.
MIN_BEAT_DURATION = 4.5

# How much bigger a shared union crop is allowed to be than the largest individual highlight's own
# solo crop before group_highlights refuses to merge them — two highlights can be far enough apart
# that their union still technically fits within the screenshot (no hard overflow) but only by
# zooming out so far neither highlight reads clearly on screen; this catches that "fits but useless"
# case, distinct from the hard overflow compute_focus_pan exists for.
MERGE_MAX_GROWTH = 1.6

# --no-audio preview mode has no real narration audio to measure, so a beat's on-screen duration is
# estimated from its word count at a comfortable narration pace instead. Deliberately a bit slower
# than average conversational speech (~165wpm) since the real TTS pass tends to read calmly, not
# rushed — better to slightly overestimate here than have the preview's pacing feel faster than the
# real thing will end up being.
PREVIEW_WORDS_PER_MINUTE = 140

# How long the push-in itself takes — after this, zoom holds steady (see zoompan_filter) so the
# frame is fully readable while the narration explains, not still drifting the whole time.
ZOOM_RAMP_SECONDS = 0.6

# Render the already-cropped content canvas at this multiple of the final output size, then let
# ffmpeg's zoompan crop+downscale it back to real size for its mild push-in. The resolved crop is
# frequently a small fraction of the original screenshot (a tight highlight), so composing
# straight at final resolution then having zoompan crop a further sliver off *that* would stack two
# lossy resamples on top of an already-small source region. Working at 2x keeps real detail in
# reserve for zoompan's own crop, so its final resize is a downscale (which sharpens) rather than
# an upscale (which blurs).
CONTENT_SUPERSAMPLE = 2

DEFAULT_MACOS_VOICE = "Evan (Enhanced)"
DEFAULT_OPENAI_VOICE = "onyx"   # calm, deep — matches a measured "guide" narration tone
OPENAI_TTS_MODEL = "tts-1-hd"
DEFAULT_ELEVENLABS_VOICE = "TX3LPaxmHKxFdv7VOQHJ"
ELEVENLABS_TTS_MODEL = "eleven_multilingual_v2"
DEFAULT_GEMINI_VOICE = "Orus"   # one of ~30 prebuilt voices (see the Gemini API speech-generation docs); same as the editor's default, so its previews sound like the render
# On a free-tier key, every Gemini TTS model (3.1 flash, 2.5 flash, 2.5 pro) is capped at
# GenerateRequestsPerDayPerProjectPerModel = 10/day or less — confirmed exhausted live on two
# models in one session, with no retry/backoff able to recover it within the same day. A paid-tier
# key removes that wall, which is what GEMINI_API_KEY in testing/.env now holds.
#
# The model decides both how a request must be shaped and whether inline pause tags work, so each
# supported model is described here (mirrored by GEMINI_TTS_MODELS in the reels editor, so the
# editor's preview and this render speak a line the same way). gemini-3.8-flash-tts speaks its
# input verbatim — send the narration as-is (a "Say ...:" instruction would be read aloud) — and
# honors inline `<short pause>` / `<long pause>` tags (measured live: about 1.6s / 2.6s of silence,
# never spoken aloud). The older 3.1 preview needs the "Say ...:" instruction or it answers the text
# instead of speaking it (400 "Model tried to generate text"), and its pause tags are inconsistent.
GEMINI_TTS_MODELS = {
    "gemini-3.8-flash-tts": {"prompt_prefix": ""},
    "gemini-3.1-flash-tts-preview": {"prompt_prefix": "Say in a natural, conversational tone: "},
}
DEFAULT_GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"   # 3.8 supports pause tags but adds a hiss to the end of every line
GEMINI_TTS_MODEL = DEFAULT_GEMINI_TTS_MODEL   # main() overwrites this from --gemini-model
# Retries still matter on a paid key for a genuine transient rate-limit or network blip — just not
# as a way to wait out a daily quota anymore.
GEMINI_MIN_SECONDS_BETWEEN_CALLS = 5.0
GEMINI_MAX_RETRIES = 5


def load_env_var(name: str) -> str | None:
    # This folder's own .env, not a parent project's — see README.md's setup section. Copy
    # .env.example to .env here and fill in whichever engine's key you're using.
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip() or None
    return None


def require_tool(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        sys.exit(f"Missing required tool: {name}\n  Install it with: {install_hint}")


def run(cmd: list[str], step: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"Failed at step: {step}\ncommand: {' '.join(cmd)}\nstderr:\n{result.stderr}")


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"ffprobe failed on {path}:\n{result.stderr}")
    return float(result.stdout.strip())


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        candidate = f"{current} {w}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] > max_width and current:
            lines.append(current)
            current = w
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def content_energy_map(shot: Image.Image) -> np.ndarray:
    """A cheap per-pixel "is there actually something here" score: edge magnitude on grayscale.
    Text, icons, borders, and buttons all produce strong edges; a flat white/gray background panel
    scores near zero. Used to steer the focus crop away from empty space instead of just centering
    blindly on the highlight, which too often let the crop's "free" side land on a blank header or
    footer strip with no content in it at all."""
    edges = shot.convert("L").filter(ImageFilter.FIND_EDGES)
    return np.asarray(edges, dtype=np.float64)


def _integral_image(energy: np.ndarray) -> np.ndarray:
    # Padded by one row/col of zeros so window_sum below never needs an "is this the first
    # row/col" special case.
    return np.pad(energy, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)


def _window_sum(integral: np.ndarray, x0: int, y0: int, w: int, h: int) -> float:
    x1, y1 = x0 + w, y0 + h
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def _best_content_offset(
    integral: np.ndarray, x_lo: float, x_hi: float, y_lo: float, y_hi: float,
    crop_w: int, crop_h: int, center_x: float, center_y: float,
) -> tuple[int, int]:
    """Prefer centered on the highlight — that's the geometrically correct framing and usually
    fine. Only deviate when centered actually lands on close-to-blank space (checked via average
    content-energy per pixel, so it's comparable across different crop sizes), and even then move
    to the *closest* position that clears the "has content" bar rather than the single densest spot
    in the whole allowed range. An earlier version maximized total content unconditionally and it
    overshot badly — it dragged a crop centered on a small target (e.g. one row in a palette list)
    toward a denser, unrelated part of the screen instead of just trimming a blank margin, so the
    intended highlight ended up off-center for no benefit."""
    cx = round(min(max(center_x, x_lo), x_hi))
    cy = round(min(max(center_y, y_lo), y_hi))
    area = crop_w * crop_h
    centered_avg = _window_sum(integral, cx, cy, crop_w, crop_h) / area
    if centered_avg >= BLANK_CROP_THRESHOLD or (x_hi <= x_lo and y_hi <= y_lo):
        return cx, cy

    xs = [round(v) for v in np.linspace(x_lo, x_hi, 9)] if x_hi > x_lo else [cx]
    ys = [round(v) for v in np.linspace(y_lo, y_hi, 9)] if y_hi > y_lo else [cy]
    candidates = sorted(
        {(x, y) for y in ys for x in xs},
        key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2,
    )
    best_score, best = centered_avg, (cx, cy)
    for x, y in candidates:
        avg = _window_sum(integral, x, y, crop_w, crop_h) / area
        if avg >= BLANK_CROP_THRESHOLD:
            return x, y
        if avg > best_score:
            best_score, best = avg, (x, y)
    return best


def _crop_geometry(
    shot_size: tuple[int, int], target_aspect: float, highlight: dict | None,
) -> tuple[int, int, float, float, float, float]:
    """The pure sizing/bounds math behind compute_focus_crop, minus the content-aware nudge —
    cheap enough to call repeatedly for tentative-union merge tests (see group_highlights) without
    paying for content_energy_map's edge detection + integral image on every candidate. Returns
    (crop_w_i, crop_h_i, x_lo, x_hi, y_lo, y_hi): the crop's clamped integer size, and the [lo,hi]
    range its origin could occupy while still fully containing the raw highlight. x_lo > x_hi (or
    y_lo > y_hi) means no single origin can contain it — the highlight/union is wider (or taller)
    than any crop at this aspect can be within the screenshot's own bounds; see compute_focus_crop
    (centers and accepts the overflow) and compute_focus_pan (pans across it instead)."""
    shot_width, shot_height = shot_size

    if highlight:
        raw_x, raw_y = highlight["x"], highlight["y"]
        raw_w, raw_h = max(highlight["width"], 1.0), max(highlight["height"], 1.0)

        # Pad the highlight's shorter dimension out to target_aspect first (centered, so no real
        # content is lost) — otherwise a wide/short or tall/narrow highlight would force the crop
        # into a shape that doesn't match the frame, and *something* would still have to give.
        pad_w, pad_h = raw_w, raw_h
        if pad_w / pad_h > target_aspect:
            pad_h = pad_w / target_aspect
        else:
            pad_w = pad_h * target_aspect

        crop_w = pad_w / FOCUS_FILL_FACTOR
        crop_h = crop_w / target_aspect

        min_crop_w = shot_width * MIN_CROP_WIDTH_FRACTION
        if crop_w < min_crop_w:
            crop_w = min_crop_w
            crop_h = crop_w / target_aspect
    else:
        raw_x, raw_y, raw_w, raw_h = shot_width / 2, shot_height / 2, 0.0, 0.0
        crop_w, crop_h = shot_width, shot_width / target_aspect
        if crop_h > shot_height:
            crop_h, crop_w = shot_height, shot_height * target_aspect

    # Never crop bigger than the screenshot itself — shrink proportionally (aspect preserved) if
    # the ideal crop would overflow either dimension.
    overflow = max(crop_w / shot_width, crop_h / shot_height, 1.0)
    crop_w, crop_h = crop_w / overflow, crop_h / overflow
    crop_w_i, crop_h_i = min(shot_width, round(crop_w)), min(shot_height, round(crop_h))

    if highlight:
        # The crop must still fully contain the *raw* highlight bbox (not the aspect-padded one —
        # that padding was only to pick a shape, never a hard requirement) — this bounds how far
        # the content search in compute_focus_crop is allowed to slide it.
        x_lo, x_hi = max(0.0, raw_x + raw_w - crop_w_i), min(shot_width - crop_w_i, raw_x)
        y_lo, y_hi = max(0.0, raw_y + raw_h - crop_h_i), min(shot_height - crop_h_i, raw_y)
    else:
        x_lo = x_hi = (shot_width - crop_w_i) / 2
        y_lo = y_hi = (shot_height - crop_h_i) / 2

    return crop_w_i, crop_h_i, x_lo, x_hi, y_lo, y_hi


def compute_focus_crop(
    shot: Image.Image, target_aspect: float, highlight: dict | None,
) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) in screenshot pixel space: the region to crop before composing,
    sized to exactly target_aspect so the composed frame is full-bleed with no letterboxing. When
    there's a highlight, the crop is sized around it and then, within whatever slack remains after
    still fully containing it, nudged toward the screenshot's real content (see
    content_energy_map) rather than just centered — centering alone regularly left the crop's
    free side sitting on a blank background strip with nothing in it. No highlight: a plain center
    crop of the whole screenshot."""
    shot_width, shot_height = shot.size
    crop_w_i, crop_h_i, x_lo, x_hi, y_lo, y_hi = _crop_geometry(shot.size, target_aspect, highlight)

    if not highlight:
        x0 = round((shot_width - crop_w_i) / 2)
        y0 = round((shot_height - crop_h_i) / 2)
        return x0, y0, x0 + crop_w_i, y0 + crop_h_i

    raw_x, raw_y = highlight["x"], highlight["y"]
    raw_w, raw_h = max(highlight["width"], 1.0), max(highlight["height"], 1.0)

    if x_lo > x_hi or y_lo > y_hi:
        # crop_w_i/crop_h_i ended up smaller than the raw highlight itself (only possible via the
        # overflow clamp above on a highlight close to the screenshot's own full size) — fall back
        # to plain centering, clamped to the image, rather than an empty range. A single-highlight
        # beat hitting this branch instead gets panned (see compute_focus_pan / plan_beat_segments)
        # so this fallback is now mainly a defensive floor, not the everyday behavior it once was.
        hx, hy = raw_x + raw_w / 2, raw_y + raw_h / 2
        x0 = min(max(hx - crop_w_i / 2, 0), shot_width - crop_w_i)
        y0 = min(max(hy - crop_h_i / 2, 0), shot_height - crop_h_i)
    else:
        integral = _integral_image(content_energy_map(shot))
        center_x, center_y = raw_x + raw_w / 2 - crop_w_i / 2, raw_y + raw_h / 2 - crop_h_i / 2
        x0, y0 = _best_content_offset(integral, x_lo, x_hi, y_lo, y_hi, crop_w_i, crop_h_i, center_x, center_y)

    return x0, y0, x0 + crop_w_i, y0 + crop_h_i


def compute_focus_pan(
    shot_size: tuple[int, int], target_aspect: float, highlight: dict,
    fixed_size: tuple[int, int] | None = None,
) -> tuple[str, int, int, float, float, float] | None:
    """When a highlight is too wide (or tall) for its crop window to contain within the
    screenshot's bounds, return a pan spec instead of silently losing the overflow off one edge.
    Returns None when the window already contains it fully (no pan needed).

    `fixed_size`, if given, is a hand-set crop's own (width, height) — even an explicit crop can
    still be too narrow/short for its own single highlight (e.g. promoted from an auto default
    before the highlight was widened, or just not noticed when set), and a saved crop deserves the
    same "don't silently clip" treatment the auto-computed one gets. When omitted, the window size
    is derived the normal way, via _crop_geometry (the auto-crop's own overflow detection).

    Returns (axis, crop_w_i, crop_h_i, start, end, fixed_origin): axis is "x" or "y" — the one that
    slides; crop_w_i/crop_h_i is the sliding window's fixed size; start/end are that axis's origin
    at the beginning/end of the pan; fixed_origin is the *other* axis's single, non-animated origin,
    centered on the highlight and clamped to the screenshot exactly as compute_focus_crop's own
    fallback centers it.

    No new geometry is needed for start/end beyond the window size itself — x_hi ("the largest
    origin whose window still shows the highlight's leading edge") and x_lo ("the smallest origin
    whose window still reaches its trailing edge") are the same bounds _crop_geometry already
    computes for the auto-crop case; the pan is just sliding from one to the other, revealing the
    highlight's leading edge first and its trailing edge last (natural reading order: left-to-right
    typically wins for a horizontal overflow since raw_x is usually less than raw_x+raw_w-crop_w,
    and top-to-bottom for a vertical one)."""
    shot_width, shot_height = shot_size
    raw_x, raw_y = highlight["x"], highlight["y"]
    raw_w, raw_h = max(highlight["width"], 1.0), max(highlight["height"], 1.0)

    if fixed_size is not None:
        crop_w_i, crop_h_i = fixed_size
        x_lo, x_hi = max(0.0, raw_x + raw_w - crop_w_i), min(shot_width - crop_w_i, raw_x)
        y_lo, y_hi = max(0.0, raw_y + raw_h - crop_h_i), min(shot_height - crop_h_i, raw_y)
    else:
        crop_w_i, crop_h_i, x_lo, x_hi, y_lo, y_hi = _crop_geometry(shot_size, target_aspect, highlight)
    x_overflow, y_overflow = x_lo > x_hi, y_lo > y_hi
    if not x_overflow and not y_overflow:
        return None

    hx, hy = raw_x + raw_w / 2, raw_y + raw_h / 2

    # Both axes overflowing at once is rare (needs a highlight close to the screenshot's own full
    # size in both dimensions) — pan whichever axis overflows more, clamp-center the other.
    if x_overflow and (not y_overflow or raw_w / crop_w_i >= raw_h / crop_h_i):
        fixed_origin = min(max(hy - crop_h_i / 2, 0), shot_height - crop_h_i)
        return "x", crop_w_i, crop_h_i, x_hi, x_lo, fixed_origin
    fixed_origin = min(max(hx - crop_w_i / 2, 0), shot_width - crop_w_i)
    return "y", crop_w_i, crop_h_i, y_hi, y_lo, fixed_origin


def cap_pan_for_duration(pan: tuple, duration: float, speed_fraction: float) -> tuple:
    """A segment's duration comes from however long its own narration takes to speak — often far
    too short to comfortably read a pan at its *natural* full distance (compute_focus_pan's
    start/end cover the highlight's entire overflow, with no regard for how long that takes to
    watch). Shortens `end` (never touches `start`, so the pan still opens on the highlight's
    leading edge exactly as before) to whatever a comfortable reading speed covers in the time
    available — it's fine if the trailing edge never fully arrives on screen. Direction-agnostic
    since compute_focus_pan doesn't guarantee start < end."""
    axis, crop_w_i, crop_h_i, start, end, fixed_origin = pan
    crop_dim = crop_w_i if axis == "x" else crop_h_i
    max_distance = speed_fraction * crop_dim * duration
    capped_delta = max(-max_distance, min(max_distance, end - start))
    return axis, crop_w_i, crop_h_i, start, start + capped_delta, fixed_origin


def resolve_highlight_crop(highlight: dict, fmt: str) -> dict | None:
    """A highlight's own optional crop override — same per-format-keyed shape as the beat-level
    `crop` (resolve_crop), but scoped to one highlight so a beat whose highlights sit in different
    areas of the screen can give each its own custom framing instead of being stuck with one crop
    for the whole beat. Two highlights sharing the *same* explicit crop (see _crops_fit_together)
    is how someone deliberately keeps them together in one shot despite being far enough apart that
    they wouldn't auto-merge on geometry alone."""
    crop = highlight.get("crop")
    return crop.get(fmt) if crop else None


def _crops_fit_together(
    shot_size: tuple[int, int], target_aspect: float, members: list[dict],
) -> bool:
    """The merge test group_highlights uses to decide whether several highlights can share one
    crop. If any member has an explicit crop (resolve_highlight_crop) — someone took manual control
    of its framing — every member must share that exact same crop to merge; a highlight with no
    explicit crop never merges into (or out of) a manually-framed group, since mixing an auto crop
    and a hand-set one into one segment wouldn't mean anything coherent. Otherwise (no member has an
    explicit crop), fall back to the geometry tests: hard — the union's _crop_geometry doesn't hit
    the overflow condition (a single crop containing all of them is possible at all); soft — the
    union crop isn't drastically bigger than the largest individual member's own solo crop
    (MERGE_MAX_GROWTH), catching the case where a union *technically* fits but only by zooming out
    so far none of the highlights read clearly.

    `independent` (set by the editor's Detach button, or by hand-adjusting a solo highlight's crop —
    see nostackai-reels' editor.html) short-circuits all of that: a highlight marked this way never
    merges with anything, in any format, regardless of geometry or matching crops. Without this
    check here, a highlight the editor keeps deliberately separate in every format (because it
    carries this flag) could still get silently merged back into a shared crop at render time the
    moment some format's geometry happens to allow it — this mirrors editor.html's cropsFitTogether()
    exactly so a beat renders the same grouping the editor showed while it was authored."""
    if any(h.get("independent") for h in members):
        return False
    explicit = [resolve_highlight_crop(h, CURRENT_FORMAT) for h in members]
    if any(c is not None for c in explicit):
        first = explicit[0]
        return first is not None and all(c == first for c in explicit)

    union = union_rect(members)
    crop_w_i, crop_h_i, x_lo, x_hi, y_lo, y_hi = _crop_geometry(shot_size, target_aspect, union)
    if x_lo > x_hi or y_lo > y_hi:
        return False
    solo_geoms = [_crop_geometry(shot_size, target_aspect, m) for m in members]
    max_solo_w = max(g[0] for g in solo_geoms)
    max_solo_h = max(g[1] for g in solo_geoms)
    return crop_w_i <= max_solo_w * MERGE_MAX_GROWTH and crop_h_i <= max_solo_h * MERGE_MAX_GROWTH


def group_highlights(
    shot_size: tuple[int, int], target_aspect: float, highlights: list[dict],
) -> list[list[dict]]:
    """Cluster a beat's highlights into groups that can share one crop, via repeated pairwise
    merging — not left-to-right greedy merging in array order, which is order-dependent and gets
    non-monotonic layouts wrong (e.g. [top-left, bottom-right, top-left-again] would greedily end up
    as 3 singleton groups even though the two top-left ones could obviously share a crop, because
    the greedy walk never reconsiders top-left once it's moved past it). Groups are returned ordered
    by the minimum original array index among their members, preserving the beat author's narration
    order regardless of merge order."""
    groups: list[list[tuple[int, dict]]] = [[(i, h)] for i, h in enumerate(highlights)]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                members = [h for _, h in groups[i]] + [h for _, h in groups[j]]
                if _crops_fit_together(shot_size, target_aspect, members):
                    groups[i] = groups[i] + groups[j]
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    groups.sort(key=lambda g: min(idx for idx, _ in g))
    return [[h for _, h in g] for g in groups]


def apply_redactions(shot: Image.Image, redactions: list[dict] | None) -> Image.Image:
    """Black out each {x,y,width,height} region (screenshot pixel space, same coordinate system
    as `highlights`) before anything else touches the image — baked into the in-memory copy, never
    the source screenshot file on disk, so a redaction is just data (editable in transcript.json /
    editor.html) instead of a one-way destructive edit to the recorded asset."""
    if not redactions:
        return shot
    shot = shot.copy()
    draw = ImageDraw.Draw(shot)
    for r in redactions:
        draw.rectangle([r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]], fill=REDACTION_FILL_COLOR)
    return shot


def _ring_box(
    rect: tuple[float, float, float, float], size: tuple[int, int], border: int,
) -> list[float] | None:
    """The padded ring box around `rect`, kept fully inside the frame: a highlight touching (or
    near) the frame's left/right/top/bottom edge has no room for the padding, and drawing the box
    past the edge is what cut the ring off there. Pulls the box in by half the border width so the
    whole stroke stays visible. None if nothing of the box is left inside the frame."""
    x0, y0, x1, y1 = rect
    pad_x, pad_y = (x1 - x0) * HIGHLIGHT_OVERLAY_PADDING, (y1 - y0) * HIGHLIGHT_OVERLAY_PADDING
    half = border / 2
    bx0, by0 = max(x0 - pad_x, half), max(y0 - pad_y, half)
    bx1, by1 = min(x1 + pad_x, size[0] - half), min(y1 + pad_y, size[1] - half)
    return [bx0, by0, bx1, by1] if bx1 > bx0 and by1 > by0 else None


def highlight_ring_layer(
    size: tuple[int, int], rects: list[tuple[float, float, float, float]],
) -> Image.Image:
    """A transparent RGBA layer holding one translucent rounded box per rect, each around a
    highlighted element's position *within the already-cropped, already-resized frame* — the crop
    alone frames the right neighborhood, but doesn't tell a viewer exactly which element the
    narration means. Each rect is (x0,y0,x1,y1) in the frame's own pixel space (see the callers for
    the crop -> frame coordinate mapping). Kept as its own layer, not painted into the shot, so the
    render can fade it in after the shot has been on screen a moment (see RING_REVEAL_DELAY_SECONDS)."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    border = max(1, round(min(size) * HIGHLIGHT_OVERLAY_BORDER_FRACTION))
    radius = min(size) * HIGHLIGHT_OVERLAY_RADIUS_FRACTION
    for rect in rects:
        box = _ring_box(rect, size, border)
        if box is None:
            continue
        ring = Image.new("RGBA", size, (0, 0, 0, 0))
        ImageDraw.Draw(ring).rounded_rectangle(
            box, radius=radius,
            fill=HIGHLIGHT_OVERLAY_COLOR + (HIGHLIGHT_OVERLAY_FILL_ALPHA,),
            outline=HIGHLIGHT_OVERLAY_COLOR + (HIGHLIGHT_OVERLAY_BORDER_ALPHA,),
            width=border,
        )
        layer = Image.alpha_composite(layer, ring)
    return layer


def ring_layer_path(content_path: Path) -> Path:
    """Where a segment's ring layer lives, next to its content PNG — derived, not passed around."""
    return content_path.with_name(f"{content_path.stem}-ring.png")


def ring_reveal_seconds(duration: float, zoom_at: float | None = None) -> float:
    """When (seconds into a segment) its ring fades in and its push-in starts. `zoom_at` is the
    crop's own setting (see resolve_group_zoom_at), used as given but never later than 90% of the
    segment so it still plays out. Without one: RING_REVEAL_DELAY_SECONDS, but never later than
    half the segment, so a short one still shows its ring."""
    if zoom_at is not None:
        return min(max(zoom_at, 0.0), duration * 0.9)
    return min(RING_REVEAL_DELAY_SECONDS, duration * 0.5)


def union_rect(rects: list[dict]) -> dict | None:
    """The smallest {x,y,width,height} box containing every rect — used to derive one crop region
    for a beat with several highlights, since the crop still has to be a single rectangle."""
    if not rects:
        return None
    x0 = min(r["x"] for r in rects)
    y0 = min(r["y"] for r in rects)
    x1 = max(r["x"] + r["width"] for r in rects)
    y1 = max(r["y"] + r["height"] for r in rects)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def hold_zoom_focus(crop: tuple[int, int, int, int], highlights: list[dict]) -> tuple[float, float]:
    """Where the push-in zooms toward, as (x, y) fractions of the cropped frame: the center of the
    highlights' union (clamped to the frame), or the frame's center when there are none."""
    union = union_rect(highlights)
    if not union:
        return 0.5, 0.5
    x0, y0, x1, y1 = crop
    fx = (union["x"] + union["width"] / 2 - x0) / (x1 - x0)
    fy = (union["y"] + union["height"] / 2 - y0) / (y1 - y0)
    return min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)


def resolve_highlights(beat: dict) -> list[dict]:
    """A beat's `highlights` — a list, so a beat can have more than one focus ring on screen at
    once (e.g. comparing two fields side by side); drawing/cropping around one region is just the
    N=1 case of drawing/cropping around several."""
    return beat.get("highlights") or []


def is_timed_highlight(h: dict) -> bool:
    """Whether a highlight (a custom slide's textbox — see slideText) manages its own independent
    on-screen window within a shared segment (resolve_highlight_timing), rather than being baked
    into the segment's static content and visible for its whole duration like an ordinary
    highlight. Mirrors editor.html's own isTimedHighlight check field-for-field — the two must stay
    in sync, or a slide authored in the editor would play back differently from how it previews."""
    return bool(h.get("startTime")) or isinstance(h.get("duration"), (int, float))


def resolve_highlight_timing(h: dict, seg_duration: float) -> tuple[float, float, float | None, float]:
    """A timed highlight's own (start, fade_in, duration, fade_out) in seconds, clamped to the
    segment's own measured duration (seg_duration — the group's narration-driven length, already
    floored by required_timed_highlight_duration so it's long enough to fit this exact window; see
    resolve_group_narration / _render_segment_clip). `duration` stays None when unset, meaning
    "stays on screen for the rest of the segment once it has faded in" — only a highlight with an
    explicit duration ever fades back out before the segment ends.

    `duration` counts the fully-opaque window *after* fade_in completes, not from `start` itself —
    mirrors editor.html's own highlightOpacityAt exactly (its fadeOutStart is
    `start + fadeIn + duration`), so the editor's preview and this real render agree on when a
    highlight fades back out."""
    start = min(max(h.get("startTime") or 0.0, 0.0), seg_duration)
    fade_in = max(h.get("fadeIn", DEFAULT_TIMED_FADE_IN_SECONDS), 0.0)
    duration = h.get("duration")
    if isinstance(duration, (int, float)):
        duration = min(max(float(duration), 0.0), max(seg_duration - start - fade_in, 0.0))
    else:
        duration = None
    fade_out = max(h.get("fadeOut", DEFAULT_TIMED_FADE_OUT_SECONDS), 0.0)
    return start, fade_in, duration, fade_out


def required_timed_highlight_duration(h: dict) -> float:
    """The minimum segment length this one timed highlight's own timing needs to fully play out —
    the group's narration-driven estimate (resolve_group_narration et al.) can easily be shorter,
    especially when the highlight carries no narration of its own (a pure on-screen caption with
    nothing spoken over it). Used to floor a segment's duration in _render_segment_clip, never to
    shorten it — a highlight that starts late, or fades out slowly, must not be silently cut off by
    a segment sized only for whatever the group happens to say out loud. Mirrors editor.html's own
    requiredTimedHighlightDuration, using the same start+fadeIn+duration+fadeOut total as
    resolve_highlight_timing's own fade-out formula."""
    start = max(h.get("startTime") or 0.0, 0.0)
    fade_in = max(h.get("fadeIn", DEFAULT_TIMED_FADE_IN_SECONDS), 0.0)
    duration = h.get("duration")
    fade_out = max(h.get("fadeOut", DEFAULT_TIMED_FADE_OUT_SECONDS), 0.0)
    if isinstance(duration, (int, float)):
        return start + fade_in + float(duration) + fade_out
    return start + fade_in + MIN_TIMED_VISIBLE_SECONDS


def resolve_group_fade_in(group: list[dict], beat: dict) -> float | None:
    """Seconds for a plain whole-content fade-in at this group's segment start, from a highlight's
    own `fadeIn` field — the simple case, for a highlight that's baked into the content and just
    wants a soft entrance rather than a hard cut, with no need for its own independent timing.
    Mirrors resolve_group_zoom/resolve_group_pan_speed's own group[0]-is-representative pattern.
    None (no extra fade) when absent, or when the group's own highlight is a timed highlight — that
    one already gets its own independent fade via resolve_highlight_timing/timed_layer_fade_chain,
    and whole-content-fading it again on top would just double up on the same effect."""
    if not group or is_timed_highlight(group[0]):
        return None
    return group[0].get("fadeIn")


def resolve_crop(beat: dict, fmt: str) -> dict | None:
    """A beat's `crop` override is keyed by format — {"shorts": {...}, "youtube": {...}} — since a
    9:16 crop and a 16:9 crop of the same screenshot are necessarily different regions; there's no
    single box that's correct for both. Also accepts the older flat {x,y,width,height} shape (pre-
    dating per-format crops) as a fallback for any format, so transcripts written before this
    existed don't silently lose their crop the first time someone renders a different format."""
    crop = beat.get("crop")
    if not crop:
        return None
    if fmt in crop:
        return crop[fmt]
    if "x" in crop:
        return crop
    return None


def _ring_rect_in_frame(
    crop_rect: tuple[int, int, int, int], canvas_width: int, content_height: int,
    highlight: dict, label: str = "",
) -> tuple[float, float, float, float]:
    """Maps a highlight's own raw (x,y,width,height) — screenshot-pixel space — into the local
    pixel space of a hold segment's already-cropped-and-resized frame, for drawing its ring there.
    Shared by _compose_static_frame's bulk ring layer and _render_segment_clip's own
    per-timed-highlight ring layer (see is_timed_highlight), so every highlight's ring is placed by
    the exact same math regardless of which path draws it."""
    x0, y0, x1, y1 = crop_rect
    crop_w, crop_h = x1 - x0, y1 - y0
    sx, sy = canvas_width / crop_w, content_height / crop_h
    hx0, hy0 = highlight["x"], highlight["y"]
    hx1, hy1 = hx0 + highlight["width"], hy0 + highlight["height"]
    # Only possible with a hand-set `crop` override, or a merge-test edge case — grouped highlights
    # are supposed to already fit their shared crop by construction. A manual crop is free to frame
    # a different region than the highlights, though, and if it ends up not overlapping one at all,
    # that ring silently never appears on screen — flag it instead of leaving someone to notice a
    # "missing" highlight only by watching the video.
    if hx1 <= x0 or hx0 >= x1 or hy1 <= y0 or hy0 >= y1:
        print(
            f"⚠️  {label}: a highlight at ({hx0:.0f},{hy0:.0f} "
            f"{highlight['width']:.0f}x{highlight['height']:.0f}) falls entirely outside "
            f"this segment's crop — its ring won't be visible in the rendered video.",
            file=sys.stderr,
        )
    return (hx0 - x0) * sx, (hy0 - y0) * sy, (hx1 - x0) * sx, (hy1 - y0) * sy


def _compose_static_frame(
    shot: Image.Image, canvas_width: int, content_height: int,
    crop_rect: tuple[int, int, int, int], highlights: list[dict], out_path: Path, label: str = "",
) -> bool:
    """Crop `shot` (already redacted) to `crop_rect` and scale to fill canvas_width x
    content_height exactly — full-bleed, no letterboxing — and save it to `out_path` (no rings).
    One ring per entry in `highlights`, mapped into the crop's local coordinate space and drawn at
    final resolution so it's crisp, goes into a separate transparent layer at
    ring_layer_path(out_path) so the render can fade it in later with the generic shared reveal
    delay (ring_fade_chain). Returns whether a ring layer was written. This is the "hold" half of a
    beat's segment timeline (see plan_beat_segments) — the crop itself is decided by the caller,
    not computed here.

    `highlights` here is only ever the NON-timed ones (see is_timed_highlight) — the caller
    (_render_segment_clip) excludes a timed highlight and builds its ring as its own separate,
    independently-timed layer instead, synced to its own text rather than this generic delay."""
    x0, y0, x1, y1 = crop_rect
    cropped = shot.crop((x0, y0, x1, y1))
    resized = cropped.resize((canvas_width, content_height), Image.LANCZOS)
    resized.save(out_path)

    ring_rects = [
        _ring_rect_in_frame(crop_rect, canvas_width, content_height, h, label)
        for h in highlights if not h.get("hideRing")
    ]
    if not ring_rects:
        return False
    highlight_ring_layer(resized.size, ring_rects).save(ring_layer_path(out_path))
    return True


def _compose_kenburns_pan_source(
    shot: Image.Image, canvas_width: int, content_height: int, axis: str,
    crop_w_i: int, crop_h_i: int, start: float, end: float, fixed_origin: float,
    zoom_out_factor: float, highlight: dict, out_path: Path,
) -> tuple[int, int, float, float, float, float, float, float]:
    """Composes the pan-source PNG for a Ken Burns pan: the camera slides from a `crop_w_i x
    crop_h_i` window (start) to one grown by `zoom_out_factor` (end) — see kenburns_pan_filter,
    which uses ffmpeg's zoompan to actually animate this (crop's own w/h can only be set once at
    filter init; only x/y re-evaluate per frame, so a growing window needs zoompan instead).
    zoompan's window is always exactly the *source image's own* aspect ratio, scaled by one `zoom`
    value — so unlike the old crop-filter-based pan, this source's own (width, height) must be
    forced to exactly crop_w_i:crop_h_i, not just "big enough," or a single zoom value couldn't
    grow both axes' window size in lockstep. Sized to contain the union of the start window and the
    (larger) end window — sound because both the window's center and size are affine in the frame
    index by construction (kenburns_pan_filter defines z that way specifically so this holds at
    render time), so the extremes of the swept area occur at the two endpoints, nowhere between.

    Returns (source_w, source_h, cx0, cy0, cx1, cy1, base_w, base_h): the composed PNG's own pixel
    dimensions; the start/end window centers in this PNG's local pixel space (cy0==cy1 unless
    axis=="y", cx0==cx1 unless axis=="x" — the fixed axis's center never moves); and the start
    window's own size scaled into that same local space."""
    shot_w, shot_h = shot.size
    move_size, fixed_size = (crop_w_i, crop_h_i) if axis == "x" else (crop_h_i, crop_w_i)

    move_center0 = start + move_size / 2
    move_center1 = end + move_size / 2
    fix_center = fixed_origin + fixed_size / 2
    move_size1 = move_size * zoom_out_factor
    fixed_size1 = fixed_size * zoom_out_factor

    move_lo = min(move_center0 - move_size / 2, move_center1 - move_size1 / 2)
    move_hi = max(move_center0 + move_size / 2, move_center1 + move_size1 / 2)
    fix_lo = min(fix_center - fixed_size / 2, fix_center - fixed_size1 / 2)
    fix_hi = max(fix_center + fixed_size / 2, fix_center + fixed_size1 / 2)

    if axis == "x":
        box_x0, box_x1, box_y0, box_y1 = move_lo, move_hi, fix_lo, fix_hi
    else:
        box_x0, box_x1, box_y0, box_y1 = fix_lo, fix_hi, move_lo, move_hi
    box_w, box_h = box_x1 - box_x0, box_y1 - box_y0

    # Force the pan-source's own aspect to exactly crop_w_i:crop_h_i (see docstring) by padding the
    # shorter side, centered — a no-op whenever the natural bounding box already matches (the
    # common case: an auto-computed crop_w_i:crop_h_i is already target_aspect). Computed on the
    # *unclamped* box: clamping first (or naively re-clamping after padding, which the previous
    # version of this function did) can cut the padded side back down at a screenshot edge —
    # confirmed live on a crop whose height already equals the full screenshot height, where the
    # silently-broken aspect made zoompan's single `zoom` value over-zoom one axis relative to the
    # other, rendering a wildly wrong crop instead of a correct-but-more-modest zoom-out.
    target_ratio = crop_w_i / crop_h_i
    if box_w / box_h > target_ratio:
        pad = (box_w / target_ratio - box_h) / 2
        box_y0, box_y1 = box_y0 - pad, box_y1 + pad
        box_h = box_y1 - box_y0
    else:
        pad = (box_h * target_ratio - box_w) / 2
        box_x0, box_x1 = box_x0 - pad, box_x1 + pad
        box_w = box_x1 - box_x0

    # If the aspect-correct box doesn't fit the screenshot, shrink it uniformly (about its own
    # center) rather than clamping each edge independently — clamping edges independently is
    # exactly what silently broke the aspect above; shrinking preserves it exactly, at the cost of
    # a more modest zoom-out/pan sweep than requested when content near a screenshot edge can't
    # supply the full amount (the same "reveal less than the ideal, never distort" trade-off
    # cap_pan_for_duration already makes for pan distance).
    shrink = min(1.0, shot_w / box_w, shot_h / box_h)
    if shrink < 1.0:
        cx, cy = (box_x0 + box_x1) / 2, (box_y0 + box_y1) / 2
        box_w, box_h = box_w * shrink, box_h * shrink
        box_x0, box_x1 = cx - box_w / 2, cx + box_w / 2
        box_y0, box_y1 = cy - box_h / 2, cy + box_h / 2

    # Box now fits within the screenshot by construction — shift (translate, not resize) it back
    # in bounds if its position pushed past an edge, which preserves box_w/box_h and so the aspect.
    if box_x0 < 0:
        box_x1 -= box_x0
        box_x0 = 0.0
    elif box_x1 > shot_w:
        box_x0 -= box_x1 - shot_w
        box_x1 = float(shot_w)
    if box_y0 < 0:
        box_y1 -= box_y0
        box_y0 = 0.0
    elif box_y1 > shot_h:
        box_y0 -= box_y1 - shot_h
        box_y1 = float(shot_h)

    scale = content_height / crop_h_i if axis == "x" else canvas_width / crop_w_i
    x0, y0, x1, y1 = round(box_x0), round(box_y0), round(box_x1), round(box_y1)
    cropped = shot.crop((x0, y0, x1, y1))
    source_w, source_h = round((x1 - x0) * scale), round((y1 - y0) * scale)
    resized = cropped.resize((source_w, source_h), Image.LANCZOS)

    resized.save(out_path)

    # The ring goes in its own layer next to the pan source (same size, so the same pan filter moves
    # both in lockstep) — see highlight_ring_layer. The caller checks hideRing to know whether to use it.
    if not highlight.get("hideRing"):
        hx0, hy0 = highlight["x"], highlight["y"]
        hx1, hy1 = hx0 + highlight["width"], hy0 + highlight["height"]
        overlay_rect = ((hx0 - x0) * scale, (hy0 - y0) * scale, (hx1 - x0) * scale, (hy1 - y0) * scale)
        highlight_ring_layer(resized.size, [overlay_rect]).save(ring_layer_path(out_path))

    base_w, base_h = crop_w_i * scale, crop_h_i * scale
    if axis == "x":
        cx0, cx1 = (move_center0 - x0) * scale, (move_center1 - x0) * scale
        cy0 = cy1 = (fix_center - y0) * scale
    else:
        cy0, cy1 = (move_center0 - y0) * scale, (move_center1 - y0) * scale
        cx0 = cx1 = (fix_center - x0) * scale

    return source_w, source_h, cx0, cy0, cx1, cy1, base_w, base_h


def kenburns_pan_filter(
    source_w: int, source_h: int, cx0: float, cy0: float, cx1: float, cy1: float,
    base_w: float, base_h: float, zoom_out_factor: float, total_frames: int, fps: int,
    final_width: int, final_height: int,
) -> str:
    """ffmpeg zoompan filter string for a Ken Burns pan: the window starts at base_w x base_h
    centered on (cx0,cy0) and grows linearly to base_w*zoom_out_factor x base_h*zoom_out_factor
    centered on (cx1,cy1) by the last frame — zoompan (not crop) because its window size is driven
    by one per-frame `zoom` value, which crop's w/h can't be (crop only re-evaluates x/y per frame).

    z is defined as start_zoom/(1+(zoom_out_factor-1)*on/last) — the reciprocal of an affine
    function of the frame index `on` — specifically so the resulting window width `iw/zoom` (and
    height `ih/zoom`) is itself affine in `on`, not just its reciprocal `zoom`. That affine-ness is
    what makes _compose_kenburns_pan_source's containment guarantee hold at render time, not just
    on paper (its docstring's endpoint-extremes argument requires both center and size to be
    affine in the frame index)."""
    last = max(total_frames - 1, 1)
    start_zoom = source_w / base_w
    frac = f"(on/{last})"
    z_expr = f"({start_zoom:.6f}/(1+({zoom_out_factor - 1:.6f})*{frac}))"
    cx_expr = f"({cx0:.3f}+({cx1 - cx0:.3f})*{frac})"
    cy_expr = f"({cy0:.3f}+({cy1 - cy0:.3f})*{frac})"
    x_expr = f"min({source_w}-iw/zoom,max(0,{cx_expr}-(iw/zoom)/2))"
    y_expr = f"min({source_h}-ih/zoom,max(0,{cy_expr}-(ih/zoom)/2))"
    return (
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}'"
        f":d={total_frames}:s={final_width}x{final_height}:fps={fps}"
    )


def resolve_group_narration(group: list[dict], beat: dict) -> str:
    """The narration spoken during one group's segment. A group with no highlights (the
    zero-highlight-beat case) speaks the beat's own narration — there's nothing else to attach text
    to. Otherwise, narration lives on the highlight itself (each highlight carries its own line,
    which is *why* a group no longer needs a manually-set duration at all: its clip's length just
    comes from however long its own narration takes to speak, exactly like a whole beat used to).
    A group's spoken line is its first member's — the representative for a merged group of 2+."""
    if not group:
        return beat.get("narration", "")
    return group[0].get("narration", "")


def resolve_group_zoom(group: list[dict], beat: dict) -> bool:
    """Whether this group's hold segment gets the push-in animation (irrelevant to a pan segment,
    which always gets its own Ken Burns zoom-out — see PAN_ZOOM_OUT_FACTOR in _render_segment_clip
    — since a pan without it is just a flat slide, not an optional flourish the way the hold's
    push-in is). Mirrors resolve_group_narration: lives on the highlight itself so different
    highlights within one beat can push in independently — a beat's overall "whole thing" push-in
    only ever made sense back when a beat had exactly one crop. Absence means on, matching
    zoompan_filter's own default."""
    if not group:
        return beat.get("zoom", True)
    return group[0].get("zoom", True)


def resolve_group_zoom_at(group: list[dict], beat: dict) -> float | None:
    """Seconds from the start of this group's segment at which its ring fades in and its push-in
    starts (see ring_reveal_seconds) — each crop sets its own moment, so a long narration can get
    its zoom where the spoken line reaches the thing being highlighted. None (the key absent) means
    the RING_REVEAL_DELAY_SECONDS default. Mirrors resolve_group_zoom/resolve_group_pan_speed."""
    if not group:
        return beat.get("zoomAt")
    return group[0].get("zoomAt")


def resolve_group_pan_speed(group: list[dict], beat: dict) -> float:
    """Per-highlight multiplier on DEFAULT_PAN_SPEED_FRACTION (default 1.0 = unscaled) — lower
    slows a pan down (more readable, less of the overflow revealed in the same duration), higher
    speeds it up. Mirrors resolve_group_zoom/resolve_group_narration; ignored for hold segments."""
    if not group:
        return beat.get("panSpeed", 1.0)
    return group[0].get("panSpeed", 1.0)


def _build_segment_for_group(shot: Image.Image, target_aspect: float, group: list[dict]) -> dict:
    """Build the (durationless) shape of the segment for one already-merge-tested group of
    highlights — its crop/pan framing, to be rendered at whatever length its own narration ends up
    taking (see resolve_group_narration / build_beat_clip). If its members share an explicit crop
    (they merged *because* they share one — see _crops_fit_together), that crop's own size is honored
    as the frame: for a lone highlight whose raw bounds don't fit inside it (hand-set narrower/
    shorter than the highlight, or the highlight was resized after the crop was saved), the camera
    pans across it instead of statically clipping the overflow away; otherwise (fits, or a merged
    group of more than one member) it's a static hold. With no explicit crop: a lone highlight whose
    own auto crop would overflow gets a pan (compute_focus_pan); anything else gets the ordinary auto
    crop around the group's union (compute_focus_crop) — only a group of exactly one highlight can
    ever need a pan, since any accepted merge, by construction, didn't overflow."""
    explicit = resolve_highlight_crop(group[0], CURRENT_FORMAT)
    if explicit:
        if len(group) == 1:
            pan = compute_focus_pan(
                shot.size, target_aspect, group[0],
                fixed_size=(explicit["width"], explicit["height"]),
            )
            if pan:
                return {"kind": "pan", "pan": pan, "highlight": group[0]}
        x0, y0 = explicit["x"], explicit["y"]
        crop_rect = (x0, y0, x0 + explicit["width"], y0 + explicit["height"])
        return {"kind": "hold", "crop": crop_rect, "highlights": group}

    union = union_rect(group)
    pan = compute_focus_pan(shot.size, target_aspect, union) if len(group) == 1 else None
    if pan:
        return {"kind": "pan", "pan": pan, "highlight": group[0]}
    crop_rect = compute_focus_crop(shot, target_aspect, union)
    return {"kind": "hold", "crop": crop_rect, "highlights": group}


def plan_beat_segments(
    shot: Image.Image, target_aspect: float, highlights: list[dict], beat_crop_override: dict | None,
) -> list[dict]:
    """Decompose one beat's visual timeline into a sequence of segments — the primitive behind both
    multi-highlight sequencing and wide-content panning. Each segment is either:
      {"kind": "hold", "crop": (x0,y0,x1,y1), "highlights": [...]}
      {"kind": "pan", "pan": (axis, crop_w_i, crop_h_i, start, end, fixed_origin), "highlight": {...}}
    Pure geometry — no duration involved. Each segment gets its length later, in build_beat_clip,
    from however long its own group's narration (resolve_group_narration) takes to actually speak —
    the same measure-the-real-audio approach a whole beat has always used, just applied per-group
    now instead of assuming one narration covers the whole beat.

    `beat_crop_override` (the beat-level `crop` field) only applies when the beat has no
    highlights at all — there's nothing else to frame around. Once a beat has highlights, crop
    control belongs to *them* individually (resolve_highlight_crop, checked inside
    _crops_fit_together/_build_segment_for_group): a beat-level override would otherwise force
    every highlight into one shared crop, which is exactly the loss-of-control this segment
    machinery exists to avoid — someone with two highlights in different corners of the screen
    needs to be able to frame each on its own, not be stuck picking one box for the whole beat the
    moment they set any custom crop at all.

    Highlights that all merge into one non-overflowing group still produce exactly one segment —
    the pre-existing single-clip behavior, unchanged — so the common case has zero regression risk
    from this machinery."""
    shot_size = shot.size

    if not highlights:
        if beat_crop_override:
            x0, y0 = beat_crop_override["x"], beat_crop_override["y"]
            crop_rect = (x0, y0, x0 + beat_crop_override["width"], y0 + beat_crop_override["height"])
        else:
            crop_rect = compute_focus_crop(shot, target_aspect, None)
        return [{"kind": "hold", "crop": crop_rect, "highlights": []}]

    groups = group_highlights(shot_size, target_aspect, highlights)
    return [_build_segment_for_group(shot, target_aspect, group) for group in groups]


def _load_caption_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    print(
        f"⚠️  No bold sans font found among {FONT_CANDIDATES} — falling back to PIL's built-in "
        "font (captions will look plainer). Install one of the above, or edit FONT_CANDIDATES.",
        file=sys.stderr,
    )
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()  # older Pillow — ignores `size`, small bitmap font


# Inline delivery tags a narration line can carry for the TTS model — `<short pause>`, `<long pause>`
# (angle brackets, what the editor's insert buttons write) or `[short pause]` (square brackets).
# They shape the audio only; they must never show up as caption or subtitle text.
DELIVERY_TAG_PATTERN = re.compile(r"<[a-z][a-z ]{0,30}>|\[(?:short|medium|long) pause\]", re.IGNORECASE)


def strip_delivery_tags(narration: str) -> str:
    """The narration as a viewer should read it: pause/delivery tags removed, spacing tidied."""
    return re.sub(r"\s{2,}", " ", DELIVERY_TAG_PATTERN.sub(" ", narration)).strip()


def compose_caption(narration: str, out_path: Path) -> None:
    canvas = Image.new("RGB", (WIDTH, CAPTION_AREA_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([FRAME_MARGIN, 0, WIDTH - FRAME_MARGIN, 4], fill=ACCENT_COLOR)

    font = _load_caption_font(CAPTION_FONT_SIZE)
    lines = wrap_text(draw, strip_delivery_tags(narration), font, WIDTH - FRAME_MARGIN * 2)
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [b[3] - b[1] for b in line_boxes]
    total_h = sum(line_heights) + CAPTION_LINE_SPACING * (len(lines) - 1)
    ty = (CAPTION_AREA_HEIGHT - total_h) // 2
    for line, box, lh in zip(lines, line_boxes, line_heights):
        tw = box[2] - box[0]
        tx = (WIDTH - tw) // 2
        draw.text((tx, ty), line, font=font, fill=CAPTION_COLOR)
        ty += lh + CAPTION_LINE_SPACING

    canvas.save(out_path)


def zoompan_filter(
    content_height: int, total_frames: int, fps: int, zoom_enabled: bool = True,
    zoom_start_seconds: float = 0.0, focus: tuple[float, float] = (0.5, 0.5),
) -> str:
    # Snap to PUSH_ZOOM_TARGET quickly, then hold — not a slow creep across the whole beat. A zoom
    # still animating throughout the narration keeps the frame in soft-focus/motion the entire
    # time, which read as "barely readable." Ramping over a fixed short window (starting at
    # `zoom_start_seconds` — the moment the ring appears, see ring_reveal_seconds — and staying at
    # exactly 1.0 until then) and holding once it arrives gives a quick push-in followed by a
    # steady, fully-readable frame for the rest of the beat.
    #
    # Zooms toward `focus` (x, y fractions of the frame — see hold_zoom_focus), not the frame's
    # center: the window is centered on the focus point, then clamped inside the frame, so a
    # highlight near an edge stays fully in view (a centered zoom trims ~7% off every edge at 1.15x,
    # which is what cut an edge highlight's ring off). Away from the edges it's the same push-in
    # aimed at the highlight instead of the middle.
    #
    # zoom_enabled=False (per-highlight/beat `"zoom": false` in transcript.json — see
    # resolve_group_zoom) skips the push-in entirely — confirmed live: a crop carefully
    # hand-positioned to show, say, the full left nav edge-to-edge still loses a few pixels off every
    # edge once the push-in animates past zoom=1.0, since that's additional magnification on top of
    # an already-exact crop, not slack the crop had to spare. target_zoom=1.0 makes zoompan a no-op
    # hold at the crop's own framing.
    target_zoom = PUSH_ZOOM_TARGET if zoom_enabled else 1.0
    start_frame = round(zoom_start_seconds * fps)
    ramp_frames = max(1, round(ZOOM_RAMP_SECONDS * fps))
    fx, fy = focus
    z_expr = f"if(lt(on,{start_frame}),1,min({target_zoom},1+({target_zoom - 1.0})*(on-{start_frame})/{ramp_frames}))"
    x_expr = f"min(iw-iw/zoom,max(0,{fx:.5f}*iw-iw/zoom/2))"
    y_expr = f"min(ih-ih/zoom,max(0,{fy:.5f}*ih-ih/zoom/2))"
    expr = f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}'"
    return f"{expr}:d={total_frames}:s={WIDTH}x{content_height}:fps={fps}"


def ring_fade_chain(motion_filter: str, reveal_seconds: float) -> str:
    """The ring layer's filter chain: the same motion filter the shot gets (zoompan or Ken Burns
    pan — identical parameters, so the ring stays glued to its element through the move), kept in
    an alpha-carrying format, faded in from fully transparent starting at `reveal_seconds`."""
    return (
        f"format=rgba,{motion_filter},format=rgba,"
        f"fade=t=in:st={reveal_seconds:.3f}:d={RING_FADE_SECONDS}:alpha=1"
    )


def timed_layer_fade_chain(
    motion_filter: str, start: float, fade_in: float, duration: float | None, fade_out: float,
) -> str:
    """A timed highlight's own overlay filter chain — mirrors ring_fade_chain exactly (same motion
    filter, so its text stays glued to the frame's zoom/pan, kept in an alpha-carrying format), but
    with its own independent fade-in at `start` and, only when `duration` is set, a fade-out
    `fade_out` seconds after the fade-in has finished and `duration` has elapsed. With no
    `duration`, it just stays at full opacity for the rest of the segment once it has faded in —
    there's nothing to fade back out to. `fade_in`/`fade_out` are floored at
    MIN_FFMPEG_FADE_SECONDS — a 0-second `fade` filter isn't meaningful to ffmpeg, so an "appear
    instantly" highlight (fadeIn unset, the default) still runs through the same filter with a
    practically-instant floor, purely to gate it invisible before `start`."""
    fade_in_d = max(fade_in, MIN_FFMPEG_FADE_SECONDS)
    chain = (
        f"format=rgba,{motion_filter},format=rgba,"
        f"fade=t=in:st={start:.3f}:d={fade_in_d:.3f}:alpha=1"
    )
    if duration is not None:
        # Mirrors editor.html's highlightOpacityAt exactly: the fade-out starts once the fade-in
        # has completed AND `duration` has elapsed since then — not `duration` seconds after `start`
        # itself (see resolve_highlight_timing's own docstring).
        fade_out_start = start + fade_in + duration
        fade_out_d = max(fade_out, MIN_FFMPEG_FADE_SECONDS)
        chain += f",fade=t=out:st={fade_out_start:.3f}:d={fade_out_d:.3f}:alpha=1"
    return chain


def _compose_overlay_layer(shots_dir: Path, h: dict, label: str) -> Path:
    """Where a timed highlight's own text-overlay PNG lives — pre-rendered and uploaded/downloaded
    by editor.html (renderSlideTextOverlayBlob) as `h["textOverlayFile"]`, not composed here; this
    just resolves and validates the path, mirroring ring_layer_path's own role for the ring layer."""
    name = h.get("textOverlayFile")
    if not name:
        sys.exit(f"{label}: timed highlight has no textOverlayFile (re-export from the editor)")
    path = shots_dir / name
    if not path.exists():
        sys.exit(f"{label}: timed highlight overlay not found: {path}")
    return path


_last_gemini_call_at: float | None = None


def _pace_gemini_call() -> None:
    """Block until at least GEMINI_MIN_SECONDS_BETWEEN_CALLS have passed since the previous Gemini
    TTS call, so a multi-beat run stays under the free tier's 3-requests/minute ceiling proactively
    instead of relying entirely on 429 retries."""
    global _last_gemini_call_at
    now = time.monotonic()
    if _last_gemini_call_at is not None:
        wait = GEMINI_MIN_SECONDS_BETWEEN_CALLS - (now - _last_gemini_call_at)
        if wait > 0:
            time.sleep(wait)
    _last_gemini_call_at = time.monotonic()


def narration_audio_hash(engine: str, model: str, voice: str, narration: str) -> str:
    """Name stem of a narration file saved by the editor: audio-<this>.wav|mp3. Must match
    narrationAudioName() in app/public/editor.html — sha1 of "engine|model|voice|text", first 16 hex."""
    key = f"{engine}|{model}|{voice}|{narration.strip()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def find_saved_narration_audio(audio_dir: Path, name_stem: str) -> Path | None:
    """A file the editor saved with the video — downloaded next to transcript.json (the project
    folder, where the Files panel puts everything) or dropped into audio/. Extensions beyond
    wav/mp3 cover a manually uploaded take (see UPLOAD_EXTENSIONS in app/public/editor.html)."""
    for folder in (audio_dir.parent, audio_dir):
        for extension in ("wav", "mp3", "m4a", "aac", "ogg", "flac"):
            candidate = folder / f"audio-{name_stem}.{extension}"
            if candidate.exists():
                return candidate
    return None


def adopt_saved_narration_audio(saved: Path, wav_path: Path, label: str) -> None:
    if saved.suffix == ".wav":
        shutil.copyfile(saved, wav_path)
    else:
        run(["ffmpeg", "-y", "-i", str(saved), str(wav_path)], f"{saved.suffix.lstrip('.')}->wav (saved audio, beat {label})")


def synthesize_narration(
    narration: str, audio_dir: Path, label: str, engine: str, voice: str, api_key: str | None,
) -> Path:
    wav_path = audio_dir / f"beat-{label}.wav"

    # A manually uploaded take (editor's ⬆ Upload button) is a human override for this exact
    # line and always wins — checked before the cache short-circuit below, so uploading a
    # replacement for a beat that was already rendered with (cached) TTS audio takes effect on
    # the next run instead of being masked by the stale cache.
    upload_cache_key = f"upload|manual|manual|{narration}"
    uploaded = find_saved_narration_audio(audio_dir, narration_audio_hash("upload", "manual", "manual", narration))
    if uploaded:
        print(f"  beat {label}: using uploaded audio {uploaded.name} (no {engine} call)")
        adopt_saved_narration_audio(uploaded, wav_path, label)
        (audio_dir / f"beat-{label}.cachekey").write_text(upload_cache_key)
        return wav_path

    # --engine saved: only audio that was made and checked in the editor is used. Never calls a
    # TTS API, so a render never spends money or sounds different from what the editor approved.
    if engine == "saved":
        sys.exit(f"No saved narration audio for beat {label}: generate and save it in the editor first.")

    # Cache key covers engine+model+voice+text, not just the beat index — a resumed run after a
    # crash (confirmed live: a 20+ call TTS run can die partway through on a transient timeout)
    # reuses already-paid-for audio instead of re-spending API calls/quota, but a genuine script
    # edit, a --voice change, or switching the underlying model constant (e.g. GEMINI_TTS_MODEL
    # from 2.5 to 3.1 — confirmed live: this was missing at first and silently served up stale
    # 2.5-generated audio under the 3.1 config) still regenerates rather than reusing stale audio.
    model = {"openai": OPENAI_TTS_MODEL, "elevenlabs": ELEVENLABS_TTS_MODEL, "gemini": GEMINI_TTS_MODEL}.get(engine, "")
    cache_key_path = audio_dir / f"beat-{label}.cachekey"
    cache_key = f"{engine}|{model}|{voice}|{narration}"
    if wav_path.exists() and cache_key_path.exists() and cache_key_path.read_text() == cache_key:
        return wav_path

    saved = find_saved_narration_audio(audio_dir, narration_audio_hash(engine, model, voice, narration))
    if saved:
        print(f"  beat {label}: using saved audio {saved.name} (no {engine} call)")
        adopt_saved_narration_audio(saved, wav_path, label)
        cache_key_path.write_text(cache_key)
        return wav_path

    if engine == "openai":
        mp3_path = audio_dir / f"beat-{label}.mp3"
        resp = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": OPENAI_TTS_MODEL, "voice": voice, "input": narration},
            timeout=60,
        )
        if resp.status_code != 200:
            sys.exit(f"OpenAI TTS failed for beat {label} ({resp.status_code}): {resp.text[:500]}")
        mp3_path.write_bytes(resp.content)
        run(["ffmpeg", "-y", "-i", str(mp3_path), str(wav_path)], f"mp3->wav (beat {label})")
    elif engine == "elevenlabs":
        mp3_path = audio_dir / f"beat-{label}.mp3"
        resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
            headers={"xi-api-key": api_key, "Accept": "audio/mpeg"},
            json={
                "text": narration,
                "model_id": ELEVENLABS_TTS_MODEL,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=60,
        )
        if resp.status_code != 200:
            sys.exit(f"ElevenLabs TTS failed for beat {label} ({resp.status_code}): {resp.text[:500]}")
        mp3_path.write_bytes(resp.content)
        run(["ffmpeg", "-y", "-i", str(mp3_path), str(wav_path)], f"mp3->wav (beat {label})")
    elif engine == "gemini":
        # How the line is sent depends on the model (see GEMINI_TTS_MODELS): the older ones need a
        # "Say ...:" instruction or they answer the text instead of speaking it; 3.8 takes it raw.
        # The narration keeps any inline pause tags — the model turns them into silence.
        _pace_gemini_call()
        prompt_prefix = GEMINI_TTS_MODELS[GEMINI_TTS_MODEL]["prompt_prefix"]
        payload = {
            "contents": [{"parts": [{"text": f"{prompt_prefix}{narration}"}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
            },
        }
        resp = None
        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent",
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=90,
                )
            except requests.exceptions.RequestException as exc:
                # A slow generation or a dropped connection is a retryable network hiccup, not a
                # reason to abort a 20-beat run outright — confirmed live: a plain ReadTimeout here
                # (uncaught, before this fix) killed an otherwise-successful run at beat 8.
                print(f"  Gemini TTS network error (beat {label}, attempt {attempt}/{GEMINI_MAX_RETRIES}): "
                      f"{exc} — retrying...", file=sys.stderr)
                resp = None
                time.sleep(GEMINI_MIN_SECONDS_BETWEEN_CALLS)
                continue
            if resp.status_code != 429:
                break
            # The free tier's own error message names a retry delay ("...retry in 10.01s") —
            # prefer that over guessing; fall back to a fixed pause if it's ever not present.
            match = re.search(r"retry in ([\d.]+)s", resp.text)
            delay = float(match.group(1)) + 2.0 if match else GEMINI_MIN_SECONDS_BETWEEN_CALLS
            print(f"  Gemini TTS rate-limited (beat {label}, attempt {attempt}/{GEMINI_MAX_RETRIES}) — "
                  f"waiting {delay:.0f}s before retrying...", file=sys.stderr)
            time.sleep(delay)
        # Note: don't write `if resp` here — a requests.Response is falsy for any status >= 400
        # (its __bool__ mirrors .ok), so that check silently discards the real error on every
        # non-2xx response and only ever prints "no response". Check identity instead.
        if resp is None or resp.status_code != 200:
            sys.exit(f"Gemini TTS failed for beat {label} "
                      f"({resp.status_code if resp is not None else '?'}): "
                      f"{resp.text[:500] if resp is not None else 'no response'}")
        part = resp.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
        # Returned as raw 16-bit PCM (mono, 24kHz) base64, not a container format — ffmpeg needs
        # to be told that explicitly (-f s16le/-ar/-ac) rather than sniffing it from a file header.
        pcm_path = audio_dir / f"beat-{label}.pcm"
        pcm_path.write_bytes(base64.b64decode(part["data"]))
        run([
            "ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", str(pcm_path), str(wav_path),
        ], f"pcm->wav (beat {label})")
    else:
        aiff_path = audio_dir / f"beat-{label}.aiff"
        run(["say", "-v", voice, "-o", str(aiff_path), narration], f"say (beat {label})")
        run(["ffmpeg", "-y", "-i", str(aiff_path), str(wav_path)], f"aiff->wav (beat {label})")

    cache_key_path.write_text(cache_key)
    return wav_path


def _render_segment_clip(
    shot: Image.Image, beat_index: int, seg_index: int, seg_count: int, content_height: int, fps: int,
    frames_dir: Path, audio_dir: Path, clips_dir: Path, captions: bool, segment: dict, narration: str,
    engine: str, voice: str, api_key: str | None, zoom: bool, pan_speed: float, no_audio: bool,
    shots_dir: Path, fade_in: float | None = None, zoom_at: float | None = None,
) -> tuple[Path, float]:
    """Render one segment (hold or pan) of a beat's visual timeline — see plan_beat_segments — as a
    complete clip with its own real audio baked in, at whatever length its own narration takes to
    speak (no manually-set duration needed: this is exactly the same measure-the-real-audio approach
    a whole beat has always used, just applied per-group now). Returns (clip_path, duration).

    `seg_count` lets a beat with exactly one segment (still the overwhelming majority) keep the
    plain beat-NN.* filenames byte-for-byte unchanged from before this multi-segment machinery
    existed — only a genuinely multi-segment beat gets the beat-NN-segNN.* naming. `shots_dir` is
    only needed to resolve a timed highlight's own text-overlay PNG (_compose_overlay_layer) — a
    custom slide's background/text otherwise, same as a real screenshot."""
    label = f"{beat_index:02d}" if seg_count == 1 else f"{beat_index:02d}-seg{seg_index:02d}"
    seg_path = clips_dir / f"beat-{label}-{CURRENT_FORMAT}.mp4"

    # A highlight with independent timing (see is_timed_highlight) gets its own ring reveal AND its
    # own text overlay (when it has slideText) synced to the exact same start/fadeIn/duration/
    # fadeOut — not the segment's generic shared ring-reveal delay (ring_reveal_seconds) or this
    # content's one shared ring layer. A timed highlight's text was also never baked into the
    # content PNG in the first place (see drawSlideCanvas's own includeTimed split in editor.html).
    # Computed up front, before duration is decided, so a highlight that starts late (or fades out
    # slowly) can floor the segment's own length instead of only ever being clamped to fit whatever
    # the group's spoken narration happens to run — see required_timed_highlight_duration.
    candidate_members = segment["highlights"] if segment["kind"] == "hold" else [segment["highlight"]]
    timed_highlights = [h for h in candidate_members if is_timed_highlight(h)]
    min_required_duration = max((required_timed_highlight_duration(h) for h in timed_highlights), default=0.0)

    if no_audio:
        # No real narration to measure yet — estimate from word count so the preview's pacing is
        # at least in the right ballpark of what the eventual real audio pass will produce.
        word_count = max(1, len(strip_delivery_tags(narration).split()))
        duration = max(word_count / (PREVIEW_WORDS_PER_MINUTE / 60), MIN_BEAT_DURATION, min_required_duration)
        wav_path = audio_dir / f"beat-{label}-silence.wav"
        run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", str(duration), str(wav_path),
        ], f"generate silence (beat {label})")
    else:
        wav_path = synthesize_narration(narration, audio_dir, label, engine, voice, api_key)
        duration = max(ffprobe_duration(wav_path), MIN_BEAT_DURATION, min_required_duration)
    total_frames = max(1, round(duration * fps))

    reveal_seconds = ring_reveal_seconds(duration, zoom_at)
    ring_path: Path | None = None

    if segment["kind"] == "hold":
        content_path = frames_dir / f"beat-{label}-{CURRENT_FORMAT}-content.png"
        # The shared ring layer only ever covers highlights WITHOUT independent timing — a timed
        # highlight gets its own separately-timed ring layer below instead (built once video_filter
        # is known, alongside its text overlay).
        shared_ring_highlights = [h for h in segment["highlights"] if not is_timed_highlight(h)]
        has_ring = _compose_static_frame(
            shot, WIDTH * CONTENT_SUPERSAMPLE, content_height * CONTENT_SUPERSAMPLE,
            segment["crop"], shared_ring_highlights, content_path, f"beat {label}",
        )
        ring_path = ring_layer_path(content_path) if has_ring else None
        # With highlights, the push-in starts as the ring appears and aims at them; with none there's
        # nothing to wait for or aim at, so it starts right away toward the center — unless the
        # crop sets its own moment (zoom_at).
        zoom_start = reveal_seconds if segment["highlights"] or zoom_at is not None else 0.0
        video_filter = zoompan_filter(
            content_height, total_frames, fps, zoom, zoom_start,
            hold_zoom_focus(segment["crop"], segment["highlights"]),
        )
    else:
        pan = cap_pan_for_duration(segment["pan"], duration, DEFAULT_PAN_SPEED_FRACTION * pan_speed)
        axis, crop_w_i, crop_h_i, start, end, fixed_origin = pan
        content_path = frames_dir / f"beat-{label}-{CURRENT_FORMAT}-pan.png"
        # Always on, unlike the hold's push-in (gated by `zoom` — resolve_group_zoom): a pan
        # without the zoom-out is just a flat lateral slide, which defeats the point of panning in
        # the first place — it isn't an optional flourish the way the hold's push-in is. A unit's
        # `zoom: false` predates this feature and only ever meant "don't push in on the hold"; it
        # shouldn't silently also flatten an unrelated pan on the same unit.
        zoom_out_factor = PAN_ZOOM_OUT_FACTOR
        source_w, source_h, cx0, cy0, cx1, cy1, base_w, base_h = _compose_kenburns_pan_source(
            shot, WIDTH * CONTENT_SUPERSAMPLE, content_height * CONTENT_SUPERSAMPLE, axis,
            crop_w_i, crop_h_i, start, end, fixed_origin, zoom_out_factor,
            segment["highlight"], content_path,
        )
        video_filter = kenburns_pan_filter(
            source_w, source_h, cx0, cy0, cx1, cy1, base_w, base_h, zoom_out_factor,
            total_frames, fps, WIDTH, content_height,
        )
        h0 = segment["highlight"]
        # A pan always has exactly one highlight, already alone in its own ring layer (just written
        # above, by _compose_kenburns_pan_source) — if it's timed, that same file is reused below
        # with its own independent fade instead of the generic shared one.
        ring_path = None if h0.get("hideRing") or is_timed_highlight(h0) else ring_layer_path(content_path)

    if fade_in is not None:
        # The simple whole-content case (resolve_group_fade_in) — a plain fade from black over the
        # base content itself, before any ring/overlay layers are chained on top of it below.
        video_filter = f"{video_filter},fade=t=in:st=0:d={fade_in:.3f}"

    caption_path = None
    if captions:
        caption_path = frames_dir / f"beat-{label}-{CURRENT_FORMAT}-caption.png"
        compose_caption(narration, caption_path)

    # Generalizes the single ring_path/ring_filter pair into an ordered list of overlay layers: the
    # shared ring (if any non-timed highlight has one) first, then each timed highlight's own ring
    # (if not hideRing) and text (if it has slideText) — both using the *same* computed timing, so
    # they move in perfect lockstep instead of the ring following the generic shared reveal delay
    # while the text follows its own start time (the bug this exists to fix). Every layer carries
    # the same base motion filter so it stays glued to the frame's zoom/pan.
    overlay_layers: list[tuple[Path, str]] = []
    if ring_path:
        overlay_layers.append((ring_path, ring_fade_chain(video_filter, reveal_seconds)))
    for idx, h in enumerate(timed_highlights):
        h_start, h_fade_in, h_duration, h_fade_out = resolve_highlight_timing(h, duration)
        h_filter = timed_layer_fade_chain(video_filter, h_start, h_fade_in, h_duration, h_fade_out)
        if not h.get("hideRing"):
            if segment["kind"] == "pan":
                h_ring_path = ring_layer_path(content_path)  # already composed alone, above
            else:
                h_ring_path = content_path.with_name(f"{content_path.stem}-ring-t{idx}.png")
                rect = _ring_rect_in_frame(
                    segment["crop"], WIDTH * CONTENT_SUPERSAMPLE, content_height * CONTENT_SUPERSAMPLE,
                    h, f"beat {label}",
                )
                highlight_ring_layer(
                    (WIDTH * CONTENT_SUPERSAMPLE, content_height * CONTENT_SUPERSAMPLE), [rect],
                ).save(h_ring_path)
            overlay_layers.append((h_ring_path, h_filter))
        if h.get("slideText"):
            overlay_path = _compose_overlay_layer(shots_dir, h, f"beat {label}")
            overlay_layers.append((overlay_path, h_filter))

    _encode_segment_clip(
        content_path, overlay_layers, caption_path, wav_path, video_filter,
        fps, duration, seg_path, label,
    )
    return seg_path, duration


def _encode_segment_clip(
    content_path: Path, overlay_layers: list[tuple[Path, str]], caption_path: Path | None, wav_path: Path,
    video_filter: str, fps: int, duration: float, seg_path: Path, label: str,
) -> None:
    """One ffmpeg pass for a segment: the shot through `video_filter`, each (path, filter) pair in
    `overlay_layers` composited on top in order — the highlight ring (ring_fade_chain) first, then
    any timed highlight's own text overlay (timed_layer_fade_chain) — the optional caption bar
    stacked under it, and the narration audio padded to the clip's length.

    -framerate on a -loop 1 still image input is required, not cosmetic — without it ffmpeg decodes
    the image as an effectively 1fps source and the filter has to duplicate/interpolate frames to
    reach fps=, which is what caused visible jitter/stutter during the push-in when this was first
    built. Forcing the input's own frame rate gives every filter a real, steady per-frame tick to
    compute each crop from — applies identically to zoompan and the pan crop filter."""
    overlay_paths = [p for p, _ in overlay_layers]
    image_inputs = [content_path] + overlay_paths + ([caption_path] if caption_path else [])
    cmd = ["ffmpeg", "-y", "-sws_flags", "lanczos"]
    for path in image_inputs:
        cmd += ["-loop", "1", "-framerate", str(fps), "-i", str(path)]
    cmd += ["-i", str(wav_path)]

    graph = [f"[0:v]{video_filter}[base]"]
    prev_tag = "base"
    for idx, (_, layer_filter) in enumerate(overlay_layers, start=1):
        layer_tag, out_tag = f"l{idx}", f"ov{idx}"
        graph += [
            f"[{idx}:v]{layer_filter}[{layer_tag}]",
            f"[{prev_tag}][{layer_tag}]overlay=format=auto[{out_tag}]",
        ]
        prev_tag = out_tag
    if caption_path:
        caption_index = 1 + len(overlay_layers)
        graph += [
            f"[{caption_index}:v]scale={WIDTH}:{CAPTION_AREA_HEIGHT}[cap]", f"[{prev_tag}][cap]vstack=inputs=2[v]",
        ]
    else:
        graph.append(f"[{prev_tag}]null[v]")
    graph.append(f"[{len(image_inputs)}:a]apad[a]")

    cmd += [
        "-filter_complex", ";".join(graph),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
        "-t", str(duration),
        str(seg_path),
    ]
    run(cmd, f"render clip (beat {label})")


def build_beat_clip(
    feature_dir: Path, beat_index: int, beat: dict,
    engine: str, voice: str, api_key: str | None, fps: int, captions: bool,
    crop: dict | None, no_audio: bool,
) -> tuple[Path, str]:
    """Returns (clip_path, spoken_text): spoken_text is every segment's narration in reading order,
    joined — used for write_captions_srt's per-beat SRT cue, since SRT timing stays per-beat (not
    per-segment: a beat's segments play back to back with no gap a viewer would need a separate cue
    for) even though each segment now has its own narration and its own on-screen caption bar."""
    # Screenshots sit directly in the project folder, alongside transcript.json — not in a nested
    # shots/ subfolder — matching how the board uploads/downloads them (every file for a video,
    # images and transcript.json alike, shares one flat shots_context "folder").
    shots_dir = feature_dir
    frames_dir = feature_dir / "frames"
    audio_dir = feature_dir / "audio"
    clips_dir = feature_dir / "clips"
    for d in (frames_dir, audio_dir, clips_dir):
        d.mkdir(exist_ok=True)

    screenshot = beat["screenshot"]
    image_path = shots_dir / screenshot
    if not image_path.exists():
        sys.exit(f"Beat {beat_index}: screenshot not found: {image_path}")

    content_height = HEIGHT - CAPTION_AREA_HEIGHT if captions else HEIGHT
    highlights = resolve_highlights(beat)
    shot = apply_redactions(Image.open(image_path).convert("RGB"), beat.get("redactions"))
    segments = plan_beat_segments(shot, WIDTH / content_height, highlights, crop)
    groups = [seg["highlights"] if seg["kind"] == "hold" else [seg["highlight"]] for seg in segments]
    narrations = [resolve_group_narration(g, beat) for g in groups]
    zooms = [resolve_group_zoom(g, beat) for g in groups]
    pan_speeds = [resolve_group_pan_speed(g, beat) for g in groups]
    zoom_ats = [resolve_group_zoom_at(g, beat) for g in groups]
    fade_ins = [resolve_group_fade_in(g, beat) for g in groups]

    seg_paths = [
        _render_segment_clip(
            shot, beat_index, i, len(segments), content_height, fps, frames_dir, audio_dir, clips_dir,
            captions, seg, narration, engine, voice, api_key, zoom, pan_speed, no_audio,
            shots_dir, fade_in, zoom_at,
        )[0]
        for i, (seg, narration, zoom, pan_speed, fade_in, zoom_at)
        in enumerate(zip(segments, narrations, zooms, pan_speeds, fade_ins, zoom_ats), start=1)
    ]

    if len(seg_paths) == 1:
        # _render_segment_clip already wrote directly to this exact path (seg_count == 1 uses the
        # plain beat-NN.* filename) — no concat needed for the overwhelmingly common case.
        clip_path = seg_paths[0]
    else:
        # Multiple segments (a beat whose highlights split into more than one crop) — concatenate
        # them hard-cut, exactly like the beat-to-beat concat below; every segment shares identical
        # codec/pix_fmt/fps by construction, so a stream-copy concat is safe here for the same
        # reason it's safe there. Each segment clip already carries its own audio, so — unlike the
        # duration-allocation era this replaced — there's no separate "mux the shared track on
        # afterward" step: concatenating is the entire job.
        clip_path = clips_dir / f"beat-{beat_index:02d}-{CURRENT_FORMAT}.mp4"
        concat_list = clips_dir / f"beat-{beat_index:02d}-{CURRENT_FORMAT}-segments.txt"
        concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in seg_paths))
        run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(clip_path),
        ], f"concatenate segments (beat {beat_index})")

    return clip_path, strip_delivery_tags(" ".join(t for t in narrations if t))


def concatenate_with_transitions(
    clip_paths: list[Path], transition: str, transition_duration: float, output_path: Path,
) -> None:
    """Cross-fade (video via xfade, audio via acrossfade) between consecutive clips instead of a
    hard cut. Requires re-encoding — ffmpeg's concat *demuxer* (the hard-cut path, stream-copy only)
    can't blend across a clip boundary, only butt clips together. Each xfade's `offset` is where in
    the running, already-merged timeline that transition should start; the whole thing has to be
    computed up front from each clip's real duration since the filter graph is one static string."""
    durations = [ffprobe_duration(p) for p in clip_paths]
    # A transition can't be longer than the clips it's blending, or the offset math goes negative —
    # clamp to a safe fraction of the shortest clip rather than letting ffmpeg fail on an edge case.
    t = min(transition_duration, min(durations) * 0.4)

    inputs = []
    for p in clip_paths:
        inputs += ["-i", str(p)]

    video_chain, audio_chain = [], []
    prev_v, prev_a = "0:v", "0:a"
    running_duration = durations[0]
    for i in range(1, len(clip_paths)):
        offset = running_duration - t
        out_v = f"v{i}" if i < len(clip_paths) - 1 else "vout"
        out_a = f"a{i}" if i < len(clip_paths) - 1 else "aout"
        video_chain.append(f"[{prev_v}][{i}:v]xfade=transition={transition}:duration={t:.3f}:offset={offset:.3f}[{out_v}]")
        audio_chain.append(f"[{prev_a}][{i}:a]acrossfade=d={t:.3f}[{out_a}]")
        prev_v, prev_a = out_v, out_a
        running_duration += durations[i] - t

    run([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(video_chain + audio_chain),
        "-map", f"[{prev_v}]", "-map", f"[{prev_a}]",
        "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", str(output_path),
    ], "concatenate with transitions")


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def beat_cue_times(
    durations: list[float], transition: str, transition_duration: float,
) -> list[tuple[float, float]]:
    """Each beat's (start, end) in the final concatenated timeline. A hard cut (transition=none) is
    just a running sum; a cross-fade pulls every later beat's start forward by the blend duration,
    since that's when the next clip's content actually starts appearing on screen — the same
    accumulation `concatenate_with_transitions` uses to compute its own xfade offsets."""
    if not durations:
        return []
    t = min(transition_duration, min(durations) * 0.4) if transition != "none" else 0.0
    starts = [0.0]
    running = durations[0]
    for dur in durations[1:]:
        starts.append(running - t)
        running += dur - t
    return [(s, s + d) for s, d in zip(starts, durations)]


def write_captions_srt(
    caption_texts: list[str], durations: list[float], transition: str, transition_duration: float, out_path: Path,
) -> None:
    """A standalone SRT sidecar, one cue per beat, timed to each beat's real (post-TTS) duration —
    for YouTube's optional caption upload. The video itself ships with no burned-in captions by
    default; this is how the narration still reaches viewers who watch muted or want real
    (editable, translatable) subtitles instead of text baked into the pixels. `caption_texts` is
    each beat's full spoken text (build_beat_clip already joins a multi-segment beat's several
    narrations into one string here, since SRT timing stays per-beat, not per-segment)."""
    cues = beat_cue_times(durations, transition, transition_duration)
    lines = []
    for i, ((start, end), text) in enumerate(zip(cues, caption_texts), start=1):
        lines.append(f"{i}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n{text}\n")
    out_path.write_text("\n".join(lines))


def assemble(
    project_dir: str, engine: str, voice: str, api_key: str | None, captions: bool, fps: int,
    no_audio: bool = False, output_suffix: str = "", transition: str = "none",
    transition_duration: float = 0.4,
) -> Path:
    feature_dir = Path(project_dir)
    if not feature_dir.is_dir():
        sys.exit(f"Not a folder: {feature_dir}")
    # Purely a filename prefix (and error-message label) — the folder's own name, so the output
    # video/SRT are named after whatever the editor called their project folder, not a repo-wide
    # "feature" concept this standalone tool has no other reason to know about.
    feature = feature_dir.name
    transcript_path = feature_dir / "transcript.json"
    if not transcript_path.exists():
        sys.exit(f"No transcript.json found in {feature_dir}")

    all_beats = json.loads(transcript_path.read_text())
    if not all_beats:
        sys.exit(f"transcript.json in {feature_dir} is empty — nothing to assemble.")
    beats = [b for b in all_beats if not b.get("excluded")]
    if not beats:
        sys.exit(f"Every beat in {feature_dir} is marked excluded — nothing to assemble.")
    if len(beats) < len(all_beats):
        print(f"Skipping {len(all_beats) - len(beats)} excluded beat(s).")

    results = [
        build_beat_clip(
            feature_dir, i, beat, engine, voice, api_key, fps, captions,
            resolve_crop(beat, CURRENT_FORMAT), no_audio,
        )
        for i, beat in enumerate(beats, start=1)
    ]
    clip_paths = [p for p, _ in results]
    caption_texts = [t for _, t in results]

    # A preview render gets its own filename — never overwrite a final, narrated cut with a
    # silent placeholder just because someone re-ran with --no-audio to check a screenshot fix.
    # Likewise a non-default format (see output_suffix in main()) never overwrites shorts' default
    # <feature>.mp4 — both formats can exist side by side from the same transcript.json.
    output_path = feature_dir / f"{feature}{output_suffix}{'-preview' if no_audio else ''}.mp4"

    if transition == "none":
        concat_list = feature_dir / "concat.txt"
        concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in clip_paths))
        run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(output_path),
        ], "concatenate final video")
    else:
        concatenate_with_transitions(clip_paths, transition, transition_duration, output_path)

    if not no_audio:
        srt_path = feature_dir / f"{feature}{output_suffix}.srt"
        write_captions_srt(
            caption_texts, [ffprobe_duration(p) for p in clip_paths], transition, transition_duration, srt_path,
        )
        print(f"Captions file (upload separately to YouTube if you want subtitles): {srt_path}")

    if CURRENT_FORMAT == "shorts":
        total_duration = ffprobe_duration(output_path)
        if total_duration > SHORTS_MAX_SECONDS:
            mins, secs = divmod(round(total_duration), 60)
            print(
                f"\n⚠️  WARNING: this shorts (9:16) video is {mins}:{secs:02d} — over YouTube's "
                f"{SHORTS_MAX_SECONDS // 60}:00 cutoff for a Short. It will still upload fine, just "
                f"as a regular video, not eligible for Shorts placement/feed. Trim beats or drop "
                f"one from the transcript (or check `excluded`) to bring it under 3:00.\n",
                file=sys.stderr,
            )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_dir", help="Path to the folder containing transcript.json and its screenshots")
    parser.add_argument("--engine", choices=["saved", "gemini", "openai", "elevenlabs", "macos"], default="saved",
                         help="Narration source. saved (default) uses only audio files saved by the "
                              "Reeldesk editor (audio-<hash>.* next to transcript.json) and never calls "
                              "a TTS API. Otherwise a TTS engine: gemini needs GEMINI_API_KEY in this folder's "
                              ".env; openai needs an API key (OPENAI_API_KEY env var, or "
                              "AI_OPENAI_KEY in .env) and sounds natural; elevenlabs needs "
                              "ELEVENLABS_API_KEY in .env; macos is offline and free but noticeably "
                              "robotic. See .env.example.")
    parser.add_argument("--voice", default=None,
                         help=f"Voice name/ID. Default: {DEFAULT_OPENAI_VOICE!r} for openai, "
                              f"{DEFAULT_ELEVENLABS_VOICE!r} for elevenlabs, "
                              f"{DEFAULT_GEMINI_VOICE!r} for gemini, {DEFAULT_MACOS_VOICE!r} for macos.")
    parser.add_argument("--gemini-model", choices=list(GEMINI_TTS_MODELS), default=DEFAULT_GEMINI_TTS_MODEL,
                         help="Gemini TTS model (only with --engine gemini). Default: "
                              f"{DEFAULT_GEMINI_TTS_MODEL} — the one where inline <short pause>/<long pause> "
                              "tags in a narration line reliably become silence.")
    parser.add_argument("--format", choices=list(FORMATS), default="shorts",
                         help="Output shape. shorts (default): 1080x1920 portrait for TikTok/"
                              "Shorts/Reels — heavily crops to each beat's highlight (or its "
                              "`crop` override, if set) since a landscape screenshot can't fill a "
                              "portrait frame otherwise. youtube: 1920x1080 landscape — crops much "
                              "more gently, since a desktop screenshot is already close to that "
                              "shape. Both can be rendered from the same transcript.json without "
                              "clobbering each other's output files or cached frames.")
    parser.add_argument("--captions", action="store_true",
                         help="Burn narration text into the video as an on-screen caption bar. Off by "
                              "default — a standalone <feature>.srt sidecar is always written next to "
                              "the video instead, ready to upload as a real (editable, translatable) "
                              "YouTube caption track without baking text into the pixels.")
    parser.add_argument("--no-audio", action="store_true",
                         help="Render a silent preview (<feature>-preview.mp4) with per-beat duration "
                              "estimated from word count, instead of real narration — for checking "
                              "screenshots, zoom, and pacing before spending any TTS API calls. No "
                              "--engine/key needed in this mode.")
    parser.add_argument("--transition", choices=TRANSITIONS, default="none",
                         help="Cross-fade style between beats. none (default): a hard cut, via "
                              "fast stream-copy concatenation. Any other value re-encodes the whole "
                              "video to blend each beat boundary over --transition-duration seconds.")
    parser.add_argument("--transition-duration", type=float, default=0.4,
                         help="Seconds each cross-fade takes (ignored when --transition is none).")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    require_tool("ffmpeg", "brew install ffmpeg")
    require_tool("ffprobe", "brew install ffmpeg")

    global WIDTH, HEIGHT, CAPTION_AREA_HEIGHT, CAPTION_FONT_SIZE, CURRENT_FORMAT, GEMINI_TTS_MODEL
    GEMINI_TTS_MODEL = args.gemini_model
    fmt = FORMATS[args.format]
    WIDTH, HEIGHT = fmt["width"], fmt["height"]
    CAPTION_AREA_HEIGHT = round(HEIGHT * fmt["caption_fraction"])
    CAPTION_FONT_SIZE = round(CAPTION_AREA_HEIGHT * 0.144)  # keeps the same proportions as shorts' 46/320
    CURRENT_FORMAT = args.format
    output_suffix = "" if args.format == "shorts" else f"-{args.format}"

    if args.no_audio:
        output_path = assemble(args.project_dir, args.engine, args.voice or "", None,
                                args.captions, args.fps, no_audio=True, output_suffix=output_suffix,
                                transition=args.transition, transition_duration=args.transition_duration)
        print(f"Done (silent preview): {output_path}")
        return

    api_key = None
    if args.engine == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or load_env_var("GEMINI_API_KEY")
        if not api_key:
            sys.exit(
                "Gemini TTS selected but no key found. Set the GEMINI_API_KEY env var, or fill in "
                "GEMINI_API_KEY in this folder's .env (copy .env.example to .env first). "
                "Or pass --engine macos/openai/elevenlabs instead."
            )
    elif args.engine == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or load_env_var("AI_OPENAI_KEY")
        if not api_key:
            sys.exit(
                "OpenAI TTS selected but no key found. Set the OPENAI_API_KEY env var, or fill in "
                "AI_OPENAI_KEY in this folder's .env. Or pass --engine macos instead."
            )
    elif args.engine == "elevenlabs":
        api_key = os.environ.get("ELEVENLABS_API_KEY") or load_env_var("ELEVENLABS_API_KEY")
        if not api_key:
            sys.exit(
                "ElevenLabs TTS selected but no key found. Set the ELEVENLABS_API_KEY env var, or "
                "fill in ELEVENLABS_API_KEY in this folder's .env. Or pass --engine macos/gemini instead."
            )
    elif args.engine == "macos":
        require_tool("say", "already built into macOS — check your PATH")

    voice = args.voice or {
        "saved": "",
        "gemini": DEFAULT_GEMINI_VOICE,
        "openai": DEFAULT_OPENAI_VOICE,
        "elevenlabs": DEFAULT_ELEVENLABS_VOICE,
        "macos": DEFAULT_MACOS_VOICE,
    }[args.engine]

    output_path = assemble(args.project_dir, args.engine, voice, api_key, args.captions, args.fps,
                            output_suffix=output_suffix, transition=args.transition,
                            transition_duration=args.transition_duration)
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
