# nostackai-reel-mp4-app

Renders [Reeldesk](https://reels.nostackai.com) videos to mp4. A Reeldesk video is a set of
screenshots and slides, a `transcript.json` (highlights, crops, narration) and the narration audio
each line was given and approved in Reeldesk's editor. This repository turns that into a finished
video and uploads it back to the video, as a GitHub Actions workflow.

It never generates narration. It uses only the audio saved in Reeldesk, so a render sounds exactly
like the editor's preview and costs nothing. A line with no saved audio stops the render and the
video's note names the beat.

## How a render starts

| From | What happens |
|---|---|
| Reeldesk's **Render video** button | The video is marked `render_status: queued`; a nostackai automation runs this workflow for it |
| Moving a card to **Ready for preview** | Same as the button |
| The schedule (every 30 minutes) | Renders any video still queued after 10 minutes (a run that never started) |
| By hand | Actions → **Render video** → Run workflow, with a video id or empty for the whole queue |

For each video the job:
1. signs in to Reeldesk as the render user;
2. marks it `rendering`;
3. downloads its transcript, screenshots and saved audio;
4. renders each format the video asks for (`shorts`, `youtube` or both);
5. uploads each mp4 as the video's `video_files` row of kind `video`, one per format and reused on a re-render;
6. points `video_content_id` at it, marks it `done` and posts a note.

A failure marks it `failed`, with the reason in `error_message` and a note.

## Setup

### 1. The render user (in the Reeldesk org)

An app user that only the job signs in as, with an access policy that allows exactly what it does.
In the Reeldesk org this is the `renderer` policy:

| Statement | Actions |
|---|---|
| Videos | `getvideos`, `listvideos`, `updatevideos` |
| Video files | `getvideo_files`, `listvideo_files`, `createvideo_files`, `updatevideo_files` |
| Notes | `createvideo_notes` |
| Upload | `getuploadurl` |

It needs nothing else: no playlists, no deletes, no user management. The previous mp4 of a
re-rendered format is replaced on its `video_files` row; removing the old upload itself is attempted
but needs content delete, which this policy does not grant, so it is skipped quietly.

### 2. Repository secrets

Settings → Secrets and variables → Actions:

| Secret | Value |
|---|---|
| `REELDESK_APP_URL` | The org's app base URL, e.g. `https://t001.nostackai.com` |
| `REELDESK_ORG_CODE` | The org code |
| `REELDESK_API_KEY` | The app's `x-api-key` |
| `REELDESK_CLIENT_ID` | The app's `client_id` |
| `REELDESK_RENDER_EMAIL` | The render user's email |
| `REELDESK_RENDER_PASSWORD` | The render user's password |

The job signs in with these each run and keeps the token, and the CDN cookies the same sign-in
returns, in memory only.

### 3. The trigger in nostackai

In the Reeldesk org:
1. adopt the GitHub connector, with a token that may run this repository's workflows (`actions: write`);
2. add an automation on `videos` updates where `render_status` becomes `queued`, calling the
   connector's **Trigger Workflow** action with `repo: nostackai-reel-mp4-app`,
   `workflow_id: render.yml` and the input `video_id`.

## Public logs

This repository is public, so its Actions logs are too. The job prints only video ids, counts and
step names. Narration text, file names from the transcript, URLs, tokens and cookies never reach
the log, and GitHub masks every secret.

## Running it yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # ffmpeg must be installed too (brew install ffmpeg)

# A Reeldesk video, with the same settings as the workflow in your environment:
python render_video.py --video <video id>

# A project folder on disk (transcript.json, screenshots and audio-*.wav|mp3 next to it):
python assemble.py path/to/project --format youtube
# Or generate the narration with a TTS engine instead of saved audio (needs that engine's key):
python assemble.py path/to/project --engine gemini
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The end-to-end test renders a two-beat project with real ffmpeg (skipped without it).
