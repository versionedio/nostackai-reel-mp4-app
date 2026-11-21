"""The render job's own logic, with a fake Reeldesk (no network, no ffmpeg)."""
import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import render_video as rv  # noqa: E402
from reeldesk import ReeldeskError  # noqa: E402

TRANSCRIPT = [
    {"screenshot": "a.png", "highlights": [{"narration": "First line."}, {"narration": "Second line."}]},
    {"screenshot": "b.png", "excluded": True, "highlights": [{"narration": "Skipped line."}]},
    {"screenshot": "c.png", "narration": "A beat with no highlights."},
    {"screenshot": "d.png", "highlights": [{"narration": "   "}]},
]


class FakeReeldesk:
    """Records every write; serves files from a dict of name -> bytes."""

    def __init__(self, video, files, contents):
        self.video, self.files, self.contents = video, files, contents
        self.updates, self.notes, self.added, self.updated_files, self.deleted = [], [], [], [], []

    def get_video(self, video_id):
        return dict(self.video)

    def list_videos(self, **filters):
        return [dict(self.video)] if self.video.get("render_status") == filters.get("render_status") else []

    def update_video(self, video_id, fields):
        self.updates.append(fields)
        self.video.update(fields)

    def list_files(self, video_id):
        return list(self.files)

    def download(self, row, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.contents[row["name"]])
        return dest

    def upload(self, path, name, content_type, context):
        return {"contentId": f"new-{name}", "cdnUrl": f"cdn/{name}", "uploadUrl": "x"}

    def add_file(self, row):
        self.added.append(row)

    def update_file(self, file_id, row):
        self.updated_files.append((file_id, row))

    def delete_content(self, content_id):
        self.deleted.append(content_id)

    def add_note(self, video_id, text):
        self.notes.append(text)


def audio_name(text, suffix=".mp3"):
    return rv.upload_identity_name(text, suffix)


class WhatToRender(unittest.TestCase):
    def test_formats_follow_the_video(self):
        self.assertEqual(rv.formats_for({"format": "both"}), ["shorts", "youtube"])
        self.assertEqual(rv.formats_for({"format": "shorts"}), ["shorts"])
        self.assertEqual(rv.formats_for({}), ["youtube"])

    def test_speaking_units_skip_excluded_and_empty(self):
        units = [(n, h["narration"]) for n, h in rv.narration_holders(TRANSCRIPT)]
        self.assertEqual(units, [(1, "First line."), (1, "Second line."), (3, "A beat with no highlights.")])


class SavedAudio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project = Path(self.tmp.name)
        (self.project / "saved-audio").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def saved(self, name):
        path = self.project / "saved-audio" / name
        path.write_bytes(b"audio")
        return name, path

    def test_the_line_s_recorded_file_is_used_under_the_text_identity(self):
        transcript = json.loads(json.dumps(TRANSCRIPT))
        transcript[0]["highlights"][0]["audio"] = "audio-aaaa.mp3"
        transcript[0]["highlights"][1]["audio"] = "audio-bbbb.wav"
        transcript[2]["audio"] = "audio-cccc.mp3"
        audio = dict([self.saved("audio-aaaa.mp3"), self.saved("audio-bbbb.wav"), self.saved("audio-cccc.mp3")])
        self.assertEqual(rv.place_saved_audio(transcript, audio, self.project), [])
        self.assertTrue((self.project / audio_name("First line.")).exists())
        self.assertTrue((self.project / audio_name("Second line.", ".wav")).exists())

    def test_an_uploaded_take_is_found_by_its_text(self):
        transcript = [{"screenshot": "a.png", "narration": "Uploaded take."}]
        audio = dict([self.saved(audio_name("Uploaded take.", ".m4a"))])
        self.assertEqual(rv.place_saved_audio(transcript, audio, self.project), [])

    def test_lines_without_a_saved_take_are_reported_by_beat(self):
        transcript = json.loads(json.dumps(TRANSCRIPT))
        transcript[0]["highlights"][0]["audio"] = "audio-aaaa.mp3"
        transcript[2]["audio"] = "audio-not-uploaded.mp3"
        audio = dict([self.saved("audio-aaaa.mp3")])
        self.assertEqual(rv.place_saved_audio(transcript, audio, self.project), [1, 3])


class Queue(unittest.TestCase):
    def test_sweep_leaves_a_fresh_request_to_its_dispatch(self):
        fresh = FakeReeldesk({"id": "v1", "render_status": "queued", "render_requested_at": rv.now().isoformat()}, [], {})
        self.assertEqual(rv.queued_video_ids(fresh), [])
        old = (rv.now() - rv.SWEEP_GRACE - timedelta(minutes=1)).isoformat()
        waiting = FakeReeldesk({"id": "v1", "render_status": "queued", "render_requested_at": old}, [], {})
        self.assertEqual(rv.queued_video_ids(waiting), ["v1"])

    def test_a_live_claim_is_respected_and_a_stale_one_taken_over(self):
        live = FakeReeldesk({"id": "v1", "render_status": "rendering", "render_started_at": rv.now().isoformat()}, [], {})
        self.assertFalse(rv.claim(live, live.video, force=False))
        self.assertTrue(rv.claim(live, live.video, force=True))
        stale_start = (rv.now() - rv.STALE_CLAIM - timedelta(minutes=1)).isoformat()
        stale = FakeReeldesk({"id": "v1", "render_status": "rendering", "render_started_at": stale_start}, [], {})
        self.assertTrue(rv.claim(stale, stale.video, force=False))


class RenderOne(unittest.TestCase):
    def fake(self, transcript, audio_names, video_rows=()):
        files = [{"kind": "transcript", "name": "transcript.json", "cdn_url": "t"},
                 {"kind": "shot", "name": "a.png", "cdn_url": "s"}]
        files += [{"kind": "audio", "name": n, "cdn_url": n} for n in audio_names]
        files += list(video_rows)
        contents = {"transcript.json": json.dumps(transcript).encode(), "a.png": b"png", **{n: b"a" for n in audio_names}}
        return FakeReeldesk({"id": "v1", "format": "both", "shots_context": "ctx"}, files, contents)

    def fake_render(self, project, formats):
        outputs = []
        for fmt in formats:
            path = rv.output_path(project, fmt)
            path.write_bytes(b"mp4")
            outputs.append((fmt, path))
        return outputs

    def test_happy_path_claims_renders_uploads_and_notes(self):
        transcript = [{"screenshot": "a.png", "narration": "Hello.", "audio": "audio-1.mp3"}]
        rd = self.fake(transcript, ["audio-1.mp3"])
        with mock.patch.object(rv, "render_formats", side_effect=self.fake_render):
            self.assertTrue(rv.render_video(rd, "v1"))
        self.assertEqual([u.get("render_status") for u in rd.updates], ["rendering", "done"])
        self.assertEqual([r["name"] for r in rd.added], ["v1-shorts.mp4", "v1-youtube.mp4"])
        self.assertEqual(rd.updates[-1]["video_content_id"], "new-v1-youtube.mp4")
        self.assertIn("Rendered shorts and youtube", rd.notes[-1])

    def test_a_re_render_reuses_the_row_and_removes_the_old_file(self):
        transcript = [{"screenshot": "a.png", "narration": "Hello.", "audio": "audio-1.mp3"}]
        old = {"kind": "video", "id": "f1", "name": "v1-youtube.mp4", "content_id": "old-c", "cdn_url": "o"}
        rd = self.fake(transcript, ["audio-1.mp3"], [old])
        with mock.patch.object(rv, "render_formats", side_effect=self.fake_render):
            rv.render_video(rd, "v1")
        self.assertEqual([f for f, _ in rd.updated_files], ["f1"])
        self.assertEqual(rd.deleted, ["old-c"])
        self.assertEqual([r["name"] for r in rd.added], ["v1-shorts.mp4"])

    def test_missing_audio_fails_with_the_beat_numbers_and_renders_nothing(self):
        transcript = [{"screenshot": "a.png", "narration": "Hello."}, {"screenshot": "a.png", "narration": "Again."}]
        rd = self.fake(transcript, [])
        with mock.patch.object(rv, "render_formats") as render:
            self.assertFalse(rv.render_video(rd, "v1"))
            render.assert_not_called()
        self.assertEqual(rd.video["render_status"], "failed")
        self.assertIn("beat(s) 1, 2", rd.video["error_message"])
        self.assertTrue(rd.notes[-1].startswith("Render failed:"))

    def test_a_platform_error_marks_the_video_failed(self):
        rd = self.fake([{"screenshot": "a.png", "narration": "Hello.", "audio": "audio-1.mp3"}], ["audio-1.mp3"])
        with mock.patch.object(rv, "render_formats", side_effect=ReeldeskError("Upload failed (500).")):
            self.assertFalse(rv.render_video(rd, "v1"))
        self.assertEqual(rd.video["render_status"], "failed")


if __name__ == "__main__":
    unittest.main()
