"""assemble.py with --engine saved, for real: a tiny project, real ffmpeg, no TTS and no network.
Skipped when ffmpeg is not installed."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import render_video as rv  # noqa: E402

LINES = ["The first beat.", "The second beat, a little longer."]


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is not installed")
class SavedAudioRender(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project = Path(self.tmp.name) / "v1"
        self.project.mkdir()
        transcript = []
        for i, line in enumerate(LINES, start=1):
            Image.new("RGB", (1600, 1000), (240, 240 - i * 40, 240)).save(self.project / f"beat-{i}.png")
            transcript.append({"screenshot": f"beat-{i}.png",
                               "highlights": [{"x": 200, "y": 200, "width": 600, "height": 300, "narration": line}]})
        (self.project / "transcript.json").write_text(json.dumps(transcript))

    def tearDown(self):
        self.tmp.cleanup()

    def tone(self, line, seconds=1.5):
        target = self.project / rv.upload_identity_name(line, ".wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                        str(target)], check=True)

    def test_renders_every_format_from_saved_audio_only(self):
        for line in LINES:
            self.tone(line)
        outputs = rv.render_formats(self.project, ["youtube", "shorts"])
        self.assertEqual([fmt for fmt, _ in outputs], ["youtube", "shorts"])
        for fmt, path in outputs:
            self.assertTrue(path.exists(), fmt)
            duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                             "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout)
            self.assertGreater(duration, 2.5, fmt)

    def test_a_line_without_saved_audio_stops_the_render(self):
        self.tone(LINES[0])
        with self.assertRaises(rv.RenderError) as err:
            rv.render_formats(self.project, ["youtube"])
        self.assertIn("No saved narration audio", str(err.exception))
        self.assertNotIn(LINES[1], str(err.exception))


if __name__ == "__main__":
    unittest.main()
