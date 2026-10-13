"""Shared fixtures: synthetic assets (no media files needed for the core tests)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from autocut.core.project import Asset, Project  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def pytest_configure(config):
    # isolate app data/cache from the developer's real one
    import tempfile
    os.environ.setdefault("AUTOCUT_HOME", tempfile.mkdtemp(prefix="autocut-test-home-"))
    try:       # QSettings would otherwise write the developer's real settings (the registry on Windows)
        from PySide6.QtCore import QSettings
        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, os.path.join(os.environ["AUTOCUT_HOME"], "settings"))
    except Exception:                                    # pragma: no cover - PySide6 missing
        pass


def pytest_collection_modifyitems(config, items):
    if HAVE_FFMPEG:
        return
    skip = pytest.mark.skip(reason="ffmpeg/ffprobe not on PATH")
    for it in items:
        if "ffmpeg" in it.keywords:
            it.add_marker(skip)


def fake_asset(aid: str = "a1", duration: float = 20.0, w: int = 1920, h: int = 1080, audio: bool = True,
               name: str | None = None) -> Asset:
    return Asset(id=aid, path=f"/nonexistent/{aid}.mp4", name=name or f"{aid}.mp4", kind="video",
                 duration=duration, width=w, height=h, fps=30.0, has_video=True, has_audio=audio,
                 audio_channels=2 if audio else 0, sample_rate=48000 if audio else 0, fingerprint=aid)


@pytest.fixture
def proj() -> Project:
    p = Project.new("test")
    p.assets["a1"] = fake_asset("a1", 20.0)
    p.assets["a2"] = fake_asset("a2", 10.0)
    p.assets["m1"] = Asset(id="m1", path="/nonexistent/m1.mp3", name="m1.mp3", kind="audio", duration=60.0,
                           has_video=False, has_audio=True, audio_channels=2, sample_rate=44100)
    return p


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory) -> Path:
    """Tiny synthetic clips made with ffmpeg lavfi (colour bars / tones), session-wide."""
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("media")

    def make(name: str, spec: list[str]) -> None:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *spec, str(d / name)], check=True)

    # landscape 16:9, 8 s, a moving red square on gradient + 440 Hz tone
    make("land.mp4", ["-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=8",
                      "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=8",
                      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "15", "-c:a", "aac", "-shortest"])
    # portrait 9:16, 5 s, solid blue + 880 Hz
    make("port.mp4", ["-f", "lavfi", "-i", "color=c=blue:size=720x1280:rate=30:duration=5",
                      "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=44100:duration=5",
                      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"])
    # landscape without audio, 4 s, solid green
    make("silent.mp4", ["-f", "lavfi", "-i", "color=c=green:size=1280x720:rate=25:duration=4",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p"])
    # music-ish audio only
    make("music.wav", ["-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100:duration=20"])
    return d


# --------------------------------------------------------------------------- GUI helpers (shared)
@pytest.fixture(scope="session")
def qapp():
    """One offscreen QApplication with the dark theme for every GUI test (do not create your own)."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from autocut.gui.theme import apply_theme
    apply_theme(app)
    return app


@pytest.fixture
def real_proj(media_dir) -> Project:
    """A project whose assets were probed from the synthetic ``media_dir`` files (land, port, silent, music)."""
    from autocut.media.probe import probe_asset
    p = Project.new("real")
    for name in ("land.mp4", "port.mp4", "silent.mp4", "music.wav"):
        a = probe_asset(media_dir / name)
        p.assets[a.id] = a
    return p
