"""Command line tests: every ``autocut`` subcommand run for real on small synthetic ffmpeg media, both in-process
(``autocut.cli.main``) and as a child process (``python -m autocut``).  Paths contain spaces and Chinese characters on
purpose.  Nothing here opens a window (``studio`` is only asked for ``--help``)."""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from autocut import cli
from autocut.core import ops
from autocut.core.project import Project, validate_sequence
from autocut.media.ffmpeg import ffmpeg_info
from autocut.media.probe import probe_asset

SRC = Path(__file__).resolve().parents[1] / "src"
REPO = Path(__file__).resolve().parents[1]
ffmpeg = pytest.mark.ffmpeg                         # skipped by conftest when ffmpeg is not installed
posix_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc and POSIX signals")


# ------------------------------------------------------------------------------------------------ helpers
def _ff(*args: str) -> None:
    subprocess.run([ffmpeg_info().ffmpeg, "-y", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> Path:
    """Synthetic sources.  ``長 片 a.mp4``: 24 s, two scenes, 440 Hz bursts (2 s on / 1 s off); ``short 12.mp4`` 12 s;
    ``tiny 6.mp4`` 6 s; ``毛片 clips``: landscape / portrait / silent clips for assemble."""
    pytest.importorskip("numpy")
    try:
        ffmpeg_info()
    except Exception:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("素材 media")
    tone = "aevalsrc='0.5*sin(2*PI*440*t)*lt(mod(t,3),2)':s=16000:d={d}"
    _ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=12", "-f", "lavfi", "-i", "smptebars=s=640x360:r=25:d=12",
        "-f", "lavfi", "-i", tone.format(d=24), "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p[v]",
        "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
        str(d / "長 片 a.mp4"))
    _ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=12", "-f", "lavfi", "-i", tone.format(d=12),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "short 12.mp4"))
    _ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=6", "-f", "lavfi", "-i", tone.format(d=6),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "tiny 6.mp4"))
    (d / "毛片 clips").mkdir()
    _ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=8", "-f", "lavfi", "-i", "sine=f=440:sample_rate=44100:d=8",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(d / "毛片 clips" / "clip 1 land.mp4"))
    _ff("-f", "lavfi", "-i", "testsrc2=s=360x640:r=25:d=6", "-f", "lavfi", "-i", "sine=f=660:sample_rate=44100:d=6",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(d / "毛片 clips" / "clip 2 port.mp4"))
    _ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=5", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(d / "毛片 clips" / "clip 3 silent.mp4"))
    _ff("-f", "lavfi", "-i", "sine=f=220:sample_rate=44100:d=10", str(d / "music.wav"))
    (d / "bad").mkdir()
    (d / "bad" / "corrupt.mp4").write_bytes(os.urandom(4000))
    (d / "bad" / "zero.mp4").write_bytes(b"")
    (d / "bad" / "notes.txt").write_text("not media", encoding="utf-8")
    (d / "subs.srt").write_text("1\n00:00:01,000 --> 00:00:02,800\n第一行字幕\n\n2\n00:00:03,500 --> 00:00:05,500\n第二行字幕 hello\n"
                               "\n3\n00:00:06,000 --> 00:00:09,000\n第三行字幕測試\n", encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def child_home(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("child home")


def env_for(home: Path, **extra: str) -> dict:
    env = dict(os.environ)
    env.pop("AUTOCUT_DEBUG", None)
    env.pop("AUTOCUT_FFMPEG_DIR", None)
    env.update(AUTOCUT_HOME=str(home), PYTHONPATH=str(SRC) + os.pathsep + env.get("PYTHONPATH", ""),
               PYTHONIOENCODING="utf-8", QT_QPA_PLATFORM="offscreen")
    env.update(extra)
    return env


def run_proc(args, home: Path, *, cwd=None, env=None, timeout: float = 300) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "autocut", *[str(a) for a in args]], cwd=cwd, env=env or env_for(home),
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def run_main(capsys, *args) -> tuple[int, str, str]:
    capsys.readouterr()
    try:
        rc = cli.main([str(a) for a in args])
    except SystemExit as e:                                     # argparse: --help (0) / bad arguments (2)
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def assert_honest(text: str) -> None:
    """The machine may say 技術檢查, never that a human review happened or passed."""
    assert not re.findall(r"(?<!不代表)審片通過", text), text
    assert "已通過審片" not in text and "人工審片：通過" not in text and "人工審片通過" not in text, text
    assert "traceback" not in text.lower(), text


def children_of(pid: int) -> list[tuple[int, str]]:
    out = []
    if not os.path.isdir("/proc"):
        return out
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                st = Path(f"/proc/{d}/stat").read_text()
                if int(st[st.rindex(")") + 2:].split()[1]) == pid:
                    out.append((int(d), st[st.index("(") + 1:st.rindex(")")]))
            except (OSError, ValueError, IndexError):
                continue
    return out


def processes_mentioning(token: str) -> list[int]:
    found = []
    for d in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if d.isdigit() and int(d) != os.getpid():
            try:
                if token in Path(f"/proc/{d}/cmdline").read_bytes().decode("utf-8", "replace"):
                    found.append(int(d))
            except OSError:
                continue
    return found


def make_project(path: Path, media: Path, *, cue: bool = True, seconds: float = 3.0) -> Project:
    """A tiny project written through the API: one 640x360 sequence, one clip of ``clip 1 land.mp4`` (+ one caption)."""
    proj = Project.new("測試 專案")
    seq = proj.active_sequence()
    asset = ops.add_asset(proj, probe_asset(media / "毛片 clips" / "clip 1 land.mp4"))
    ops.set_sequence_format(proj, seq, 640, 360, 25.0)
    ops.place_clip(proj, seq, asset.id, seq.video_tracks[0].id, start=0.0, src_in=0.0, src_out=seconds)
    if cue:
        ops.add_cue(proj, seq, 0.5, 2.0, "你好 字幕")
    proj.save(path)
    return proj


def duration_of(path: Path) -> float:
    return probe_asset(path).duration


# ---------------------------------------------------------------------------------------- parser / help / usage
def test_help_is_traditional_chinese(capsys):
    rc, out, _ = run_main(capsys, "--help")
    assert rc == 0 and "用法：" in out and "結束代碼" in out
    for name in ("doctor", "probe", "qa", "render", "shorts", "assemble", "studio"):
        assert name in out
        rc, sub, _ = run_main(capsys, name, "--help")
        assert rc == 0 and "用法：" in sub and has_cjk(sub), name
    rc, out, _ = run_main(capsys, "--version")
    assert rc == 0 and "AutoCut Studio" in out


def test_studio_help_never_loads_qt(child_home):
    """``studio --help`` exits 0 without importing PySide6 (so it cannot open a window)."""
    code = ("import sys\nfrom autocut.cli import main\n"
            "try:\n    main(['studio', '--help'])\nexcept SystemExit as e:\n    print('EXIT', e.code)\n"
            "print('QT_LOADED', any(m.startswith('PySide6') for m in sys.modules))\n")
    p = subprocess.run([sys.executable, "-c", code], env=env_for(child_home), capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    assert "EXIT 0" in p.stdout and "QT_LOADED False" in p.stdout, p.stdout + p.stderr
    p = run_proc(["studio", "--help"], child_home, timeout=60)
    assert p.returncode == 0 and "桌面" in p.stdout


def test_argument_errors_are_chinese(capsys, media):
    src = media / "長 片 a.mp4"
    for args in (["shorts"], ["nonsense"], [], ["shorts", src, "-o", "x", "--count", "0"],
                 ["shorts", src, "-o", "x", "--count", "abc"], ["shorts", src, "-o", "x", "--platform", "nope"],
                 ["shorts", src, "-o", "x", "--target", "-5"], ["qa", src, "--size", "10"], ["render", "p.acproj"]):
        rc, out, err = run_main(capsys, *args)
        assert rc == 2, args
        assert "參數錯誤" in err and has_cjk(err), (args, err)
        assert not re.search(r"\b(the following arguments|invalid choice|expected one argument)\b", err), err


def test_argument_message_translation():
    z = cli._zh_arg_error
    assert z("the following arguments are required: input, -o/--output") == "缺少必要的參數：input, -o/--output"
    assert z("argument --count: invalid int value: 'x'") == "參數 --count：'x' 不是有效的整數"
    assert "不接受" in z("argument --platform: invalid choice: 'x' (choose from 'a', 'b')")
    assert z("unrecognized arguments: --foo") == "無法辨識的參數：--foo"
    assert z("something new") == "something new"


def test_safe_name_is_windows_legal():
    s = cli._safe_name
    assert s('a<b>c:d"e/f\\g|h?i*j') == "a-b-c-d-e-f-g-h-i-j"
    assert s("短影音 1｜00:03–00:29") == "短影音_1｜00-03–00-29"
    assert s("  ..name. ") == "name" and s("") == "output" and s("...") == "output"
    assert s("CON") == "_CON" and s("nul.txt") == "_nul.txt" and s("COM1") == "_COM1"
    assert len(s("長" * 500)) <= 100


# ------------------------------------------------------------------------------------------------------ doctor
@ffmpeg
def test_doctor_in_process_and_subprocess(capsys, child_home):
    rc, out, err = run_main(capsys, "doctor")
    assert rc == 0, out + err
    for needle in ("AutoCut Studio", "ffmpeg", "libx264", "OpenCV", "numpy", "人臉偵測模型"):
        assert needle in out, needle
    p = run_proc(["doctor", "--hw"], child_home, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "硬體編碼器" in p.stdout and "libx264" in p.stdout
    assert "Traceback" not in p.stderr


# ------------------------------------------------------------------------------------------------------- probe
@ffmpeg
def test_probe_text_json_and_folders(media, capsys, child_home):
    src = media / "長 片 a.mp4"
    rc, out, err = run_main(capsys, "probe", src)
    assert rc == 0 and src.name in out and "640x360" in out and "0:24" in out and "h264" in out, out
    rc, out, err = run_main(capsys, "probe", "--json", src, media / "short 12.mp4")
    rows = [json.loads(line) for line in out.splitlines()]
    assert rc == 0 and [r["name"] for r in rows] == ["長 片 a.mp4", "short 12.mp4"]
    assert rows[0]["width"] == 640 and rows[0]["height"] == 360 and abs(rows[0]["duration"] - 24.0) < 0.2
    rc, out, err = run_main(capsys, "probe", media / "毛片 clips")                       # a folder: every file in it
    assert rc == 0 and len(out.strip().splitlines()) == 3 and "clip 2 port.mp4" in out
    p = run_proc(["probe", "--json", src], child_home, timeout=60)                      # the real entry point, UTF-8 JSON
    assert p.returncode == 0
    row = json.loads(p.stdout)
    api = probe_asset(src)
    assert (row["width"], row["height"], row["fps"], row["has_audio"]) == (api.width, api.height, api.fps, api.has_audio)
    assert row["path"] == str(src)


@ffmpeg
def test_probe_problems_are_reported_in_chinese(media, capsys):
    bad = media / "bad"
    cases = [(media / "nope.mp4", "找不到"), (bad / "notes.txt", "不支援"), (bad / "zero.mp4", "空的"),
             (bad / "corrupt.mp4", "無法讀取"), (media / "bad" / "notes.txt", "notes.txt"),
             (media / "mp3-folder-that-does-not-exist", "找不到")]
    for path, needle in cases:
        rc, out, err = run_main(capsys, "probe", path)
        assert rc == 1 and needle in err and has_cjk(err), (path, err)
        assert out == "" and "Traceback" not in err
    # one bad file must not hide the good ones, and --json keeps stdout pure JSON lines
    rc, out, err = run_main(capsys, "probe", "--json", media / "nope.mp4", media / "short 12.mp4", bad / "corrupt.mp4")
    assert rc == 1 and "找不到" in err and "corrupt.mp4" in err
    assert [json.loads(line)["name"] for line in out.splitlines()] == ["short 12.mp4"]
    (media / "empty folder").mkdir(exist_ok=True)
    rc, out, err = run_main(capsys, "probe", media / "empty folder")
    assert rc == 1 and "資料夾裡沒有可用的" in err


# --------------------------------------------------------------------------------------------------------- qa
@ffmpeg
def test_qa_reports_technical_check_only(media, capsys, child_home):
    src = media / "tiny 6.mp4"
    rc, out, err = run_main(capsys, "qa", src, "--duration", "6", "--fps", "25", "--size", "640x360")
    assert rc == 0, out + err
    assert out.splitlines()[0].startswith("技術檢查：") and "人工審片：尚未" in out
    assert "尺寸" in out and "影格率" in out
    assert_honest(out)
    rc, out, _ = run_main(capsys, "qa", src, "--duration", "30")                      # wrong length -> fail
    assert rc == 1 and "技術檢查：未通過" in out and "✘ 長度" in out
    rc, out, _ = run_main(capsys, "qa", media / "nope.mp4")
    assert rc == 1 and "技術檢查：未通過" in out and "找不到輸出檔" in out
    rc, out, _ = run_main(capsys, "qa", media / "bad" / "corrupt.mp4")
    assert rc == 1 and "技術檢查：未通過" in out
    p = run_proc(["qa", src], child_home, timeout=120)
    assert p.returncode == 0 and "技術檢查：" in p.stdout and "人工審片：尚未" in p.stdout
    assert_honest(p.stdout)


# ------------------------------------------------------------------------------------------------------ render
@ffmpeg
def test_render_project_with_relative_paths_in_process(media, tmp_path, monkeypatch, capsys):
    """Regression: a relative ``-o`` used to be resolved against the render cache folder and fail."""
    work = tmp_path / "工作 資料夾"
    work.mkdir()
    make_project(work / "我的 專案.acproj", media)
    monkeypatch.chdir(work)
    rc, out, err = run_main(capsys, "render", "我的 專案.acproj", "-o", "輸出 資料夾/成品 1.mp4", "--preset", "draft_fast",
                            "--srt")
    assert rc == 0, out + err
    mp4 = work / "輸出 資料夾" / "成品 1.mp4"
    assert mp4.is_file() and (work / "輸出 資料夾" / "成品 1.srt").is_file()
    assert abs(duration_of(mp4) - 3.0) < 0.3
    assert str(mp4) in out and "技術檢查：" in out and "人工審片：尚未" in out
    assert "你好 字幕" in (work / "輸出 資料夾" / "成品 1.srt").read_text(encoding="utf-8")
    assert_honest(out + err)
    assert not list((work / "輸出 資料夾").glob("*.partial*")), "no half-written output may stay behind"


@ffmpeg
def test_render_in_a_child_process_bare_filename(media, tmp_path, child_home):
    work = tmp_path / "中文 工作"
    work.mkdir()
    make_project(work / "p.acproj", media, cue=False)
    p = run_proc(["render", "p.acproj", "-o", "out.mp4", "--preset", "draft_fast", "--no-captions"], child_home,
                 cwd=work, timeout=240)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (work / "out.mp4").is_file() and abs(duration_of(work / "out.mp4") - 3.0) < 0.3
    assert "技術檢查：" in p.stdout and "人工審片：尚未" in p.stdout
    assert_honest(p.stdout + p.stderr)


@ffmpeg
def test_render_output_naming_and_sequence_choice(media, tmp_path, capsys):
    proj_file = tmp_path / "兩個 序列.acproj"
    proj = make_project(proj_file, media, cue=False, seconds=2.0)
    second = ops.create_sequence(proj, "第二 序列", 640, 360, 25.0, activate=False)
    ops.place_clip(proj, second, next(iter(proj.assets)), second.video_tracks[0].id, start=0.0, src_in=0.0, src_out=1.0)
    proj.save(proj_file)
    outdir = tmp_path / "資料夾 輸出"
    outdir.mkdir()
    rc, out, err = run_main(capsys, "render", proj_file, "-o", outdir, "--sequence", "第二 序列", "--preset", "draft_fast")
    assert rc == 0, out + err
    produced = outdir / "兩個_序列.mp4"                       # -o is a folder: <project name>.mp4 inside it
    assert produced.is_file() and abs(duration_of(produced) - 1.0) < 0.3
    rc, out, err = run_main(capsys, "render", proj_file, "-o", tmp_path / "no extension", "--sequence", second.id,
                            "--preset", "draft_fast")
    assert rc == 0 and (tmp_path / "no extension.mp4").is_file()                    # a missing extension becomes .mp4
    rc, out, err = run_main(capsys, "render", proj_file, "-o", tmp_path / "x.mp4", "--sequence", "沒有這個序列")
    assert rc == 1 and "沒有序列" in err and second.id in err and "第二 序列" in err


@ffmpeg
def test_render_refuses_bad_input_politely(media, tmp_path, capsys):
    rc, out, err = run_main(capsys, "render", tmp_path / "nope.acproj", "-o", tmp_path / "x.mp4")
    assert rc == 1 and "找不到專案檔" in err and "Traceback" not in err
    garbage = tmp_path / "garbage.acproj"
    garbage.write_text("{not json", encoding="utf-8")
    rc, out, err = run_main(capsys, "render", garbage, "-o", tmp_path / "x.mp4")
    assert rc == 1 and "無法開啟專案檔" in err and "Traceback" not in err
    other = tmp_path / "other.acproj"
    other.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    rc, out, err = run_main(capsys, "render", other, "-o", tmp_path / "x.mp4")
    assert rc == 1 and "無法開啟專案檔" in err
    empty = tmp_path / "empty.acproj"
    proj = Project.new("空", with_sequence=False)
    proj.save(empty)
    rc, out, err = run_main(capsys, "render", empty, "-o", tmp_path / "x.mp4")
    assert rc == 1 and "沒有任何序列" in err
    # the output may never be one of the project's own source files (it would be overwritten)
    src_copy = tmp_path / "素材 copy.mp4"
    shutil.copy(media / "毛片 clips" / "clip 1 land.mp4", src_copy)
    proj = Project.new("p")
    seq = proj.active_sequence()
    asset = ops.add_asset(proj, probe_asset(src_copy))
    ops.place_clip(proj, seq, asset.id, seq.video_tracks[0].id, start=0.0, src_in=0.0, src_out=2.0)
    proj.save(tmp_path / "p.acproj")
    before = src_copy.read_bytes()
    rc, out, err = run_main(capsys, "render", tmp_path / "p.acproj", "-o", src_copy)
    assert rc == 1 and "素材" in err and src_copy.read_bytes() == before
    rc, out, err = run_main(capsys, "render", tmp_path / "p.acproj", "-o", tmp_path / "x.mp4", "--preset", "nope")
    assert rc == 2 and "參數錯誤" in err


# ------------------------------------------------------------------------------------------------------ shorts
def _plan(outdir: Path, stem: str) -> dict:
    return json.loads((outdir / f"{stem}.plan.json").read_text(encoding="utf-8"))


@ffmpeg
def test_shorts_writes_plan_receipt_and_project(media, tmp_path, capsys):
    src = media / "長 片 a.mp4"
    outdir = tmp_path / "輸出 資料夾" / "巢狀 nested" / "shorts"           # created on demand, nested
    rc, out, err = run_main(capsys, "shorts", src, "-o", outdir, "--count", "1")
    assert rc == 0, out + err
    stem = "長_片_a"
    for suffix in (".acproj", ".plan.json", ".receipt.json"):
        assert (outdir / f"{stem}{suffix}").is_file(), suffix
    assert "提出 1 支短影音" in out and "計畫指紋：" in out and "尚未輸出影片" in out
    assert "分析" in err                                                        # progress goes to stderr
    assert_honest(out + err)
    proj = Project.load(outdir / f"{stem}.acproj")
    seqs = proj.sequences
    assert len(seqs) == 1 and (seqs[0].width, seqs[0].height) == (1080, 1920)
    assert not [i for i in validate_sequence(proj, seqs[0]) if i.level == "error"]
    for _, clip in seqs[0].all_clips():
        assert 0.0 <= clip.src_in < clip.src_out <= 24.05
    run = proj.meta["ai_runs"][0]
    assert run["human_review"] == "pending" and run["technical_qa"] == "not_run"
    receipt = json.loads((outdir / f"{stem}.receipt.json").read_text(encoding="utf-8"))
    assert receipt["accepted"] is True and receipt["plan_sha256"] == _plan(outdir, stem)["plan_sha256"]
    assert not list(outdir.glob("*.mp4"))                                       # nothing rendered without --render
    # the finding list is shown in Chinese, with the human-review reminder
    assert "[資訊] human.pending" in out and "尚未經人工審片" in out


@ffmpeg
def test_shorts_render_child_process_default_preset(media, tmp_path, child_home):
    """The real entry point with a RELATIVE output folder and the default 1080x1920 preset."""
    outdir_name = "輸出 relative"
    p = run_proc(["shorts", media / "tiny 6.mp4", "-o", outdir_name, "--count", "1", "--render"], child_home, cwd=tmp_path,
                 timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    outdir = tmp_path / outdir_name
    mp4s = list(outdir.glob("*.mp4"))
    assert len(mp4s) == 1, list(outdir.iterdir())
    info = probe_asset(mp4s[0])
    assert (info.width, info.height) == (1080, 1920) and info.has_audio
    assert not list(outdir.glob("*.partial*"))
    assert re.search(r"技術檢查：通過", p.stdout) and "人工審片：尚未" in p.stdout and "不代表審片通過" in p.stdout
    assert_honest(p.stdout + p.stderr)
    # the project file remembers the machine's check, and the human review is still pending (only a person may change it)
    run = Project.load(outdir / "tiny_6.acproj").meta["ai_runs"][0]
    assert run["technical_qa"] in ("pass", "warn") and run["human_review"] == "pending"
    assert "尚未人工審片" in run["technical_qa_summary"] and run["outcomes"] == []


@ffmpeg
def test_a_technical_check_that_did_not_finish_is_not_a_pass_in_the_exit_code_or_the_project(media, tmp_path, capsys,
                                                                                            monkeypatch):
    from autocut.core.plan import sequence_sha256
    from autocut.render import qa as qa_mod
    real, calls = qa_mod._run_ffmpeg, []

    def flaky(cmd, timeout=3600):                                              # the decode pass works, the detectors do not
        calls.append(cmd)
        return real(cmd, timeout) if len(calls) % 2 == 1 else (None, "", "ffmpeg 超過 60 分鐘還沒結束，已放棄")

    monkeypatch.setattr(qa_mod, "_run_ffmpeg", flaky)
    outdir = tmp_path / "沒跑完"
    rc, out, err = run_main(capsys, "assemble", media / "毛片 clips", "-o", outdir, "--target", "10", "--render",
                            "--preset", "draft_fast", "--transition", "cut")
    assert rc == 1, out + err                                                  # EXIT_FAIL: do not publish on this "result"
    assert "沒有跑完" in out and "技術檢查：通過" not in out
    assert_honest(out + err)
    proj = Project.load(outdir / "assembled.acproj")
    run = proj.meta["ai_runs"][0]
    seq = proj.sequences[0]
    assert run["technical_qa"] == "not_run" and run["human_review"] == "pending"
    slot = run["by_sequence"][seq.id]
    assert slot["technical_qa"] == "not_run" and slot["technical_qa_sha"] == sequence_sha256(seq)


@ffmpeg
def test_shorts_strict_turns_band_violations_into_errors(media, tmp_path, capsys):
    src = media / "short 12.mp4"                                                # a 10 s short is < the 13 s platform band
    loose, strict = tmp_path / "loose", tmp_path / "strict"
    rc, out, err = run_main(capsys, "shorts", src, "-o", loose, "--count", "1")
    assert rc == 0 and "[提醒] shorts.band" in out and (loose / "short_12.acproj").is_file()
    rc, out, err = run_main(capsys, "shorts", src, "-o", strict, "--count", "1", "--strict")
    assert rc == 2, out + err
    assert "[錯誤] shorts.band" in out and "審核未通過，沒有套用" in out
    assert (strict / "short_12.plan.json").is_file() and (strict / "short_12.receipt.json").is_file()
    assert not (strict / "short_12.acproj").exists()                              # fail-closed: nothing applied
    assert json.loads((strict / "short_12.receipt.json").read_text(encoding="utf-8"))["accepted"] is False


@ffmpeg
def test_shorts_caption_modes(media, tmp_path, capsys):
    src, srt = media / "short 12.mp4", media / "subs.srt"

    def cues(outdir: Path) -> list[str]:
        return [c["text"] for c in _plan(outdir, "short_12")["sequences"][0].get("cues", [])]

    rc, out, err = run_main(capsys, "shorts", src, "-o", tmp_path / "off", "--count", "1", "--captions", "off",
                            "--no-tighten")
    assert rc == 0 and cues(tmp_path / "off") == [] and "caption.rate" not in out
    rc, out, err = run_main(capsys, "shorts", src, "-o", tmp_path / "file", "--count", "1", "--captions", "file",
                            "--caption-file", srt, "--no-tighten")
    assert rc == 0, out + err
    text = "".join(cues(tmp_path / "file"))
    assert "第一行字幕" in text and "第二行字幕 hello" in text and "第三行字幕測試" in text
    rc, out, err = run_main(capsys, "shorts", src, "-o", tmp_path / "implied", "--count", "1", "--caption-file", srt,
                            "--no-tighten")                                       # --caption-file alone means 'file'
    assert rc == 0 and cues(tmp_path / "implied") == cues(tmp_path / "file")
    # auto without speech recognition installed degrades to "no captions" and SAYS so (no invented captions)
    rc, out, err = run_main(capsys, "shorts", src, "-o", tmp_path / "auto", "--count", "1", "--no-tighten")
    assert rc == 0 and cues(tmp_path / "auto") == []
    assert "說明：尚未安裝語音辨識" in out and "說明：沒有逐字稿，所以沒有加字幕" in out
    # bad combinations
    rc, out, err = run_main(capsys, "shorts", src, "-o", tmp_path / "x", "--captions", "file")
    assert rc == 1 and "--caption-file" in err
    rc, out, err = run_main(capsys, "shorts", src, "-o", tmp_path / "x", "--caption-file", tmp_path / "nope.srt")
    assert rc == 1 and "找不到字幕檔" in err
    rc, out, err = run_main(capsys, "shorts", src, "-o", tmp_path / "x", "--music", tmp_path / "nope.mp3")
    assert rc == 1 and "找不到配樂檔" in err


@ffmpeg
def test_shorts_plan_hash_is_reproducible(media, tmp_path, capsys, child_home):
    """Same material + same parameters -> same plan hash: cold cache vs warm cache vs another process."""
    src = media / "長 片 a.mp4"

    def fingerprint(text: str) -> str:
        return re.search(r"計畫指紋：([0-9a-f]{12})", text).group(1)

    rc, out1, _ = run_main(capsys, "shorts", src, "-o", tmp_path / "run1", "--count", "2", "--no-tighten")
    assert rc == 0
    rc, out2, _ = run_main(capsys, "shorts", src, "-o", tmp_path / "run2", "--count", "2", "--no-tighten")    # warm cache
    p = run_proc(["shorts", src, "-o", tmp_path / "run3", "--count", "2", "--no-tighten"], child_home, timeout=120)
    assert rc == 0 and p.returncode == 0, p.stdout + p.stderr
    h1, h2, h3 = fingerprint(out1), fingerprint(out2), fingerprint(p.stdout)
    assert h1 == h2 == h3
    plans = [_plan(tmp_path / f"run{i}", "長_片_a") for i in (1, 2, 3)]
    assert plans[0]["plan_sha256"].startswith(h1)
    assert plans[0]["plan_sha256"] == plans[1]["plan_sha256"] == plans[2]["plan_sha256"]
    # a different parameter really changes it (the fingerprint is not a constant)
    rc, out4, _ = run_main(capsys, "shorts", src, "-o", tmp_path / "run4", "--count", "1", "--no-tighten")
    assert fingerprint(out4) != h1


@ffmpeg
def test_shorts_input_problems_are_reported_in_chinese(media, tmp_path, capsys):
    bad = media / "bad"
    out_ok = tmp_path / "o"
    cases = [(media / "nope.mp4", "沒有任何可用", "找不到"), (bad / "notes.txt", "沒有任何可用", "不支援"),
             (bad / "zero.mp4", "沒有任何可用", "空的"), (bad / "corrupt.mp4", "沒有任何可用", "無法讀取"),
             (media / "毛片 clips", "一次只處理一支影片", ""), (media, "一次只處理一支影片", "")]
    for path, needle, detail in cases:
        rc, out, err = run_main(capsys, "shorts", path, "-o", out_ok, "--count", "1")
        assert rc == 1 and needle in err and detail in err and has_cjk(err), (path, err)
        assert "Traceback" not in err
    assert not list(out_ok.glob("*.acproj"))
    rc, out, err = run_main(capsys, "shorts", media / "music.wav", "-o", out_ok)             # audio only: no picture to reframe
    assert rc == 1 and "沒有影像" in err and "Traceback" not in err
    tiny = tmp_path / "tiny 3s.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tiny))
    rc, out, err = run_main(capsys, "shorts", tiny, "-o", out_ok)
    assert rc == 1 and "太短" in err
    blocker = tmp_path / "a file"
    blocker.write_text("x", encoding="utf-8")
    rc, out, err = run_main(capsys, "shorts", media / "short 12.mp4", "-o", blocker)
    assert rc == 1 and has_cjk(err) and "Traceback" not in err


# ----------------------------------------------------------------------------------------------------- assemble
@ffmpeg
def test_assemble_folder_plan_hash_and_project(media, tmp_path, capsys):
    clips = media / "毛片 clips"
    rc, out1, err = run_main(capsys, "assemble", clips, "-o", tmp_path / "長片 一", "--target", "12")
    assert rc == 0, out1 + err
    assert "提出 1 支長片" in out1 and "尚未輸出影片" in out1
    proj = Project.load(tmp_path / "長片 一" / "assembled.acproj")
    seq = proj.sequences[0]
    assert len(proj.sequences) == 1 and len(proj.assets) == 3
    assert not [i for i in validate_sequence(proj, seq) if i.level == "error"]
    assert 5.0 <= seq.duration <= 14.0
    assert proj.meta["ai_runs"][0]["human_review"] == "pending"
    rc, out2, _ = run_main(capsys, "assemble", clips / "clip 1 land.mp4", clips / "clip 2 port.mp4",
                           clips / "clip 3 silent.mp4", "-o", tmp_path / "長片 二", "--target", "12")
    assert rc == 0
    fp = lambda t: re.search(r"計畫指紋：([0-9a-f]{12})", t).group(1)                 # noqa: E731
    assert fp(out1) == fp(out2)                                  # a folder and the same files listed -> the same plan
    assert _plan(tmp_path / "長片 一", "assembled")["plan_sha256"] == _plan(tmp_path / "長片 二", "assembled")["plan_sha256"]


@ffmpeg
def test_assemble_render_in_process(media, tmp_path, capsys):
    outdir = tmp_path / "長片 輸出"
    rc, out, err = run_main(capsys, "assemble", media / "毛片 clips", "-o", outdir, "--target", "10", "--render",
                            "--preset", "draft_fast", "--transition", "cut")
    assert rc == 0, out + err
    mp4s = list(outdir.glob("*.mp4"))
    assert len(mp4s) == 1
    proj = Project.load(outdir / "assembled.acproj")
    assert abs(duration_of(mp4s[0]) - proj.sequences[0].duration) < 0.4
    assert "技術檢查：" in out and "人工審片：尚未" in out and "不代表審片通過" in out
    assert_honest(out + err)
    run = proj.meta["ai_runs"][0]
    assert run["technical_qa"] in ("pass", "warn") and run["human_review"] == "pending"


@ffmpeg
def test_assemble_problems_are_reported_in_chinese(media, tmp_path, capsys):
    rc, out, err = run_main(capsys, "assemble", media / "nope.mp4", media / "bad" / "notes.txt", "-o", tmp_path / "o")
    assert rc == 1 and "沒有任何可用" in err and "略過" in err and "Traceback" not in err
    rc, out, err = run_main(capsys, "assemble", media / "bad" / "corrupt.mp4", media / "bad" / "zero.mp4", "-o", tmp_path / "o")
    assert rc == 1 and "沒有任何可用" in err
    rc, out, err = run_main(capsys, "assemble", media / "毛片 clips", "-o", tmp_path / "o", "--music", tmp_path / "nope.mp3")
    assert rc == 1 and "找不到配樂檔" in err
    # a bad file next to good ones is skipped, the rest is assembled
    rc, out, err = run_main(capsys, "assemble", media / "bad" / "corrupt.mp4", media / "毛片 clips" / "clip 1 land.mp4",
                            "-o", tmp_path / "mixed", "--target", "5")
    assert rc == 0 and "略過 corrupt.mp4" in err and (tmp_path / "mixed" / "assembled.acproj").is_file()


# ----------------------------------------------------------------------------------- environment: no / broken ffmpeg
def _no_ffmpeg_env(home: Path, tmp_path: Path, **extra: str) -> dict:
    empty = tmp_path / "empty path dir"
    empty.mkdir(exist_ok=True)
    return env_for(home, PATH=str(empty), **extra)


@pytest.mark.skipif((REPO / "tools" / "ffmpeg").exists(), reason="a bundled ffmpeg is found before PATH")
@pytest.mark.parametrize("command", ["probe", "qa", "shorts", "assemble", "render"])
def test_missing_ffmpeg_is_a_clear_chinese_error(command, tmp_path, child_home):
    target = tmp_path / "x.mp4"
    target.write_bytes(b"not really a video")
    argv = {"probe": ["probe", target], "qa": ["qa", target], "shorts": ["shorts", target, "-o", tmp_path / "o"],
            "assemble": ["assemble", target, "-o", tmp_path / "o"], "render": ["render", tmp_path / "p.acproj", "-o", tmp_path / "o.mp4"]}
    p = run_proc(argv[command], child_home, env=_no_ffmpeg_env(tmp_path / "fresh home", tmp_path), timeout=60)
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
    assert "找不到 ffmpeg" in p.stderr and has_cjk(p.stderr) and "Traceback" not in p.stderr
    assert not (tmp_path / "o").exists()                                         # nothing half-created


@pytest.mark.skipif((REPO / "tools" / "ffmpeg").exists(), reason="a bundled ffmpeg is found before PATH")
def test_doctor_and_help_work_without_ffmpeg(tmp_path, child_home):
    env = _no_ffmpeg_env(tmp_path / "fresh home", tmp_path)
    p = run_proc(["doctor"], child_home, env=env, timeout=60)
    assert p.returncode == 1 and re.search(r"錯誤\s+ffmpeg\s+找不到 ffmpeg", p.stdout), p.stdout + p.stderr
    assert "numpy" in p.stdout and "Traceback" not in p.stderr
    p = run_proc(["shorts", "--help"], child_home, env=env, timeout=60)
    assert p.returncode == 0 and "用法：" in p.stdout


@pytest.mark.skipif(sys.platform.startswith("win"), reason="shell-script stand-ins for ffmpeg")
def test_broken_ffmpeg_is_reported_not_a_traceback(tmp_path, child_home):
    fake = tmp_path / "fake ffmpeg"
    fake.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        (fake / name).write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        (fake / name).chmod(0o755)
    target = tmp_path / "x.mp4"
    target.write_bytes(b"x")
    env = _no_ffmpeg_env(tmp_path / "fresh home", tmp_path, AUTOCUT_FFMPEG_DIR=str(fake))
    p = run_proc(["probe", target], child_home, env=env, timeout=60)
    assert p.returncode == 3 and "ffmpeg 無法執行" in p.stderr and "Traceback" not in p.stderr, p.stderr


@ffmpeg
def test_legacy_code_page_stdout_does_not_crash(media, child_home):
    """A Taiwanese Windows console redirected to a file uses cp950, which cannot encode ✔ ✘: print ? instead of dying."""
    env = env_for(child_home, PYTHONIOENCODING="cp950")
    p = subprocess.run([sys.executable, "-m", "autocut", "qa", str(media / "tiny 6.mp4")], env=env, capture_output=True,
                       timeout=120)
    out, err = p.stdout.decode("cp950", "replace"), p.stderr.decode("cp950", "replace")
    assert p.returncode == 0 and "UnicodeEncodeError" not in err, err
    assert "技術檢查" in out and "人工審片：尚未" in out
    p = subprocess.run([sys.executable, "-m", "autocut", "probe", str(media / "nope.mp4")], env=env, capture_output=True,
                       timeout=60)
    assert p.returncode == 1 and "找不到" in p.stderr.decode("cp950", "replace")      # the ✘ message survives too


# ------------------------------------------------------------------------------------------------------- Ctrl+C
@ffmpeg
@posix_only
def test_ctrl_c_cancels_cleanly_in_process(media, tmp_path, monkeypatch, capsys):
    """SIGINT during the analysis -> exit code 130, "已取消", no ffmpeg child left (os.kill delivers it to ourselves)."""
    caught = []
    previous = signal.signal(signal.SIGINT, lambda *_: caught.append(1))           # safety net: never kill pytest
    sent = []

    def fake_progress(prefix=""):
        def cb(frac, msg):
            if frac >= 0.2 and not sent:
                sent.append(frac)
                os.kill(os.getpid(), signal.SIGINT)
        return cb

    monkeypatch.setattr(cli, "_progress_printer", fake_progress)
    try:
        rc, out, err = run_main(capsys, "shorts", media / "長 片 a.mp4", "-o", tmp_path / "o", "--count", "1")
    finally:
        signal.signal(signal.SIGINT, previous)
    assert sent, "the analysis never reported 20 % progress"
    assert rc == 130 and "已取消" in err and not caught, (rc, err)
    assert not (tmp_path / "o" / "長_片_a.acproj").exists()
    time.sleep(0.2)
    assert [c for c in children_of(os.getpid()) if "ffmpeg" in c[1]] == []
    assert signal.getsignal(signal.SIGINT) is previous                              # the handler was restored


@ffmpeg
@posix_only
def test_ctrl_c_cancels_a_child_process_and_kills_ffmpeg(media, tmp_path, child_home):
    token = f"ctrlc-{uuid.uuid4().hex[:10]}"
    src = tmp_path / f"{token}.mp4"
    shutil.copy(media / "長 片 a.mp4", src)
    log = tmp_path / "stderr.txt"
    with open(log, "wb") as err_file:
        proc = subprocess.Popen([sys.executable, "-m", "autocut", "shorts", str(src), "-o", str(tmp_path / "o"), "--count", "1"],
                                env=env_for(child_home), stdout=subprocess.DEVNULL, stderr=err_file)
        try:
            deadline = time.time() + 60
            while time.time() < deadline and b"%" not in log.read_bytes():           # first progress line = analysis running
                assert proc.poll() is None, "the command ended before reporting progress"
                time.sleep(0.02)
            proc.send_signal(signal.SIGINT)
            rc = proc.wait(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
    text = log.read_text(encoding="utf-8", errors="replace")
    assert rc == 130, (rc, text)
    assert "已取消" in text and "Traceback" not in text
    assert not (tmp_path / "o" / f"{token}.acproj").exists()
    for _ in range(20):
        if not processes_mentioning(token):
            break
        time.sleep(0.1)
    assert processes_mentioning(token) == [], "an ffmpeg child outlived the cancelled command"
