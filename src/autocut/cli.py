"""``autocut`` command line: the same engine as the desktop app, scriptable and testable without a display.

Exit codes: 0 ok · 1 error or the technical check failed · 2 bad arguments, or the audit rejected the plan ·
3 ffmpeg missing / unusable · 130 interrupted (Ctrl+C).  Every message for the user is 繁體中文; set
``AUTOCUT_DEBUG=1`` to get the Python traceback of an unexpected error.
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import platform
import re
import signal
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .util import collect_media_files, human_duration, human_size, sha256_hex

EXIT_OK, EXIT_FAIL, EXIT_REJECTED, EXIT_NO_FFMPEG, EXIT_INTERRUPTED = 0, 1, 2, 3, 130
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_QA_LABEL = {"pass": "通過", "warn": "有警告", "fail": "未通過", "not_run": "沒有跑完（有項目未執行）"}
_SEVERITY = {"error": "錯誤", "warn": "提醒", "info": "資訊"}


class CliError(Exception):
    """Something the user can fix; the message is 繁體中文 and is printed as it is (``✘ …``)."""

    def __init__(self, message: str, code: int = EXIT_FAIL) -> None:
        super().__init__(message)
        self.code = code


# ------------------------------------------------------------------------------------ helpers
def _abs(path) -> Path:
    """Absolute path without following symlinks.  Render / ffmpeg run in other working directories, so every path
    that leaves this module must be absolute (a relative ``-o out.mp4`` used to end up in the render cache)."""
    return Path(os.path.abspath(os.path.expanduser(str(path).strip().strip('"'))))


def _safe_name(text: str, limit: int = 100) -> str:
    """A file name that is legal on Windows too (no ``<>:"/\\|?*``, no trailing dot/space, no reserved device names)."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", str(text).replace("·", "-"))
    s = re.sub(r"\s+", "_", s).strip(" ._") or "output"
    s = s[:limit].rstrip(" ._") or "output"
    return "_" + s if s.split(".")[0].upper() in _WINDOWS_RESERVED else s


def _has_cjk(text: str) -> bool:
    return any("㐀" <= c <= "鿿" or "＀" <= c <= "￯" or "　" <= c <= "〿" for c in text)


def _oneline(exc: BaseException, limit: int = 160) -> str:
    text = " ".join(str(exc.args[0] if isinstance(exc, KeyError) and exc.args else exc).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _progress_printer(prefix: str = ""):
    """``fn(fraction, message)`` writing to stderr: a live ``\\r`` line on a terminal, one line per 10 % otherwise."""
    tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    last = {"pct": -1}

    def cb(frac: float, msg: str) -> None:
        try:
            pct = int(max(0.0, min(1.0, float(frac))) * 100)
        except (TypeError, ValueError):
            return
        if tty:
            if pct != last["pct"]:
                last["pct"] = pct
                sys.stderr.write(f"\r{prefix}{pct:3d}%  {msg[:60]:<60}")
                sys.stderr.flush()
                if pct >= 100:
                    sys.stderr.write("\n")
        else:
            step = pct // 10 * 10
            if step != last["pct"]:
                last["pct"] = step
                sys.stderr.write(f"{prefix}{step:3d}%  {msg[:60]}\n")
                sys.stderr.flush()

    return cb


@contextlib.contextmanager
def _cancel_on_ctrl_c():
    """Yield a ``threading.Event`` that Ctrl+C sets, so analysis / render stop cleanly (ffmpeg children are killed and
    reaped by the engine).  A second Ctrl+C is an emergency exit (``KeyboardInterrupt``)."""
    cancel = threading.Event()
    old = None

    def handler(signum, frame) -> None:                       # noqa: ANN001
        if cancel.is_set():
            raise KeyboardInterrupt
        cancel.set()
        sys.stderr.write("\n正在取消…（再按一次 Ctrl+C 可強制結束）\n")
        sys.stderr.flush()

    try:
        old = signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):                              # not the main thread: keep the default behaviour
        old = None
    try:
        yield cancel
    finally:
        if old is not None:
            try:
                signal.signal(signal.SIGINT, old)
            except (ValueError, OSError):
                pass


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        from .media.ffmpeg import Cancelled
        raise Cancelled()


def _require_ffmpeg():
    """The :class:`FFmpegInfo`, or a clear 繁中 error with exit code 3 (checked up front: ``probe_asset`` would otherwise
    turn a missing ffmpeg into one 'cannot read this file' message per file)."""
    from .media.ffmpeg import FFmpegMissing, ffmpeg_info
    try:
        return ffmpeg_info()
    except FFmpegMissing:
        raise
    except Exception as exc:                                  # exists but cannot run (damaged file, wrong platform ...)
        raise CliError("ffmpeg 無法執行（檔案可能已損毀，或不是這個作業系統用的版本）。請重新執行安裝程式，"
                       "或把正確的 ffmpeg 的 bin 資料夾加入 PATH。", EXIT_NO_FFMPEG) from exc


def _expand_inputs(raw: Sequence[str]) -> tuple[list[Path], list[str]]:
    """``(media files, complaints)`` for the command-line paths; each argument that yields nothing gets a 繁中 reason."""
    files: list[Path] = []
    problems: list[str] = []
    seen: set[str] = set()
    for r in raw:
        p = _abs(r)
        if not p.exists():
            problems.append(f"找不到檔案或資料夾：{r}")
            continue
        found = collect_media_files([p])
        if not found:
            problems.append(f"資料夾裡沒有可用的影片、音訊或圖片檔：{r}" if p.is_dir() else
                            f"不支援的檔案格式「{p.suffix or '（沒有副檔名）'}」：{p.name}（支援常見的影片、音訊與圖片格式）")
        for f in found:
            key = str(f.resolve())
            if key not in seen:
                seen.add(key)
                files.append(f)
    return files, problems


def _import_assets(proj, paths: Sequence[str]) -> list[str]:
    """Probe and add every usable file; unusable ones are reported on stderr.  Raises :class:`CliError` when nothing is left.

    Asset ids come from the file (fingerprint + path) instead of ``probe_asset``'s random ones, so the same material gives
    the same plan hash on every run (spec §0 rule 7)."""
    from .core import ops
    from .media.probe import ProbeError, probe_asset
    files, problems = _expand_inputs(paths)
    for msg in problems:
        print(f"略過：{msg}", file=sys.stderr)
    ids: list[str] = []
    for f in files:
        try:
            if f.stat().st_size == 0:
                raise ProbeError("檔案是空的（0 位元組）")
            asset = probe_asset(f)
        except (ProbeError, OSError) as e:
            print(f"略過 {f.name}：{_probe_failure(e)}", file=sys.stderr)
            continue
        asset.id = "a_" + sha256_hex(f"{asset.fingerprint}|{asset.path}")[:8]
        ids.append(ops.add_asset(proj, asset).id)
    if not ids:
        raise CliError("沒有任何可用的影片、音訊或圖片檔，所以沒有東西可以處理。請確認檔案存在、格式正確且沒有損毀。")
    return ids


def _probe_failure(exc: BaseException) -> str:
    """One 繁中 line for a file ffprobe could not read (the raw ffprobe text is long, English and full of paths)."""
    text = str(exc)
    if not text.startswith("無法讀取媒體資訊"):
        return _oneline(exc)
    raw = [ln.strip() for ln in text.splitlines() if ln.strip()]
    tail = re.sub(r"^.*?: (?=[A-Z])", "", raw[-1]) if raw else ""
    detail = f"（ffprobe：{tail[:80]}）" if tail else ""
    return f"無法讀取這個檔案（可能已損毀、尚未下載完成，或格式不支援）{detail}"


def _qa_headline(rep) -> str:
    """The first line of every technical-QA report.  The machine only ever speaks about the technical check."""
    return f"技術檢查：{_QA_LABEL.get(rep.status, rep.status)}（人工審片：尚未，機器不會代替你判斷好不好看）"


def _guard_not_source(proj, out: Path) -> None:
    """Refuse to write the output over one of the project's own source files."""
    try:
        target = out.resolve()
    except OSError:
        return
    for a in proj.assets.values():
        try:
            if Path(a.path).resolve() == target:
                raise CliError(f"輸出檔不可以和素材檔是同一個檔案（會蓋掉原始素材）：{out.name}")
        except OSError:
            continue


def _explain(exc: BaseException) -> tuple[str, int]:
    """``(繁中 message, exit code)`` for an exception that reached :func:`main`."""
    from .media.ffmpeg import FFmpegMissing
    if isinstance(exc, FFmpegMissing):
        return str(exc), EXIT_NO_FFMPEG
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "磁碟空間不足，無法寫入輸出檔。請清出空間或換一個輸出位置。", EXIT_FAIL
    if isinstance(exc, FileNotFoundError):
        return f"找不到檔案或資料夾：{exc.filename or exc}", EXIT_FAIL
    if isinstance(exc, PermissionError):
        return f"沒有權限讀寫：{exc.filename or exc}", EXIT_FAIL
    if isinstance(exc, FileExistsError):
        return f"同名的檔案已存在，無法建立資料夾：{exc.filename or exc}", EXIT_FAIL
    if isinstance(exc, (IsADirectoryError, NotADirectoryError)):
        return f"路徑類型不對（檔案與資料夾搞混了）：{exc.filename or exc}", EXIT_FAIL
    if isinstance(exc, json.JSONDecodeError):
        return "檔案內容不是有效的 JSON，檔案可能已損毀。", EXIT_FAIL
    text = _oneline(exc, 400)
    if _has_cjk(text):                                         # the engine's own messages are already 繁體中文
        return text, EXIT_FAIL
    return f"發生未預期的錯誤（{type(exc).__name__}）：{text}。設定環境變數 AUTOCUT_DEBUG=1 可看到完整追蹤。", EXIT_FAIL


# ------------------------------------------------------------------------- commands
def cmd_doctor(args) -> int:
    ok = True

    def line(name: str, status: str, detail: str = "") -> None:
        print(f"{status:<4} {name:<22} {detail}")

    print(f"AutoCut Studio {__version__}  ·  Python {platform.python_version()}  ·  {platform.platform()}")
    from .util import app_data_dir
    line("資料夾", "ok", str(app_data_dir()))
    try:
        from .media.ffmpeg import detect_hw_encoder, ffmpeg_info
        info = ffmpeg_info()
        line("ffmpeg", "ok", f"{info.version_text.split(' Copyright')[0]}  ({info.ffmpeg})")
        line("libass(燒入字幕)", "ok" if info.has_ass else "警告", "可用" if info.has_ass else "缺少：字幕只能輸出 .srt")
        line("libx264", "ok" if info.has_encoder("libx264") else "錯誤", "")
        ok &= info.has_encoder("libx264")
        if args.hw:
            hw = detect_hw_encoder()
            line("硬體編碼器", "ok", "沒有可用的硬體編碼器，輸出用 libx264" if hw == "libx264" else f"{hw}（實測可用）")
        else:
            line("硬體編碼器", "略過", "沒有實測（加 --hw 才會實測）")
    except Exception as e:  # noqa: BLE001
        ok = False
        line("ffmpeg", "錯誤", _oneline(e, 300))
    for mod, label, required, hint in (("numpy", "numpy", True, ""), ("cv2", "OpenCV", True, ""),
                                       ("PySide6", "PySide6(桌面介面)", False, "只有 studio 需要"),
                                       ("faster_whisper", "faster-whisper(字幕)", False,
                                        "要自動字幕才需要：pip install -r requirements-speech.txt"),
                                       ("opencc", "opencc(簡轉繁)", False, "")):
        try:
            m = __import__(mod)
            line(label, "ok", getattr(m, "__version__", ""))
        except Exception:  # noqa: BLE001
            line(label, "錯誤" if required else "選用", ("未安裝　" + hint).strip())
            ok &= not required
    try:
        import cv2
        casc = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        line("人臉偵測模型", "ok" if not casc.empty() else "警告", "OpenCV 內建 Haar")
    except Exception:  # noqa: BLE001
        pass
    return EXIT_OK if ok else EXIT_FAIL


def cmd_probe(args) -> int:
    _require_ffmpeg()
    from .media.probe import ProbeError, probe_asset
    files, problems = _expand_inputs(args.paths)
    rc = EXIT_OK
    for msg in problems:
        print(f"✘ {msg}", file=sys.stderr)
        rc = EXIT_FAIL
    for f in files:
        try:
            if f.stat().st_size == 0:
                raise ProbeError("檔案是空的（0 位元組）")
            a = probe_asset(f)
        except (ProbeError, OSError) as e:
            print(f"✘ {f.name}：{_probe_failure(e)}", file=sys.stderr)
            rc = EXIT_FAIL
            continue
        if args.json:
            print(json.dumps(a.to_dict(), ensure_ascii=False))
        else:
            vid = f"{a.width}x{a.height} @{a.fps:.2f}fps {a.vcodec}" if a.has_video else "無影像"
            aud = f"{a.acodec} {a.audio_channels}ch {a.sample_rate}Hz" if a.has_audio else "無聲音"
            print(f"{a.name}  [{a.kind}]  {human_duration(a.duration)}  {vid}  {aud}  {human_size(a.size_bytes)}")
    return rc


def cmd_qa(args) -> int:
    _require_ffmpeg()
    from .render.qa import technical_qa
    rep = technical_qa(_abs(args.file), expect_duration=args.duration, expect_size=args.size, expect_fps=args.fps)
    print(_qa_headline(rep))
    print(rep.summary())
    return EXIT_OK if rep.status in ("pass", "warn") else EXIT_FAIL          # an unfinished check is not a pass


def cmd_render(args) -> int:
    _require_ffmpeg()
    from .core.project import Project
    from .render.qa import technical_qa
    from .render.runner import RenderSettings, render_sequence
    pf = _abs(args.project)
    if not pf.is_file():
        raise CliError(f"找不到專案檔：{args.project}")
    try:
        proj = Project.load(pf)
    except (ValueError, KeyError, TypeError, OSError) as e:
        reason = _oneline(e) if _has_cjk(str(e)) else "檔案可能已損毀"
        raise CliError(f"無法開啟專案檔（{reason}）：{pf.name}") from e
    if not proj.sequences:
        raise CliError("這個專案裡沒有任何序列，沒有東西可以輸出。")
    if args.sequence:
        seq = next((s for s in proj.sequences if args.sequence in (s.id, s.name)), None)
        if seq is None:
            have = "、".join(f"{s.id}（{s.name}）" for s in proj.sequences)
            raise CliError(f"專案裡沒有序列「{args.sequence}」。可用的序列：{have}")
    else:
        seq = proj.active_sequence()
    rng = tuple(seq.meta["range"]) if args.use_range and "range" in seq.meta else None
    out = _abs(args.output)
    if out.is_dir():
        out = out / f"{_safe_name(pf.stem)}.mp4"
    elif not out.suffix:
        out = out.with_suffix(".mp4")
    _guard_not_source(proj, out)
    with _cancel_on_ctrl_c() as cancel:
        res = render_sequence(proj, seq, out,
                              RenderSettings(preset=args.preset, encoder=args.encoder, burn_captions=not args.no_captions,
                                             write_srt=args.srt, frame_range=rng),
                              _progress_printer("渲染 "), cancel)
        print(f"輸出：{res.path}（{human_duration(res.duration)}，{res.width}x{res.height}，快取命中 {res.cached}/{res.segments}）")
        for w in res.warnings:
            print("提醒：", w)
        rep = technical_qa(res.path, expect_duration=res.duration, expect_size=(res.width, res.height), expect_fps=res.fps)
    print(_qa_headline(rep))
    print(rep.summary())
    return EXIT_OK if rep.status in ("pass", "warn") else EXIT_FAIL          # an unfinished check is not a pass


def _run_plan(proj, plan, args, outdir: Path, stem: str, cancel: threading.Event) -> int:
    from .autoedit.api import apply_plan, audit_plan
    from .autoedit.config import AuditRules
    from .core.editor import Editor
    from .core.plan import set_technical_qa
    from .render.qa import technical_qa, worst_status
    from .render.runner import RenderSettings, render_sequence
    stem = _safe_name(stem)
    receipt = audit_plan(plan, proj, AuditRules(strict_platform_bands=getattr(args, "strict", False)))
    print(f"計畫指紋：{plan.sha256()[:12]}（同樣的素材與參數會得到同樣的指紋）")
    for note in plan.notes:                                    # the honest degradations (no speech model, no faces ...)
        print(f"  說明：{note}")
    for f in receipt.findings:
        print(f"  [{_SEVERITY.get(f.severity, f.severity)}] {f.rule}：{f.message}（{f.where}）")
    plan.save(outdir / f"{stem}.plan.json")
    (outdir / f"{stem}.receipt.json").write_text(json.dumps(receipt.to_dict(), ensure_ascii=False, indent=1), "utf-8")
    _check_cancel(cancel)
    if not receipt.accepted:
        print("審核未通過，沒有套用。請看上面的錯誤項目。（計畫與審核結果已存檔）")
        return EXIT_REJECTED
    editor = Editor(proj)
    seq_ids = apply_plan(editor, plan, receipt)
    project_file = outdir / f"{stem}.acproj"
    proj.save(project_file)
    print(f"已建立 {len(seq_ids)} 個序列 → {project_file}")
    if not args.render:
        print("尚未輸出影片（加 --render 才會輸出並做技術檢查）。機器審核不等於審片：成品請你自己看過。")
        return EXIT_OK
    rc = EXIT_OK
    used: set[str] = set()
    reports: list = []
    for sid in seq_ids:
        _check_cancel(cancel)
        seq = proj.sequence(sid)
        base = f"{stem}_{_safe_name(seq.name)}"
        name, n = base, 1
        while name.lower() in used:
            n += 1
            name = f"{base}_{n}"
        used.add(name.lower())
        out = outdir / f"{name}.mp4"
        _guard_not_source(proj, out)
        res = render_sequence(proj, seq, out, RenderSettings(preset=args.preset, encoder=args.encoder),
                              _progress_printer(f"{seq.name} "), cancel)
        for w in res.warnings:
            print("提醒：", w)
        rep = technical_qa(res.path, expect_duration=res.duration, expect_size=(res.width, res.height),
                           expect_fps=res.fps)
        print(f"{res.path}\n  {_qa_headline(rep)}")
        for c in rep.checks:
            if c.status in ("warn", "fail"):
                print(f"   - {c.name}：{c.detail}")
        reports.append((sid, rep))
        rc = max(rc, EXIT_OK if rep.status in ("pass", "warn") else EXIT_FAIL)     # fail, or a check that did not finish
    # Remember the machine's technical result in the project, one entry per sequence (never the human review: that stays
    # "pending").  The run-level fields keep the worst result of the batch.
    worst = worst_status([rep.status for _sid, rep in reports])
    summary = f"{len(reports)} 個成品的技術檢查：{_QA_LABEL.get(worst, worst)}（尚未人工審片）"
    runs: dict[str, None] = {}
    for sid, rep in reports:
        run_id = (proj.sequence(sid).meta.get("ai") or {}).get("run_id")
        if not run_id:
            continue
        runs[run_id] = None
        one = f"技術檢查：{_QA_LABEL.get(rep.status, rep.status)}（尚未人工審片）"
        editor.do("記錄技術檢查結果", lambda p, r=run_id, st=rep.status, sm=one, q=sid: set_technical_qa(
            p, r, st, sm, sequence=p.sequence(q)))
    for run_id in runs:
        editor.do("記錄技術檢查結果", lambda p, r=run_id: set_technical_qa(p, r, worst, summary))
    proj.save(project_file)
    print("以上只是技術檢查，不代表審片通過；請你親自看過成品再發布。")
    return rc


def _existing_file(value: Optional[str], what: str) -> str:
    if value and not _abs(value).is_file():
        raise CliError(f"找不到{what}：{value}")
    return str(_abs(value)) if value else ""


def cmd_shorts(args) -> int:
    _require_ffmpeg()
    from .autoedit.api import propose_shorts
    from .autoedit.config import ShortsParams
    from .core.project import Project
    src = _abs(args.input)
    if src.is_dir() or len(_expand_inputs([args.input])[0]) > 1:
        raise CliError("shorts 一次只處理一支影片，請指定單一影片檔（很多支毛片請用 assemble）。")
    captions = args.captions or ("file" if args.caption_file else "auto")
    if captions == "file" and not args.caption_file:
        raise CliError("--captions file 需要同時用 --caption-file 指定字幕檔（.srt／.ass／.vtt）。")
    caption_file = _existing_file(args.caption_file, "字幕檔")
    music = _existing_file(args.music, "配樂檔")
    proj = Project.new(src.stem, with_sequence=False)
    ids = _import_assets(proj, [args.input])
    params = ShortsParams(platform=args.platform, count=args.count, target_seconds=args.target,
                          captions=captions, caption_file=caption_file, reframe=args.reframe,
                          language=args.language, tighten=not args.no_tighten, music_path=music,
                          transcript_file=_existing_file(args.subs, "字幕檔"), highlight_engine=args.highlight)
    outdir = _abs(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    with _cancel_on_ctrl_c() as cancel:
        plan = propose_shorts(proj, ids[0], params, _progress_printer("分析 "), cancel)
        print(f"提出 {len(plan.sequences)} 支短影音")
        return _run_plan(proj, plan, args, outdir, src.stem, cancel)


def cmd_assemble(args) -> int:
    _require_ffmpeg()
    from .autoedit.api import propose_assemble
    from .autoedit.config import AssembleParams
    from .core.project import Project
    music = _existing_file(args.music, "配樂檔")
    proj = Project.new("長片", with_sequence=False)
    ids = _import_assets(proj, args.inputs)
    params = AssembleParams(target_seconds=args.target, aspect=args.aspect, order=args.order, pace=args.pace,
                            transition=args.transition, music_path=music, captions=args.captions,
                            language=args.language)
    outdir = _abs(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    with _cancel_on_ctrl_c() as cancel:
        plan = propose_assemble(proj, ids, params, _progress_printer("分析 "), cancel)
        print(f"提出 1 支長片，約 {human_duration(plan.total_duration)}")
        return _run_plan(proj, plan, args, outdir, "assembled", cancel)


def cmd_longform(args) -> int:
    _require_ffmpeg()
    from .autoedit.api import propose_longform
    from .autoedit.config import LongformParams
    from .core.project import Project
    src = _abs(args.input)
    if src.is_dir() or len(_expand_inputs([args.input])[0]) > 1:
        raise CliError("longform 一次只處理一支影片，請指定單一影片檔（很多支毛片請用 assemble）。")
    if not (0.05 <= args.keep <= 1.0):
        raise CliError("--keep 需介於 0.05 與 1.0 之間（例如 0.5 = 保留一半的說話內容）。", EXIT_REJECTED)
    proj = Project.new(src.stem, with_sequence=False)
    ids = _import_assets(proj, [args.input])
    params = LongformParams(mode="clean" if args.clean else "highlights", keep_ratio=args.keep,
                            target_seconds=args.target or 0.0, remove_fillers=not args.keep_fillers,
                            speaker_zoom=not args.no_zoom, zoom_min_turn=args.zoom_after,
                            captions=args.captions, transcript_file=_existing_file(args.subs, "字幕檔"),
                            language=args.language, highlight_engine=args.highlight)
    outdir = _abs(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    with _cancel_on_ctrl_c() as cancel:
        plan = propose_longform(proj, ids[0], params, _progress_printer("分析 "), cancel)
        main = next((d for d in plan.decisions if d.get("kind") == "longform"), {})
        print(f"長片：{main.get('summary', '')}")
        return _run_plan(proj, plan, args, outdir, src.stem + "_longform", cancel)


def cmd_subtitle(args) -> int:
    _require_ffmpeg()
    from .analysis.speech import SpeechUnavailable
    from .autoedit.subtitle_gen import make_subtitle_file
    src = _abs(args.input)
    if not src.is_file():
        raise CliError(f"找不到影片檔：{args.input}")
    with _cancel_on_ctrl_c() as cancel:
        try:
            out, n = make_subtitle_file(src, _abs(args.output) if args.output else None, language=args.language,
                                        traditional=not args.no_traditional, model_size=args.model,
                                        allow_download=args.download, overwrite=args.force,
                                        progress=_progress_printer("字幕 "), cancel=cancel)
        except FileExistsError as exc:
            raise CliError(f"{exc}；加 --force 覆蓋。") from exc
        except SpeechUnavailable as exc:
            hint = "（第一次使用可加 --download 允許下載語音模型）" if "下載" in str(exc) else ""
            raise CliError(f"{exc}{hint}") from exc
    print(f"已產生 {n} 條字幕 → {out}")
    print("可以先用任何文字編輯器修正錯字；之後 shorts／longform 會自動使用這個同名字幕檔來找重點。")
    return EXIT_OK


def cmd_studio(args) -> int:
    try:
        from .gui.app import main as gui_main
    except ImportError as e:
        raise CliError("無法開啟桌面介面：沒有安裝 PySide6。請先執行 Setup-AutoCut.bat，或 pip install -r requirements.txt。") from e
    return gui_main([str(_abs(p)) for p in args.paths])


# ---------------------------------------------------------------------------- parser
_ARG_ERRORS = (
    (r"^the following arguments are required: (.*)$", r"缺少必要的參數：\1"),
    (r"^argument (.+?): invalid int value: (.*)$", r"參數 \1：\2 不是有效的整數"),
    (r"^argument (.+?): invalid float value: (.*)$", r"參數 \1：\2 不是有效的數字"),
    (r"^argument (.+?): invalid choice: (.*?) \(choose from (.*)\)$", r"參數 \1：不接受 \2（可用：\3）"),
    (r"^argument (.+?): expected one argument$", r"參數 \1：需要一個值"),
    (r"^argument (.+?): expected at least one argument$", r"參數 \1：至少需要一個值"),
    (r"^unrecognized arguments: (.*)$", r"無法辨識的參數：\1"),
    (r"^ambiguous option: (.*)$", r"選項不明確：\1"),
    (r"^argument (.+?): (.*)$", r"參數 \1：\2"),
)


def _zh_arg_error(message: str) -> str:
    for pat, rep in _ARG_ERRORS:
        if re.match(pat, message):
            return re.sub(pat, rep, message, count=1)
    return message


class _Parser(argparse.ArgumentParser):
    """argparse with 繁體中文 headings and error messages (sub-parsers inherit this class)."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("add_help", False)
        super().__init__(*args, **kwargs)
        self.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="顯示這則說明並結束")
        self._positionals.title = "參數"
        self._optionals.title = "選項"

    def format_usage(self) -> str:
        return super().format_usage().replace("usage: ", "用法：", 1)

    def format_help(self) -> str:
        return super().format_help().replace("usage: ", "用法：", 1)

    def error(self, message: str) -> None:                     # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(EXIT_REJECTED, f"{self.prog}：參數錯誤：{_zh_arg_error(message)}\n")


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} 不是整數") from None
    if v < 1:
        raise argparse.ArgumentTypeError("必須是 1 以上的整數")
    return v


def _positive_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} 不是數字") from None
    if not v > 0 or v == float("inf"):
        raise argparse.ArgumentTypeError("必須大於 0")
    return v


def _size_arg(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{2,5})[xX×](\d{2,5})", text.strip())
    if not m:
        raise argparse.ArgumentTypeError("格式應為 寬x高，例如 1080x1920")
    return int(m.group(1)), int(m.group(2))


def build_parser() -> argparse.ArgumentParser:
    from .render.presets import PRESETS
    p = _Parser(prog="autocut", description="AutoCut Studio — 自動剪輯（短影音 / 長片）",
                epilog="結束代碼：0 成功　1 失敗或技術檢查未通過　2 參數錯誤或審核未通過　3 找不到 ffmpeg　130 已中止。"
                       "機器只做技術檢查；成品好不好看要由你審片。")
    p.add_argument("--version", action="version", version=f"AutoCut Studio {__version__}", help="顯示版本並結束")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="命令", title="命令")

    def add_cmd(name: str, text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=text, description=text)

    s = add_cmd("doctor", "檢查環境（ffmpeg、OpenCV、PySide6、語音）")
    s.add_argument("--hw", action="store_true", help="實測硬體編碼器")
    s.set_defaults(fn=cmd_doctor)

    s = add_cmd("probe", "讀取素材資訊")
    s.add_argument("paths", nargs="+", metavar="路徑", help="素材檔或資料夾")
    s.add_argument("--json", action="store_true", help="每個檔案輸出一行 JSON")
    s.set_defaults(fn=cmd_probe)

    s = add_cmd("qa", "對輸出檔做技術檢查（不代表審片通過）")
    s.add_argument("file", metavar="檔案")
    s.add_argument("--duration", type=_positive_float, help="預期長度（秒）")
    s.add_argument("--fps", type=_positive_float, help="預期影格率")
    s.add_argument("--size", type=_size_arg, help="預期尺寸，例如 1080x1920")
    s.set_defaults(fn=cmd_qa)

    s = add_cmd("render", "把專案（.acproj）輸出成影片")
    s.add_argument("project", metavar="專案檔")
    s.add_argument("-o", "--output", required=True, help="輸出的影片檔（.mp4）")
    s.add_argument("--sequence", help="序列 id 或名稱（預設使用中序列）")
    s.add_argument("--preset", default="youtube_1080p", choices=sorted(PRESETS), help="輸出預設")
    s.add_argument("--encoder", default="libx264", choices=["libx264", "auto", "h264_nvenc", "h264_qsv", "h264_amf"],
                   help="編碼器（沒有的會改用 libx264）")
    s.add_argument("--no-captions", action="store_true", help="不燒入字幕")
    s.add_argument("--srt", action="store_true", help="另存 .srt")
    s.add_argument("--use-range", action="store_true", help="只輸出入出點範圍")
    s.set_defaults(fn=cmd_render)

    def add_run_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-o", "--output", required=True, help="輸出資料夾（會自動建立）")
        sp.add_argument("--render", action="store_true", help="套用後直接輸出成品並做技術檢查")
        sp.add_argument("--preset", default="shorts_1080x1920", choices=sorted(PRESETS), help="輸出預設")
        sp.add_argument("--encoder", default="libx264", help="編碼器（沒有的會改用 libx264）")
        sp.add_argument("--strict", action="store_true", help="平台時長區間違規視為錯誤")
        sp.add_argument("--language", default="zh", help="語音／字幕語言")
        sp.add_argument("--music", help="配樂檔")

    s = add_cmd("shorts", "16:9 長影片 → 9:16 短影音")
    s.add_argument("input", metavar="影片檔")
    add_run_opts(s)
    s.add_argument("--count", type=_positive_int, default=3, help="要幾支短影音")
    s.add_argument("--platform", default="ig_reels", choices=["yt_shorts", "ig_reels", "fb_reels", "tiktok"])
    s.add_argument("--target", type=_positive_float, default=35.0, help="每支的目標秒數")
    s.add_argument("--captions", default=None, choices=["auto", "file", "off"],
                   help="字幕：auto 語音辨識（要另外安裝）、file 用 --caption-file、off 不要（預設 auto；有給 --caption-file 就是 file）")
    s.add_argument("--caption-file", help="字幕檔（.srt／.ass／.vtt）")
    s.add_argument("--reframe", default="face", choices=["face", "motion", "center", "blur"], help="9:16 取景方式")
    s.add_argument("--no-tighten", action="store_true", help="不剪掉過長的停頓")
    s.add_argument("--subs", help="用這個字幕檔（.srt／.vtt／.ass）的文字找爆點（預設：影片旁同名字幕檔或語音辨識）")
    s.add_argument("--highlight", default="local", choices=["local", "claude"],
                   help="爆點判斷：local 本機規則（預設）、claude 由 AI 判斷（要另外安裝並設定 API 金鑰，只傳送字幕文字）")
    s.set_defaults(fn=cmd_shorts)

    s = add_cmd("longform", "長影片 → 精華長片（重點＋去水詞＋講者特寫）")
    s.add_argument("input", metavar="影片檔")
    add_run_opts(s)
    s.set_defaults(preset="youtube_1080p")
    s.add_argument("--keep", type=float, default=0.5, help="保留多少比例的說話內容（0.05–1.0，預設 0.5）")
    s.add_argument("--target", type=_positive_float, help="目標秒數（指定時取代 --keep）")
    s.add_argument("--clean", action="store_true", help="不挑重點，只剪掉水詞與停頓")
    s.add_argument("--keep-fillers", action="store_true", help="不剪「嗯、呃、那個」這類水詞")
    s.add_argument("--no-zoom", action="store_true", help="不做講者特寫")
    s.add_argument("--zoom-after", type=_positive_float, default=3.0, help="一個人連續說超過幾秒就放大（預設 3）")
    s.add_argument("--captions", default="off", choices=["off", "auto"], help="是否加字幕")
    s.add_argument("--subs", help="用這個字幕檔的文字找重點（預設：影片旁同名字幕檔或語音辨識）")
    s.add_argument("--highlight", default="local", choices=["local", "claude"], help="重點判斷：local 本機規則、claude AI")
    s.set_defaults(fn=cmd_longform)

    s = add_cmd("subtitle", "一鍵字幕：語音辨識整支影片，存成同名 .srt")
    s.add_argument("input", metavar="影片檔")
    s.add_argument("-o", "--output", help="字幕檔路徑（預設：影片旁的同名 .srt）")
    s.add_argument("--language", default="zh", help="語言（預設 zh）")
    s.add_argument("--model", default="small", help="語音模型大小（tiny／base／small／medium／large-v3）")
    s.add_argument("--no-traditional", action="store_true", help="不轉成繁體")
    s.add_argument("--download", action="store_true", help="允許第一次下載語音模型（需要網路；影片不會上傳）")
    s.add_argument("--force", action="store_true", help="覆蓋已存在的字幕檔")
    s.set_defaults(fn=cmd_subtitle)

    s = add_cmd("assemble", "多支毛片 → 一支長片")
    s.add_argument("inputs", nargs="+", metavar="素材", help="影片／音訊／圖片檔，或放它們的資料夾")
    add_run_opts(s)
    s.set_defaults(preset="youtube_1080p")
    s.add_argument("--target", type=_positive_float, default=180.0, help="目標秒數")
    s.add_argument("--aspect", default="auto", choices=["auto", "16:9", "9:16", "1:1", "4:5"])
    s.add_argument("--order", default="chronological", choices=["chronological", "name", "as_given"])
    s.add_argument("--pace", default="medium", choices=["fast", "medium", "slow"])
    s.add_argument("--transition", default="crossfade", choices=["cut", "crossfade"])
    s.add_argument("--captions", default="off", choices=["off", "auto"])
    s.set_defaults(fn=cmd_assemble)

    s = add_cmd("studio", "開啟桌面軟體")
    s.add_argument("paths", nargs="*", metavar="路徑", help="要開啟的專案或素材")
    s.set_defaults(fn=cmd_studio)
    return p


def _safe_streams() -> None:
    """A redirected stdout on a legacy Windows code page (cp950 …) cannot encode ✔ ✘ …; print ``?`` instead of crashing."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")              # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _safe_streams()
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        print("\n已中止", file=sys.stderr)
        return EXIT_INTERRUPTED
    except CliError as e:
        print(f"✘ {e}", file=sys.stderr)
        return e.code
    except Exception as e:  # noqa: BLE001
        from .media.ffmpeg import Cancelled
        if isinstance(e, Cancelled):
            print("已取消（沒有寫入未完成的成品）", file=sys.stderr)
            return EXIT_INTERRUPTED
        if os.environ.get("AUTOCUT_DEBUG"):
            raise
        message, code = _explain(e)
        print(f"✘ {message}", file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
