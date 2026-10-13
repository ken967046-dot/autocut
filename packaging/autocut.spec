# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for AutoCut Studio ("one folder" build, no console window).
#
#   build:  powershell -NoProfile -ExecutionPolicy Bypass -File packaging\Build-Windows.ps1
#   result: dist\AutoCutStudio\AutoCutStudio.exe   (copy the whole folder; it needs no Python installed)
#
# STATUS: written but NOT verified on Windows until someone builds it there and starts the exe
# (see TASKS.md WIN-1).
#
# ffmpeg is NOT bundled by default: the builds the setup downloads are mostly GPL, and whoever hands such a folder to other
# people takes on the GPL duties for it (THIRD_PARTY_NOTICES.md says this project does not ship ffmpeg).  The exe looks for
# ffmpeg in ffmpeg\bin next to itself, in %LOCALAPPDATA%\AutoCutStudio\ffmpeg and on PATH.  To bundle it anyway (for your own
# use, or after you have checked the licence of your build), run the build with the environment variable
# AUTOCUT_BUNDLE_FFMPEG=1: tools\ffmpeg\bin\ffmpeg.exe and ffprobe.exe then go into the folder.
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent

hiddenimports = collect_submodules("autocut")
for optional in ("faster_whisper", "ctranslate2", "opencc", "tokenizers", "onnxruntime", "av"):
    try:                                                 # speech recognition is optional: bundle it only when installed
        __import__(optional)
        hiddenimports += collect_submodules(optional)
    except Exception:
        pass

datas = [(str(ROOT / name), ".") for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md")]
datas += collect_data_files("cv2", includes=["data/*.xml"])          # Haar face cascades
for optional in ("faster_whisper", "opencc"):
    try:
        datas += collect_data_files(optional)
    except Exception:
        pass

binaries = []
if os.environ.get("AUTOCUT_BUNDLE_FFMPEG") == "1":
    ffmpeg_dir = ROOT / "tools" / "ffmpeg" / "bin"
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        if not (ffmpeg_dir / name).is_file():
            raise SystemExit(f"AUTOCUT_BUNDLE_FFMPEG=1 but {ffmpeg_dir / name} does not exist (run Setup-AutoCut.bat first)")
        binaries.append((str(ffmpeg_dir / name), "ffmpeg"))
    print("NOTE: ffmpeg is bundled. Builds like this are usually GPL: if you give the folder to other people, GPL applies to it.")
else:
    print("ffmpeg is not bundled (set AUTOCUT_BUNDLE_FFMPEG=1 to bundle it); the exe looks next to itself, then in %LOCALAPPDATA%.")

a = Analysis(
    [str(ROOT / "packaging" / "autocut_studio_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="AutoCutStudio",
    console=False,
    disable_windowed_traceback=False,
    icon=None,                                           # the window icon is drawn in code (autocut.gui.app.build_app_icon)
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="AutoCutStudio")
