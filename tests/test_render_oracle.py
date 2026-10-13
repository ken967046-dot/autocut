"""Pixel oracle: after random edits, every rendered frame must show what the timeline says it shows.

The source clips are *barcodes*: the top row of every frame carries its frame number (Gray coded, so a frame off by one
differs in one bit) and the number of the source file, as black/white blocks that survive H.264 at a tiny size.  A random
chain of edits (split, trim, ripple, speed, crossfade, slip, range delete, move ...) is applied with the editing
operations, the result is rendered with the real renderer and EVERY frame is decoded and compared with the frame the
timeline model predicts:

* one clip under the frame  -> its barcode (right file, right source frame, +-1 source frame for rounding);
* two clips (a crossfade)   -> the blend of both barcodes: the fade weight is ESTIMATED from the blocks where the two
  barcodes differ and must be within 0.12 of the weight the model gives (the renderer is about 0.03 below it, never more
  than 0.07 away), and the blocks must fit that blend almost exactly;
* nothing under the frame   -> black (a hole in the timeline);
* the rendered length equals the timeline length.

The sound is checked the same way: every source has a rising tone (200 Hz + 100 Hz per second of source time), so the pitch
heard at any moment says which second of the source it is - through trims, speed changes (the pitch is kept) and ripples.
A hole must be silent.

Frames within two frames of a cut or of the start of a fade are skipped (rounding of the boundary is the encoder's).
A clip whose start lies BETWEEN two frames (a ripple or a speed change leaves such starts) is put on the nearest frame - the
renderer does the same for picture and sound - so the model counts a clip's frames from that frame, not from the exact
start: at x2 the two differ by up to 2.4 source frames, far more than the tolerance (the first 25 fps seeds that had such a
clip failed on this, not on the picture).
This is what the structural fuzz test (``test_core_fuzz``) cannot see: that the *pictures and sound* are the right ones.
"""
from __future__ import annotations

import collections
import itertools
import random
import subprocess

import numpy as np
import pytest

from autocut.core import ops
from autocut.core.ops import OpError
from autocut.core.project import Project, ordered_clips, validate_sequence
from autocut.core.timecode import frame_of
from autocut.media.ffmpeg import ffmpeg_info
from autocut.media.probe import probe_asset
from autocut.render.runner import RenderSettings, render_sequence

pytestmark = pytest.mark.ffmpeg

W, H, FPS = 160, 90, 30.0
SRC_FPS = 30.0                                              # the barcode files are 30 fps whatever the sequence runs at
ALPHA_TOL, FIT_TOL = 0.12, 0.04                             # crossfade weight off by at most this; mean block error of the blend
BLOCK, FRAME_BITS, BITS = 12, 10, 12                       # 12 blocks of 12 px: 10 for the frame number, 2 for the file
DURS = (16.0, 12.0, 20.0)
SEEDS = range(6)
SKIP = 2                                                    # frames next to a boundary are not compared


def barcode_bits(frame: int, asset: int) -> np.ndarray:
    """The 12 block values (0/1) of source frame ``frame`` of file ``asset`` (Gray code of frame + 1: never all black)."""
    m = max(0, frame) + 1
    gray = m ^ (m >> 1)
    return np.array([(gray >> k) & 1 for k in range(FRAME_BITS)] + [(asset >> k) & 1 for k in range(BITS - FRAME_BITS)],
                    dtype=np.float32)


def make_sources(d) -> list:
    """Three barcode + rising-tone files (see the module docstring) in directory ``d``, probed."""
    out = []
    for i, dur in enumerate(DURS):
        bit = ("if(lt(floor(X/12),10),mod(floor((N+1)/pow(2,floor(X/12)))+floor((N+1)/pow(2,floor(X/12)+1)),2),"
               f"mod(floor({i}/pow(2,floor(X/12)-10)),2))")
        graph = (f"color=c=black:s={W}x{H}:r=30:d={dur},format=yuv420p,"
                 f"geq=lum='if(lt(Y,12)*lt(X,144),16+219*{bit},96)':cb=128:cr=128")
        tone = f"aevalsrc='0.5*sin(2*PI*(200*t+50*t*t))':s=48000:d={dur}"
        path = d / f"bar{i}.mp4"
        subprocess.run([ffmpeg_info().ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", graph, "-f", "lavfi", "-i", tone,
                        "-c:v", "libx264", "-crf", "10", "-g", "15", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                        "-shortest", str(path)], check=True)
        out.append(probe_asset(path))
    return out


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    return make_sources(tmp_path_factory.mktemp("barcodes"))


def build(seed: int, sources, fps: float = FPS):
    rng = random.Random(seed)
    proj = Project.new(f"oracle{seed}")
    for a in sources:
        proj.assets[a.id] = a
    seq = proj.active_sequence()
    ops.set_sequence_format(proj, seq, W, H, fps)
    v1 = seq.video_tracks[0]
    log: list[str] = []
    for _ in range(rng.randint(3, 5)):
        a = rng.choice(sources)
        s_in = round(rng.uniform(0, a.duration - 5), 2)
        ops.place_clip(proj, seq, a.id, v1.id, src_in=s_in, src_out=s_in + round(rng.uniform(2.0, 4.5), 2), mode="append")
    for clip in list(v1.clips)[1:]:
        if rng.random() < 0.6:
            try:
                ops.set_crossfade(proj, seq, clip.id, rng.choice((0.3, 0.5, 0.8)))
            except OpError:
                pass
    log.append("start: " + " ".join(f"[{c.start:.2f}-{c.end:.2f} x{c.speed}]" for c in ordered_clips(v1.clips)))
    for step in range(rng.randint(4, 8)):
        clips = ordered_clips(v1.clips)
        if not clips:
            break
        c = rng.choice(clips)
        kinds = ["split", "trim_start", "trim_end", "trim_start", "trim_end", "speed", "crossfade", "remove", "slip", "range",
                 "move", "place", "close_gaps"]
        if len(clips) < 2:                                   # keep something to render
            kinds = [k for k in kinds if k not in ("remove", "range")]
        kind = rng.choice(kinds)
        ripple = rng.random() < 0.6
        t = rng.choice((c.start + rng.uniform(0.1, max(0.2, c.duration - 0.1)), c.start + rng.uniform(-0.8, 0.8),
                        c.end + rng.uniform(-0.8, 0.8)))
        try:
            if kind == "split":
                ops.split_clip(proj, seq, c.id, t)
            elif kind == "trim_start":
                ops.trim_clip_start(proj, seq, c.id, t, ripple=ripple)
            elif kind == "trim_end":
                ops.trim_clip_end(proj, seq, c.id, t, ripple=ripple)
            elif kind == "speed":
                ops.set_speed(proj, seq, c.id, rng.choice((0.5, 0.75, 1.0, 1.5, 2.0)), ripple=ripple)
            elif kind == "crossfade":
                ops.set_crossfade(proj, seq, c.id, rng.choice((0.0, 0.3, 0.6, 1.0)))
            elif kind == "remove":
                ops.remove_clips(proj, seq, [c.id], ripple=ripple)
            elif kind == "slip":
                ops.slip_clip(proj, seq, c.id, rng.uniform(-2, 2))
            elif kind == "range":
                t0 = rng.uniform(0, max(0.5, seq.duration - 1))
                ops.ripple_delete_range(proj, seq, t0, t0 + rng.uniform(0.3, 2.0))
            elif kind == "move":
                ops.move_clip(proj, seq, c.id, max(0.0, t), resolve=rng.choice(("overwrite", "insert", "reject")))
            elif kind == "place":
                a = rng.choice(sources)
                s_in = round(rng.uniform(0, a.duration - 4), 2)
                ops.place_clip(proj, seq, a.id, v1.id, start=max(0.0, t), src_in=s_in, src_out=s_in + rng.uniform(0.8, 3.0),
                               mode=rng.choice(("overwrite", "insert", "append")))
            else:
                ops.close_gaps(proj, seq, v1.id)
            log.append(f"#{step} {kind} {c.id[-4:]} t={t:.2f} ripple={ripple} -> "
                       + " ".join(f"[{c.start:.2f}-{c.end:.2f} x{c.speed}]" for c in ordered_clips(v1.clips)))
        except OpError as exc:
            log.append(f"#{step} {kind} refused: {exc}")
    return proj, seq, log


def rendered_blocks(path):
    """Decode the whole video: ``(frames, 12)`` block values 0..1 of the barcode row and the mean grey of each frame."""
    raw = subprocess.run([ffmpeg_info().ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, H, W, 3).astype(np.float32) / 255.0
    bar = frames[:, :BLOCK, : BLOCK * BITS, :].mean(axis=3)                      # (n, 12 rows, 144 columns)
    blocks = bar.reshape(len(frames), BLOCK, BITS, BLOCK).mean(axis=(1, 3))      # (n, 12)
    return blocks, frames.mean(axis=(1, 2, 3))


AUDIO_RATE = 16000


def rendered_audio(path) -> np.ndarray:
    raw = subprocess.run([ffmpeg_info().ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(AUDIO_RATE),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def dominant_hz(x: np.ndarray) -> float:
    n = 1 << 15
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n))
    k = int(spec.argmax())
    if 0 < k < len(spec) - 1:                                              # parabolic interpolation of the peak
        a, b, c = np.log(spec[k - 1:k + 2] + 1e-12)
        k += 0.5 * (a - c) / (a - 2 * b + c)
    return k * AUDIO_RATE / n


def expected_layers(seq, fps):
    """The model: per video frame, the clips under it (bottom first) with their frame ranges."""
    items = []
    for c in ordered_clips(seq.video_tracks[0].clips):
        cs = max(0, frame_of(c.start, fps))
        items.append((c, cs, max(cs + 1, frame_of(c.end, fps))))
    return items


def source_frame(c, cs: int, f: int, fps: float) -> int:
    """The source frame the model puts under output frame ``f`` of clip ``c``, whose first frame is ``cs`` (its start put on
    the nearest frame, as the renderer does): the middle of the frame's interval (the renderer may pick the next or the
    previous source frame: tolerance 1)."""
    return int((c.src_in + (f - cs + 0.5) / fps * c.speed) * SRC_FPS)


def decode(row: np.ndarray) -> tuple[int, int, float]:
    """``(source frame, file number, how far the blocks are from pure black / white)`` of a one-layer barcode."""
    bits = (row > 0.5).astype(int)
    gray = sum(int(bits[k]) << k for k in range(FRAME_BITS))
    n = 0
    while gray:
        n ^= gray
        gray >>= 1
    return n - 1, int(bits[FRAME_BITS]) + 2 * int(bits[FRAME_BITS + 1]), float(np.abs(row - bits).max())


def blend_estimate(row: np.ndarray, layers, f: int, fps: float, asset_index: dict):
    """``(estimated weight, the model's weight, mean block error of that blend)`` for the best fit over one frame of slack on
    each layer.  The weight is read off the blocks where the two barcodes differ (``(row - under) / (over - under)``), so it
    is what the renderer really did, not what it was allowed to do.  ``None`` for the estimate when the two barcodes are
    (almost) the same picture and nothing can be told."""
    (lo, lcs, lce), (hi, hcs, hce) = layers
    overlap = max(1, min(lce, hce) - hcs)
    model = (f + 0.5 - hcs) / overlap
    best = None
    for dl in (-1, 0, 1):
        for dh in (-1, 0, 1):
            under = barcode_bits(source_frame(lo, lcs, f, fps) + dl, asset_index[lo.asset_id])
            over = barcode_bits(source_frame(hi, hcs, f, fps) + dh, asset_index[hi.asset_id])
            diff = over - under
            differing = np.abs(diff) > 0.5
            if differing.sum() >= 3:
                a = float(np.mean((row[differing] - under[differing]) / diff[differing]))
                a = min(1.0, max(0.0, a))
            else:
                a = min(1.0, max(0.0, model))
            err = float(np.abs(row - ((1 - a) * under + a * over)).mean())
            if best is None or err < best[2]:
                best = (a if differing.sum() >= 3 else None, model, err)
    return best


COVERAGE: collections.Counter = collections.Counter()      # what the comparisons really looked at, over all the tests
COVERAGE_WANTED = {("frames", 1): 400, ("frames", 2): 60, "blend, weight read": 50, "sound: pitch": 60}      # the least it must see


def compare(items, blocks: np.ndarray, grey: np.ndarray, asset_index: dict, fps: float = FPS, only=None,
            stats: collections.Counter | None = None) -> list[str]:
    """Every frame of the video against the model ``items`` (from :func:`expected_layers`); the list of disagreements.
    ``only(f)``: look at the frames it accepts (default all); ``stats`` counts what was looked at."""
    stats = COVERAGE if stats is None else stats
    total = max(ce for _c, _cs, ce in items)
    edges = sorted({b for _c, cs, ce in items for b in (cs, ce)})
    bad: list[str] = []
    for f in range(min(total, len(blocks))):
        if (only is not None and not only(f)) or any(abs(f - e) <= SKIP for e in edges):
            continue
        layers = [it for it in items if it[1] <= f < it[2]]
        where = f"frame {f} ({f / fps:.2f}s)"
        stats[("frames", len(layers))] += 1
        if not layers:
            if grey[f] > 0.1:
                bad.append(f"{where}: a hole in the timeline, but the picture is not black (grey {grey[f]:.2f})")
        elif len(layers) == 1:
            c, cs = layers[0][0], layers[0][1]
            n, asset, unclear = decode(blocks[f])
            want_n, want_asset = source_frame(c, cs, f, fps), asset_index[c.asset_id]
            if unclear > 0.3 or asset != want_asset or abs(n - want_n) > 1:
                bad.append(f"{where}: shows file {asset} frame {n}, the timeline says file {want_asset} frame {want_n}"
                           f" (clip {c.id[-4:]} {c.start:.2f}-{c.end:.2f} x{c.speed})")
        elif len(layers) == 2:
            a_hat, a_model, err = blend_estimate(blocks[f], layers, f, fps, asset_index)
            names = [(c.id[-4:], round(c.start, 2), round(c.end, 2)) for c, _s, _e in layers]
            if a_hat is None:                                          # same picture twice: only "not garbage" can be said
                stats["blend, weight unreadable"] += 1
                if err > 0.08:
                    bad.append(f"{where}: not the crossfade of {names} (error {err:.2f})")
            else:
                stats["blend, weight read"] += 1
                if err > FIT_TOL or abs(a_hat - a_model) > ALPHA_TOL:
                    bad.append(f"{where}: crossfade of {names}: the picture is a {a_hat:.2f} blend, the timeline says "
                               f"{a_model:.2f} (block error {err:.3f})")
        else:
            bad.append(f"{where}: {len(layers)} layers in the model")
    return bad


def compare_audio(items, audio: np.ndarray, fps: float = FPS, stats: collections.Counter | None = None) -> list[str]:
    """Pitch heard in 0.2 s windows against the source second the model puts there; holes must be silent.  Windows that
    touch a cut or a crossfade (0.2 s guard) are skipped."""
    stats = COVERAGE if stats is None else stats
    total_s = max(ce for _c, _cs, ce in items) / fps
    bad: list[str] = []
    t = 0.4
    while t < total_s - 0.4:
        lo, hi = t - 0.2, t + 0.2
        touching = [it for it in items if it[1] / fps < hi and it[2] / fps > lo]
        window = audio[int((t - 0.1) * AUDIO_RATE): int((t + 0.1) * AUDIO_RATE)]
        if not touching:
            stats["sound: silence"] += 1
            if float(np.sqrt((window ** 2).mean())) > 0.02:
                bad.append(f"{t:.2f}s: a hole in the timeline, but there is sound")
        elif len(touching) == 1 and touching[0][1] / fps <= lo and touching[0][2] / fps >= hi:
            c, cs = touching[0][0], touching[0][1]
            stats["sound: pitch"] += 1
            want = 200 + 100 * (c.src_in + (t - cs / fps) * c.speed)
            got = dominant_hz(window)
            if abs(got - want) > 15:
                bad.append(f"{t:.2f}s: pitch {got:.0f} Hz, the timeline says {want:.0f} Hz "
                           f"(source second {(want - 200) / 100:.2f}, clip {c.id[-4:]} {c.start:.2f}-{c.end:.2f} x{c.speed})")
        t += 0.25
    return bad


def render_scenario(seed: int, sources, tmp_path, fps: float = FPS):
    proj, seq, log = build(seed, sources, fps)
    where = f"seed {seed}\n" + "\n".join(log)
    if not seq.video_tracks[0].clips:
        pytest.skip("the random edits emptied the timeline")
    assert not [i for i in validate_sequence(proj, seq) if i.level == "error"], where
    out = tmp_path / "o.mp4"
    res = render_sequence(proj, seq, out, RenderSettings(preset="draft_fast", normalize=False))
    items = expected_layers(seq, fps)
    total = max(ce for _c, _cs, ce in items)
    assert res.frames == total, f"{where}\nrendered {res.frames} frames, the timeline has {total}"
    blocks, grey = rendered_blocks(out)
    assert len(blocks) == total, where
    audio = rendered_audio(out)
    assert abs(len(audio) / AUDIO_RATE - total / fps) < 0.15, f"{where}\nthe sound is {len(audio) / AUDIO_RATE:.2f} s long"
    return seq, items, blocks, grey, audio, where


def check_scenario(seed, sources, tmp_path, fps=FPS):
    _seq, items, blocks, grey, audio, where = render_scenario(seed, sources, tmp_path, fps)
    index = {a.id: i for i, a in enumerate(sources)}
    bad = compare(items, blocks, grey, index, fps) + compare_audio(items, audio, fps)
    assert not bad, where + "\n" + "\n".join(bad[:12]) + (f"\n... {len(bad)} frames in all" if len(bad) > 12 else "")


@pytest.mark.parametrize("seed", SEEDS)
def test_every_rendered_frame_shows_what_the_timeline_says(seed, sources, tmp_path):
    check_scenario(seed, sources, tmp_path)


@pytest.mark.parametrize("fps, seed", [(25.0, 20), (25.0, 21), (25.0, 22), (24.0, 23), (24.0, 24), (60.0, 25)])
def test_the_same_holds_at_other_frame_rates(fps, seed, sources, tmp_path):
    """The barcode files run at 30 fps; the sequence at 25 / 24 / 60: every output frame must still show the source frame
    the timeline names (the renderer drops or repeats frames, the model says which)."""
    check_scenario(seed, sources, tmp_path, fps)


def test_the_oracle_notices_a_wrong_timeline(sources, tmp_path):
    """The comparison is only worth something if it objects when the picture and the model disagree."""
    seq, items, blocks, grey, audio, _where = render_scenario(1, sources, tmp_path)
    index = {a.id: i for i, a in enumerate(sources)}
    assert not compare(items, blocks, grey, index) and not compare_audio(items, audio)
    first, rest = items[0], items[1:]
    clip = first[0]
    old = clip.src_in
    clip.src_in = old + 0.3                                              # nine frames of the first clip: the wrong ones
    try:
        assert len(compare(items, blocks, grey, index)) > 10
        assert len(compare_audio(items, audio)) > 3                      # 0.3 s of source = 30 Hz of pitch
    finally:
        clip.src_in = old
    shifted = [(first[0], first[1] + 20, first[2] + 20)] + rest          # the first clip starts two thirds of a second late
    assert len(compare(shifted, blocks, grey, index)) > 10
    assert compare_audio(shifted, audio)                                 # sound where the model has a hole
    others = [c for c, _cs, _ce in items if c.asset_id != clip.asset_id]
    if others:                                                           # the model names another file for that stretch
        assert len(compare([(others[0], first[1], first[2])] + rest, blocks, grey, index)) > 10


def two_layer_frames(items):
    edges = sorted({b for _c, cs, ce in items for b in (cs, ce)})
    return [f for f in range(max(ce for _c, _cs, ce in items))
            if sum(1 for it in items if it[1] <= f < it[2]) == 2 and not any(abs(f - e) <= SKIP for e in edges)]


def test_the_oracle_notices_a_wrong_crossfade(sources, tmp_path):
    """The two-layer branch had no sensitivity check: a crossfade that is the wrong length, the wrong way round, or a wrong
    weight must be reported (this scenario has a crossfade of 16 frames)."""
    seq, items, blocks, grey, audio, _where = render_scenario(1, sources, tmp_path)
    index = {a.id: i for i, a in enumerate(sources)}
    assert len(two_layer_frames(items)) >= 10
    assert not compare(items, blocks, grey, index)
    hi = next(it for it in items if any(o is not it and o[1] < it[1] < o[2] for o in items))      # the clip fading in over another
    lo = next(o for o in items if o is not hi and o[1] < hi[1] < o[2])
    # 1. the model fades twice as fast: the weight it names is off by up to 0.5
    faster = [(c, (cs + (ce - cs) // 2) if (c is hi[0]) else cs, ce) if c is hi[0] else (c, cs, ce) for c, cs, ce in items]
    assert len(compare(faster, blocks, grey, index)) >= 3
    # 2. the model starts the fade 8 frames late: the weight is off by 8 / overlap
    overlap = min(lo[2], hi[2]) - hi[1]
    late = [(c, cs + 8, ce) if c is hi[0] else (c, cs, ce) for c, cs, ce in items]
    assert len(compare(late, blocks, grey, index)) >= 3, f"a fade started 8 frames late (overlap {overlap}) went unnoticed"
    # 3. the layers the other way round: the fade-in is the OTHER picture
    swapped = [(hi[0], lo[1], lo[2]) if it is lo else (lo[0], hi[1], hi[2]) if it is hi else it for it in items]
    assert len(compare(swapped, blocks, grey, index)) >= 3


def test_the_crossfade_weight_the_renderer_uses_is_close_to_the_model(sources, tmp_path):
    """Not just 'within tolerance': the estimated weights have no trend and stay far inside the tolerance, so a change in
    the renderer's fade curve shows up here before it shows up as a failure of the tolerance above."""
    index = {a.id: i for i, a in enumerate(sources)}
    errors = []
    for seed in (0, 1, 4, 5):
        _seq, items, blocks, _grey, _audio, _where = render_scenario(seed, sources, tmp_path / f"s{seed}")
        for f in two_layer_frames(items):
            layers = [it for it in items if it[1] <= f < it[2]]
            a_hat, a_model, _err = blend_estimate(blocks[f], layers, f, FPS, index)
            if a_hat is not None:
                errors.append((a_hat - a_model, a_model))
    assert len(errors) >= 40, f"only {len(errors)} crossfade frames could be read"
    diff = np.array([e for e, _m in errors])
    assert abs(float(np.median(diff))) < 0.06 and float(np.abs(diff).max()) < 0.10, (float(np.median(diff)), float(np.abs(diff).max()))
    slope = float(np.polyfit([m for _e, m in errors], diff, 1)[0])        # the error does not grow along the fade
    assert abs(slope) < 0.08, slope


# ---------------------------------------------------------------------------------------------- picture against sound
# The checks above are coarse about WHEN things happen: a picture may be one source frame off, a pitch 15 Hz (that is 0.15 s of
# source) off.  Neither would notice the picture and the sound of one clip drifting a frame apart - which is what the person sees as
# lip-sync that is "a bit off".  This measures both against the same model: the source time the clip's placed frame names.
def sound_offset(audio: np.ndarray, c, cs: int, t0: float, t1: float, fps: float, span: float = 0.12) -> float:
    """How much later in the source than the timeline says the sound of clip ``c`` (first frame ``cs``) plays between ``t0`` and
    ``t1``: the rising tone is slid along the sound until it fits best (the tone's pitch changes all the time, so there is
    one best fit), first in steps of 1 ms within +-``span`` seconds, then 0.1 ms."""
    i0, i1 = int(t0 * AUDIO_RATE), int(t1 * AUDIO_RATE)
    seg = audio[i0:i1].astype(np.float64)
    src = c.src_in + (np.arange(i0, i1) / AUDIO_RATE - cs / fps)         # (speed 1: a faster clip is not the same waveform)

    def fit(lag: float) -> float:
        s = src + lag
        return float(np.dot(seg, np.sin(2 * np.pi * (200 * s + 50 * s * s))))

    coarse = max(np.arange(-span, span, 0.001), key=fit)
    return float(max(np.arange(coarse - 0.001, coarse + 0.00101, 0.0001), key=fit))


def av_offsets(items, blocks: np.ndarray, audio: np.ndarray, asset_index: dict, fps: float) -> list[tuple]:
    """``(clip, picture offset, sound offset)`` in seconds for every speed-1 clip with a stretch of its own (no other clip under
    it, away from the cuts) of at least 0.8 s: how far the source time its pictures show, and the source time its sound plays,
    are from the source time the timeline names for the clip's placed frame (positive = later in the source).  The picture
    is a whole frame, so it is only known to half a source frame (17 ms); the sound to a fraction of a millisecond."""
    out = []
    for c, cs, ce in items:
        if abs(c.speed - 1.0) > 1e-9:
            continue
        own = [f for f in range(cs + SKIP + 1, ce - SKIP - 1) if sum(1 for it in items if it[1] <= f < it[2]) == 1]
        runs = [[f for _i, f in g] for _k, g in itertools.groupby(enumerate(own), key=lambda p: p[1] - p[0])]
        run = max(runs, key=len, default=[])
        if len(run) < int(0.8 * fps):
            continue
        shown, ideal = [], []
        for f in run:
            n, which, unsure = decode(blocks[f])
            if unsure > 0.3 or which != asset_index[c.asset_id]:
                break
            shown.append((n + 0.5) / SRC_FPS)
            ideal.append(c.src_in + (f + 0.5 - cs) / fps)
        else:
            t0, t1 = run[0] / fps + 0.1, (run[-1] + 1) / fps - 0.1
            mid = (t0 + t1) / 2
            t0, t1 = max(t0, mid - 0.5), min(t1, mid + 0.5)
            if t1 - t0 >= 0.5:
                out.append((c, float(np.mean(np.array(shown) - np.array(ideal))), sound_offset(audio, c, cs, t0, t1, fps)))
    return out


@pytest.mark.parametrize("fps", [30.0, 25.0, 24.0])
def test_picture_and_sound_of_a_clip_agree_to_well_under_a_frame(fps, sources, tmp_path):
    """Measured at the time of writing: the sound is a constant 5 ms behind (the AAC decoder and the resampler), the picture
    within half a source frame, and the two within 19 ms of each other at 30 fps and 10 ms at 25 / 24 fps.  A picture or a sound
    moved by one frame (33-42 ms) is outside all three limits."""
    index = {a.id: i for i, a in enumerate(sources)}
    rows = []
    for seed in (300, 301, 302):
        _seq, items, blocks, _grey, audio, _where = render_scenario(seed, sources, tmp_path / f"s{seed}", fps)
        rows += [(seed, c.id[-4:], c.start, v, a) for c, v, a in av_offsets(items, blocks, audio, index, fps)]
    assert len(rows) >= 6, f"only {len(rows)} clips could be measured"
    for seed, cid, start, v, a in rows:
        say = f"seed {seed} clip {cid} (start {start:.3f}): picture {v * 1000:+.1f} ms, sound {a * 1000:+.1f} ms"
        assert abs(a) < 0.015, say
        assert abs(v) < 0.022, say
        assert abs(v - a) < 0.025, say


def late_sound_source(d, delay: float):
    """A barcode file (10 s) whose sound starts ``delay`` seconds after its picture (the container says so: the audio stream's
    start time is later).  Its rising tone is delayed with it, so at any moment of the file the tone is ``delay`` behind the
    picture - which is what a clip taken from it must keep."""
    graph = (f"color=c=black:s={W}x{H}:r=30:d=10,format=yuv420p,geq=lum='if(lt(Y,12)*lt(X,144),16+219*"
             "if(lt(floor(X/12),10),mod(floor((N+1)/pow(2,floor(X/12)))+floor((N+1)/pow(2,floor(X/12)+1)),2),0),96)'"
             ":cb=128:cr=128")
    path = d / f"late{int(delay * 1000)}.mp4"
    subprocess.run([ffmpeg_info().ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", graph, "-itsoffset", str(delay), "-f", "lavfi",
                    "-i", "aevalsrc='0.5*sin(2*PI*(200*t+50*t*t))':s=48000:d=10", "-c:v", "libx264", "-crf", "10", "-g", "15",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", str(path)], check=True)
    return probe_asset(path)


def render_clip(asset, src_in: float, speed: float, d):
    proj = Project.new("late")
    proj.assets[asset.id] = asset
    seq = proj.active_sequence()
    ops.set_sequence_format(proj, seq, W, H, FPS)
    clip = ops.place_clip(proj, seq, asset.id, seq.video_tracks[0].id, start=0.0, src_in=src_in, src_out=src_in + 4.0 * speed,
                          speed=speed)
    out = d / f"o{src_in}_{speed}.mp4"
    render_sequence(proj, seq, out, RenderSettings(preset="draft_fast", normalize=False))
    return clip, expected_layers(seq, FPS)[0][1], rendered_audio(out)


@pytest.mark.parametrize("delay", [0.1, 0.25])
def test_sound_that_starts_late_in_the_file_keeps_its_offset_for_a_clip_cut_from_the_very_start(delay, tmp_path):
    """A clip taken from the start of the file and one taken from the middle must have the SAME picture-to-sound relation (the
    file's own).  Rebasing the sound to start at zero made the first one's sound ``delay`` too early (measured 77 ms of 78 and
    226 ms of 255 before the fix); the second one was right all along."""
    asset = late_sound_source(tmp_path, delay)
    offsets = []
    for src_in in (0.0, 3.0):
        clip, cs, audio = render_clip(asset, src_in, 1.0, tmp_path)
        offsets.append(sound_offset(audio, clip, cs, 1.0, 2.5, FPS, span=0.4))
    assert abs(offsets[0] - offsets[1]) < 0.012, f"from the start {offsets[0] * 1000:+.0f} ms, from the middle {offsets[1] * 1000:+.0f} ms"
    assert abs(offsets[0] + delay) < 0.03, f"the file's delay is {delay} s, the sound is {offsets[0] * 1000:+.0f} ms off"


def test_late_sound_stays_late_in_a_fast_clip_too(tmp_path):
    """At x2 the delay is half as long on the timeline: the sound of the clip cut from the very start must still begin
    ``delay / 2`` after the picture (before the fix it began at once, 0.011 s after)."""
    asset = late_sound_source(tmp_path, 0.25)
    _clip, _cs, audio = render_clip(asset, 0.0, 2.0, tmp_path)
    hop = AUDIO_RATE // 100
    rms = np.sqrt((audio[: len(audio) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1))
    onset = float(np.argmax(rms > 0.02)) / 100
    assert abs(onset - 0.125) < 0.03, f"the sound begins at {onset:.2f} s, expected about 0.125 s"


def test_zz_the_scenarios_looked_at_every_kind_of_frame_and_sound():
    """If the random scenarios drifted into only easy cases (no crossfade, no hole, no silence) the oracle would still pass.
    Needs the tests above to have run in this process; skipped when this test is run alone."""
    if not COVERAGE:
        pytest.skip("run together with the other oracle tests")
    short = {k: (COVERAGE[k], n) for k, n in COVERAGE_WANTED.items() if COVERAGE[k] < n}
    assert not short, f"the oracle looked at too little: {short} (got, wanted); all counts: {dict(COVERAGE)}"
