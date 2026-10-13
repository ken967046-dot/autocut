# API 速查（給協作者：先看這份，需要細節再讀原始碼）

## 資料模型（`autocut.core.project`）
- `Project`: `id, name, assets{id→Asset}, sequences[], active, meta, revision, path`；`sequence(seq_id=None)`、`asset(id)`、`find_asset_by_path(p)`、`content_json()/content_sha256()`、`to_dict()/from_dict()`、`save(path)/load(path)`。複製：`Project.from_dict(p.to_dict())`。
- `Asset`: `id, path, name, kind(video|audio|image), duration, width, height（**顯示尺寸**，已含旋轉與像素比）, fps, has_video, has_audio, audio_channels, sample_rate, rotation, sar, fingerprint, created, missing(property)`。
- `Sequence`: `id, name, width, height, fps, sample_rate, video_tracks[]（由下到上，V1 在最下）, audio_tracks[], cues[], markers[], caption_style, background, meta`；`tracks()`、`find_track(id)`、`find_clip(id)->(track, clip)`、`find_cue(id)`、`all_clips()->(track, clip)`、`duration`、`aspect_label`、`is_portrait`；`Sequence.create(name,w,h,fps,video_tracks=2,audio_tracks=2,seq_id=None)`（軌道 id 是**隨機**的）。
- `Track`: `id, name, kind(video|audio), clips[], muted(視訊軌=隱藏), locked, volume`；`end()`、`clip_at(t)`。
- `Clip`: `id, asset_id, start, src_in, src_out, speed, volume, muted, enabled, fade_in, fade_out, transform, look, label, color, note`；`duration=(src_out-src_in)/speed`、`end`、`src_at(t)`。
  **視訊軌上的片段同時帶畫面與聲音**；音訊軌放配樂。**同一軌道上交疊＝交叉淡化**（後開始的在上、淡入；最多同時兩層）。
- `Transform`: `fit(cover|contain|blur), zoom(>=1), cx, cy, keys=[[t_local,cx,cy],...], flip_h`。`cover`＝在來源畫面上取與輸出同比例的視窗（9:16 取景）；`keys` 的時間是**片段內秒數**。`Look`: `brightness(-1..1), contrast(0..3), saturation(0..3)`。
- `CaptionCue(id,start,end,text,style)`、`Marker(id,time,name,color)`、`CaptionStyle(font,size,bold,color,outline_color,outline,shadow,position(bottom|center|top),margin,max_chars,box,box_color,box_opacity)`（size/outline/margin 皆為畫面高度比例）。
- `validate_sequence(proj, seq) -> list[Issue(level error|warn, code, message, where)]`。

## 修改一律走 `core.ops`（純函式，丟 `OpError`，訊息為繁中）
GUI 透過 `state.op(label, "<函式名>", *args)`（`seq` 由 state 帶入）或 `state.do(label, lambda p: ...)`（多個 op＝一次復原步驟）。時間會自動對齊影格。
```
add_track(kind, name=None, index=None)   remove_track(track_id, force=False)   set_track_props(track_id, **props: name/muted/locked/volume)
place_clip(asset_id, track_id, start=None, src_in=0.0, src_out=None, mode="overwrite"|"insert"|"append"|"reject"|"free", **props) -> Clip
split_clip(clip_id, at)   split_at_time(at, clip_ids=None, track_ids=None) -> 新右半片段   # 兩者都拒絕（OpError，什麼都不改）：落在交叉淡化內、落在片段自己的淡入/淡出內、離邊緣太近、或切完有一半和另一個片段起迄完全一樣
trim_clip_start(clip_id, new_start, ripple=False)   trim_clip_end(clip_id, new_end, ripple=False)   slip_clip(clip_id, delta_src)
move_clip(clip_id, new_start, track_id=None, resolve="overwrite"|"insert"|"reject"|"free")   move_clips(clip_ids, delta, resolve="overwrite")   # 移到原位＝沒改；輕推交叉淡化中的片段只改交疊長度（夥伴保持完整，除非落點上還有別的片段會被它蓋住）；未知 resolve 在動手前就拒絕
remove_clips(clip_ids, ripple=False)   ripple_delete_range(t0, t1, track_ids=None, captions=True)   close_gaps(track_id)   # 在主軌 V1 上的波紋，字幕與標記跟著走
set_speed(clip_id, speed, ripple=True)   set_crossfade(clip_id, dur)（讓此片段與前一片段交疊 dur 秒；dur=0 且本來就沒有交疊＝什麼都不做）
set_clip_props(clip_id, **props)  # volume, muted, enabled, fade_in, fade_out, label, color, note, fit, zoom, cx, cy, flip_h, brightness, contrast, saturation
set_transform_key(clip_id, t_local, cx, cy)   remove_transform_key(clip_id, index)   clear_transform_keys(clip_id)
add_cue(start, end, text="")  update_cue(cue_id, start=None, end=None, text=None)  remove_cues(ids)  split_cue(cue_id, at, text_left=None, text_right=None)
merge_cues(ids)  shift_cues(delta, ids=None)  replace_cues(cues)  set_caption_style(**kw)   # 字幕時間對齊影格、至少 MIN_CUE；split_cue 的合法性用查詢函式 cue_split_ok(seq, cue, at)（介面據此啟用「分割字幕」）
add_marker(time, name="", color=...)  remove_marker(marker_id)  set_sequence_range(t_in, t_out)  # 存在 seq.meta["range"]
create_sequence(name, width, height, fps, activate=True, video_tracks=2, audio_tracks=2)  rename_sequence(name)  delete_sequence(seq_id)  duplicate_sequence()
set_sequence_format(width=None, height=None, fps=None)  normalize_sequence()
add_asset(asset)（路徑相同就沿用既有）  remove_asset(asset_id, force=False)  relink_asset(asset_id, info: Asset)
```
（專案層級的 op——`create_sequence`、`delete_sequence`、`add_asset`、`remove_asset`、`relink_asset`——第一個參數是 `proj`、沒有 `seq`，用 `state.project_op(label, name, ...)`；其餘都吃 `(proj, seq, ...)`，用 `state.op(...)`（作用在使用中的序列；要作用在別的序列用 `state.editor.op(label, name, seq_id, ...)`）。）
常數：`ops.MIN_CLIP=0.1`、`ops.MIN_CUE=0.2`、`ops.IMAGE_DEFAULT=5.0`（圖片預設秒數）；`core.project.MIN_FRAGMENT=0.03`（短於此的碎片會被清掉）；`core.timecode.EPS=1e-6`、`TOUCH=1e-4`（「相接」的容許值，存檔只留 6 位小數）。
查詢函式（不改專案、不套 NaN 檢查）：`trim_limits(proj, track, clip, edge, ripple)`——拖曳修剪的可動範圍，**介面的拖曳預覽也用它**；`cue_split_ok(seq, cue, at)`。
所有其他公開 op 都會拒絕 NaN/inf（`OpError`，什麼都不改）。

## `core.timecode` / `core.geometry`
`snap_time(t, fps)`、`frame_of(t, fps)`、`fmt_timecode(t, fps)`（HH:MM:SS:FF）、`fmt_clock(t, millis=True)`、`parse_timecode(text, fps)`；
`interp_keys(keys, t, cx0, cy0)`、`base_window(dw, dh, out_w, out_h) -> (ww, wh)`（顯示尺寸內、與輸出同比例的最大視窗，單位=來源像素）、`crop_window(dw,dh,out_w,out_h,zoom)`、`crop_topleft(dw,dh,cw,ch,cx,cy)`（視窗中心 (cx,cy) 為 0..1，會夾在畫面內）、`crop_rect_px(...)`、`contain_rect(...)`、`split_keys/shift_keys/limit_keys/scale_keys`。

## `autocut.gui.state.AppState`（QObject；所有元件共用）
屬性：`editor, project, seq_id, playhead, selection(set[clip id]), cue_selection(set), fps, dirty, title, project_path`。
訊號：`project_replaced, edited(kind do|undo|redo|reset, label), sequence_changed, selection_changed, cue_selection_changed, playhead_changed(float), reframe_requested(str), status(str), dirty_changed(bool), title_changed(str)`。
方法：`sequence()`、`op(label, name, *args, **kw)`（OpError 會發 `status` 並重新丟出）、`project_op(label, name, ...)`、`do(label, fn)`、`undo()/redo()`、`set_sequence(id)`、`set_playhead(t, snap=True)`、`select(ids, mode=replace|add|toggle|remove)`、`clear_selection()`、`select_cues(ids, mode)`、`selected_clips()`、`request_reframe(clip_id)`、`get_range()/set_range(t_in,t_out)`、`new_project/open_project(path)/save_project(path=None)/replace_project(p)`、`run_for_sequence(seq_id=None)`、`record_technical_qa(run_id, status, summary)`、`human_review(run_id, verdict, notes)`（**只能由使用者按鈕觸發**）。
## `autocut.gui.workers`
`TaskRunner(fn(progress, cancel), parent=None, name="")`：訊號 `progress(float,str) finished(object) failed(str) cancelled() done()`；`.start() .cancel() .running .wait()`；`run_task(fn, parent=, on_progress=, on_done=, on_error=, on_cancel=)`；`friendly_error(exc)`；測試用 `wait_for_idle(app, [runners])`。
## `autocut.gui.widgets`
`TimecodeEdit(fps)`（`set_time/time/set_fps`、訊號 `timeCommitted(float)`）、`ClickableLabel`、`HumanReviewBox(state)`。`gui.theme`: 色彩常數（`BG PANEL PANEL_ALT BORDER TEXT TEXT_DIM ACCENT ACCENT_DARK OK WARN ERROR TL_* CLIP_VIDEO CLIP_AUDIO CLIP_IMAGE CLIP_CAPTION CLIP_SELECTED PLAYHEAD MARKER SNAP_LINE RANGE_FILL`）、`apply_theme(app)`。

## 媒體與渲染
- `media.probe.probe_asset(path) -> Asset`（`ProbeError`）；`media.cache.thumbnail(asset, t=0.0, height=96) -> Path(jpg)`、`filmstrip(asset, count=12, height=54) -> Path(單張橫條 jpg，count 格並排)`、`waveform(asset, per_second=50) -> np.ndarray(0..1 峰值)`、`decode_audio_mono(path, rate, start, duration)`。
- `media.ffmpeg`: `ffmpeg_info()`、`ffmpeg_available()`、`detect_hw_encoder()`、`run_ffmpeg(...)`、`Cancelled`、`FFmpegError`、`FFmpegMissing`、`_NOWINDOW`（Windows 隱藏視窗旗標）。
- `render.runner.render_sequence(proj, seq, out_path, settings=None, progress=None, cancel=None) -> RenderResult(path, duration, frames, width, height, fps, segments, cached, elapsed, encoder, ffmpeg, warnings)`；
  `RenderSettings(preset, encoder, workers, burn_captions, write_srt, frame_range, use_cache, loudness, normalize)`；`render_preview(proj, seq, progress=None, cancel=None, frame_range=None)`（快取檔 `cache_dir("render","preview")/preview_<seq id>.mp4`）；`render.presets.PRESETS`（`youtube_1080p, shorts_1080x1920, source_quality, draft_fast, preview_540p`）；`render.compiler.MissingMediaError`（素材遺失）。
- `render.qa.technical_qa(path, expect_duration=, expect_size=, expect_fps=, expect_audio=True, intended_black=(), deep=True) -> QAReport`：`.status`（"pass"/"warn"/"fail"/"not_run"）、`.checks[(name,status,detail)]`、`.summary()`、`human_review` 永遠 "pending"。
- 同一份渲染語意（取景/淡化/交叉淡化/軌道順序/背景/速度）以 `render/compiler.py` 為準。

## 自動剪輯（引擎）
`autocut.autoedit.api`: `propose_shorts(proj, asset_id, params, progress, cancel) -> EditPlan`、`propose_assemble(proj, asset_ids, params, progress, cancel)`、`audit_plan(plan, proj, rules=None) -> AuditReceipt`、`apply_plan(editor, plan, receipt) -> [新序列 id]`、`preview_project(proj, plan, index) -> (scratch Project, seq_id)`、`AutoEditError`、`Cancelled`；
參數類別在 `autoedit/config.py`（`ShortsParams, AssembleParams, AuditRules, PLATFORM_RULES, ASPECTS`）。`EditPlan(kind, brief, sources, assets, sequences, decisions, notes)`、`.subset(indexes)`、`.sha256()`、`.to_dict()`；`AuditReceipt(.accepted, .errors, .warnings, .findings[Finding(rule,severity,message,where)])`。
`core.plan.find_run(project, run_id)`、`project.meta["ai_runs"]`（每筆：`run_id, kind, plan_sha256, applied_at, sequence_ids, warnings, technical_qa, human_review, outcomes`）。序列的 `meta["ai"]["run_id"]` 指向其 run。

## 測試
`QT_QPA_PLATFORM=offscreen /home/claude/venvs/autocut/bin/python -m pytest tests/<檔> -q`。共用 fixture（`tests/conftest.py`，**不要改**）：`qapp`、`proj`（假素材 a1 20s/a2 10s 視訊、m1 60s 音訊）、`media_dir`（land.mp4 1280x720 8s 有聲、port.mp4 720x1280 5s 藍、silent.mp4 4s 無聲綠、music.wav 20s）、`real_proj`（以上檔案 probe 後的專案）。截圖存 `screenshots/<區域>_<名稱>.png`（已在 .gitignore），並用 Read 工具實際看圖。
