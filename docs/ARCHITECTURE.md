# 架構

```
素材(拖放) ─▶ probe ─▶ Asset ─┐
                              ├─▶ Project ──▶ Sequence(V 軌/A 軌/字幕/標記) ──▶ 渲染 ──▶ 技術 QA ──▶ 人工審片
自動剪輯：evidence ─▶ EditPlan(hash) ─▶ AuditReceipt(綁 hash + revision) ─▶ apply(單一復原步驟) ─┘
```

## 資料模型（`core/project.py`，schema `autocut.project/1`，副檔名 `.acproj`，JSON）
- `Project`：`assets{id→Asset}`、`sequences[]`、`active`、`meta`、`revision`（每次變更/復原/重做 +1，單調遞增）。
- `Asset`：路徑、種類(video/audio/image)、長度、**顯示尺寸**（已套用旋轉與像素比）、fps、聲音資訊、指紋。
- `Sequence`：`width/height/fps`、`video_tracks`（由下到上）、`audio_tracks`、`cues`（字幕；`style == {"kind": "card"}` 的是字卡，
  自己一層、樣式由 `card_style(caption_style)` 推導、不寫進 .srt）、`markers`、`caption_style`、`meta`。
- `Clip`：`start`（時間軸位置）、`src_in/src_out`（素材範圍）、`speed`、`volume`、`fade_in/out`、`transform`、`look`。
  `duration=(src_out-src_in)/speed`。**視訊軌上的片段同時帶畫面與聲音**；音訊軌放配樂/旁白。
- `Transform`：`fit`(cover/contain/blur)、`zoom`、`cx/cy`（取景窗中心，0..1）、`keys=[[t,cx,cy]...]`（取景運鏡關鍵格，片段內秒數）、`flip_h`。
  9:16 取景 = `cover` 模式在 16:9 畫面上取一個 9:16 視窗；`keys` 讓視窗跟著人物移動。
- **同軌交疊 = 交叉淡化**（後開始的片段在上面淡入；聲音等功率淡入淡出），同一時刻最多兩層。

## 修改一律走 `ops` + `Editor`
`core/ops.py`（純函式，丟 `OpError`）→ `core/editor.py::Editor.op(label, name, seq_id, ...)` 包成一個 `History` 交易：
前後做 canonical JSON 快照，沒變就不記錄，失敗就還原，成功就 `revision += 1`。復原/重做用 zlib 快照。
主要操作：`place_clip`、`split_clip`、`split_at_time`、`trim_clip_start/end`（可波紋）、`slip_clip`、`move_clip(s)`（覆蓋/插入/拒絕）、
`remove_clips`、`ripple_delete_range`、`close_gaps`、`set_speed`、`set_crossfade`、`set_clip_props`、關鍵格、字幕與標記操作、序列與素材操作。

### 剪輯規則（`ops` 的行為契約；測試在 `tests/test_core*.py`）
- **失敗就什麼都不改**：被拒絕的操作不留復原步驟，沒改到內容的操作也不留。`ops._guarded` 的安全網（`OpInvalid`）只是最後一道防線——
  隨機測試（`test_core_fuzz.py`）要求它**永遠不會被觸發**：正常操作必須自己擋掉會弄壞時間軸的情況。
- **字幕與標記跟著主軌 V1**：V1 上改變「後面素材何時播放」的編輯（插入、波紋刪除／修剪、變速、關閉空隙…）會讓後面的字幕與標記一起移動或收縮——
  字幕黏在它原本對著的那一刻畫面上（`test_core_sync.py` 用一份獨立的 oracle 檢查）。其他軌道（配樂、疊圖）的編輯不動字幕。
  字幕最短 `MIN_CUE`、起迄對齊影格，順序以存檔的 6 位小數為準（復原／重做後順序不變）。
- **移動**：移到原位＝完全沒改。輕推一個正在交叉淡化的片段，只會改交疊長度，夥伴片段保持完整；如果落點上還有別的片段，切掉它會讓夥伴把它整段蓋住，
  就改成一般的覆蓋（夥伴也被切）——做法是先在複本上試算（`_keeps_it_valid`）。
- **分割（剃刀）**會被拒絕的情形：落在交叉淡化裡、落在片段自己的淡入／淡出裡、離邊緣太近、切完有一半和另一個片段的位置與長度完全一樣。
- **數字**：NaN／inf 一律拒絕（所有公開 op 外包一層 `_finite`；`trim_limits`、`cue_split_ok` 這類純查詢除外）。
- **時間軸最長 24 小時**（`core/timecode.py::MAX_TIMELINE`，等於 `HH:MM:SS:FF` 時鐘的長度）：片段結尾、字幕、標記、入出點都不能超過。
  呼叫端給的時間一律經 `ops._sn`，超過就丟 `OpError`（「時間軸最長 24 小時…」）；波紋／插入／變速把後面的片段推過去，
  由 `_guarded` 的 `too_long` 擋下（一般的拒絕，不是 `OpInvalid`）。`validate_sequence` 也把超過的片段／字幕列為錯誤（輸出前的檢查會看到）。
  原因：在「開始」欄輸入 `1000000000` 曾讓時間軸的捲動條（32 位元）溢位而丟例外。損壞或手改的專案檔仍可能帶超大數字，
  所以時間軸另有保險（捲動範圍以 24 小時為上限、`QRect` 座標夾在 ±2²⁸ 內）；`TimecodeEdit` 也把輸入夾在 24 小時內（`1e308` 連顯示都會算爆）。
- **讀專案檔**：`Project.load` 對任何壞檔只丟 `ProjectFileError`（是 `ValueError`），訊息是給使用者看的；
  NaN／Infinity 不是時間、欄位型別不對（字串、數字、少了欄位…）都包成同一種錯誤。介面讀不了的自動存檔會改名成 `<名稱>.unreadable[N]`
  （內容不刪、不再詢問）。
- **一格底線**：修剪／變速／交叉淡化之後，片段除了交叉淡化的部分，至少還留一格自己的畫面；`set_crossfade(c, 0)` 在沒有交疊時什麼都不做。
- **技術 QA 與人工審片綁序列內容**：`meta["ai_runs"][i]["by_sequence"][seq_id]` 記 `technical_qa_sha`、`human_review_sha`（＝當時的 `sequence_sha256(seq)`）；
  序列改過之後狀態變成 `stale`，要重跑 QA、重新審片。機器只會說「技術檢查通過」，不會說「審片通過」。
- **怎麼驗證這些規則**：`test_core_hardening.py`（逐條規則）、`test_core_sync.py`（字幕）、`test_core_fuzz.py`（隨機剪輯序列）、
  `test_render_oracle.py`（真的渲染再逐格比對畫面與聲音）；更長時間的隨機測試用 `scripts/dev/campaign.py`。
  已知缺口：`ripple_delete_range` 目前只有核心與測試，**介面還沒接**（用剃刀＋波紋刪除可以做到同樣的事）。

## 渲染（`render/`）
1. `compile_sequence` 把序列在每個片段邊界切成 **segment**；段內每個作用中的片段都涵蓋整段 → 每段只是個小 ffmpeg 工作（沒有時間偏移運算）。
2. 每段的 `segment_signature`（素材路徑/大小/mtime、所有變換參數、字幕、編碼設定）決定快取檔名 `seg_<sig>.mov`（H.264 + PCM float）。
3. `runner.render_sequence`：快取命中就跳過，其餘平行跑 ffmpeg；最後用 concat demuxer **串流複製影像**，音訊一次 AAC 編碼，
   可選兩段式 `loudnorm`（`measured_*` + `linear=true`）。硬體編碼失敗自動整批改用 libx264。
4. 取景：`crop` 的 `x/y` 用「依時間的分段線性運算式」做運鏡（`piecewise_expr`，平衡 if 樹），只帶入該段需要的關鍵格。
5. 淡入淡出只部分落在段內時，會多解碼一小段「pre-roll」讓斜率精準。
   **片段放在哪一格**：`cs = frame_of(start)`（起點四捨五入到最近的格；漣漪剪輯或變速後起點可能落在兩格之間，例如 25 fps 的 124.5 格），
   畫面與聲音都以 `cs` 為準（`-ss src_in + S/fps*speed`，影像鏈裡**不**做 `setpts=PTS-STARTPTS`，改靠 `fps=...:start_time=0` 補第一格；
   不然 `-ss` 之後的第一格可能晚一個來源格，畫面會比聲音早約 33 ms）。`RENDER_VERSION` 因此升到 6，舊快取不會被誤用。
   對照測試（`tests/test_render_oracle.py`）的模型同樣以「放定的那一格」算每格該出現哪個來源格（容許 ±1 來源格）。
   **畫面與聲音同步**另有一個永久測試（`test_picture_and_sound_of_a_clip_agree_to_well_under_a_frame`，30／25／24 fps）：
   速度 1、有一段只有自己的片段，畫面用條碼量、聲音用「上升音高」滑動比對（0.1 ms 解析度）；實測聲音固定晚約 5 ms（AAC 解碼＋重取樣）、
   畫面與聲音相差最多 19 ms（30 fps，來源格量化）／10 ms（25、24 fps）；限值 15／22／25 ms，畫面或聲音差一格（33–42 ms）一定超過
   （把修正拿掉測試會失敗）；三個限值依序是「聲音」「畫面」「兩者之差」。
   **聲音比畫面晚開始的檔案**（螢幕錄影、部分相機；容器裡聲音串流的起始時間比較晚）：音訊鏈是 `aresample=48000:first_pts=0`，
   **不**做 `asetpts=PTS-STARTPTS`、也不帶 `async`——從檔案最開頭剪出來的片段，那段延遲用靜音補上、保留下來；
   原本的做法會把聲音整個往前拉（量到 78 ms 的延遲少了 77 ms、255 ms 的少了 226 ms，也就是聲音提早出現）。從中間剪的片段不受影響。
   `RENDER_VERSION` 因此升到 7。測試：`test_sound_that_starts_late_in_the_file_keeps_its_offset_for_a_clip_cut_from_the_very_start`
   （延遲 0.1／0.25 s：從頭剪與從中間剪的聲音對畫面關係要一樣，且等於檔案自己的延遲）與 `test_late_sound_stays_late_in_a_fast_clip_too`（2 倍速時延遲縮成一半）。
   **沒量過**：真實攝影機／手機拍的素材——測試素材都是合成的（含一組「聲音晚開始」的合成檔），沒有真實相機的容器怪癖。
6. `qa.technical_qa`：可讀、尺寸、fps、長度、完整解碼、黑畫面、凍結、靜音、響度/真峰值 → **只代表技術檢查**；`human_review` 永遠先是 `pending`。
7. `live.py`（GUI 用）：OpenCV 即時合成預覽，與渲染共用 `core/geometry.py` 的取景數學。

## 自動剪輯（`analysis/`、`autoedit/`）
見 `docs/AUTOEDIT_METHOD.md`。三種計畫：`shorts`（長片 → 多支 9:16）、`longform`（一支長片 → 精華長片：重點＋去水詞＋講者特寫，§3d）、
`assemble`（多支毛片 → 一支長片）。挑段依「說了什麼」（`autoedit/highlights.py`，§3b；文字來自字幕檔／同名字幕／語音辨識，選用 AI 判斷）。
流程：`analyze`（＋長片的 `analysis/speakers.py`）→ `propose_shorts / propose_longform / propose_assemble`（得到 `EditPlan`，純資料）→ `audit_plan`（得到綁定計畫雜湊與專案 revision 的 `AuditReceipt`）
→ `apply_plan`（以單一復原步驟把序列加入專案，並記錄 `meta["ai_runs"]`）→ 渲染 → 技術 QA → 使用者按「我已審片」。

## 介面（`gui/`）
詳見 `docs/GUI_SPEC.md`。原則：UI 只讀 `Project`、只透過 `Editor` 改；耗時工作（探測、縮圖、分析、渲染）一律在背景執行緒；時間軸/監視器共用 `AppState`。
- **鍵盤焦點**（`gui/widgets.py`、`main_window.py`）：按鈕、勾選框、滑桿、不可編輯的下拉選單只用 Tab 取得焦點（按一下滑鼠不拿走鍵盤，
  並讓正在輸入的數字框先送出）；數字框／時間框按 Enter 送出後把鍵盤還給時間軸；主視窗 `keyPressEvent` 把沒人要的按鍵轉給時間軸，
  但文字框、數字框、清單（`keeps_its_keys`）自己留著。數字框沒有自己的文字復原可做時，Ctrl+Z／Ctrl+Y 交給專案的復原（`wants_project_undo`；
  `QAbstractSpinBox` 要在 spin box 本身處理 `ShortcutOverride`，因為焦點代理的關係，在內部的 line edit 上做不到）。
- **播放跟著剪輯**：播放中序列被改短或清空，播放邊界即時更新（`_follow_the_edit`），清空就停下；**匯入不跨專案**：新增／開啟專案會取消進行中的匯入並丟掉遲到的結果（`_epoch`）。
- **版面**：預設分割比例 `SPLIT_SIZES = [470, 450]`（預覽：時間軸），1100×700 的筆電上時間軸至少看得到字幕、V2、V1 與 A1（其餘垂直捲動）。已存的分割狀態照舊還原。

## 執行緒與取消
長工作吃 `progress(fraction, message)` 與 `threading.Event cancel`；`run_ffmpeg` 會在取消時殺掉子行程並清掉未完成的暫存檔。
