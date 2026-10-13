# GUI 規格（PySide6 桌面軟體）

這是桌面軟體，不是網頁。使用者要的核心體驗：**把檔案直接拖進視窗 → 自動剪好 → 像剪輯軟體一樣逐段修正 → 輸出**。
所有文字繁體中文。深色主題。目標 Windows 11，但在 Linux 無頭環境（`QT_QPA_PLATFORM=offscreen`）完成測試與截圖。

## 0. 硬性規則
- 只讀 `Project`，**所有修改都走 `state.editor.op(label, "<ops 函式名>", seq_id, ...)` 或 `state.editor.do(label, fn)`**。不得直接改 `Clip/Track` 欄位。
  拖曳過程只畫「鬼影預覽」，放開滑鼠才呼叫一次 op（＝一次復原步驟）。
- 耗時工作（probe、縮圖、波形、分析、渲染）都在背景執行緒（`QThreadPool`/`QThread`），UI 執行緒任何一次回呼不得 > 50 ms。
- 介面不得依賴網路。語音辨識為選用功能：沒安裝時按鈕要說明如何安裝而不是崩潰。
- 不要把「機器技術檢查通過」寫成「審片通過」。「我已審片」按鈕只在使用者點擊時呼叫 `core.plan.set_human_review`。

## 1. 檔案與職責（`src/autocut/gui/`）
| 檔案 | 內容 |
|---|---|
| `app.py` | `main(argv=None) -> int`：建立 QApplication（HiDPI、App 名稱、圖示、深色主題）、解析命令列的檔案/資料夾/.acproj、顯示 `MainWindow`、`app.exec()` |
| `theme.py` | 深色調色盤 + QSS；顏色常數（影片/音訊/圖片片段色、選取色、播放頭色）；字型挑選（Microsoft JhengHei UI → 系統預設） |
| `state.py` | `AppState(QObject)`：`editor`、`seq_id`、`playhead`、`selection`(set[str])、`range`；訊號 `project_replaced`、`edited(str kind, str label)`、`selection_changed`、`playhead_changed(float)`、`sequence_changed`、`status(str)`；`open_project/save_project/new_project`；`current_sequence()` |
| `workers.py` | `Task`/`TaskRunner`：把函式丟背景跑，訊號 `progress(float,str)`、`finished(object)`、`failed(str)`、`cancelled`；內含 `threading.Event cancel` |
| `importer.py` | `ImportController`：`import_paths(paths)`（展開資料夾、背景 `probe_asset`、一次復原步驟 `add_asset`）、匯入後的**建議橫幅**邏輯（見 §4） |
| `media_bin.py` | 素材庫（縮圖、名稱、長度、解析度、遺失標記）；接受 OS 拖放；內部拖曳 MIME `application/x-autocut-asset`；右鍵選單；雙擊加入時間軸 |
| `timeline.py` | 自繪時間軸（見 §2） |
| `monitor.py` | 節目監視器 + 傳輸列 + 播放控制 + 取景編輯模式（見 §3） |
| `playback.py` | `PlaybackController`：用 `perf_counter` 時鐘驅動；靜音即時播放（`LiveCompositor`）與「含聲音預覽」（`QMediaPlayer` 播放 `render_preview` 的快取檔，失敗就退回靜音） |
| `inspector.py` | 檢查器分頁：片段 / 字幕 / 序列 / 檢查（見 §5） |
| `autoedit_dialog.py` | 兩個自動剪輯對話框（短影音、長片）與結果審閱頁（見 §6） |
| `export_dialog.py` | 輸出對話框 + 進度 + 技術 QA 報告 + 人工審片按鈕（見 §7） |
| `main_window.py` | 主視窗：選單、工具列、版面、全視窗拖放、最近檔案、自動存檔/復原、未儲存提示、`QSettings` |
| `widgets.py` | 共用小元件（時間碼輸入框、可點擊標籤…） |

另外在 `src/autocut/render/live.py`（無 Qt）：`LiveCompositor`（見 §8）。

## 2. 時間軸（`timeline.py`）
自繪 `QWidget`（`QPainter`，只畫可見區），左側軌道標頭（名稱、靜音 M、鎖定 L），上方尺規，中間軌道區。由上到下：**字幕軌 → V(n)…V1 → A1…An**。
- **縮放**：`pps`（每秒像素，2–1500）。Ctrl+滾輪以游標為中心縮放；`+`/`-`；`\` 或按鈕「符合視窗」；底部縮放滑桿。
- **捲動**：滾輪垂直、Shift+滾輪或水平滾輪橫向、中鍵拖曳平移；播放時播放頭離開可視區要自動捲動。
- **尺規**：自適應刻度（影格/秒/分），顯示 `mm:ss` 或時間碼；在尺規上點擊/拖曳 = 擦洗（scrub），對齊影格；顯示標記旗；顯示入/出點區間（`seq.meta["range"]=[in,out]`，用 `I`/`O` 設定，需新增對應 op）。
- **片段繪製**：圓角矩形，顏色依素材種類；標籤（素材名/自訂 label）與長度；影片片段背景用 `media.cache.filmstrip`（非同步，依 `src_in..src_out` 取片段）；音訊波形（`media.cache.waveform`）；
  淡入淡出三角把手；交疊區用 X 形漸層表示交叉淡化；取景關鍵格畫成片段底部的小菱形；變速徽章（如 `2×`）；靜音/停用變灰；素材遺失紅色斜線。
- **選取**：點擊選取；Ctrl 加減選；Shift 在同軌範圍選；框選（空白處拖曳）；空白處點擊 = 取消選取；`Ctrl+A` 全選。
- **移動**：拖片段本體（可跨同類型軌道），**吸附**到其他片段邊緣/播放頭/標記/0（門檻 8px，工具列「吸附」開關，按住 Alt 暫時關閉）。
  放開 → `move_clip` / `move_clips`（預設 `resolve="overwrite"`；按住 Ctrl = `insert`）。拖曳時顯示鬼影與新位置時間碼。
- **修剪**：拖片段左/右邊緣（邊緣 8px 範圍，游標變 ↔）；Shift = 波紋修剪；放開 → `trim_clip_start/end`。顯示新入出點與長度的提示。
- **滑動素材（slip）**：Alt+拖片段本體 → `slip_clip`。
- **淡入淡出**：拖左上/右上把手 → `set_clip_props(fade_in/fade_out)`。
- **交叉淡化**：右鍵「交叉淡化…」（對話框輸入秒數）→ `set_crossfade`；在交疊區拖曳也可調整。
- **右鍵選單**（片段）：在播放頭分割、刪除、波紋刪除、靜音/取消靜音、停用/啟用、速度…、淡入淡出…、交叉淡化…、取景→重設/加入關鍵格/清除關鍵格、在素材庫顯示、屬性。
  （軌道標頭）：新增視訊軌/音訊軌、刪除軌道、重新命名。（空白處）：貼上、在此新增標記、移除間隙（`close_gaps`）。
- **字幕軌**：字幕以區塊顯示（顯示文字）；拖曳移動、拖邊緣調整、雙擊在原地編輯文字（可用小型 `QLineEdit` 疊在區塊上）、右鍵分割/合併/刪除。→ `update_cue/split_cue/merge_cues/remove_cues`。
- **鍵盤**（時間軸或主視窗有焦點）：Space 播放/暫停；←/→ ±1 格；Shift+←/→ ±1 秒；Home/End；↑/↓ 上/下一個剪輯點；`S` 或 `Ctrl+K` 在播放頭分割（選取的片段，沒選就切所有未鎖定軌的片段，`split_at_time`；播放頭落在交叉淡化中間時整個拒絕並在狀態列說明，因為切過轉場會讓畫面跳回前一個片段）；
  Delete 刪除；Shift+Delete 波紋刪除；`Ctrl+Z`/`Ctrl+Y`/`Ctrl+Shift+Z`；`M` 標記；`I`/`O` 入出點；`Ctrl+A`。
- **拖放**：接受素材庫內部拖曳與 OS 檔案（先走 `ImportController` 再放置）。游標位置決定軌道與時間；影片→視訊軌，純音訊→音訊軌；多檔依序排。放開前畫鬼影與目標軌高亮。
- 空序列時顯示提示字：「把影片拖到這裡，或用『自動剪輯』」。

## 3. 監視器與取景編輯（`monitor.py`, `playback.py`）
- 畫面來自 `LiveCompositor.frame_at(t)`；等比縮放置中；可開「安全區/中心線」；字幕用 `QPainter` 依 `caption_style` 疊在畫面上（近似即可，最終以渲染為準）。
- 傳輸列：⏮ ◀| ▶/⏸ |▶ ⏭、時間碼（可點擊輸入跳轉）、循環、入出點播放；右側顯示「預覽快取：最新／過期／渲染中」與按鈕「渲染預覽（含聲音）」(Ctrl+Enter)。
- 播放：`PlaybackController` 以 `perf_counter` 為時鐘，慢了就丟格不拖慢；預設靜音即時播放；有最新預覽快取時改播快取檔並有聲音（`QMediaPlayer`+`QAudioOutput`，任何錯誤退回靜音並提示一次）。
- **取景模式（重新取景）**：選取 `cover` 片段時，監視器上方顯示切換「取景」。進入後畫面顯示該片段在播放頭時間的**完整來源畫面**（`LiveCompositor.source_frame_at`），
  外圍壓暗，中間是與輸出同比例的取景框：拖曳框移動、滾輪或拉角縮放(zoom)、方向鍵微調；「自動關鍵格」開啟時每次放開滑鼠在該時間加關鍵格
  （`set_transform_key`），關閉時改靜態 `set_clip_props(cx,cy,zoom)`；按鈕「上/下一個關鍵格」「刪除此關鍵格」「清除全部」；旁邊小預覽顯示實際輸出構圖。
  這是「像剪輯軟體般修正 9:16 運鏡」的主要工具。

## 4. 匯入與拖放（`importer.py`, `main_window.py`）
- 整個主視窗接受 OS 拖放（檔案與資料夾）：拖入時顯示半透明覆蓋層「放開以匯入」。副檔名依 `util.MEDIA_EXTS`，字幕檔 `.srt/.ass/.vtt` 另外處理（附加到選取序列為字幕，詢問確認）。
- 匯入 = 背景 probe（進度條）→ 一次復原步驟 `add_asset`（重複路徑自動略過）。失敗的檔案列在結果訊息裡，不中斷其他檔案。
- 匯入完成後在監視器上方顯示**建議橫幅**（可關閉，不擋操作）：
  - 恰好 1 支影片、橫式、長度 ≥ 45 秒 → 「偵測到 1 支 16:9 長影片，要自動剪成 9:16 短影音嗎？」［自動剪成短影音］［放到時間軸］［略過］
  - ≥ 2 支影片 → 「偵測到 N 支素材，要自動剪成一支長片嗎？」［自動合併成長片］［依順序放到時間軸］［略過］
  - 其他（單支短片、音訊、圖片）→ 直接加入素材庫並提示「雙擊或拖到時間軸」。
- 「放到時間軸」：視訊→V1 尾端（依序），音訊→A1；若目前序列是空的，序列尺寸隨第一支影片方向自動調整（直式 1080×1920、橫式 1920×1080，fps 取素材最常見值，限制 24–60），整個動作是一次可復原步驟。

## 5. 檢查器（`inspector.py`）
- **片段**：開始/長度/入點/出點（時間碼輸入）、速度、音量(dB)、淡入/淡出、填滿方式（cover/contain/blur）、縮放、取景中心 X/Y 滑桿、水平翻轉、亮度/對比/飽和、交叉淡化長度、
  「加入取景關鍵格」「清除關鍵格」。多選時只顯示共同可改的欄位。數值在 `editingFinished` / 滑桿放開時才提交（避免每個刻度都產生復原步驟）。
- **字幕**：表格（開始、結束、文字，可編輯）、新增/分割/合併/刪除、位移全部、匯入 SRT/ASS/VTT、匯出 SRT、樣式（字型、大小、顏色、描邊、位置、底距、每行字數、底框）。
  「自動產生字幕」按鈕：呼叫語音辨識（未安裝則說明如何 `pip install -r requirements-speech.txt`）。
- **序列**：名稱、尺寸預設（16:9 / 9:16 / 1:1 / 4:5 / 自訂）、fps、背景色；多序列分頁切換在時間軸上方（新增/複製/刪除/重新命名）。
- **檢查**：顯示最近一次自動剪輯的 `ai_runs` 紀錄（計畫雜湊、警告、技術 QA 狀態）、`validate_sequence` 的問題清單（點擊跳到該片段），以及**人工審片區**（見 §7）。

## 6. 自動剪輯對話框（`autoedit_dialog.py`）
呼叫 `autoedit.api`：`propose_shorts / propose_assemble / audit_plan / apply_plan`（都吃 `progress` 與 `cancel`，在背景跑）。
- **短影音**：參數表單（平台、數量、目標長度、取景方式 face/motion/center/blur、取景靈敏度、字幕來源、緊湊化停頓、配樂…，預設值取自 `ShortsParams`）→［開始分析］→ 進度（可取消）→ **結果審閱頁**：
  每個候選一列：縮圖、時間範圍、長度、分數、「為什麼選這段」（`plan.decisions`）、勾選是否納入；下方列出 `AuditReceipt.findings`（錯誤紅、警告琥珀）；
  ［預覽此段］（把候選序列暫時載入監視器：以 `LiveCompositor` 播放，不改專案）、［套用選取的片段］（`plan.subset(indexes)` → 重新 `audit_plan` → `apply_plan`）。
- **長片**：參數表單（目標長度、比例、順序、節奏、轉場、丟掉品質差的片段、配樂、字幕…）→ 同樣的分析/審閱/套用流程；審閱頁顯示每支來源被使用的片段與總長。
- 套用後：新序列成為使用中序列、狀態列顯示「已建立 N 個序列（Ctrl+Z 可整批復原）」，檢查器「檢查」分頁出現該次紀錄。

## 7. 輸出（`export_dialog.py`）
預設（`render.presets`）、輸出檔案路徑（`QFileDialog`）、範圍（整個序列／入出點）、編碼器（自動偵測硬體／CPU libx264）、燒入字幕、另存 SRT、響度標準化開關。
背景呼叫 `render.runner.render_sequence`（進度條＋取消）。完成後自動跑 `render.qa.technical_qa`（預期長度/尺寸/fps 取自序列），顯示 `QAReport.summary()`，並提供：
［開啟資料夾］［播放］，以及**人工審片區**：「我看過了」→［通過］［需修改］［退回］三個按鈕，點擊才呼叫 `core.plan.set_human_review`（若這個序列來自某次 `ai_runs`）；
在使用者點擊之前一律顯示「尚未審片（機器不會代替你判斷好不好看）」。

## 8. `render/live.py`（無 Qt，可單元測試）
```python
class LiveCompositor:
    def __init__(self, project: Project, max_side: int = 720): ...
    def set_sequence(self, seq_id: str) -> None
    def frame_at(self, t: float) -> "np.ndarray"            # BGR uint8，序列比例，長邊 <= max_side
    def source_frame_at(self, clip_id: str, t_local: float) -> "np.ndarray"   # 來源完整畫面（顯示尺寸，取景編輯用）
    def invalidate(self) -> None                             # 專案改動後清除快取
    def close(self) -> None
```
實作：OpenCV `VideoCapture` 池（LRU，最多 6 個）、順序讀取最佳化（下一格直接 `read()`，小幅向前 `grab()`，否則 `CAP_PROP_POS_FRAMES` seek）、圖片快取；
`asset.sar` 修正；`core.geometry` 的取景數學（cover/contain/blur + 關鍵格內插，與 ffmpeg 渲染一致）；亮度/對比/飽和；水平翻轉；淡入淡出與同軌交疊的 alpha 混合；多視訊軌疊加；空檔為背景色。
需有單元測試（合成素材比對像素）。

## 9. 持久化與安全
- `QSettings("AutoCutStudio","AutoCutStudio")`：視窗幾何/狀態、最近專案、上次輸出資料夾。
- 自動存檔：每 60 秒、有未儲存變更時寫到 `util.app_data_dir()/autosave/<project id>.acproj`；啟動時若發現比正式檔新的自動存檔，詢問是否復原。
- 關閉/新建/開啟時若有未儲存變更要詢問；存檔用 `Project.save`（原子寫入，保留 `.bak`）。
- 專案內素材路徑遺失時：素材庫標紅，時間軸片段畫斜線；右鍵「重新連結…」→ `probe_asset` 新檔 → `relink_asset`；渲染前檢查並用 `MissingMediaError` 訊息提示。

## 10. 測試與截圖要求（`tests/test_gui_foundation.py`、`test_main_window.py`、`test_timeline.py`、`test_inspector.py`、`test_export_dialog.py` 等，`QT_QPA_PLATFORM=offscreen`）
- 以 `media_dir` 合成素材建立專案，建構 `MainWindow`，用 `QTest`/直接送 `QMouseEvent` 模擬：點選、拖移、修剪、分割（鍵盤 `S`）、刪除、復原/重做、縮放、擦洗；斷言 `Project` 內容改變正確（用 `Project.content_json()` 或讀 clips）。
- 模擬 OS 拖放（`QMimeData` + `QDropEvent`）匯入 1 支橫式長影片 → 出現建議橫幅；匯入多支 → 建議長片。
- 檢查器改值會產生對應的復原步驟；字幕表格編輯；取景模式拖曳取景框會產生關鍵格。
- 每個主要畫面 `widget.grab().save("screenshots/<name>.png")` 並**實際開圖檢查**版面（文字不被截斷、對齊、深色主題一致）。截圖目錄已在 `.gitignore`。

## 11. 元件介面契約（多人/多代理同時開發時以此為準）
地基已存在且有測試：`gui/state.py`（`AppState`）、`gui/workers.py`（`TaskRunner`/`run_task`）、`gui/theme.py`、`gui/widgets.py`（`TimecodeEdit`、`ClickableLabel`、`HumanReviewBox`），
測試用 `tests/conftest.py` 的 `qapp`、`real_proj`、`media_dir` fixture（**不要改 conftest.py**；需要別的 fixture 請放自己的測試檔）。

| 元件 | 檔案 | 建構子 | 備註 |
|---|---|---|---|
| `MainWindow` | `main_window.py` | `MainWindow(state: AppState | None = None, paths: list[str] | None = None)` | 組裝下面所有元件；`app.main()` 建立它 |
| `MediaBin` | `media_bin.py` | `MediaBin(state, importer, parent=None)` | 內部拖曳 MIME `application/x-autocut-asset`，payload 為 UTF-8 JSON `{"assets": ["<asset id>", ...]}` |
| `ImportController` | `importer.py` | `ImportController(state, parent=None)` | `import_paths(paths) -> TaskRunner`；訊號 `imported(list[str] asset_ids, list[str] failed_messages)`、`suggestion(kind: str, list[str] asset_ids)`（kind = "shorts" / "assemble" / "none"） |
| `PlaybackController` | `playback.py` | `PlaybackController(state, parent=None)` | `play()/pause()/toggle()/is_playing`；訊號 `playing_changed(bool)`；播放時更新 `state.set_playhead(...)` |
| `MonitorWidget` | `monitor.py` | `MonitorWidget(state, playback, parent=None)` | 內含取景模式；監聽 `state.reframe_requested` |
| `TimelineWidget` | `timeline.py` | `TimelineWidget(state, parent=None)` | 訊號 `externalFilesDropped(list[str] paths, float t, str track_id)`（OS 檔案拖到時間軸；由 MainWindow 交給 importer 再放置）。時間軸內部處理素材庫的內部拖曳（`place_clip`）。可選 `set_playing(bool)` 供自動捲動 |
| `InspectorPanel` | `inspector.py` | `InspectorPanel(state, parent=None)` | 分頁：片段 / 字幕 / 序列 / 檢查（檢查頁內含 `HumanReviewBox`） |
| `ShortsDialog` | `autoedit_dialog.py` | `ShortsDialog(state, asset_id, parent=None)` | `exec()`；成功套用後 `state.set_sequence(new_id)` |
| `AssembleDialog` | `autoedit_dialog.py` | `AssembleDialog(state, asset_ids, parent=None)` | 同上 |
| `ExportDialog` | `export_dialog.py` | `ExportDialog(state, parent=None)` | 內含 `HumanReviewBox`；輸出後呼叫 `state.record_technical_qa`（若序列有 ai run） |
| `LiveCompositor` | `render/live.py` | 見 §8 | 無 Qt；`MonitorWidget` 與 `autoedit_dialog` 的預覽共用 |

規則：
- 一個動作＝一次 `state.op(...)`；多個 op 要合成一個復原步驟時用 `state.do("標題", lambda p: ...)`（在裡面直接呼叫 `core.ops.*(p, p.sequence(seq_id), ...)`）。
- 背景工作一律 `TaskRunner`/`run_task`；結束訊號在 GUI 執行緒送達。取消 = `runner.cancel()`。
- 不要在別的代理負責的檔案裡加功能；缺少的功能在回報中提出，或用現有 API 在自己的檔案內組合出來。
- `autoedit.api` 的函式（`propose_*`、`audit_plan`、`apply_plan`、`preview_project`）與 `autoedit.captions.generate_cues_for_sequence`、`analysis.speech.{speech_available,transcribe,SpeechUnavailable}`
  由「引擎」代理同時開發；GUI 端請只依賴 `docs/AUTOEDIT_METHOD.md` 與 `autoedit/api.py` 中的簽名，測試時用假物件（monkeypatch）驗證流程，不要等引擎。

### 11.1 快捷鍵分工（避免重複綁定）
- **MainWindow 以 `QAction` 綁定全域快捷鍵**：Ctrl+N / Ctrl+O / Ctrl+S / Ctrl+Shift+S、Ctrl+I（匯入）、Ctrl+E（輸出）、Ctrl+Enter（渲染預覽）、Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z、**Space（播放/暫停）**、Ctrl+Q、F11。
  選單裡屬於時間軸的指令（分割、刪除、標記…）只用「文字提示」顯示快捷鍵（例如 `"在播放頭分割\tS"`），**不要真的綁定**，並呼叫 `TimelineWidget` 的公開方法。
- **TimelineWidget 自己在 `keyPressEvent` 處理**：←/→、Shift+←/→、Home/End、↑/↓、S、Ctrl+K、Delete、Shift+Delete、M、I、O、Ctrl+A、+、-、\\、Esc；
  並提供公開方法：`split_at_playhead()`、`delete_selection(ripple: bool = False)`、`add_marker_at_playhead()`、`set_in_at_playhead()`、`set_out_at_playhead()`、`clear_range()`、
  `step_frames(n: int)`、`goto_prev_cut()`、`goto_next_cut()`、`zoom_in()`、`zoom_out()`、`zoom_fit()`、`select_all()`、`set_snapping(on: bool)`。
- 文字輸入元件（QLineEdit/QSpinBox…）取得焦點時，字母/Delete 鍵本來就歸輸入框，不得攔截。

### 11.2 鍵盤焦點規則（審查發現後補的；測試在 `tests/test_main_window.py` 的「焦點／復原」一節與 `tests/test_inspector.py`）
- **按鈕、勾選框、滑桿、不可編輯的下拉選單只用 Tab 取得焦點**（`widgets.keep_keyboard_on_timeline`）：按一下滑鼠不拿走鍵盤，否則方向鍵會去拉滑桿、空白鍵會再按一次按鈕。
  按下去時先把鍵盤還給時間軸，也讓正在輸入的數字框先送出。
- **數字框／時間框按 Enter 送出後**，鍵盤還給時間軸（`release_focus` → `MainWindow.focus_timeline`）；方向鍵、S、Delete、空白鍵繼續剪輯。
- **主視窗 `keyPressEvent`** 把沒人要的按鍵轉給時間軸；但文字框、數字框、清單／表格（`widgets.keeps_its_keys`）自己留著鍵（Delete＝刪這一列）。
- **Ctrl+Z / Ctrl+Y**：框裡有自己還沒送出的輸入時，復原的是那段輸入；沒有時（剛送出的數字）就是專案的復原／重做
  （`widgets.wants_project_undo`；數字框的 `NumField.event` 處理 `ShortcutOverride`，因為焦點其實在 spin box 而不是裡面的 line edit）。
- **時間輸入有上限**：超過 24 小時（`MAX_TIMELINE`）的開始時間會被核心拒絕，狀態列說明，欄位回到原值，時間軸不受影響。
