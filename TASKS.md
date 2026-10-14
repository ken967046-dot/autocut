# TASKS

狀態：`todo` → `doing` → `review` → `done`（或 `blocked`）。負責人欄位填 `claude` / `codex` / `user`。
「done」只表示**在 Linux 無頭環境用自動測試驗證過**；Windows 實機的事一律另列（WIN-*、SPEECH-1），沒跑過就不寫 done。

| 代號 | 任務 | 負責 | 狀態 | 驗收 / 證據 |
|---|---|---|---|---|
| CORE-1 | 專案模型、復原、剪輯操作、計畫/審核收據 | claude | done | `tests/test_core.py`；`tests/test_core_fuzz.py`（250 組隨機剪輯序列：每次編輯後時間軸仍合法、被拒絕的編輯完全不改專案、波紋編輯不留黑畫面空洞）|
| REND-1 | 渲染引擎（分段快取、交叉淡化、字幕、響度、技術檢查） | claude | done | `tests/test_render.py`、`tests/test_e2e_gui.py` 的實際輸出量測；`tests/test_render_oracle.py`：隨機剪輯後實際輸出，逐格解碼畫面條碼＋聽音高，對照時間軸模型；最終版渲染器另用 `scripts/dev/campaign.py oracle` 離線跑 30／25／24／60／29.97／23.976 fps 共 640 組（約 17.9 萬格畫面、1.7 萬處聲音音高、7.9 千個交叉淡化比例）零失敗；畫面與聲音同步另有量測測試（含聲音晚開始的檔案）|
| AUTO-1 | 分析（聲音/鏡頭/畫面/人臉/語音）與短影音、長片規劃、審核、套用 | claude | done | `tests/test_analysis.py`、`test_shorts.py`、`test_assemble.py`、`test_audit_apply.py` 等；啟發式規則，**品質要人看**，真實素材上的效果未驗證 |
| GUI-1 | 桌面介面：時間軸、預覽、素材庫、檢查器、拖放匯入、自動剪輯對話框、輸出 | claude（子代理）| done | `tests/test_timeline.py`、`test_inspector.py`、`test_media_bin.py`、`test_export_dialog.py`…（offscreen 模擬滑鼠／鍵盤＋截圖）|
| CLI-1 | `autocut` 指令列 | claude | done | `tests/test_cli.py` |
| E2E-1 | 端對端驗收：拖入 → 自動剪輯 → 手動修正 → 輸出 → 技術檢查 → 人工審片按鈕 | claude | done | `tests/test_e2e_gui.py`（3 條流程，真視窗、真引擎、真 ffmpeg）|
| FUZZ-1 | 隨機剪輯測試與修剪／變速／分割／波紋刪除對「交叉淡化」的處理 | claude | done | `tests/test_core_fuzz.py`；另以 2250 組×40 步與 400 組×150 步離線跑過，零失敗；2026-10-09 動到 `ops` 規則（24 小時上限）後用新種子再跑：fuzz base 3000、wide 2000、glue 兩種模式各 4000 組，零失敗 |
| REVIEW-1 | 第一輪交叉審查（Claude 全新上下文審查員；Codex 受網路政策阻擋） | claude-reviewer | review | 兩位全新上下文審查員（A：剪輯核心；B：Windows 安裝／打包）各自獨立讀碼與實驗；每項發現都先重現再修，修正與回歸測試見下方「審查紀錄」。這是**同一家模型的審查，不是 Codex**；Codex 那一輪仍待 REVIEW-2 |
| REVIEW-2 | Codex 在使用者電腦審查全專案 | codex / user | todo | 用 `scripts/windows/Codex-Review.ps1`；報告放 `.codex-runs\` 並貼回此處 |
| WIN-1 | Windows 實機：安裝腳本、啟動、拖放、輸出 | codex / user | todo | 照 `docs/ACCEPTANCE.md` 的 19 步清單（約 30 分鐘）在 Windows 11 實際跑：`Setup-AutoCut.bat`、`Run-Tests.ps1`、開啟軟體、拖放、輸入法、縮放、輸出 |
| WIN-2 | Windows 打包（PyInstaller）驗證 | codex / user | todo | `packaging/Build-Windows.ps1`（尚未在任何機器上跑過）|
| WIN-3 | 硬體編碼器（NVENC/QSV/AMF）實機驗證 | codex / user | todo | 三種至少測使用者機器上有的那種 |
| SPEECH-1 | 語音辨識（faster-whisper）實機驗證與模型選擇 | codex / user | todo | 中文影片產生字幕、轉繁體 |
| PLUGIN-1 | 選用外掛「動態圖層」（HyperFrames）| claude | blocked | **核心驗收之後才做**；先在使用者電腦量測 Node／Chrome／渲染速度 |
| HL-1 | 短影音依「說了什麼」找爆點（字幕檔／同名字幕／語音辨識；選用 AI 判斷） | claude | review | `tests/test_highlights.py`（句子切分、爆點分數、視窗索引對照暴力解、水詞與疊字、文字來源優先序、AI 判斷用假客戶端：覆寫／快取／拒答／沒金鑰退回本機）、`tests/test_shorts.py`（無文字時權重比例與改版前相同）；**真實影片上的效果未驗證**（→ CODEX-2）|
| SUBS-1 | 一鍵字幕：整支影片 → 同名 `.srt`＋`.words.json`（CLI `autocut subtitle`、對話框按鈕） | claude | review | `tests/test_cli.py`（假語音引擎：寫檔、不覆蓋、`--force`、模型未下載提示）、`tests/test_longform_dialog.py`（按鈕流程、覆蓋前詢問、下載前詢問）；**真實 faster-whisper 沒跑過**（SPEECH-1）|
| LONG-1 | 長片精華：重點＋去水詞＋講者特寫（`autocut longform`、「長片精華…」對話框） | claude | review | `tests/test_longform.py`（選句／鋪陳／銜接、水詞與疊字、特寫觸發 3 秒、換人先全景、長段切回全景、爆點句推近、一人或兩人同時講不放大、審核→套用→一次復原；**真的 ffmpeg 渲染後量測特寫面積 ≈ 倍率²**；講者偵測用合成畫面＋替身人臉偵測器）、`tests/test_longform_dialog.py`（真引擎跑對話框、截圖已檢視）；**真實雙人訪談未驗證**（→ CODEX-3）|
| REVIEW-3 | Codex 審查 HL-1／SUBS-1／LONG-1（唯讀、只看改動的檔案，省額度） | codex | done | **真正由 Codex 執行**（雲端環境開放 OpenAI 網域後，用使用者的 ChatGPT 方案以裝置碼登入）：第一輪 changes-requested（6 major、1 minor），全部重現、補測試、修正；複查兩輪後 **approve**。見下方審查紀錄 |
| CODEX-2 | 用真實影片＋字幕校準爆點詞庫 | claude + codex / user | doing | 第一輪（2026-10-14）：使用者第一支影片的字幕（9 分 16 秒、381 條無標點）＋**AI 評審共識**當暫代標註（**不是使用者的標註**）；見下方紀錄。等使用者標出自己心目中的爆點後再做第二輪 |
| CUT-1 | 剪輯清單：照使用者決定的段落剪短影音（冷開場、跳過口誤、字卡；CLI `--cutlist`、對話框「剪輯清單」）| claude | review | `tests/test_cutlist.py`（時間格式、逐欄位繁中錯誤、超出長度、冷開場順序、字卡 at／source、略過說明、審核→套用→一次復原、存檔再讀、CLI 真 ffmpeg）、`tests/test_autoedit_dialog.py`（欄位驗證、支數／目標長度變灰、不記住檔案）；**這集的 6 支**用真字幕＋同長度合成影片端對端跑過（6 支都套用、輸出、技術檢查通過）；**真影片畫面未看過** |
| CARD-1 | 字卡：渲染、預覽、時間軸、檢查器、匯出 | claude | review | `tests/test_render.py`（ASS 字卡樣式與圖層、分段快取只影響有字卡的段、字卡樣式算進簽章〔拿掉就失敗〕、**實際輸出量測**：上方 30% 有黃字與不透明黑框、底部字幕不變、.srt 沒有字卡）、`test_monitor.py`（預覽上方出現字卡、下方不變；截圖已檢視）、`test_timeline.py`（字卡帶、重疊時上下都點得到、拖曳一次復原；截圖已檢視）、`test_inspector.py`（字卡按鈕、匯出／匯入保留字卡；截圖已檢視）、`test_core.py`（op 規則、復原）、`test_export_dialog.py` |
| WRAP-1 | 直式字幕超出畫面（**既有問題**，在這集輸出畫面上發現）＋平均分行 | claude | review | 9:16 每行 16 字、字高 5.2% 時頭尾的字被切掉 → `wrap_units` 以畫面寬度為上限、兩行平均分配、標點規則；`tests/test_render.py` 實際輸出量測字不碰邊（拿掉修正就失敗）；`RENDER_VERSION` 8 |
| CODEX-3 | 真實雙人訪談驗證講者特寫並微調 | codex / user | todo | `Codex-Task.ps1 -Task CODEX-3` |
| CODEX-4 | 講者分析在 1 小時影片上的速度 | codex | todo | `Codex-Task.ps1 -Task CODEX-4` |
| CODEX-5 | 平滑推近（取代硬切特寫）——先提案 | codex | todo | 只寫提案，等同意 |

## 審查紀錄
- 2026-10-08：端對端測試與隨機剪輯測試找到並修正一整類「交叉淡化邊界」問題（波紋刪除／修剪／變速沒有帶動淡入的下一段、
  修剪可以短於轉場、分割落在轉場中間、覆蓋後留下零碎片段、起點只差百萬分之一秒時排序翻轉）。
  修法集中在 `core/ops.py`（`trim_limits`、`_neighbours`、`_tidy_cut`、`_guarded`）與 `core/project.py`（`ordered_clips`、`geometry_issues`）。
- 2026-10-08：實際輸出紅→綠漸變後在轉場中間用剃刀（S）切開，畫面會在切點「跳回紅色」再重新漸變（切過轉場＝兩段短轉場）。
  這種分割現在一律拒絕（整個動作不改任何軌道），狀態列說明原因；`split_clip` 原本就拒絕。
- 2026-10-08：新增「像素＋聲音對照」測試（`tests/test_render_oracle.py`）：素材畫面上方是影格號＋檔案號的條碼、聲音是隨來源時間上升的音高；
  隨機剪輯後輸出並逐格比對，並有一個「故意給錯誤模型，確認測試會抗議」的測試，避免這個測試空轉。
- 2026-10-09（REVIEW-1 修正）：兩位審查員的發現逐項**先重現再修**，每項都有回歸測試，並用「把修正拿掉、測試就會失敗」確認測試有牙齒：
  - 字幕與標記在主軌 V1 的波紋剪輯（插入、波紋刪除／修剪、變速、關閉空隙…）後會漂離原本對著的畫面 → 現在跟著走
    （`tests/test_core_sync.py`，內含獨立 oracle；另離線跑兩種模式各 1 萬組隨機序列，各約 60 萬次字幕檢查，零真實失敗）。
  - 移動／輕推：移到原位會切掉交叉淡化夥伴 → 現在完全不改；輕推只改交疊長度。落點上還有別的片段、保留夥伴會把它整段蓋住時，改成一般覆蓋（先在複本上試算）。
  - 剃刀：落在片段自己的淡入／淡出、或切完有一半和另一個片段位置長度完全相同 → 拒絕。
  - NaN／inf 能寫進專案檔 → 所有公開 op 一律拒絕；`clamp` 對 NaN 安全。
  - 片段在交叉淡化之外至少留一格自己的畫面；`set_crossfade(c, 0)` 在沒有交疊時是空操作（原本會把使用者留的空隙關掉）。
  - 字幕分割／平移／排序：對齊影格、不短於 0.2 秒、順序以存檔的 6 位小數為準（復原／重做後不變）。
  - 大型隨機測試（修正後）：基本 1 萬組、進階 6 千組，零失敗；修正前找到的 14 組失敗種子重放全過。
  - 關閉視窗現在會取消並等待所有背景工作（原本只取消匯入）；序列背景色走 `ops`（驗證 `#RRGGBB`）；另存新檔的專案名稱走 `Editor.set_document`，
    且「復原」不再把名稱倒回舊的；`natural_key` 遇到 ②、² 不再當掉；`doctor` 不再把沒實測的硬體編碼器標成 ok；
    打包版 exe 也會找 exe 旁邊的 ffmpeg。
  - 渲染對照測試加嚴：交叉淡化的混合比例改成從畫面反推（實測與模型差約 −0.03、最大 0.063；容許 0.12）；新增 25／24／60 fps 場景、
    「交叉淡化被改錯一定被發現」的敏感度測試、涵蓋率檢查。
  - Windows 腳本與打包（**只在 Linux 上讀碼審查，沒有在 Windows 跑過**）：ffmpeg 下載逐一核對發布者的 SHA-256（不符就中止、不退而求其次）；
    ffmpeg 失敗不再讓整個安裝失敗，改以結束代碼 2 回報（測試失敗為 3）；winget 結果改用重新偵測 Python 判斷；`Start-AutoCut.bat` 啟動前先載入一次程式、失敗就顯示原因；
    路徑一律當字面值（資料夾名稱含 `[ ]` 也行）；補 `LICENSE`、`pyproject.toml` 改 SPDX 授權欄位（最新 setuptools 在乾淨環境實際打包成功，中繼資料已確認）；
    PyInstaller 預設不再內附（多半是 GPL 的）ffmpeg，要 `-BundleFfmpeg` 才放；README 的「安裝要連網、哪些東西裝在資料夾外」改成與實際一致。
- 2026-10-09（REVIEW-1 第二輪：像使用者一樣操作整個視窗，加上渲染同步）：全新上下文審查員用模擬滑鼠／鍵盤操作離線視窗；
  每項發現一樣**先重現再修**、有回歸測試、並用「把修正拿掉、測試就會失敗」確認（變異檢查）：
  - **鍵盤焦點與復原**：在檢查器的數字／時間框輸入並按 Enter 後，Ctrl+Z 不會復原專案，方向鍵／S／Delete／空白鍵也會作用到剛碰過的滑桿或按鈕 →
    按鈕、勾選框、滑桿、不可編輯下拉選單只用 Tab 取得焦點；Enter 把鍵盤還給時間軸；主視窗把沒人要的按鍵轉給時間軸（文字框、數字框、清單自己留著）；
    數字框沒有自己的輸入可復原時，Ctrl+Z／Y 交給專案（`NumField.event`）。**只用模擬按鍵驗證；真實鍵盤與輸入法（IME）在 Windows 上的行為沒測過。**
  - 播放中序列被改短或清空：播放邊界即時更新、清空就停（`playback._follow_the_edit`）。匯入進行中新增／開啟專案：取消匯入並丟掉遲到的結果（`_epoch`）。
  - 壞的 `.acproj`／自動存檔（欄位型別錯、NaN、巢狀太深、不是文字…）：一律 `ProjectFileError` 與清楚訊息；讀不了的自動存檔改名 `.unreadable` 不再反覆詢問（內容不刪）。
  - **時間軸最長 24 小時**：在「開始」欄輸入 `1000000000` 曾讓時間軸的 32 位元捲動條溢位、在槽函式裡丟例外。核心現在拒絕任何超過 24 小時的位置
    （`ops._sn`、`_guarded` 的 `too_long`、`validate_sequence`），訊息在狀態列、欄位回到原值、不留復原步驟；損壞檔裡的超大數字另由時間軸保險（捲動範圍、`QRect` 座標）接住；
    輸入框把 `1e308` 夾在上限內（它連顯示成時間碼都會算爆）。動到 `ops` 規則，所以依 `AGENTS.md` 用**新種子**重跑隨機測試：
    glue 兩種模式各 4000 組（每種約 24.2 萬則字幕檢查）、fuzz base 3000 組、fuzz wide 2000 組，**零失敗**。
  - 版面：預設分割改成預覽 470：時間軸 450；1100×700 與 1366×768 的 offscreen 截圖已檢視（時間軸看得到字幕、V2、V1、A1，其餘垂直捲動）。
  - **渲染對照測試的模型**：片段起點落在兩格之間時（波紋或變速之後，如 25 fps 的第 124.5 格），渲染器把片段放在最近的那一格（畫面與聲音一致）；
    測試模型原本用「確切起點」算，造成 25 fps 兩組誤報 → 改模型（不是渲染器），並確認「把渲染器移兩個來源格仍會被抓到」。
  - **畫面與聲音同步**：新增永久測試（30／25／24 fps；實測聲音固定晚約 5 ms、畫面與聲音相差最多 19 ms／10 ms；畫面或聲音差一格一定被抓到）。
    同一輪還量到並修正：音訊比畫面晚開始的檔案（螢幕錄影、部分相機），從檔案最開頭剪的片段聲音會提早約等於那個延遲
    （合成檔：78 ms 延遲 → 提早 77 ms；255 ms → 提早 226 ms；從檔案中間剪的片段原本就對）→ 改用 `aresample first_pts=0`、`RENDER_VERSION` 7、三個回歸測試。
    修正後用最終版渲染器重跑六種幀率的離線對照（新種子 500 起）：30 fps 160 組、25／24 fps 各 150 組、60／29.97／23.976 fps 各 60 組，共 640 組，**零失敗**。
    **真實攝影機／手機素材沒量過**，只有合成檔。
  - **完整測試**：開發環境（PySide6 6.11.2）一輪 2036 項通過、1 項失敗——開發用 `campaign.py` 後來改成完整列出前三個失敗的細節，
    對應的測試還在檢查舊格式（工具與測試不同步，不是軟體問題），已改測試、補一個測試，並確認工具改回舊格式測試就會失敗。
    再從提交（`d3977bd`）用 `git archive` 打 zip、解壓到別處、建全新 venv 只照 `requirements-dev.txt` 安裝（PySide6 **6.12.0**，比開發環境新一版）、
    `pip install -e . --no-deps` 後跑 `autocut doctor` 與完整測試：**2038 項通過、2 項跳過（10 分鐘長影片、檔案權限）、0 失敗**（9 分 44 秒）。
    警告 121 則，全部來自測試程式用舊式建構子建立 `QDragEnterEvent`／`QDragMoveEvent`（新版 PySide6 標為棄用），軟體本身沒有棄用警告。
  - 已知的小問題（沒修）：編輯後又做相反的編輯、內容回到存檔時的樣子，標題仍顯示「未儲存」（髒標記以編輯次數計，不是比對內容）；
    「記錄技術檢查結果」是一個復原步驟（Ctrl+Z 會先復原那筆紀錄）；自動剪輯對話框預設 1000×720（最小 900×640），在很小的螢幕上可能超出（沒實測）。

### 交接：HL-1 / SUBS-1 / LONG-1　作者：Claude（2026-10-13）
- 改了什麼：
  - `autoedit/highlights.py`（新）：句子切分、爆點分數（用詞＋語氣−水詞）、視窗索引、水詞／疊字、文字來源（字幕檔 > 同名字幕 > 語音辨識）。
  - `autoedit/scoring.py`：`keywords` 換成 `content`（0.42）＋`content_hook`（0.10）；沒文字時其餘權重比例與改版前相同。
  - `autoedit/llm_judge.py`（新，選用）：Claude 逐句評分，只送字幕文字；快取；任何失敗退回本機規則。
  - `autoedit/longform.py`（新）、`analysis/speakers.py`（新）、`LongformParams`、`propose_longform`、審核規則 `longform.zoom`。
  - `autoedit/subtitle_gen.py`（新）：一鍵字幕。CLI：`longform`、`subtitle`、`shorts --subs/--highlight`。
  - GUI：「長片精華…」（選單、工具列、素材庫右鍵）、對話框的「找重點」區（字幕檔、一鍵產生字幕、AI 判斷與金鑰、狀態說明）；
    修正設定頁捲動區裡換行的提示文字被裁掉（QScrollArea 不考慮 height-for-width）。
  - `Setup-AutoCut.bat -WithAI`、`requirements-ai.txt`、`pyproject` extra `ai`；`scripts/windows/Codex-Task.ps1`（新）、`Codex-Review.ps1 -Paths`、`docs/CODEX_TASKS.md`。
- 為什麼：使用者試用後回報短影音「重點沒抓到、很無聊」，並要求第二個功能「長影片自動剪輯（重點＋去水詞）＋兩人以上對談時講者自動放大」。
- 驗證：見各列「驗收／證據」。完整測試（Linux、Python 3.13、PySide6 6.12.0、ffmpeg 6.1.1、offscreen）：收集 2067 項，**2065 通過、2 跳過（10 分鐘長影片、root 權限）、0 失敗**，pytest 結束代碼 0。過程中自己抓到並修正：浮點雜訊被百分位排名放大成名次差、`tighten_ranges` 不剪段落開頭的水詞、
  每人各自正規化讓「一直講」和「都沒講」無法分辨、疊字（聊聊）被當成口吃、清除 AI 金鑰沒有從環境變數移除、提示文字被裁掉。
- 沒驗證／已知限制：真實影片的選段品質、真實訪談的講者判斷（只用合成畫面與替身人臉偵測器）、真實 faster-whisper、真實 Claude API 回應
  （只驗證了 SDK 1.13 能組出請求、用假客戶端測流程）、Windows 實機、`Codex-Task.ps1` 只在 Linux 的 PowerShell 7.4 做過語法解析與 `-DryRun`。
- 想請審查者特別看：`docs/CODEX_TASKS.md` CODEX-1 列出的五點。

- 2026-10-13（REVIEW-3，**Codex** 審查，`codex-cli 0.162.1`，以使用者 ChatGPT 帳號登入，唯讀沙盒，只看 7 個改動的檔案）：
  第一輪 changes-requested。每項都先在記憶體重現、補「修正前會失敗」的回歸測試、再修：
  - major：「對對方很好」逐字切開時兩個「對」被當水詞 → 情境水詞改成「整串前後都停頓才算」，重複本身不算。
  - major：「非常非常好」被當口吃 → 同一詞連續三次以上才算重複字（兩次是語法或強調）。
  - major：字幕時間超出影片長度會產生超出素材的片段 → `kept_ranges` 夾在 0..長度、規劃時裁掉超出的字幕。
  - major：`anthropic.Anthropic()` 初始化失敗不在例外處理內 → 移入，退回本機規則。
  - major：錯誤訊息可能把 API 金鑰寫進 `plan.notes`（會存進專案檔）→ 先改成遮蔽，Codex 複查指出黑名單仍會漏
    （`api_key=customToken123`）→ 改成**只寫固定描述**（HTTP 狀態類別、連線逾時、回覆格式錯誤或錯誤類型），完全不保存原始錯誤文字。
  - major：覆寫字幕但新結果沒有逐字時間時留下舊 `.words.json`（剪水詞會用錯時間）→ 刪除舊檔。
  - minor：取樣數少於平滑視窗時 `speaker_labels` 崩潰 → 卷積結果裁回原長度。
  複查一：6 項已修正、1 項部分修正、修正未引入新問題；複查二：**approve**。完整測試（收集 2074 項）：2072 通過、2 跳過、0 失敗。
  Codex 自己標明未做：pytest、實際渲染、GUI、Windows、真實 Whisper／Claude API（這些由 Claude 的測試或仍為 not_run）。

- 2026-10-14（CODEX-2 第一輪：真實字幕校準＋REVIEW-4）：
  - 素材：使用者「發聲炮臺」開台集的字幕（**沒有影片**：聲音／畫面用中性值，只測「說了什麼」）。字幕內容不放進 repo（隱私）；測試只用合成句子。
  - 暫代標註：工作流程讓 4 位不同角度的 AI 剪輯編輯（爆款、紀實、綜藝、品牌）＋ **Codex**（以使用者 ChatGPT 帳號）各自挑片段，
    合併成共識（第 1 名 5/5 票），再由「單獨看得懂嗎」與「法律／品牌風險」兩位審查員檢查。**這是 AI 共識，不是使用者的判斷。**
  - 原版引擎：3 支裡 1 支落在 5 位評審一致認為「全集最該剪」的區段；最高分句子都是「其實…」開頭。原因與修正（皆為通用規則、各有測試）：
    無標點字幕碎片 → `merge_fragments`；「其實／但是」→ 弱轉折 0.25；「一個」不算數字；新增被騙／詐騙／不公不義等詞；附和語算水詞；
    解釋性「這也是我為什麼」不算提問；「已經／再次強調」扣分；`_rank` 四捨五入（固定機位的動作量浮點雜訊曾把相同數值排成 0.36–0.93）。
    另外抓到 REVIEW-3 修正的副作用：有逐字時間的「對對對」不再被算成水詞 → 水詞比例改用同一套整串規則。
  - 結果：3 支 0 支落在該剪區；5 支命中共識第 1、2、5、6 名。**限制**：前幾名分數非常接近（0.77–0.81），詞庫看不懂語意衝突
    （例如「打手」），共識第 3、4 名仍挑不到；要更準需要 AI 判斷模式，或使用者的實際標註。
  - REVIEW-4（**Codex** 審查校準改動）：changes-requested 4 點 → 2 點修正（附和語被併入別人的句子、數字規則）、
    1 點反駁獲接受（結尾空白不會跨句合併，補測試）、1 點取捨獲接受（「你知道為什麼」仍算提問）；複查指出 speaker 判斷不對稱 → 修正 → **approve**。
  - 完整測試：校準提交後全部通過（pytest 結束代碼 0）；最後兩個合併規則的小改動後，相關測試檔全部通過。
