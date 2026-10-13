# 自動剪輯方法（`analysis/` + `autoedit/`）

本文件同時是**實作規格**與**使用者可讀的方法說明**。方法改編自開源專案 *video-autopilot-kit*（MIT，© 2026 Hao0321 Studio，見 `THIRD_PARTY_NOTICES.md`）：
「證據 → 計畫 → 審核 → 套用 → 渲染 → 技術 QA → 人工審片」這條**不讓機器自己幫自己蓋章**的流程，以及「數值門檻只是範例校準值、要能調」的態度。
本專案是獨立實作的桌面軟體，資料模型與演算法都是自己的；上游的數值（例如首個剪點 ≤ 2.05 秒、字幕閱讀速度 5/7 字每秒）在 `autoedit/config.py` 裡全部是**可調參數**，不是法律。

## 0. 原則（違反就是 bug）
1. **計畫是純資料**：`propose_*` 只讀專案與素材、產生 `EditPlan`，絕不修改專案。計畫內容用 canonical JSON 取 sha256。
2. **審核綁定**：`audit_plan` 產生 `AuditReceipt`，綁計畫雜湊 + 專案 `revision`；計畫或專案一改，收據就失效（`core.plan.check_receipt`，fail-closed）。
3. **套用＝一次復原步驟**：`apply_plan` 在單一 `Editor.do(...)` 交易內完成；失敗整批回滾。
4. **機器不代替人判斷好不好看**：計畫/審核/技術 QA 都只說「規則與技術檢查結果」；`human_review` 永遠從 `pending` 開始，只有使用者在介面按鈕才會改（`core.plan.set_human_review`）。
5. **誠實降級**：沒有語音辨識、偵測不到臉、素材沒有聲音……都要退化到較簡單的做法，並在 `plan.notes` 與對應 `decision.why` 寫明（繁體中文），不得假裝有證據。
6. **不連網、不上傳**：分析全在本機。語音模型若未下載，預設**不自動下載**（要使用者明確同意，見 §2.4）。
7. **可重現**：同樣的素材與參數 → 同樣的計畫雜湊。計畫內的 id 用 `util.IdGen("p")` 產生（確定性），時間取到影格格點並四捨五入；不要把時間戳、亂數、絕對路徑以外的環境資訊放進計畫內容。
8. **可取消、可回報進度**：長工作都吃 `progress(fraction, message)` 與 `cancel: threading.Event`；取消時丟 `media.ffmpeg.Cancelled`（`autoedit.api.Cancelled` 是同一個類別）並清掉子行程。

## 1. 管線
```
素材 ──analyze_asset──▶ Evidence（可快取：聲音/畫面/鏡頭/人臉/語音）
Evidence ──plan_shorts / plan_assemble──▶ EditPlan（sources、assets、sequences、decisions、notes）
EditPlan ──audit──▶ AuditReceipt（錯誤＝不可套用；警告＝提醒）
EditPlan + Receipt ──apply──▶ 專案新增序列（一次復原）＋ project.meta["ai_runs"] 紀錄
序列 ──render_sequence──▶ mp4 ──technical_qa──▶ 技術 QA ──使用者按「我已審片」──▶ human_review
```

## 2. `analysis/`（證據）
所有模組只依賴 numpy / OpenCV / ffmpeg（`media.ffmpeg`），**不依賴 Qt**。

### 2.1 `analysis/evidence.py`（資料模型，已寫好）與 `analysis/pipeline.py`（`analyze_asset`、`refine_cut`）
資料模型以 `analysis/evidence.py` 為準（欄位只能新增、不可改名；比下面的摘要多了 `motion_cx/motion_cy`＝每個取樣的「變化像素重心」、`notes`、`silence_thr_db`、`voice_thr_db`）。
`analyze_asset`、`refine_cut` 在 `analysis/pipeline.py`，並由 `analysis/__init__.py` 匯出：`from autocut.analysis import analyze_asset, refine_cut, Evidence, AnalyzeParams`。
```python
EVIDENCE_VERSION = 1

@dataclass
class AnalyzeParams:
    audio_fps: float = 20.0       # 聲音特徵每秒幾格
    audio_sr: int = 16000
    visual_fps: float = 5.0       # 畫面取樣率（低解析度）
    frame_width: int = 256        # 取樣畫面寬（高度依顯示比例）
    face_fps: float = 2.0         # 人臉偵測取樣率（<= visual_fps）
    faces: bool = True
    silence_db: float = -38.0     # 靜音絕對下限（會與雜訊底噪自適應取較嚴者，見 2.2）
    speech: bool = False          # 是否同時做語音辨識（選用）
    language: str = "zh"; model_size: str = "small"; traditional: bool = True; allow_download: bool = False

@dataclass
class Evidence:
    asset_id: str; path: str; fingerprint: str; duration: float; width: int; height: int; fps: float; has_audio: bool
    params: dict
    audio_fps: float; rms_db: list[float]; voice: list[float]; silences: list[list[float]]   # [[t0,t1],...] ≥ 0.3 s
    noise_floor_db: float
    visual_fps: float; motion: list[float]; sharp: list[float]; bright: list[float]
    faces: list[list[list[float]]]   # 每個取樣一個臉框清單 [[x,y,w,h],...]，座標為顯示畫面 0..1（左上角起算）；沒做偵測的取樣為 []
    face_sample_mask: list[int]      # 1 = 該取樣有跑人臉偵測
    scenes: list[float]              # 鏡頭切換時間（秒）
    speech: list[dict] | None        # [{"start","end","text","words":[{"start","end","word"}]}]；None = 沒做/不可用
    def sha256(self) -> str          # canonical JSON 雜湊（計畫的 sources[].evidence_sha256 用）
    def to_dict / from_dict / save(path) / load(path)

def analyze_asset(asset: Asset, params: AnalyzeParams | None = None, *, progress=None, cancel=None, use_cache: bool = True) -> Evidence
```
- 快取：`util.cache_dir("evidence") / f"{asset.fingerprint or <path,size,mtime 雜湊>}_{EVIDENCE_VERSION}_{params 雜湊}.json"`。命中就直接讀，不重跑。
- 圖片素材 / 無影像素材：只回有意義的欄位（其餘為空串列），不得丟例外。
- `progress` 的 fraction 以「這個素材」為 0..1；訊息繁體中文（「分析聲音…」「分析畫面 35%」…）。

### 2.2 `analysis/audio.py`
- 串流解碼（`ffmpeg -vn -ac 1 -ar 16000 -f f32le -`，Popen，區塊讀取；**不要把一小時的音訊整個載入記憶體**）。
- 每 `1/audio_fps` 秒一格（視窗 ≈ 64 ms，hop = 1/audio_fps）：`rms_db`（相對滿刻度）、語音頻帶比例（300–3400 Hz 能量 / 全頻能量，numpy `rfft`）。
- **雜訊底噪自適應**：`noise_floor = rms_db 的第 10 百分位`；`silence_thr = clamp(max(params.silence_db, noise_floor + 8), 上限 -25 dB)`；`voice_thr = max(noise_floor + 10, params.silence_db)`。
  （餐廳/廚房這類吵雜環境不能只靠固定 -38 dB。）
- `voice[i]`：`rms_db > voice_thr` 且語音頻帶比例 > 0.45，再做遲滯與平滑（最短 0.15 s、補洞 0.2 s）→ 0..1。
- `silences`：低於 `silence_thr` 連續 ≥ 0.3 s 的區間。
- 沒有音軌：`rms_db=[]`、`voice=[]`、`silences=[]`、`has_audio=False`。

### 2.3 `analysis/visual.py`、`analysis/scenes.py`、`analysis/faces.py`
- **單次解碼**：`ffmpeg -i in -vf "fps={visual_fps},scale={W}:{H}:flags=area,setsar=1" -f rawvideo -pix_fmt bgr24 -`（`W=frame_width`，`H` 依 `asset.width/height` 的**顯示比例**；ffmpeg 預設已自動旋轉）。Popen + 逐格讀固定位元組；Windows 用 `CREATE_NO_WINDOW`（參考 `media.ffmpeg`）。
- 每個取樣算：`bright`（灰階平均 0..1）、`sharp`（Laplacian 變異數，**之後以該素材中位數正規化**）、`motion`（與上一取樣灰階差的平均 / 255）。
- **鏡頭切換**：灰階直方圖（32 bins）L1 距離 + 灰階平均絕對差，兩者同時超過自適應門檻（`max(固定下限, 中位數 + 6×MAD)`）才算；同一秒內去重；時間取兩取樣的中點。
  `refine_cut(path, t_est, fps, window=0.6) -> float`：只解碼 `t_est±window` 的小範圍（低解析度全 fps），找最大亮度差那格 → 精準到影格。**規劃器最後選定的片段邊界若落在鏡頭切換上，必須用它校正**（避免閃到前一個鏡頭）。
- **人臉**：OpenCV Haar（`cv2.data.haarcascades + "haarcascade_frontalface_default.xml"`；`opencv-python-headless` 內含），只在 `face_fps` 取樣上跑，輸入為已縮小的灰階畫面（直方圖等化）、`minSize` 約畫面寬的 6%；回傳正規化框。
  沒有級聯檔或載入失敗 → 優雅降級（`faces` 全空，`plan.notes` 說明「未偵測人臉，改用動作重心」）。
  已知限制：Haar 對側臉/低頭/口罩/逆光很弱；**不要宣稱能可靠追蹤**。

### 2.4 `analysis/speech.py`（選用）
```python
class SpeechUnavailable(RuntimeError): ...        # 訊息為繁體中文，可直接顯示給使用者
def speech_available() -> tuple[bool, str]         # (可用?, 不可用的原因/安裝指示)
def model_cached(model_size: str = "small") -> bool
def transcribe(path, *, language="zh", model_size="small", traditional=True, allow_download=False,
               start=0.0, duration=None, progress=None, cancel=None) -> list[dict]
```
- `faster_whisper` 以 `import` 偵測（缺少 → `(False, "尚未安裝語音辨識：請執行 pip install -r requirements-speech.txt")`），**延遲 import**，沒裝也不能讓其他功能壞掉。
- 模型未快取且 `allow_download=False` → 丟 `SpeechUnavailable("尚未下載語音模型（約 N MB）…")`；GUI 會詢問使用者後以 `allow_download=True` 重試。快取判斷用 `WhisperModel(..., local_files_only=True)` 嘗試載入。
- 參數：`vad_filter=True`、`word_timestamps=True`、`compute_type="int8"`、`device="auto"`；`language="zh"` 時若 `traditional` 且 `opencc` 可用，轉成繁體（`s2twp`）；無 opencc 時原文輸出並在回傳註記無法轉換。
- 回傳的時間為**素材時間**（`start` 偏移已加回）。

## 3. 短影音：16:9 長片 → 多支 9:16（`autoedit/shorts.py`、`scoring.py`、`reframe.py`、`captions.py`）
輸入：`asset_id` + `ShortsParams`。輸出：一份 `EditPlan(kind="shorts")`，每個候選短片 = 一個序列（1080×1920@fps，V1 主畫面、V2 空軌、A1 配樂、A2 空軌）。

### 3.1 流程
1. 取 `Evidence`（`use_speech` 且可用才做語音辨識；否則降級並記錄）。
2. **長度帶**：`lo, hi = resolve_band(params)`；目標 `T = clamp(target_seconds, lo, hi)`。`yt_shorts` 有範例死區 25.001–44.999 s（上游：13–25 s 單一節拍、45–60 s 教學型）：目標落在死區就移到最近的允許區並寫進 `notes`。
   **長度以「緊湊化之後」計算。**
3. **邊界候選**（可剪的好位置）：句末（語音辨識標點/段落結尾，權重 1.0）、靜音區間中點（0.7）、鏡頭切換（0.5）、（無以上時）每秒格點（0.1）。每個邊界記錄類型與權重。
4. **視窗搜尋**：起點取自邊界；終點取自 `[s + lo', s + hi']` 內的邊界（`lo'/hi'` 是把緊湊化的縮短量換算回原始時間）。用前綴和（prefix sums）O(1) 評分，避免 O(n²) 的重算。
5. **評分**（全部換成 0..1；以該素材的**百分位排名**去尺度化，權重放在 `brief["weights"]`，使用者可調）：
   | 項目 | 權重 | 定義 |
   |---|---|---|
   | `voice` | 0.22 | 視窗內有人聲的比例（無音軌時改看 `motion`） |
   | `energy` | 0.10 | 有聲段平均響度的百分位 |
   | `motion` | 0.10 | 平均動作量的百分位（適中最好，過高視為晃動而降分） |
   | `faces` | 0.10 | 有臉的取樣比例 |
   | `hook` | 0.18 | 前 3 秒：1 秒內就有人聲、響度/動作/臉高於中位數（上游「開頭鉤子」概念） |
   | `complete` | 0.18 | 起點在句首/靜音後、終點在句末：兩端都是強邊界=1；切在句中=0（有逐字時間就檢查是否切在字內） |
   | `keywords` | — | （2026-10-13 起併入 `content`，見 §3b；下表其餘數字是改版前的權重，現在每項都乘以 0.5，`content` 0.42、`content_hook` 0.10） |
   | `length` | 0.05 | 與目標長度接近程度 |
   | `quality` | 0.03 | 不模糊（`sharp` 相對值）、不太暗 |
6. **選取**：貪婪取前 `count` 個，彼此**不重疊**且間隔 ≥ `min_gap_between`；依分數排序命名「短影音 1｜mm:ss–mm:ss」。素材短於長度帶下限 → 只做 1 支（整段）並警告；短於 5 秒 → `AutoEditError("影片太短，無法剪成短影音")`。
7. **緊湊化（jump cut）**：`tighten=True` 時，視窗內長度 ≥ `pause_min` 的靜音，保留中間 `pause_keep` 秒（兩端各留一半），其餘剪掉；切出來的段落各成一個 V1 片段，**小於 `min_clip_seconds`（0.25 s）的不切**；
   每個片段 `fade_in=fade_out=0.015`（避免喀聲）。
8. **開頭的第一個剪點（上游「2 秒規則」）**：V1 第一個片段若 > `first_cut_max`（2.05 s）且之後沒有任何剪點/變焦，就在 ≤ 1.9 s 處（優先靠近低能量點）切開，後半**改變取景（punch-in zoom 1.12）**形成視覺剪點。
   有緊湊化的剪點時，相鄰片段交替 `zoom 1.0 / 1.12`（遮掩跳接）；`reframe="blur"` 時不做 punch-in。
9. **取景（虛擬攝影機）`reframe.py`**：
   - `face`：每個取樣挑「主體」（面積 × 與上一取樣的連續性）；多人時，若所有臉的聯集放得進取景窗就取聯集中心，否則跟最大/最連續的臉；沒有臉的取樣 → 用動作重心（兩取樣差分的重心）；仍沒有 → 保持上一位置。
   - `motion`：只用動作重心。`center`：靜態 `cx=0.5`。`blur`：`fit="blur"`（整張 16:9 畫面放中間、背景模糊放大），不裁切。
   - 先把目標中心**夾進合法範圍**（`core.geometry.base_window` 算出取景窗寬 `ww`，`cx ∈ [ww/2, 1-ww/2]`，`zoom>1` 時 `cy` 同理）再平滑，避免攝影機頂到邊緣；
     死區（主體在窗中心 ±15% 窗寬內不動，`reframe_strength` 調整）→ 臨界阻尼彈簧/指數平滑 + 最大平移速度（每秒 ≤ 0.25×窗寬×strength 倍率）。
   - **鏡頭切換處不平滑**：在切換點放兩個相隔 1 影格的關鍵格讓取景直接跳。
   - 用 Ramer–Douglas–Peucker（ε≈0.01）簡化成 `Transform.keys=[[t_local,cx,cy],...]`（每片段 ≤ ~40 個），首尾必有關鍵格。`punch-in` 時 `cy = 臉心y + 0.12×窗高`（臉落在窗的上中段）。
   - 每個取景決策寫進 `decisions`（`kind="reframe"`）：用了臉/動作/置中、臉偵測比例是多少。
10. **字幕**（`captions.py`，`ShortsParams.captions`）：`auto`=用逐字稿、`file`=讀 `.srt/.ass/.vtt`（`core.subtitles.load_subtitle_file`，時間為素材時間）、`off`。
    素材時間 → 時間軸時間要走**緊湊化後的片段對應**；落在被剪掉區間的字幕丟棄、跨邊界的字幕裁切。切段規則：每條 ≤ `max_cue_chars`（預設 16 個全形字；CJK 感知，優先在標點/停頓處斷）、
    顯示 `end = min(說話結束 + CAP_PAD(0.15), 下一條開始 − CAP_GAP(0.12))`、最短 `min_cue_seconds`（0.4 s）、閱讀速度超過 `reading_warn_cps` 就延長到不撞下一條為止。
    短影音序列的 `caption_style`：`size=0.052, bold, outline, position="bottom", margin=0.20`（底部約 20% 常被平台介面遮住）。
11. **配樂**：`music_path` 有值 → 以 `media.probe.probe_asset` 取得 `Asset` 放進 `plan.assets`（確定性 id），A1 放一個（或重複排滿）片段，`volume = 10**(music_gain_db/20)`，結尾 `fade_out=1.5`。
    **沒有動態閃避（ducking）**：靜態音量；有人聲的短片用 `music_gain_db`，整段沒人聲則 +6 dB。這是已知限制，寫在 `notes`。
12. **輸出的決策紀錄**（`plan.decisions`）：每支短片一筆 `{"id","kind":"short","sequence":<seq id>,"score":0..1,"source":{"asset_id","t0","t1"},"summary":"mm:ss–mm:ss · 35 秒 · 分數 0.82","why":"…（繁中，具體：句首句尾完整、前 3 秒有人聲與臉、動作量高於中位數 78%…）","evidence":{各分項分數、邊界類型}}`；
    另有 `tighten`（剪掉幾處停頓共幾秒）、`reframe`、`caption`、`music` 各一筆（`sequence` 指向該短片）。

## 3b. 找「爆點」：依說了什麼挑段落（`autoedit/highlights.py`、`autoedit/llm_judge.py`，2026-10-13）
**為什麼加**：實際試用時短影音「重點沒抓到、很無聊」。原本的評分幾乎只看「有人在講話、聲音大、有臉、有動作」，
說話內容只透過 4% 權重的關鍵字正規式參與；講話類影片每一段在這些指標上都差不多，挑出來的自然是平淡的段落。

1. **文字從哪來**（`choose_transcript`，優先順序）：使用者指定的字幕檔（`transcript_file`／CLI `--subs`）→ 影片旁同名字幕檔
   （`<影片>.srt`、`<影片>.zh-TW.srt`…；「一鍵產生字幕」寫的就是它）→ 語音辨識逐字稿。來源與文字雜湊寫進 `plan.brief["transcript"]`，
   文字一改計畫雜湊就變。字幕檔的文字＋語音辨識的逐字時間可以並用（逐字時間只用來避免切在字中間、剪水詞）。
2. **句子單位**（`sentences`）：在 。！？等句末標點切開；沒有標點的字幕檔一條就是一句；重疊的字幕會被裁開。
3. **每句的爆點分數**（`score_sentences`，0..1）＝ 0.65 × 用詞 ＋ 0.35 × 語氣 − 0.5 × 水詞比例；不到 2 個字的句子為 0。
   - 用詞：提問（1.0）、反轉（1.3：沒想到、竟然、結果、但是、其實、原來…）、情緒（1.1）、乾貨（1.0：重點、關鍵、秘訣、步驟…）、
     利害（0.8：賠、賺、第一次、後悔…）、具體數字（0.8：300 萬、3 個步驟…）、鮮明觀點（0.6）、笑點（0.8）；每類最多算兩次，
     `1 − exp(−總和/1.6)` 飽和。詞庫含繁體、簡體與英文。理由（「反轉、具體數字」）會寫進決策說明。
   - 語氣：這句的響度在整支影片裡的百分位（0.7）＋語速接近「偏快但不趕」（0.3）。排名前先四捨五入，避免浮點雜訊變成名次差。
4. **視窗分數**：新元件 `content`（0.42）＝ 0.5 × 視窗內時間加權平均分數 /（本片第 90 百分位句子分數，至少 0.3）＋ 0.5 × 視窗內最高分句子；
   `content_hook`（0.10）＝ 開頭 3 秒內開始的最高分句子（開頭金句）。兩者用前綴和＋稀疏表（range max）一次算完上百萬個視窗。
   **沒有文字時**這兩項不可用，其餘權重依比例放大——和改版前的相對比例完全相同（原本的 `keywords` 已併入 `content`）。
5. **AI 判斷（選用、預設關閉）**：`highlight_engine="claude"` 時，把每句的**文字與時間碼**（不含影音）送給 Claude（`claude-opus-5-5`，
   結構化輸出 JSON、伺服器端 fallback），每句 0–10 分取代「用詞」部分（語氣與水詞仍在本機算）。需要選用套件 `anthropic`
   （`requirements-ai.txt`）與 API 金鑰；沒裝／沒金鑰／斷線／被拒答 → 在 `notes` 說明並退回本機規則，**規劃不會因此失敗**。
   結果快取在 `<cache>/highlights/<雜湊>.json`（鍵＝句子、模型、`PROMPT_VERSION`），同一支影片再規劃不再連線、計畫雜湊不變。
   這是「不連網」原則的**明確例外**：只有使用者在介面選了 AI 判斷才會連線，介面與 notes 都寫明只傳文字。
6. **說明**：每支短影音的決策多了「爆點句：『…』（反轉、具體數字，0.95）」與 `highlights` 清單；序列名稱帶上最佳句的前 12 字。

## 3c. 一鍵字幕（`autoedit/subtitle_gen.py`；CLI `autocut subtitle`；對話框按鈕）
語音辨識整支影片 → `<影片>.srt`（16 個全形字一條，CJK 斷句規則同 §3.10）＋ `<影片>.words.json`（逐字時間：SRT 存不下，
長片剪水詞要用；`highlights.word_segments` 會讀它）。已存在時不覆蓋（CLI 要 `--force`，介面會先問）；模型未下載時先徵求同意。
使用者可先用記事本修正錯字再剪——字幕檔的文字優先於語音辨識。

## 3d. 長片精華：一支長影片 → 一支較短的長片（`autoedit/longform.py`、`analysis/speakers.py`，計畫 kind=`longform`）
輸入：`asset_id` + `LongformParams`。輸出：一個序列（尺寸／影格率同來源，V1 片段、V2 空、A1/A2 空、可選字幕）。
1. **留什麼**：`mode="highlights"`：句子依爆點分數由高到低選，直到說話內容達 `keep_ratio`（或 `target_seconds`）；
   再補上被選句子前一句的「鋪陳」（提問或 ≤ 2.5 秒的短句）與夾在兩個被選句之間的單句「銜接」（≤ 3 秒），兩者合計不超過目標的 125%。
   被選的句子前留 0.12 秒、後留 0.22 秒，但不伸進沒被選的鄰句；相距 < 1.2 秒的保留段落併成一段（中間的停頓交給緊湊化）。
   `mode="clean"`：每一句都留，只剪下面兩項。沒有文字時以「靜音之間的人聲段落」為單位、依響度排名挑選（notes 會說明）。
2. **去水詞**（需要逐字時間）：「嗯、呃、欸、um、uh」一律剪；「那個、就是、然後、對、like…」只有前後是停頓（> 0.25 秒）或連續重複時才剪；
   重複字：同一詞連續重複只留最後一次，**單一中文字要連續三次以上**（聊聊、謝謝、天天是正常語法）。沒有逐字時間時只剪整句都是水詞的字幕。
   長停頓同短影音（`pause_min`／`pause_keep`）。剪法 `cut_ranges`：落在段落開頭／結尾的水詞也會剪，剩下不到 0.25 秒的碎片丟掉。
3. **誰在說話**（`analysis/speakers.py`）：額外一次 640 px 寬、5 fps 的灰階解碼；每 2 個取樣用 Haar 找臉，依位置連成「人」（軌跡），
   出現少於 8% 取樣的軌跡丟掉。每個取樣、每個人：**嘴部區域**（臉框高度 62%–108%、寬度 20%–80%）與上一取樣的平均灰階差，
   減去整張畫面的平均差（鏡頭移動、切換、閃爍會一起抵消）。說話判定：有人聲的取樣上，1 秒平滑後的嘴部動作最大者，
   要高於全體可見值第 75 百分位的一半、且至少是第二名的 1.35 倍才算；1.5 秒多數決平滑，同一人相隔 < 2.5 秒的段落合併。
   （不對每個人各自正規化：那樣會把「B 一直在講」和「A 都沒講」變得一樣。）結果快取在 `<cache>/speakers/`。
4. **講者特寫**：畫面上至少兩個人、某人連續說話且**剪輯後仍有 ≥ `zoom_min_turn`（3 秒）**→ 硬切到他的特寫：
   倍率＝臉高的 0.42 倍畫面（夾在 1.35–2.4；可固定），臉放在畫面中上方。**放大多久依內容決定**：
   開場與每次換人說話，先給全景至少 `zoom_back_min`（2 秒，盡量延到句子結束）再推近；特寫超過 `zoom_max`（12 秒）的 60% 後，
   在下一個句子邊界切回全景（若下一句是爆點句就撐到上限）；全景至少 `zoom_back_min` 後，遇到爆點句或全景已達上限一半就再推近；
   含爆點句的特寫多放大 1.12 倍。特寫 < 1 秒不做。每個特寫一筆 `decision(kind="zoom")` 說明是誰、說了多久、為什麼。
5. **審核**：新規則 `longform.zoom`（warn）：特寫短於 1 秒、或放大後來源只剩不到輸出高度一半的像素（會糊）。
6. **已知限制**：特寫是硬切（像多機位剪法），沒有推近動畫（要改渲染器，見 `docs/CODEX_TASKS.md` CODEX-5）；
   嘴部動作法對側臉、遮嘴、兩人同時說、鏡頭很遠的畫面判斷不出來（那些時間維持全景）；**沒有在真實訪談影片上驗證**。

## 4. 長片：很多毛片 → 一支長片（`autoedit/assemble.py`）
輸入：`asset_ids`（影片/圖片；音訊檔忽略並記在 notes）+ `AssembleParams`。輸出：`EditPlan(kind="assemble")`，**一個序列**。
1. **序列格式**：`aspect="auto"` → 依「各素材長度加權」的多數方向：橫式 1920×1080、直式 1080×1920、方形 1080×1080；`fps=0` → 素材最常見幀率，正規化到 24/25/30/50/60 並夾在 24–60。
2. **排序**：`chronological` = 容器 `creation_time`（`asset.created`），缺則檔案修改時間，再以自然排序檔名；`name` = 自然排序；`as_given` = 傳入順序。排序依據寫進 `decisions`（`kind="order"`）。
3. **逐素材切成 shot**：用 `scenes` 切；太長的連續鏡頭依 `max_shot` 再切（有人聲時切在靜音/句末）。
4. **品質過濾（`drop_bad`）**：相對於該素材中位數 `sharp < 0.45` → 模糊；`bright < 0.08` 或 `> 0.95` → 太暗/過曝；`motion≈0` 且全程靜音 → 無內容；動作量變異係數過高 → 晃動。丟棄的 shot 記 `decision(kind="drop", why=…)`（上限 50 筆，其餘合併成一筆統計）。
5. **每個 shot 挑最佳子區段**（長度依 `pace_shot_range`，`min_shot` 下限）：最大化 清晰度 + 人臉 + 人聲 + 適中動作；避開 shot 頭尾各 0.3 s；有人聲時邊界貼靜音/句末。
6. **長度預算**：`target_seconds>0` → 各來源配額 ∝ √(可用長度)，每個有可用內容的來源至少 1 個 shot，其餘依分數補到預算；來源內維持時間順序。可用內容不足 → 全收並在 `notes` 說明實際長度。`target_seconds=0` → 全收（只剔除壞段）。
7. **鋪排（V1，A 軌放配樂）**：依序排；`transition="crossfade"` → 相鄰片段交疊 `crossfade_seconds`（上限＝兩片段中較短者的一半，且同一時刻最多 2 層，否則改硬切）；格式不符的來源 `fit="blur"`（或 `cover`/`contain`）；`cover` 時用與 §3.9 相同的取景路徑。
8. **聲音**：保留原聲（`keep_audio`）；依各來源 p90 響度把 `clip.volume` 拉到共同目標（限制 0.35–2.8 倍）。有 `music_path` → A1 配樂（重複排滿、`fade_music_out` 淡出、靜態音量）。
9. **標記**：每個來源的第一個 shot 放一個 `Marker`（名稱=檔名），方便在時間軸跳轉。圖片素材 → 靜態片段 3 秒。
10. **字幕**：同 §3.10（`captions` = off/auto/file）。
11. **決策紀錄**：`shot`（為什麼選這段）、`drop`、`order`、`format`、`music`、`caption`。

## 5. 審核（`autoedit/audit.py`）
`audit(plan, proj, rules) -> AuditReceipt(plan_sha256=plan.sha256(), project_revision=proj.revision, project_sha256=proj.content_sha256(), accepted=<沒有 error>, findings, thresholds=rules.to_dict())`。
每條規則一個固定的 `rule` 代碼、`severity`（error/warn/info）、繁體中文訊息、`where`（序列/片段/字幕 id 或時間）。

| rule | 嚴重度 | 條件 |
|---|---|---|
| `plan.empty` | error | 沒有序列 |
| `plan.hash` | error | `plan.to_dict()` 內聲明的雜湊與重算不符（防竄改） |
| `source.missing` | error | 片段引用的素材不在專案也不在 `plan.assets` |
| `source.file` | error | `require_sources_exist` 且檔案不存在 |
| `source.changed` | error | 檔案指紋與 `plan.sources[].fingerprint` 不符（分析後被換掉） |
| `seq.invalid` | error/warn | `core.project.validate_sequence` 的 error/warn 逐條轉入 |
| `clip.short` | warn | 片段 < `min_clip_seconds` |
| `shorts.aspect` | error | `kind=shorts` 但序列不是直式 |
| `shorts.band` | warn（`strict_platform_bands` 時 error）| 長度超出平台長度帶 |
| `shorts.deadzone` | warn（strict 時 error）| `yt_shorts` 落在範例死區 |
| `shorts.first_cut` | warn | 第一個視覺剪點（V1 第一個片段的結尾，或第一個取景/zoom 突變）晚於 `first_cut_max` |
| `shorts.mid_word` | warn | 有逐字時間時，剪點落在字的中間 |
| `caption.reading` | warn / error | 閱讀速度 > `reading_warn_cps` / `reading_fail_cps`（字數不含空白與標點） |
| `caption.rate` | warn | 序列 ≥ 20 s 且字幕換頁率 < `caption_rate_warn_per_min`（太稀疏） |
| `caption.overlap` | error | 兩條字幕重疊超過 `max_caption_overlap` |
| `caption.short` / `caption.long` | warn | 少於 `min_cue_seconds` / 字數超過 `max_cue_chars` |
| `caption.range` | error | 字幕超出序列範圍 |
| `assemble.target` | warn | 實際長度與目標差距比例 > `target_tolerance`（來源不足時改為 info） |
| `assemble.unused` | info | 某來源有可用內容卻一個 shot 都沒用到 |
| `start.black` | warn | `black_start_warn` 且第一個片段起頭約 0.5 s 平均亮度 < 0.05（有證據時） |
| `longform.zoom` | warn | `kind=longform` 的特寫短於 1 秒，或放大後來源像素不到輸出高度一半 |
| `total.max` | error | 總長 > `max_total_seconds` |
| `human.pending` | info | 一律附上：「機器審核只檢查規則與技術面，尚未經人工審片」 |

## 6. 套用（`autoedit/apply.py`）
```python
def remap_plan_assets(project: Project, plan: EditPlan) -> dict[str, str]
    # 把 plan.assets 加進 project（路徑相同者沿用既有素材），回傳 {計畫素材 id: 專案素材 id}；供 apply 與 api.preview_project 共用
def apply(editor: Editor, plan: EditPlan, receipt: AuditReceipt) -> list[str]
```
- 先 `core.plan.check_receipt(plan, receipt, editor.project)`（不通過就丟 `PlanError`，**不改任何東西**）。
- 在**一個** `editor.do("自動剪輯：…", fn)` 內：`remap_plan_assets` → 重寫片段的 `asset_id` → 把計畫序列（`Sequence.from_dict(deepcopy)`）加進專案；序列/軌道/片段/字幕/標記 id 若與專案現有 id 衝突就重新產生（無衝突時沿用計畫 id）；
  `project.active` 設為第一個新序列；`ops.normalize_sequence`；`validate_sequence` 有 error 就丟例外（整批回滾）；
  每個新序列 `meta["ai"] = {"run_id","kind","plan_sha256","source":{...}, "score": …}`；最後 `core.plan.record_run(project, plan, receipt, seq_ids)`（`human_review="pending"`）。
- 回傳新序列 id 清單。`Ctrl+Z` 一次復原整批（含新增的素材與 `ai_runs` 紀錄）。

## 7. `autoedit/captions.py` 其他公開函式
```python
def generate_cues_for_sequence(proj, seq, progress=None, cancel=None, *, language="zh", traditional=True,
                               model_size="small", allow_download=False) -> list[CaptionCue]
```
給 GUI「自動產生字幕」用：對序列中每個**有聲、未靜音、未停用**的 V 軌片段（同一時間以最上層為準）轉錄其 `src_in..src_out`（以 `speed` 換算），
把素材時間對應到時間軸時間，套用 §3.10 的切段規則，回傳 `CaptionCue` 清單（GUI 用 `ops.replace_cues` 一次寫入）。沒有語音辨識 → 丟 `SpeechUnavailable`。

## 8. 測試要求（`tests/test_analysis.py`、`test_shorts.py`、`test_assemble.py`、`test_audit_apply.py`，ffmpeg 合成素材，沒有 ffmpeg 時整合測試跳過）
- 合成素材：音訊=「440 Hz 音爆 + 靜音」交替（可精確驗證靜音/人聲偵測與緊湊化）；影片=不同顏色段落串接（鏡頭切換時間已知）、`testsrc2` 動態、`boxblur` 模糊段、全黑段；直式素材。
- 單元測試：靜音/語音偵測落在預期區間；鏡頭切換偵測到已知時間（±0.25 s）且 `refine_cut` 精準到 ≤ 1 影格；Evidence 快取命中；`reframe` 用**假的臉框序列**（從左到右移動）驗證：跟隨、死區、夾邊界、鏡頭切換處跳變、關鍵格數量上限；
  評分/視窗搜尋用假 Evidence（含假逐字稿）驗證：不重疊、間隔、長度帶、緊湊化後長度、死區處理、句尾邊界優先。
- 整合測試：`propose_shorts` → `audit_plan` → `apply_plan` → 序列尺寸 1080×1920、片段來源範圍合法、**一次 `undo` 完全還原**、`human_review=="pending"`；`render_sequence` 第一支並 `technical_qa` 通過；
  `propose_assemble`（橫式+直式+模糊+全黑）→ 壞段被丟、直式來源 `fit=="blur"`、交疊合法、`validate_sequence` 無 error；計畫雜湊可重現（跑兩次相同）；竄改計畫 → `check_receipt` 失敗；套用後再編輯專案 → 收據失效。
- 審核規則：每條 rule 至少一個會觸發、一個不會觸發的測試（可直接構造 `EditPlan`）。
- 取消：`cancel.set()` 後 `propose_*` 丟 `Cancelled`，且沒有殘留 ffmpeg 子行程。
- 效能：90 秒合成素材的完整分析 < 30 秒（2 核心環境）；整份測試 < 5 分鐘。

## 9. 誠實聲明（會寫進 README 的限制）
- 評分與門檻是**啟發式**，在合成素材上驗證過「機制正確」，**並未在大量真實影片上校準**；請把結果當成「很快的第一版剪輯」，再用時間軸修。
- 人臉偵測用 Haar（正臉、光線好時才可靠）；語音辨識（選用）的品質取決於模型與錄音。
- 沒有動態音樂閃避；沒有轉場特效庫（只有硬切與交叉淡化）；變速只有線性。
- 上游數值（13–25/45–60 秒帶、2.05 秒首剪、字幕 5/7 字每秒…）是範例校準值，這裡全部可調。
