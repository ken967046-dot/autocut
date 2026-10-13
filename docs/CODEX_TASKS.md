# Codex 任務卡（由 `scripts/windows/Codex-Task.ps1 -Task <代號>` 讀取）

每張卡都是**一個 Codex 回合就能做完**的小範圍工作（省額度：範圍小、要讀的檔案已列出、驗收條件明確）。
Codex 會在自己的分支 `codex/<代號>` 上工作；做完後在 `TASKS.md` 對應列貼「交接說明」，再交給 Claude 審查與合併。
共同規則：先讀 `AGENTS.md`；**不要**改 `core/ops.py` 的既有行為、計畫／收據雜湊規則；`human_review` 永遠不能由程式設定；
沒驗證過的事不要寫成已驗證。

額度不夠時的順序建議：CODEX-1（只審查、最便宜）→ CODEX-2 → CODEX-3 → 其餘。

## CODEX-1
**審查 2026-10-13 的「找爆點／長片精華／講者特寫／一鍵字幕」改動（唯讀，不改程式）**

這張卡用審查腳本跑，不用 Codex-Task：

```powershell
.\scripts\windows\Codex-Review.ps1 -Base 153ba1f -Paths src/autocut/autoedit/highlights.py,src/autocut/autoedit/longform.py,src/autocut/autoedit/llm_judge.py,src/autocut/autoedit/subtitle_gen.py,src/autocut/analysis/speakers.py,src/autocut/autoedit/scoring.py,src/autocut/autoedit/shorts.py
```

特別請看：
- `highlights.repeated_word_spans` / `filler_spans`：會不會剪到正常的詞（疊字、「那個人」、「對啊」）？
- `longform.cut_ranges` 與 `split_for_closeups`：片段會不會短於 0.25 秒、重疊、超出素材範圍？
- `speakers.speaker_labels`：兩人同時講話、一人一直沒入鏡、人臉偵測時有時無時的行為。
- `llm_judge`：沒有金鑰／沒有套件／網路斷線／AI 拒答時是否一定退回本機規則、且不會把金鑰寫進計畫或專案檔。
- Windows：`subtitle_gen` 寫檔（中文路徑、檔案被佔用）、`.words.json` 編碼。

## CODEX-2
**用真實影片校準「爆點」詞庫（需要使用者提供一支影片與它的字幕檔）**

- 讀：`src/autocut/autoedit/highlights.py`（`_CATS`、`score_sentences`）、`src/autocut/autoedit/scoring.py`（`DEFAULT_WEIGHTS`）、
  `tests/test_highlights.py`。
- 做：
  1. `python -m autocut shorts <影片> --subs <字幕.srt> -o out --count 5`，把 `out\*.plan.json` 裡每支的 `highlights`（爆點句）
     與使用者認為的爆點比較，列出漏掉的句子與誤判的句子。
  2. 只調 `_CATS` 的詞與權重（或新增一類），不要改評分架構。每個改動都要能說出是哪一句話促成的。
  3. 把這些句子（去識別化）加成 `tests/test_highlights.py` 的回歸測試：「這句要比那句高分」。
- 驗收：`python -m pytest tests/test_highlights.py tests/test_shorts.py -q` 全綠；交接說明列出調整前後選到的段落。

## CODEX-3
**真實雙人對談影片上驗證講者特寫（Windows 實機）**

- 讀：`src/autocut/analysis/speakers.py`、`src/autocut/autoedit/longform.py`（`plan_closeups`）。
- 做：`python -m autocut longform <雙人訪談.mp4> -o out --render`，看成品：
  - 誰在講就放大誰？放大時臉在畫面中上方嗎？換人講話時有先回到全景嗎？
  - 記錄：`out\*.plan.json` 的 `brief.speakers`（偵測到幾個人、幾段發言、幾個特寫）與錯誤的段落時間。
- 若誤判多：只調 `MOUTH_TOP/MOUTH_BOTTOM`、`speaker_labels` 的 `margin`、`speaker_turns` 的 `min_gap/vote`，並用
  `tests/test_longform.py` 的合成方式補一個重現該情況的測試。
- 驗收：`python -m pytest tests/test_longform.py -q` 全綠；交接說明附「調整前／後」誤判段數。

## CODEX-4
**講者分析的速度（長影片）**

- 讀：`src/autocut/analysis/speakers.py`（`analyze_speakers`、`SpeakerAnalyzer.push`）。
- 做：量測 1 小時 1080p 影片的 `analyze_speakers` 時間（`python -c "..."` 或加 `--profile` 都可以）。超過 10 分鐘就優化：
  例如臉框穩定後降低偵測頻率、只解碼人臉附近的區域、或 `frame_width` 依臉的大小自動決定。不能改變 `SpeakerEvidence` 欄位意義。
- 驗收：`tests/test_longform.py` 全綠；交接說明附量測前後時間（同一支影片、同一台電腦）。

## CODEX-5（先提案，不要直接實作）
**平滑放大（推近／拉遠的動畫），取代目前的硬切特寫**

目前特寫是「硬切」到放大的片段（跟多機位剪法一樣）。平滑推近需要 `Transform.keys` 帶 zoom，會動到 `core/project.py`、
`core/geometry.py`、`render/compiler.py` 與 `render/live.py`，屬於架構變更：請在 `TASKS.md` 寫提案（資料格式、相容舊專案、
渲染快取鍵、要加的對照測試），等 Claude／使用者同意後才做。
