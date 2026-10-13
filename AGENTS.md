# AGENTS.md — AutoCut Studio（給 Codex / Claude / 任何 AI 協作者）

AutoCut Studio 是一套 **桌面自動剪輯軟體**（不是網頁）：把 16:9 長影片自動剪成 9:16 短影音，或把很多支毛片自動剪成一支長片；
素材可直接拖進視窗，剪完後可像一般剪輯軟體一樣在時間軸上逐段修正，最後輸出。
目標平台 Windows 11（開發與自動測試在 Linux 無頭環境）。使用者介面文字一律 **繁體中文**。

## 先讀這些
1. `docs/ARCHITECTURE.md` — 模組地圖、資料模型、渲染管線、自動剪輯流程、GUI 結構
2. `COLLABORATION.md` — Claude ↔ Codex 分工、交接與交叉審查流程
3. `TASKS.md` — 目前待辦與誰負責
4. `docs/AUTOEDIT_METHOD.md` — 自動剪輯方法（改編自 video-autopilot-kit，MIT）

## 指令
```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest                                       # 全部測試（沒有 ffmpeg 時整合測試會自動跳過）
python -m pytest tests/test_core.py -q                 # 純 Python 核心
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_timeline.py tests/test_main_window.py   # 介面測試（無頭；其餘見 tests/test_*gui*、test_inspector、test_export_dialog…）
python scripts/dev/campaign.py --help                  # 長時間隨機測試（fuzz / glue / oracle / alpha / coverage）；見下方「完成的定義」
python -m autocut doctor                               # 檢查 ffmpeg / PySide6 / OpenCV / Whisper
python -m autocut studio [檔案或資料夾...]              # 開啟桌面軟體
```

## 目錄
```
src/autocut/
  util.py                 id、雜湊(canonical JSON)、原子寫檔、素材掃描
  core/                   專案資料模型、復原/重做、剪輯操作、剪輯計畫與審核收據（不依賴 Qt / ffmpeg）
  media/                  ffmpeg 定位與執行、ffprobe -> Asset、縮圖/波形快取
  render/                 時間軸 -> 可快取的渲染片段 -> ffmpeg；ASS 字幕；技術 QA；live.py（OpenCV 即時合成預覽）
  analysis/               聲音 / 鏡頭 / 畫面 / 人臉 / 語音證據
  autoedit/               短影音規劃、長片組裝、審核、套用（evidence -> plan -> audit -> apply）
  gui/                    PySide6 桌面介面
  cli.py                  `autocut` 指令列
tests/                    pytest；整合測試用 ffmpeg lavfi 合成素材
```

## 不可違反的規則
1. **所有修改專案的動作都走 `core.ops` + `core.editor.Editor`**（一個動作＝一次可復原的交易）。GUI 絕不直接改 `Clip`/`Track` 欄位。
2. 時間以秒為單位並對齊影格（`snap_time`）；同軌片段交疊＝交叉淡化，最多兩層（`validate_sequence` 會檢查）。
3. **機器只能說「技術檢查通過」，不能說「審片通過」。** `human_review` 只能由使用者在介面上按鈕設定（`core.plan.set_human_review`）。
4. 自動剪輯一律走 `propose_* -> audit_plan -> apply_plan`；收據綁計畫雜湊與專案 revision，過期就重審（fail-closed）。
5. 沒驗證過的事不要寫成已驗證；沒跑的檢查標 `not_run`。報告裡列出「已驗證 / 未驗證」。
6. 預設不連網、不上傳素材。語音辨識是選用（`faster-whisper`），沒安裝時要優雅降級。
7. Windows 相容：不要用 POSIX 專屬路徑或 shell 語法；subprocess 用參數陣列；中文路徑要能運作；ffmpeg 檔名參數用 cwd 相對的簡單名稱。
8. 不新增相依套件，除非說明必要性與授權；保留 `THIRD_PARTY_NOTICES.md` 的上游 MIT 聲明。
9. 使用者介面字串用繁體中文；程式碼註解與 docstring 用英文或中文皆可，但同一檔案保持一致。

## 完成的定義（Definition of Done）
- 新行為有測試；`python -m pytest` 全綠（或明確說明哪些因環境跳過）。
- 動到渲染的改動，要用 `tests/test_render.py` 風格的合成素材實際輸出並量測（長度、解析度、畫面/聲音內容），不是只看 ffmpeg 指令字串。
- 動到 `core/ops.py` 的剪輯規則（修剪、分割、移動、波紋、交叉淡化、字幕跟隨…）：除了 `pytest`，另跑
  `python scripts/dev/campaign.py fuzz base`、`fuzz wide`、`glue`、`glue --exact`（各用新的種子、至少幾千組），要零失敗。
  失敗的種子先重現、修好，再補成固定的回歸測試，並確認「把修正拿掉、測試就會失敗」。
- 動到 GUI 的改動，要在 `QT_QPA_PLATFORM=offscreen` 下 `widget.grab().save("x.png")` 截圖並檢視，並寫模擬滑鼠/鍵盤的測試。
- 在 `TASKS.md` 更新狀態，在 PR/交接說明列出：改了什麼、怎麼驗證、哪些沒驗證。

## 提交
小而明確的 commit；訊息說明「為什麼」。AI 協作者在訊息結尾加上 `Co-Authored-By:` 行。
