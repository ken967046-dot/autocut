# 第三方聲明（Third-party notices）

AutoCut Studio 本身以 MIT 授權釋出（見 `LICENSE`）。它是獨立實作的桌面軟體。它的**資料模型、渲染管線、介面與演算法都是自己寫的**；
下面列出「方法上借鏡」與「執行時使用」的第三方專案。散布本軟體前，請再核對各專案目前的授權條款（本表是依開發當時的理解整理，不是法律意見）。

## 方法來源（改編，非複製程式碼）

### video-autopilot-kit（影片自動化工具包）
- 來源：<https://github.com/Hao0321/video-autopilot-kit>（介紹頁：freetwai.com）
- 授權：MIT License，Copyright (c) 2026 Hao0321 Studio
- 借用的是**工作流程的想法**：「證據 → 計畫（canonical JSON 雜湊）→ 審核收據（綁計畫雜湊與專案版本）→ 原子套用 → 渲染 → 技術 QA → 人工審片 → 結果紀錄」、
  「失敗就關閉（fail-closed）」、「沒跑的檢查標 `not_run`」、「機器不代替人判斷好不好看」，以及短影音規則中的**範例校準值**
  （長度帶 13–25 / 45–60 秒、首個剪點 ≤ 2.05 秒、字幕閱讀速度 5/7 字每秒、`CAP_PAD`/`CAP_GAP` 等）。
- 本專案把這些數值全部做成**可調參數**（`autocut/autoedit/config.py`），不是規定。

```
MIT License

Copyright (c) 2026 Hao0321 Studio

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 執行時相依

| 元件 | 用途 | 授權（開發當時的理解） | 備註 |
|---|---|---|---|
| PySide6 / Qt for Python | 桌面介面 | LGPL-3.0（或 GPL-3.0／商業授權） | 以動態連結方式使用；若用 PyInstaller 打包散布，需遵守 LGPL（保留聲明、允許使用者替換 Qt 函式庫） |
| NumPy | 數值運算 | BSD-3-Clause | |
| OpenCV（`opencv-python-headless`） | 即時預覽解碼與合成、鏡頭/人臉偵測 | Apache-2.0 | 官方 wheel 內含以 LGPL 方式建置的 FFmpeg 函式庫；Haar 人臉級聯檔隨 OpenCV 提供 |
| FFmpeg / ffprobe | 解碼、渲染、分析（**以獨立行程呼叫**，不連結） | LGPL-2.1+ 或 GPL-2.0+/3.0+（依建置版本而定） | 本專案**不內附** FFmpeg：安裝腳本（`scripts/windows/Get-Ffmpeg.ps1`）從 ffmpeg.org 下載頁連結的建置（gyan.dev、BtbN）下載，並核對發布者列出的 SHA-256；這些建置多半是 GPL 版。打包（`packaging/Build-Windows.ps1`）預設也不把 ffmpeg 放進資料夾，加 `-BundleFfmpeg` 才會放；若你把含 FFmpeg 的資料夾交給別人，需遵守該建置的授權（GPL 版要附上對應原始碼的取得方式等） |
| faster-whisper（選用） | 語音辨識（字幕） | MIT | 需另外 `pip install -r requirements-speech.txt`；模型檔第一次使用時才下載，預設不連網 |
| CTranslate2（選用，faster-whisper 相依） | 推論引擎 | MIT | |
| OpenAI Whisper 模型權重（選用） | 語音辨識模型 | MIT | |
| OpenCC（選用，`opencc-python-reimplemented`） | 簡體轉繁體 | Apache-2.0／MIT（依套件） | 沒安裝時字幕維持辨識原文並提示 |
| Anthropic Python SDK（選用，`anthropic`） | AI 判斷爆點（Claude API） | MIT | 需另外 `pip install -r requirements-ai.txt` 與 API 金鑰；預設關閉，只傳送字幕文字與時間碼 |
| pytest（僅開發） | 測試 | MIT | |
| PyInstaller（僅打包） | 產生 Windows 執行檔 | GPL-2.0 含打包例外（produced executables 不受 GPL 限制） | |

## 不使用、但評估過的項目
- **HyperFrames（HeyGen，Apache-2.0）**：HTML → 影片的動態圖形框架（需 Node.js ≥ 22、FFmpeg 與無頭 Chrome）。
  目前**沒有**整合；若之後做成選用的「動態圖層」外掛，它只會是獨立行程，產生的影片/透明 WebM 當一般素材匯入。
