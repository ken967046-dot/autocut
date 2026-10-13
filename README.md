# AutoCut Studio

**桌面剪輯軟體（不是網頁）**：把檔案直接拖進視窗，自動剪好，再像一般剪輯軟體那樣在時間軸上逐段修正，最後輸出。

- **16:9 長影片 → 9:16 短影音**：分析聲音、畫面、鏡頭與人臉，挑出精華、自動取景成直式，可選配字幕與配樂。
- **很多支毛片 → 一支長片**：自動排序、去掉模糊／過暗／晃動的部分、加轉場、控制總長度。
- **像剪輯軟體一樣修正**：時間軸分割、修剪、移動、波紋刪除、交叉淡化、變速、字幕、取景框關鍵格，全部可復原／重做。
- **誠實的檢查**：輸出後機器只做「**技術檢查**」（能不能讀、長度、尺寸、黑畫面、凍結畫面、靜音、響度）。
  剪得好不好看**一定是你審片**；軟體不會替你按「通過」。

介面全部是繁體中文，深色主題，目標平台 Windows 11（開發與自動測試在 Linux 無頭環境完成，Windows 實機狀況見最下方「驗證狀態」）。

---

## 安裝（Windows）

1. 把整個資料夾放在你喜歡的位置（路徑可以有中文或空格）。
2. 雙擊 **`Setup-AutoCut.bat`**。它會：找到（或用 winget 安裝）Python 3.10 以上 → 在資料夾內建立 `.venv` → 安裝套件
   → 下載 ffmpeg 到 `tools\ffmpeg`（會核對發布者列出的 SHA-256，不符就不安裝）→ 在桌面建立「AutoCut Studio」捷徑。
   - **資料夾內**：`.venv`（套件）、`tools\ffmpeg`（ffmpeg）。
   - **資料夾外**：Python 本身（找不到時用 winget 為目前使用者安裝）、pip 的下載快取、桌面捷徑、軟體設定（目前使用者的登錄檔）、
     以及 `%LOCALAPPDATA%\AutoCutStudio`（快取與記錄）。
   - **安裝時需要網路**（pypi.org、ffmpeg 的下載來源 gyan.dev 或 GitHub 上的 BtbN、winget）。ffmpeg 下載失敗不會讓整個安裝失敗：
     軟體會裝好，結束時明確告訴你 ffmpeg 還沒裝（沒有 ffmpeg 時無法匯入／預覽／輸出）。
   - 想確認過程：`Start-AutoCut.bat` 啟動前會先檢查程式能不能載入，載入失敗時會顯示原因，而不是視窗默默不出現。
3. 想要**自動產生字幕**（語音辨識）：在終端機執行 `Setup-AutoCut.bat -WithSpeech`。語音模型第一次使用時才下載，需要網路，軟體會先問你。
4. 開啟軟體：雙擊桌面捷徑或 **`Start-AutoCut.bat`**；也可以把影片直接拖到 `Start-AutoCut.bat` 上。

已經有 Python 與 ffmpeg 的人：

```bash
python -m venv .venv && .venv\Scripts\activate        # macOS / Linux: . .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps
python -m autocut doctor                                # 檢查 ffmpeg / PySide6 / OpenCV / 語音是否就緒
python -m autocut studio                                # 開啟桌面軟體
```

---

## 三分鐘上手

1. **拖進來**：把影片（或整個資料夾）拖到視窗任何地方，放開就匯入。支援影片、音訊、圖片、字幕檔（.srt / .ass / .vtt）與專案檔（.acproj）。
2. **看建議**：
   - 拖進**一支 45 秒以上的橫式影片** → 上方出現橫幅「要自動剪成 9:16 短影音嗎？」
   - 拖進**兩支以上的影片** → 「要自動剪成一支長片嗎？」
   - 橫幅只是建議：按「略過」什麼都不會改；按「放到時間軸」就是手動剪。
3. **自動剪輯**（選單「自動剪輯」或橫幅按鈕）：先設定（支數、目標長度、字幕、取景方式…），按「開始分析」。
   分析**不會動你的專案**；結果頁列出每一支候選（縮圖、長度、為什麼選它、審核意見），可取消勾選、按「預覽此段」，
   確定後按「套用選取的片段」→ 變成新的序列（**一次 Ctrl+Z 就整批復原**）。
4. **在時間軸修正**：點片段 → 右側「檢查器」調整音量、淡入淡出、速度、取景、字幕…；拖邊緣修剪、拖本體移動；
   `S` 在播放頭分割（不能落在交叉淡化的中間，請把播放頭移到轉場外）、`Delete` 刪除、`Shift+Delete` 波紋刪除（後面的片段會補位）。
5. **輸出**（`Ctrl+E`）：選預設（短影音 9:16 / YouTube 1080p / 原尺寸 / 快速草稿）與檔名 → 開始輸出 → 完成後自動做技術檢查。
6. **審片**：看完成品，在輸出結果或檢查器的「檢查」分頁按「通過 / 需修改 / 退回」。在你按之前，狀態一直是「尚未審片」。

---

## 快速鍵

| 動作 | 按鍵 |
|---|---|
| 新增／開啟／儲存／另存專案 | `Ctrl+N` / `Ctrl+O` / `Ctrl+S` / `Ctrl+Shift+S` |
| 匯入素材／輸出影片 | `Ctrl+I` / `Ctrl+E` |
| 復原／重做 | `Ctrl+Z` / `Ctrl+Y`（或 `Ctrl+Shift+Z`） |
| 播放／暫停 | `Space` |
| 渲染預覽（含聲音） | `Ctrl+Enter` |
| 在播放頭分割 | `S`（或 `Ctrl+K`） |
| 刪除／波紋刪除 | `Delete` / `Shift+Delete` |
| 前後一格／前後一秒 | `←` `→` ／ `Shift+←` `Shift+→` |
| 上一個／下一個剪輯點 | `↑` / `↓` |
| 到開頭／結尾 | `Home` / `End` |
| 標記／入點／出點 | `M` / `I` / `O` |
| 全選／複製／貼上 | `Ctrl+A` / `Ctrl+C` / `Ctrl+V` |
| 時間軸放大／縮小／符合視窗 | `+` / `-` / `\` |
| 全螢幕 | `F11` |

時間軸上拖曳時按住 `Alt` 可暫時關閉吸附；`Shift+拖邊緣` 是波紋修剪；`Alt+拖片段` 是滑動素材（slip）。

---

## 指令列

```bash
python -m autocut doctor [--hw]                           # 檢查環境（--hw 實測硬體編碼器）
python -m autocut probe 素材.mp4                          # 讀取素材資訊
python -m autocut shorts 長片.mp4 -o 輸出資料夾 --count 3 --render      # 16:9 → 9:16 短影音，並輸出 + 技術檢查
python -m autocut assemble 毛片資料夾 -o 輸出資料夾 --target 180 --render   # 多支毛片 → 一支長片
python -m autocut render 專案.acproj -o 成品.mp4 --preset youtube_1080p
python -m autocut qa 成品.mp4 --size 1080x1920            # 只做技術檢查
python -m autocut studio [專案或素材...]                  # 開啟桌面軟體
```

結束代碼：`0` 成功、`1` 失敗或技術檢查未通過、`2` 參數錯誤或審核未通過、`3` 找不到 ffmpeg、`130` 已中止。

---

## 它怎麼自動剪（以及它不做什麼）

流程是「**證據 → 計畫 → 審核 → 套用 → 渲染 → 技術檢查 → 人工審片**」：

1. **證據**：從素材分析出聲音能量、鏡頭切換、畫面清晰度／亮度／晃動、人臉位置（以及選用的語音逐字稿）。
2. **計畫**：依證據提出剪法（每一段取自哪裡、取景框怎麼動、字幕與配樂），計畫有雜湊值，內容一變就不同。
3. **審核**：套用前先自動檢查（長度、第一個剪點、字幕可讀性、素材是否存在…），產生綁定計畫與專案版本的收據；專案在審核後改過就**作廢重審**。
4. **套用**：一次完成、一次復原；失敗時什麼都不會被改。
5. **技術檢查**：輸出後檢查檔案本身；沒執行的項目標示「–」而不是假裝通過。
6. **人工審片**：只有你能按。

重要限制（請當真）：

- 取景、挑段、轉場都是**啟發式規則**，不是會「看懂劇情」的模型。成品一定要自己看；它是快速的第一版，不是最終版。
- 人臉取景使用 OpenCV 內建的 Haar 偵測，側臉、遮擋、很小的臉會偵測不到；找不到臉時改用畫面動態或置中。
- 語音辨識是**選用**功能，需另外安裝（`-WithSpeech`）並下載模型；沒安裝時字幕選項會明確告訴你，而不是假裝有字幕。
- **軟體執行時**預設不連網、不上傳任何素材；只有「下載語音模型」會連網，而且會先徵求你的同意。（**安裝**時需要網路，見上方「安裝」。）
- 即時預覽是靜音的；要聽聲音請按「渲染預覽（含聲音）」。
- 輸出格式是 MP4（H.264 + AAC）。硬體編碼器（NVENC / QSV / AMF）可選，失敗會自動改用 CPU。
- 還沒有的功能：多機位剪輯、進階調色、動態圖層／動畫字卡（規劃中的選用外掛）。

---

## 資料放在哪裡

- 專案檔：`.acproj`（JSON，可以用文字編輯器打開；素材只存路徑，不複製）。素材搬家後用素材庫右鍵「重新連結…」。
- 自動存檔：每 60 秒存有改動的專案；當機或忘記存檔，下次開啟會問你要不要復原。
- 快取、記錄、自動存檔：Windows 在 `%LOCALAPPDATA%\AutoCutStudio`（可用環境變數 `AUTOCUT_HOME` 改位置）；選單「說明 → 開啟資料與記錄資料夾」可直接打開。

---

## 給開發者

```bash
pip install -r requirements-dev.txt
python -m pytest                                          # 全部測試（沒有 ffmpeg 時整合測試會自動跳過）
python -m pytest -m "not e2e"                             # 跳過最慢的端到端驗收流程
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_e2e_gui.py    # 從拖入檔案到輸出成品的完整流程（真視窗、真引擎、真 ffmpeg）
python scripts/dev/campaign.py fuzz wide --seeds 3000     # 長時間隨機測試：測試套件只跑幾百組種子，這支從套件用完的種子之後接著跑
python scripts/dev/campaign.py --help                     # glue（字幕黏貼）、fuzz、oracle（真的渲染再逐格比對）、alpha、coverage
```

- 架構與模組地圖：`docs/ARCHITECTURE.md`；自動剪輯方法：`docs/AUTOEDIT_METHOD.md`；介面規格：`docs/GUI_SPEC.md`；API 速查：`docs/API_CHEATSHEET.md`。
- **Windows 實機驗收清單**（約 30 分鐘，每步寫了「應該看到什麼」）：`docs/ACCEPTANCE.md`。
- 協作規則（Claude ↔ Codex 輪流寫、互相審）：`AGENTS.md`、`COLLABORATION.md`、`TASKS.md`。
- 所有修改專案的動作都走 `core.ops` ＋ `core.editor.Editor`（一個動作＝一次可復原的交易）；GUI 不直接改資料。
- 第三方與方法來源聲明：`THIRD_PARTY_NOTICES.md`（自動剪輯流程改編自 video-autopilot-kit，MIT © 2026 Hao0321 Studio）。

授權：MIT（見 `LICENSE`；第三方元件各依其授權，見 `THIRD_PARTY_NOTICES.md`）。打包成獨立 exe 見 `packaging/Build-Windows.ps1`
（**預設不把 ffmpeg 放進去**，原因見該檔說明與 `THIRD_PARTY_NOTICES.md`；尚未在 Windows 實機驗證）。

---

## 驗證狀態

> 這一節只寫「真的跑過」的事。最後更新：2026-10-09。逐項紀錄與證據在 `TASKS.md`。

**在 Linux 無頭環境跑過**（Python 3.13；沒有真實螢幕、沒有 Windows）：

- 全部自動測試，在全新建立的環境（只照 `requirements-dev.txt` 安裝：PySide6 6.12.0、ffmpeg 6.1.1）：
  **2038 項通過、2 項因環境跳過（10 分鐘長影片測試、檔案權限測試）、0 項失敗**。
  含端到端流程：拖入 → 自動剪輯 → 手動修正 → 輸出 → 技術檢查 → 審片按鈕（真視窗、真引擎、真 ffmpeg）。
- 長時間隨機測試（`scripts/dev/campaign.py`，用測試套件沒跑過的新種子，全部零失敗）：
  剪輯規則 fuzz base 3000 組、wide 2000 組；字幕黏貼 glue 兩種模式各 4000 組（每種約 24 萬次字幕檢查）；
  渲染對照 oracle（真的輸出，再逐格解碼畫面條碼、聽音高）30／25／24／60／29.97／23.976 fps 共 640 組（約 17.9 萬格畫面、1.7 萬處聲音）。
- 畫面與聲音同步：合成素材量到聲音固定晚約 5 ms、畫面與聲音相差最多 19 ms；聲音比畫面晚開始的合成檔也保留那段延遲。
- 介面：offscreen 截圖逐張檢視（1100×700、1366×768、1920×1080）＋模擬滑鼠／鍵盤的測試。

**沒有驗證過**（需要在你的 Windows 電腦上做；逐步清單見 `docs/ACCEPTANCE.md`）：

- 真實的 Windows 視窗：螢幕縮放（DPI）、中文輸入法、從檔案總管拖放、原生檔案對話框、全螢幕、工作列。鍵盤焦點的修正只用模擬按鍵測過。
- `Setup-AutoCut.bat`、`Get-Ffmpeg.ps1`、`Start-AutoCut.bat`、`Run-Tests.ps1`、`Codex-Review.ps1`、`packaging\Build-Windows.ps1`（PyInstaller）：
  只在 Linux 上讀碼審查，沒在 Windows PowerShell 5.1 跑過；ffmpeg 下載來源與雜湊格式沒有實際連線核對。
- 硬體編碼器（NVENC／QSV／AMF）。
- 語音辨識（faster-whisper）與簡轉繁（opencc）：沒安裝、沒用真實中文語音跑過。
- 自動剪輯的**品質**：取景（Haar 人臉）、選段、順序、轉場都是啟發式規則，沒有用真實素材評估；
  真實手機／相機素材的畫面聲音同步也沒量過（只量過合成素材）。
- 真正的聲音播放（QtMultimedia）與預覽的音畫同步。
- Codex 交叉審查：雲端環境的網路政策擋住 Codex 登入；目前只做過兩輪 Claude 全新上下文審查員的審查（不是 Codex）。
