# Claude ↔ Codex 協作流程

目標：兩個 AI 互相補位——一個寫，另一個審；任何「完成」都要有證據，而不是自我宣告。

## 角色
| 角色 | 預設負責 | 不要做 |
|---|---|---|
| **Claude（作者／整合者）** | 架構、`core/`、`render/`、`autoedit/` 的決策邏輯、跨模組整合、最終驗證與交付說明 | 自己審自己剛寫的大改動（請交給 Codex） |
| **Codex（實作者／審查者）** | 依 `TASKS.md` 的獨立任務實作（例如 GUI 細節、分析演算法、Windows 打包）；審查 Claude 的 diff，專找 bug、邊界條件、Windows 相容性 | 改動 `core/ops.py` 的既有行為、改變計畫/收據的雜湊規則（需先在 TASKS 提案） |
| **使用者** | 驗收、審片（`human_review`）、決定取捨 | — |

> 實際狀態（2026-10-06，2026-10-13 再確認一次：`api.openai.com` 仍回 403）：雲端工作環境的組織網路政策擋住 `auth.openai.com`／`api.openai.com`，Codex 在那裡無法登入。
> 因此首輪交叉審查由「全新上下文的 Claude 審查員」完成並明確標示；Codex 審查請在使用者自己的 Windows 電腦上進行（見下方「在你的電腦上跑 Codex」）。

## 工作迴圈
1. **領任務**：在 `TASKS.md` 把一項任務的「負責人」改成自己，狀態改 `doing`。
2. **分支**：`git switch -c <claude|codex>/<任務代號>`。
3. **先寫測試（共同契約）**：測試就是兩邊共同遵守的規格。改行為先改測試。
4. **實作**，小步提交。
5. **自我檢查**：`python -m pytest` 全綠；渲染/介面改動依 `AGENTS.md`「完成的定義」附證據。
6. **交接**：在 `TASKS.md` 該任務下貼「交接說明」（範本如下），狀態改 `review`。
7. **交叉審查**：另一方依「審查範本」審 diff，結論只有三種：`approve`／`changes-requested`／`blocked`，每個問題要有檔案行號與可重現步驟。
8. **修正、合併**：作者處理所有 `changes-requested` 後才合併；合併後狀態改 `done`，並記錄「已驗證／未驗證」。

## 交接說明範本
```
### 交接：<任務代號>  作者：<Claude|Codex>
- 改了什麼：…
- 為什麼：…
- 驗證（附指令與結果）：python -m pytest … → N passed
- 沒驗證／已知限制：…
- 想請審查者特別看：…
```

## 審查範本（貼給審查者）
```
請審查分支 <branch> 相對 main 的 diff。
重點：1) 正確性與邊界條件（時間為 0、片段重疊、空序列、素材遺失、中文路徑）
      2) 是否違反 AGENTS.md「不可違反的規則」
      3) Windows 相容性（路徑、subprocess、編碼）
      4) 測試是否真的覆蓋行為（不是只執行程式碼）
輸出：每個問題一行「嚴重度 | 檔案:行 | 問題 | 重現步驟」，最後給 approve / changes-requested / blocked。
不要改程式碼，只回報。
```

## 在你的電腦上跑 Codex（Windows）
1. 安裝 Codex CLI 並登入（你電腦上已有 `.codex` 設定的話通常已完成）。
2. 在專案資料夾開 PowerShell：`.\scripts\windows\Codex-Review.ps1`（對目前分支相對 `main` 做審查，輸出到 `.codex-runs\`）。
3. 或手動：`codex` 進入互動模式，貼上上面的「審查範本」；Codex 會自動讀取本專案的 `AGENTS.md`。
4. 把審查結果貼回 `TASKS.md` 對應任務，交給 Claude 修正。
5. **讓 Codex 當實作者**（一張任務卡一回合，省額度）：`.\scripts\windows\Codex-Task.ps1 -List` 看任務卡（`docs/CODEX_TASKS.md`），
   `.\scripts\windows\Codex-Task.ps1 -Task CODEX-2` 讓 Codex 在 `codex/CODEX-2` 分支上做，完成後在 `TASKS.md` 留交接說明，再交給 Claude 審查合併。
6. **額度不夠時**：審查只看改動的檔案（`Codex-Review.ps1 -Paths a.py,b.py`），任務卡已列出要讀的檔案、提示裡也要求不要多讀；
   一次只跑一張卡，先跑 CODEX-1（唯讀審查）。

## 誠實記錄
- 每次交付都要分開列「已驗證」「未驗證」。
- 無頭環境無法驗證的事（Windows 視窗實際操作、硬體編碼器、真實語音辨識模型、PyInstaller 打包）一律標 `not_run` 或「未驗證」。
- 任何 AI 都不得替使用者填 `human_review`。
