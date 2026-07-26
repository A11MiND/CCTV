# Edcosys 零售 CCTV 疑似盜竊風險偵測 Prototype

Author: Edcosys

這個工作區包含一份繁體中文 Proposal、兩個可操作 UI 流程、以真實超市影片在本機 NVIDIA RTX 4060 Laptop GPU 執行的 YOLO26s＋ByteTrack 感知 Demo，以及完整的來源、指標與重跑腳本。

## 主要交付物

- `Edcosys_智能零售防損系統_Proposal.docx`：可編輯正式 Proposal。
- `output/pdf/Edcosys_智能零售防損系統_Proposal.pdf`：已逐頁驗收的 PDF。
- `src/`：React／Vite 互動 Prototype。
- `public/assets/video/yolo26-retail-demo.mp4`：真實超市影片的 YOLO26 人員偵測與 ByteTrack 追蹤結果。
- `assets/video/yolo26-run-metrics.json`：本機 GPU 實測設定與指標。
- `scripts/run_yolo_retail_demo.py`：可重現的本機推論腳本。
- `docs/proposal-content.md`：Proposal 的繁體中文可追蹤內容。
- `docs/frontend-qa-results.json`：前端自動化驗收結果。
- `docs/fidelity-ledger.md`：真實、推論及模擬內容的邊界。

## 啟動互動 Prototype

```powershell
pnpm install
pnpm run dev
```

瀏覽 `http://127.0.0.1:5173`。若要檢查正式建置：

```powershell
pnpm run build
pnpm run preview -- --port 4173
```

## 重跑 YOLO26 真實影片 Demo

既有 `.venv-yolo` 已包含本機執行時所需套件：

```powershell
.\.venv-yolo\Scripts\python.exe .\scripts\run_yolo_retail_demo.py
```

預設輸入為 `assets/video/pexels-hong-kong-supermarket.mp4`，輸出為 `public/assets/video/yolo26-retail-demo.mp4`。

素材來自 Pexels 的 Suika Chan「Customers Shopping at Supermarket」實拍影片，來源與授權：

- https://www.pexels.com/video/customers-shopping-at-supermarket-10901926/
- https://www.pexels.com/license/

## 正確解讀

這次本機實測證明的是「真實影片可在本機 GPU 完成人員偵測、追蹤及 UI 播放」，不是 shoplifting accuracy test。原片沒有盜竊 ground truth，也不是固定高位 CCTV；介面中的 92%、事件編號、取物／停留／靠近衣袋／未放回等證據鏈均清楚標示為產品流程模擬。系統不應由影像推斷身份、犯罪意圖或是否已犯罪，正式試點必須採 human-in-the-loop。

## 建議下一步

先以 1 店、2–4 個高損耗區鏡頭進行 10–14 週分段試點；驗證 RTSP／ONVIF、像素與遮擋條件，收集連續正常時段、受控演練與合法取得的已確認事件，最後以事件級召回、每鏡頭每小時誤報、P95 提示延遲及人工覆核時間決定 Go／Conditional Go／No-Go。
