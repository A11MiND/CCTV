export const primaryCamera = {
  id: "aisle",
  label: "貨架通道 03",
  shortLabel: "貨架通道 03 · CM532",
  model: "IPC-HDBW2249E-S-IL",
  code: "CM532",
  stream: "1080p · 5 FPS 分析",
  bitrate: "1.8 Mbps",
  status: "online",
  image: "/assets/video/yolo26-preview.jpg",
  video: "/assets/video/yolo26-retail-demo.mp4",
  performanceLabel: "RTX 4060 實跑",
  source:
    "https://www.pexels.com/video/customers-shopping-at-supermarket-10901926/",
};

const trainingCamera = ({
  id,
  label,
  file,
  datasetLabel,
  actorGroup,
}) => ({
  id,
  label,
  shortLabel: datasetLabel,
  model: "Shoplifting Dataset 2022",
  code: actorGroup,
  stream: "640×480 · 15 FPS preview",
  bitrate: "Dataset clip",
  status: "online",
  image: `/assets/cctv/training/${file}.jpg`,
  video: `/assets/cctv/training/${file}.mp4`,
  datasetLabel,
  source:
    "https://data.mendeley.com/datasets/r3yjf35hzr/1",
});

export const datasetCameraSets = [
  {
    id: "mixed",
    shortLabel: "A · Mixed",
    label: "Mixed training examples",
    cameras: [
      trainingCamera({
        id: "entrance",
        label: "Training Shoplifting 08",
        file: "set-a-shoplifting-8",
        datasetLabel: "TRAIN · SHOPLIFTING 08",
        actorGroup: "A_checkered",
      }),
      trainingCamera({
        id: "checkout",
        label: "Training Normal 06",
        file: "set-a-normal-6",
        datasetLabel: "TRAIN · NORMAL 06",
        actorGroup: "C_purple",
      }),
      trainingCamera({
        id: "drinks",
        label: "Training Shoplifting 01",
        file: "set-a-shoplifting-1",
        datasetLabel: "TRAIN · SHOPLIFTING 01",
        actorGroup: "D_navy_stripe",
      }),
    ],
  },
  {
    id: "positive",
    shortLabel: "B · Positive",
    label: "Shoplifting-positive training examples",
    cameras: [
      trainingCamera({
        id: "entrance",
        label: "Training Shoplifting 10",
        file: "set-b-shoplifting-10",
        datasetLabel: "TRAIN · SHOPLIFTING 10",
        actorGroup: "B_light_blue",
      }),
      trainingCamera({
        id: "checkout",
        label: "Training Shoplifting 14",
        file: "set-b-shoplifting-14",
        datasetLabel: "TRAIN · SHOPLIFTING 14",
        actorGroup: "C_purple",
      }),
      trainingCamera({
        id: "drinks",
        label: "Training Shoplifting 16",
        file: "set-b-shoplifting-16",
        datasetLabel: "TRAIN · SHOPLIFTING 16",
        actorGroup: "D_navy_stripe",
      }),
    ],
  },
  {
    id: "normal",
    shortLabel: "C · Normal",
    label: "Normal-control training examples",
    cameras: [
      trainingCamera({
        id: "entrance",
        label: "Training Normal 02",
        file: "set-c-normal-2",
        datasetLabel: "TRAIN · NORMAL 02",
        actorGroup: "B_light_blue",
      }),
      trainingCamera({
        id: "checkout",
        label: "Training Normal 07",
        file: "set-c-normal-7",
        datasetLabel: "TRAIN · NORMAL 07",
        actorGroup: "D_navy_stripe",
      }),
      trainingCamera({
        id: "drinks",
        label: "Training Normal 11",
        file: "set-c-normal-11",
        datasetLabel: "TRAIN · NORMAL 11",
        actorGroup: "E_older_plaid",
      }),
    ],
  },
];

export const cameras = [primaryCamera, ...datasetCameraSets[0].cameras];

export const events = [
  {
    id: "EV-20260726-0142",
    type: "疑似藏匿動作",
    score: 92,
    time: "14:32:18",
    cameraId: "aisle",
    severity: "high",
    evidence: "取貨 → 停留 → 手部靠近衣袋 → 商品疑似消失",
    clipLength: 12,
    status: "new",
  },
  {
    id: "EV-20260726-0138",
    type: "快速掃貨動作",
    score: 78,
    time: "14:28:44",
    cameraId: "aisle",
    severity: "medium",
    evidence: "短時間連續取貨 → 放入袋中 → 離開貨架",
    clipLength: 10,
    status: "new",
  },
  {
    id: "EV-20260726-0129",
    type: "區域長時間徘徊",
    score: 61,
    time: "14:22:07",
    cameraId: "drinks",
    severity: "low",
    evidence: "同一區域停留超過 90 秒",
    clipLength: 8,
    status: "reviewed",
  },
];

export const evidenceSteps = [
  { id: "take", time: "14:32:06", label: "取貨", icon: "basket" },
  { id: "pause", time: "14:32:10", label: "停留", icon: "clock" },
  { id: "overlap", time: "14:32:14", label: "手部靠近衣袋", icon: "hand" },
  {
    id: "missing",
    time: "14:32:18",
    label: "商品疑似消失",
    icon: "focus",
  },
];

export const datasets = [
  {
    name: "RetailAction",
    scope: "21,000 組雙視角片段、41 小時",
    role: "零售人與商品互動、時空定位",
    fit: "高",
    caveat: "不等同真實盜竊；須核對取得方式與使用條款",
  },
  {
    name: "PoseLift",
    scope: "真實零售店、匿名人體姿態",
    role: "私隱友善的異常行為基準",
    fit: "中高",
    caveat: "只保留姿態，商品與場景訊號較弱",
  },
  {
    name: "UCF-Crime",
    scope: "1,900 段、128 小時、13 類異常",
    role: "弱監督異常／Shoplifting 研究基準",
    fit: "中",
    caveat: "來源授權不清、影片級弱標註、場景偏差大",
  },
  {
    name: "店內自有數據",
    scope: "建議 10–20 個攝影機日負樣本 + 演練正例",
    role: "最終訓練、閾值校準及驗收",
    fit: "必要",
    caveat: "需同意、去識別化、分日期／鏡頭隔離測試",
  },
];

export const navItems = [
  { id: "live", label: "即時態勢", icon: "monitor" },
  { id: "review", label: "事件研判", icon: "clipboard" },
  { id: "cameras", label: "攝影機", icon: "camera" },
  { id: "data", label: "數據閉環", icon: "refresh" },
  { id: "architecture", label: "架構說明", icon: "book" },
];
