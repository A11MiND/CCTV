"""Render code-native Traditional Chinese architecture diagrams for the proposal."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/diagrams"
OUTPUT.mkdir(parents=True, exist_ok=True)

FONT_REGULAR = "C:/Windows/Fonts/msjh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msjhbd.ttc"

COLORS = {
    "paper": "#F7F9FB",
    "ink": "#17242C",
    "muted": "#5D6B75",
    "line": "#91A1AC",
    "blue_fill": "#E8F1FF",
    "blue": "#2E74B5",
    "teal_fill": "#E5F7F3",
    "teal": "#138A78",
    "amber_fill": "#FFF4DE",
    "amber": "#B66B00",
    "purple_fill": "#F2EBFA",
    "purple": "#7B4FA0",
    "red_fill": "#FCEBEC",
    "red": "#B94850",
    "gray_fill": "#EDF1F4",
    "gray": "#52616B",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def text_width(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont) -> float:
    return draw.textbbox((0, 0), text, font=text_font)[2]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for character in paragraph:
            candidate = current + character
            if current and text_width(draw, candidate, text_font) > max_width:
                lines.append(current)
                current = character
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines or [""]


def draw_title(
    draw: ImageDraw.ImageDraw,
    title: str,
    subtitle: str,
    width: int,
) -> None:
    draw.text((70, 45), title, font=font(44, True), fill=COLORS["ink"])
    draw.text((70, 106), subtitle, font=font(25), fill=COLORS["muted"])
    draw.line((70, 153, width - 70, 153), fill=COLORS["blue"], width=4)


def draw_node(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    detail: str,
    tone: str = "blue",
    eyebrow: str | None = None,
) -> None:
    x1, y1, x2, y2 = box
    fill = COLORS[f"{tone}_fill"]
    stroke = COLORS[tone]
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=stroke, width=3)
    cursor_y = y1 + 22
    if eyebrow:
        draw.text((x1 + 24, cursor_y), eyebrow, font=font(19, True), fill=stroke)
        cursor_y += 31
    draw.text((x1 + 24, cursor_y), title, font=font(28, True), fill=COLORS["ink"])
    cursor_y += 45
    detail_font = font(21)
    for line in wrap_text(draw, detail, detail_font, x2 - x1 - 48)[:4]:
        draw.text((x1 + 24, cursor_y), line, font=detail_font, fill=COLORS["muted"])
        cursor_y += 31


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str | None = None,
    width: int = 5,
) -> None:
    arrow_color = color or COLORS["line"]
    draw.line((*start, *end), fill=arrow_color, width=width)
    x2, y2 = end
    x1, y1 = start
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        points = [(x2, y2), (x2 - direction * 18, y2 - 11), (x2 - direction * 18, y2 + 11)]
    else:
        direction = 1 if y2 > y1 else -1
        points = [(x2, y2), (x2 - 11, y2 - direction * 18), (x2 + 11, y2 - direction * 18)]
    draw.polygon(points, fill=arrow_color)


def add_footer(draw: ImageDraw.ImageDraw, width: int, height: int, note: str) -> None:
    draw.text((70, height - 58), note, font=font(19), fill=COLORS["muted"])
    draw.text(
        (width - 250, height - 58),
        "Author: Edcosys",
        font=font(19, True),
        fill=COLORS["blue"],
    )


def save(image: Image.Image, name: str) -> None:
    image.save(OUTPUT / name, dpi=(180, 180), optimize=True)


def render_user_workflow() -> None:
    width, height = 2200, 720
    image = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw_title(draw, "店員事件處理流程", "提示只描述可觀察行為；人工覆核保留最終判斷", width)
    boxes = [
        (80, 230, 410, 430, "AI 候選事件", "跨幀訊號達門檻\n去重＋冷卻", "blue"),
        (470, 230, 800, 430, "8–15 秒短片", "事件前後文、鏡頭、時間及原因", "teal"),
        (860, 230, 1190, 430, "店員覆核", "播放、跳到訊號、查看風險理由", "amber"),
        (1250, 185, 1570, 330, "需關注", "依門店 SOP 升級\n不自動指控", "red"),
        (1250, 370, 1570, 515, "誤報／無法判斷", "回寫資料品質\n不直接作訓練真值", "gray"),
        (1660, 230, 2090, 430, "可審計結果", "事件、模型／規則版本、操作人、備註與保留期", "purple"),
    ]
    for x1, y1, x2, y2, title, detail, tone in boxes:
        draw_node(draw, (x1, y1, x2, y2), title, detail, tone)
    draw_arrow(draw, (410, 330), (470, 330))
    draw_arrow(draw, (800, 330), (860, 330))
    draw_arrow(draw, (1190, 300), (1250, 255))
    draw_arrow(draw, (1190, 355), (1250, 440))
    draw_arrow(draw, (1570, 255), (1660, 300))
    draw_arrow(draw, (1570, 440), (1660, 360))
    add_footer(draw, width, height, "安全邊界：不得以單一提示作攔截、搜查、處罰或報警的唯一依據。")
    save(image, "01-user-workflow.png")


def render_sidecar() -> None:
    width, height = 2200, 800
    image = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw_title(draw, "旁路 AI 接入架構", "保留現有大華 NVR 的連續錄影職責；AI 系統只讀串流", width)
    draw_node(draw, (80, 270, 420, 490), "現有攝影機", "H.264／H.265\n主碼流＋子碼流", "blue")
    draw_node(draw, (600, 190, 990, 390), "現有 NVR", "16 路 PoE；原始錄影、回放與正式取證", "gray")
    draw_node(draw, (600, 450, 990, 650), "AI 邊緣主機", "子碼流分析、GPU 推理、環形緩衝、事件 metadata", "teal")
    draw_node(draw, (1190, 270, 1570, 490), "事件服務", "狀態機、去重、短片、稽核與離線佇列", "amber")
    draw_node(draw, (1770, 190, 2120, 390), "店員 Web UI", "提示、證據、人工回饋", "purple")
    draw_node(draw, (1770, 450, 2120, 650), "雲端控制面", "配置、RBAC、模型治理、監控；只需出站 TLS", "blue")
    draw_arrow(draw, (420, 335), (600, 290))
    draw_arrow(draw, (420, 425), (600, 550), COLORS["teal"])
    draw_arrow(draw, (990, 550), (1190, 410), COLORS["teal"])
    draw_arrow(draw, (1570, 335), (1770, 290))
    draw_arrow(draw, (1570, 425), (1770, 550))
    draw.text((458, 245), "主碼流／ONVIF", font=font(20, True), fill=COLORS["gray"])
    draw.text((445, 520), "RTSP 子碼流 2–5 FPS", font=font(20, True), fill=COLORS["teal"])
    add_footer(draw, width, height, "現場必驗：NVR PoE 私有網段、唯讀帳戶、串流 URL、韌體、並發拉流與時間同步。")
    save(image, "02-sidecar-architecture.png")


def render_model_network() -> None:
    width, height = 2400, 1180
    image = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw_title(draw, "YOLO26＋時序行為網絡", "YOLO26 是感知前端；行為事件由追蹤、時間分支、規則與融合器完成", width)
    row1 = [
        (70, 210, 360, 390, "影像輸入", "1080p／ROI\n硬件解碼／抽幀", "gray"),
        (430, 210, 750, 390, "Backbone", "Conv + C3k2\nP3／P4／P5", "blue"),
        (820, 210, 1140, 390, "Neck", "SPPF + C2PSA\nFPN／PAN", "blue"),
        (1210, 185, 1550, 415, "雙 Head", "One-to-One：免 NMS\nOne-to-Many：傳統部署\n另接 Pose Head", "teal"),
        (1620, 210, 1940, 390, "多目標追蹤", "ByteTrack／BoT-SORT\nTrack ID＋軌跡", "teal"),
        (2010, 210, 2330, 390, "每人特徵", "手－貨架／袋距離\n停留、取放、遮擋", "purple"),
    ]
    for item in row1:
        draw_node(draw, item[:4], item[4], item[5], item[6])
    for left, right in zip(row1, row1[1:]):
        draw_arrow(draw, (left[2], 300), (right[0], 300))
    row2 = [
        (310, 590, 720, 805, "時間分支 A", "TCN／輕量 Transformer\n學習 2–8 秒行為序列", "purple"),
        (820, 590, 1230, 805, "異常分支 B", "只學正常行為分布\n輸出場景相對異常分數", "purple"),
        (1330, 590, 1740, 805, "可解釋規則 C", "ROI、持續時間、取放順序、出口／POS 訊號", "amber"),
    ]
    for item in row2:
        draw_node(draw, item[:4], item[4], item[5], item[6])
    draw_arrow(draw, (2170, 390), (515, 590), COLORS["purple"])
    draw_arrow(draw, (2170, 390), (1025, 590), COLORS["purple"])
    draw_arrow(draw, (2170, 390), (1535, 590), COLORS["amber"])
    draw_node(draw, (780, 900, 1250, 1080), "校準融合器", "分鏡頭門檻＋hysteresis\n輸出原因碼及風險分數", "teal")
    draw_node(draw, (1400, 900, 1880, 1080), "事件狀態機", "去重／冷卻 → 前後短片 → 人工覆核", "red")
    draw_arrow(draw, (515, 805), (900, 900), COLORS["purple"])
    draw_arrow(draw, (1025, 805), (1015, 900), COLORS["purple"])
    draw_arrow(draw, (1535, 805), (1130, 900), COLORS["amber"])
    draw_arrow(draw, (1250, 990), (1400, 990), COLORS["teal"])
    add_footer(draw, width, height, "首輪建議：YOLO26n/s detect／pose 比較；FP16 起步；INT8 只在本店校準集證明可接受後啟用。")
    save(image, "03-model-network.png")


def render_system() -> None:
    width, height = 2400, 1120
    image = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw_title(draw, "端－邊－雲系統架構", "門店斷網仍可錄影與產生本地事件；雲端只作控制面與治理", width)
    columns = [
        (70, 220, 650, 980, "門店視頻網", "blue"),
        (720, 220, 1660, 980, "AI 邊緣主機", "teal"),
        (1730, 220, 2330, 980, "雲端控制面", "purple"),
    ]
    for x1, y1, x2, y2, label, tone in columns:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=22, fill=COLORS[f"{tone}_fill"], outline=COLORS[tone], width=3)
        draw.text((x1 + 24, y1 + 22), label, font=font(30, True), fill=COLORS[tone])
    left_nodes = [
        ((110, 330, 610, 500), "大華攝影機", "主／子碼流；NTP；不直接對公網"),
        ((110, 620, 610, 790), "NVR4216-16P-4KS3", "全天錄影、回放、原始證據；AI 故障不受影響"),
    ]
    edge_nodes = [
        ((770, 320, 1160, 500), "Ingest／Decode", "RTSP 重連、硬件解碼、抽幀、motion gate"),
        ((1220, 320, 1610, 500), "IVA Pipeline", "YOLO26、tracker、pose／互動、時間模型"),
        ((770, 600, 1160, 780), "事件與緩衝", "環形 buffer、SQLite、clip、reason codes"),
        ((1220, 600, 1610, 780), "Edge Agent", "配置版本、健康、離線佇列、回滾"),
    ]
    cloud_nodes = [
        ((1780, 320, 2280, 490), "API／RBAC／配置", "出站 TLS；多租戶；最小權限；審計"),
        ((1780, 560, 2280, 730), "事件 Metadata／物件儲存", "短片保留策略；原始全天錄影留在 NVR"),
        ((1780, 800, 2280, 940), "MLOps／Observability", "簽署模型、灰度、監控、drift、回滾"),
    ]
    for box, title, detail in left_nodes:
        draw_node(draw, box, title, detail, "blue")
    for box, title, detail in edge_nodes:
        draw_node(draw, box, title, detail, "teal")
    for box, title, detail in cloud_nodes:
        draw_node(draw, box, title, detail, "purple")
    draw_arrow(draw, (610, 415), (770, 410))
    draw_arrow(draw, (610, 705), (770, 690), COLORS["teal"])
    draw_arrow(draw, (1160, 410), (1220, 410), COLORS["teal"])
    draw_arrow(draw, (1415, 500), (965, 600), COLORS["teal"])
    draw_arrow(draw, (1160, 690), (1220, 690), COLORS["teal"])
    draw_arrow(draw, (1610, 680), (1780, 630), COLORS["purple"])
    draw_arrow(draw, (1610, 410), (1780, 405), COLORS["purple"])
    draw_arrow(draw, (2030, 730), (2030, 800), COLORS["purple"])
    add_footer(draw, width, height, "關鍵 SLA：p95 告警延遲、clip 完整率、每鏡頭每小時誤報、7 日穩定運行與自動重連。")
    save(image, "04-system-architecture.png")


def render_data_loop() -> None:
    width, height = 2300, 860
    image = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw_title(draw, "資料與模型閉環", "公開資料只建立底座；店內 hard negatives、演練與人工回饋才決定可用性", width)
    nodes = [
        (70, 270, 360, 470, "合法收集", "連續正常時段\n受控演練／已確認事件", "gray"),
        (420, 270, 710, 470, "最小化", "目的限制、保留期\n必要時遮罩", "blue"),
        (770, 270, 1060, 470, "雙人標註", "可觀察行為標籤\n衝突裁決", "amber"),
        (1120, 270, 1410, 470, "版本／切分", "dataset card\n按日期／人物／鏡頭隔離", "purple"),
        (1470, 270, 1760, 470, "訓練／校準", "偵測、時序、規則\n事件級 KPI", "teal"),
        (1820, 270, 2110, 470, "影子部署", "先觀察不通知\n錯誤分桶／drift", "red"),
    ]
    for item in nodes:
        draw_node(draw, item[:4], item[4], item[5], item[6])
    for left, right in zip(nodes, nodes[1:]):
        draw_arrow(draw, (left[2], 370), (right[0], 370))
    draw_node(draw, (650, 600, 1080, 760), "店員回饋", "需關注／誤報／無法判斷\n保留模型、規則及操作版本", "amber")
    draw_node(draw, (1260, 600, 1690, 760), "質量閘門", "去偏差、重標、凍結驗收集\n審批後才進下一版本", "blue")
    draw_arrow(draw, (1965, 470), (1475, 600), COLORS["red"])
    draw_arrow(draw, (1260, 680), (1080, 680), COLORS["blue"])
    draw_arrow(draw, (650, 680), (215, 470), COLORS["amber"])
    add_footer(draw, width, height, "不得逐幀隨機切分；同一事件相鄰幀不可跨 train／validation／test。")
    save(image, "05-data-loop.png")


def main() -> None:
    render_user_workflow()
    render_sidecar()
    render_model_network()
    render_system()
    render_data_loop()
    for path in sorted(OUTPUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
