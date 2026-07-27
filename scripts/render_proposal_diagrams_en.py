"""Render high-resolution English diagrams for the Edcosys CCTV proposal."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "diagrams" / "en"
OUTPUT.mkdir(parents=True, exist_ok=True)

FONT_REGULAR = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/seguisb.ttf"

COLORS = {
    "paper": "#F7F9FC",
    "white": "#FFFFFF",
    "ink": "#17242C",
    "muted": "#5C6B76",
    "line": "#94A3AE",
    "blue_fill": "#E8F1FF",
    "blue": "#2E74B5",
    "teal_fill": "#E4F7F3",
    "teal": "#138A78",
    "amber_fill": "#FFF4DD",
    "amber": "#B66B00",
    "purple_fill": "#F1EBFA",
    "purple": "#7550A2",
    "red_fill": "#FCEBED",
    "red": "#B94850",
    "gray_fill": "#EDF1F4",
    "gray": "#52616B",
    "green_fill": "#EAF6E9",
    "green": "#477D45",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.FreeTypeFont,
) -> float:
    box = draw.textbbox((0, 0), text, font=text_font)
    return box[2] - box[0]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if text_width(draw, candidate, text_font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines or [""]


def draw_title(
    draw: ImageDraw.ImageDraw,
    title: str,
    subtitle: str,
    width: int,
) -> None:
    draw.text((82, 52), title, font=font(49, True), fill=COLORS["ink"])
    draw.text((82, 122), subtitle, font=font(26), fill=COLORS["muted"])
    draw.line((82, 174, width - 82, 174), fill=COLORS["blue"], width=5)


def draw_node(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    detail: str,
    tone: str = "blue",
    eyebrow: str | None = None,
    max_detail_lines: int = 4,
    title_size: int = 29,
    detail_size: int = 22,
) -> None:
    x1, y1, x2, y2 = box
    fill = COLORS[f"{tone}_fill"]
    stroke = COLORS[tone]
    draw.rounded_rectangle(
        box,
        radius=20,
        fill=fill,
        outline=stroke,
        width=3,
    )
    cursor_y = y1 + 22
    if eyebrow:
        draw.text(
            (x1 + 25, cursor_y),
            eyebrow.upper(),
            font=font(18, True),
            fill=stroke,
        )
        cursor_y += 31
    title_font = font(title_size, True)
    for line in wrap_text(draw, title, title_font, x2 - x1 - 50)[:2]:
        draw.text((x1 + 25, cursor_y), line, font=title_font, fill=COLORS["ink"])
        cursor_y += title_size + 8
    cursor_y += 6
    detail_font = font(detail_size)
    for line in wrap_text(draw, detail, detail_font, x2 - x1 - 50)[:max_detail_lines]:
        draw.text((x1 + 25, cursor_y), line, font=detail_font, fill=COLORS["muted"])
        cursor_y += detail_size + 10


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
        points = [
            (x2, y2),
            (x2 - direction * 20, y2 - 12),
            (x2 - direction * 20, y2 + 12),
        ]
    else:
        direction = 1 if y2 > y1 else -1
        points = [
            (x2, y2),
            (x2 - 12, y2 - direction * 20),
            (x2 + 12, y2 - direction * 20),
        ]
    draw.polygon(points, fill=arrow_color)


def draw_elbow_arrow(
    draw: ImageDraw.ImageDraw,
    points: Iterable[tuple[int, int]],
    color: str | None = None,
    width: int = 5,
) -> None:
    point_list = list(points)
    arrow_color = color or COLORS["line"]
    draw.line(point_list, fill=arrow_color, width=width, joint="curve")
    draw_arrow(
        draw,
        point_list[-2],
        point_list[-1],
        color=arrow_color,
        width=width,
    )


def draw_arrow_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    label: str,
    tone: str = "gray",
) -> None:
    x, y = position
    text_font = font(19, True)
    width = int(text_width(draw, label, text_font)) + 28
    draw.rounded_rectangle(
        (x, y, x + width, y + 36),
        radius=10,
        fill=COLORS["white"],
        outline=COLORS["line"],
        width=1,
    )
    draw.text((x + 14, y + 5), label, font=text_font, fill=COLORS[tone])


def draw_group(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    tone: str,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(
        box,
        radius=26,
        fill=COLORS["white"],
        outline=COLORS[tone],
        width=3,
    )
    draw.rounded_rectangle(
        (x1 + 20, y1 + 18, x1 + 20 + 255, y1 + 62),
        radius=12,
        fill=COLORS[f"{tone}_fill"],
    )
    draw.text(
        (x1 + 36, y1 + 24),
        label.upper(),
        font=font(20, True),
        fill=COLORS[tone],
    )


def add_footer(draw: ImageDraw.ImageDraw, width: int, height: int, note: str) -> None:
    draw.text((82, height - 60), note, font=font(20), fill=COLORS["muted"])
    label = "Author: Edcosys"
    label_width = text_width(draw, label, font(20, True))
    draw.text(
        (width - 82 - label_width, height - 60),
        label,
        font=font(20, True),
        fill=COLORS["blue"],
    )


def save(image: Image.Image, name: str) -> None:
    image.save(OUTPUT / name, dpi=(180, 180), optimize=True)


def render_user_workflow() -> None:
    width, height = 2500, 1030
    image = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw_title(
        draw,
        "Store Associate Event Workflow",
        "A short evidence package moves each signal from detection to a recorded human decision.",
        width,
    )

    top_y1, top_y2 = 260, 500
    nodes = [
        ((80, top_y1, 430, top_y2), "Candidate event", "Detection, tracking and scene rules create a scored signal.", "blue", "1"),
        ((490, top_y1, 840, top_y2), "Evidence clip", "8–15 seconds before and after the event, camera and reason code.", "teal", "2"),
        ((900, top_y1, 1250, top_y2), "Triage queue", "Rank by risk, freshness, zone and confidence.", "amber", "3"),
        ((1310, top_y1, 1660, top_y2), "Human review", "Play, scrub, jump cameras and inspect the event rationale.", "purple", "4"),
    ]
    for box, title, detail, tone, eyebrow in nodes:
        draw_node(draw, box, title, detail, tone, f"STEP {eyebrow}")
    for left, right in zip(nodes, nodes[1:]):
        draw_arrow(draw, (left[0][2], 380), (right[0][0], 380))

    draw_node(
        draw,
        (1770, 220, 2150, 420),
        "Escalate",
        "Apply store SOP and assign follow-up.",
        "red",
        "REVIEW OUTCOME",
    )
    draw_node(
        draw,
        (1770, 470, 2150, 670),
        "Dismiss / relabel",
        "Record the decision and correction reason.",
        "gray",
        "REVIEW OUTCOME",
    )
    draw_arrow(draw, (1660, 350), (1770, 320), COLORS["red"])
    draw_arrow(draw, (1660, 420), (1770, 570), COLORS["gray"])

    draw_node(
        draw,
        (2210, 310, 2420, 590),
        "Audit record",
        "Event, model, rule, user, timestamp and retention state.",
        "purple",
        "SYSTEM",
        title_size=27,
        detail_size=20,
        max_detail_lines=5,
    )
    draw_arrow(draw, (2150, 320), (2210, 390), COLORS["red"])
    draw_arrow(draw, (2150, 570), (2210, 520), COLORS["gray"])

    draw_node(
        draw,
        (670, 720, 1160, 900),
        "Store feedback",
        "False alarm reason, missed event and workflow notes.",
        "amber",
        "LEARNING INPUT",
    )
    draw_node(
        draw,
        (1335, 720, 1825, 900),
        "Threshold and model update",
        "Calibrated release through versioned approval.",
        "green",
        "CONTROLLED CHANGE",
    )
    draw_arrow(draw, (1550, 670), (950, 720), COLORS["amber"])
    draw_arrow(draw, (1160, 810), (1335, 810), COLORS["green"])

    add_footer(
        draw,
        width,
        height,
        "Each signal remains reviewable, explainable and attributable to a model and ruleset version.",
    )
    save(image, "01-user-workflow-en.png")


def render_sidecar() -> None:
    width, height = 2500, 1110
    image = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw_title(
        draw,
        "Sidecar Integration with Existing CCTV",
        "The NVR keeps recording. The edge host reads camera substreams and emits reviewable events.",
        width,
    )

    draw_group(draw, (70, 230, 865, 980), "Existing store video", "blue")
    draw_group(draw, (930, 230, 1785, 980), "Edcosys edge sidecar", "teal")
    draw_group(draw, (1850, 230, 2430, 980), "Operations", "purple")

    draw_node(
        draw,
        (120, 340, 485, 560),
        "IP cameras",
        "H.264 / H.265\nMain stream + substream",
        "blue",
    )
    draw_node(
        draw,
        (520, 340, 815, 560),
        "Existing NVR",
        "Full-time recording, playback and evidence retention.",
        "gray",
        detail_size=20,
    )
    draw_node(
        draw,
        (285, 690, 650, 875),
        "Store video LAN",
        "Private PoE/VLAN segment with NTP time sync.",
        "blue",
    )

    draw_node(
        draw,
        (980, 335, 1325, 565),
        "Stream ingest",
        "RTSP reconnect, hardware decode, frame sampling and ROI.",
        "teal",
    )
    draw_node(
        draw,
        (1380, 335, 1735, 565),
        "Vision pipeline",
        "YOLO26, tracking, temporal features and event scoring.",
        "teal",
    )
    draw_node(
        draw,
        (980, 680, 1325, 875),
        "Local buffer",
        "Ring buffer, encrypted clips and retry queue.",
        "gray",
    )
    draw_node(
        draw,
        (1380, 680, 1735, 875),
        "Event service",
        "State machine, de-duplication, clip assembly and metadata.",
        "amber",
        detail_size=20,
    )

    draw_node(
        draw,
        (1900, 335, 2380, 565),
        "Review web app",
        "Live view, event queue, evidence playback and reviewer feedback.",
        "purple",
    )
    draw_node(
        draw,
        (1900, 680, 2380, 875),
        "Cloud control plane",
        "Configuration, RBAC, fleet health, model rollout and monitoring.",
        "blue",
    )

    draw_arrow(draw, (485, 420), (520, 420), COLORS["blue"])
    draw_arrow(draw, (300, 560), (400, 690), COLORS["blue"])
    draw_arrow(draw, (650, 785), (980, 450), COLORS["teal"])
    draw_arrow(draw, (1325, 450), (1380, 450), COLORS["teal"])
    draw_arrow(draw, (1555, 565), (1555, 680), COLORS["amber"])
    draw_arrow(draw, (1380, 780), (1325, 780), COLORS["gray"])
    draw_arrow(draw, (1735, 450), (1900, 450), COLORS["purple"])
    draw_arrow(draw, (1735, 780), (1900, 780), COLORS["purple"])
    draw_arrow_label(draw, (570, 372), "MAIN STREAM", "blue")
    draw_arrow_label(draw, (710, 605), "RTSP SUBSTREAM", "teal")
    draw_arrow_label(draw, (1747, 388), "EVENT + CLIP", "purple")
    draw_arrow_label(draw, (1742, 724), "OUTBOUND TLS", "purple")

    add_footer(
        draw,
        width,
        height,
        "Pilot checks: stream URLs, codec profiles, time sync, substream capacity and sustained reconnect behavior.",
    )
    save(image, "02-sidecar-architecture-en.png")


def render_model_network() -> None:
    width, height = 2700, 1560
    image = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw_title(
        draw,
        "YOLO26 Perception and Event Network",
        "Per-frame perception feeds identity tracking, temporal evidence and a calibrated event state machine.",
        width,
    )

    top_nodes = [
        ((65, 245, 390, 470), "Video input", "Substream, ROI, decode and adaptive frame sampling.", "gray", "INGEST"),
        ((450, 245, 800, 470), "YOLO26 backbone", "Multi-scale visual features from each sampled frame.", "blue", "PERCEPTION"),
        ((860, 245, 1210, 470), "Feature neck", "Cross-scale feature aggregation for small and occluded people.", "blue", "PERCEPTION"),
        ((1270, 220, 1660, 495), "Detection / pose head", "Person boxes, confidence, keypoints and task-specific classes.", "teal", "PERCEPTION"),
        ((1720, 245, 2070, 470), "Multi-object tracking", "ByteTrack or BoT-SORT maintains identity and trajectory.", "teal", "TRACKING"),
        ((2130, 245, 2555, 470), "Track-level features", "Zone, dwell, hand–shelf distance, motion and object interaction.", "purple", "FEATURES"),
    ]
    for box, title, detail, tone, eyebrow in top_nodes:
        draw_node(draw, box, title, detail, tone, eyebrow)
    for left, right in zip(top_nodes, top_nodes[1:]):
        draw_arrow(draw, (left[0][2], 355), (right[0][0], 355))

    draw.text(
        (120, 590),
        "EVENT EVIDENCE BRANCHES",
        font=font(22, True),
        fill=COLORS["muted"],
    )
    branch_nodes = [
        ((130, 650, 720, 895), "Temporal action model", "A TCN or lightweight transformer scores short sequences such as take, conceal, replace and exit.", "purple", "A"),
        ((795, 650, 1385, 895), "Scene-relative anomaly", "Learns the normal distribution for each camera and produces a relative anomaly score.", "purple", "B"),
        ((1460, 650, 2050, 895), "Explainable scene rules", "ROI transitions, dwell, event order, exit direction and camera-specific thresholds.", "amber", "C"),
        ((2125, 650, 2590, 895), "Store context", "POS state, staffing schedule, shelf zone and optional EAS signals.", "gray", "D"),
    ]
    for box, title, detail, tone, eyebrow in branch_nodes:
        draw_node(draw, box, title, detail, tone, f"BRANCH {eyebrow}")

    hub_x, hub_y = 2340, 520
    for box, *_ in branch_nodes:
        center_x = (box[0] + box[2]) // 2
        draw_elbow_arrow(
            draw,
            [(hub_x, hub_y), (hub_x, 590), (center_x, 590), (center_x, box[1])],
            COLORS["purple"] if center_x < 1450 else COLORS["amber"],
            width=4,
        )

    draw_node(
        draw,
        (455, 1060, 1115, 1300),
        "Calibrated fusion",
        "Combines branch scores with camera-specific thresholds, hysteresis and reason codes.",
        "teal",
        "DECISION LAYER",
    )
    draw_node(
        draw,
        (1270, 1060, 1930, 1300),
        "Event state machine",
        "Candidate → confirmed → cool-down → clip assembly → review queue.",
        "red",
        "WORKFLOW LAYER",
    )
    draw_node(
        draw,
        (2085, 1060, 2590, 1300),
        "Human review",
        "Reviewer action becomes labeled operational feedback.",
        "purple",
        "CONTROL POINT",
    )

    branch_centers = [425, 1090, 1755, 2357]
    for center_x in branch_centers:
        draw_elbow_arrow(
            draw,
            [(center_x, 895), (center_x, 980), (785, 980), (785, 1060)],
            COLORS["amber"] if center_x > 1400 else COLORS["purple"],
            width=4,
        )
    draw_arrow(draw, (1115, 1180), (1270, 1180), COLORS["teal"])
    draw_arrow(draw, (1930, 1180), (2085, 1180), COLORS["red"])
    draw_elbow_arrow(
        draw,
        [(2338, 1300), (2338, 1405), (785, 1405), (785, 1300)],
        COLORS["purple"],
        width=4,
    )
    draw_arrow_label(draw, (1260, 1370), "FEEDBACK FOR CALIBRATION", "purple")

    add_footer(
        draw,
        width,
        height,
        "Pilot baseline: YOLO26n/s detect, optional pose comparison, FP16 deployment and store-specific calibration.",
    )
    save(image, "03-model-network-en.png")


def render_system() -> None:
    width, height = 2700, 1530
    image = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw_title(
        draw,
        "Edge–Cloud System Architecture",
        "The store produces events locally; the cloud manages configuration, fleet health and model releases.",
        width,
    )

    draw_group(draw, (70, 235, 760, 1380), "Store video network", "blue")
    draw_group(draw, (825, 235, 1895, 1380), "Edge AI host", "teal")
    draw_group(draw, (1960, 235, 2630, 1380), "Cloud control plane", "purple")

    store_nodes = [
        ((120, 345, 710, 565), "IP cameras", "Main stream to NVR; substream to the edge host. Time synchronized.", "blue"),
        ((120, 690, 710, 910), "Existing NVR", "Continuous recording, evidence playback and store retention.", "gray"),
        ((120, 1035, 710, 1255), "Store review station", "Browser access to events, clips and review decisions.", "purple"),
    ]
    edge_nodes = [
        ((875, 335, 1340, 565), "Ingest and decode", "RTSP session manager, hardware decode, ROI, sampling and motion gate.", "teal"),
        ((1380, 335, 1845, 565), "Vision pipeline", "YOLO26, tracker, pose option, temporal model and scene rules.", "teal"),
        ((875, 685, 1340, 915), "Event service", "Score fusion, state machine, de-duplication, reason codes and notifications.", "amber"),
        ((1380, 685, 1845, 915), "Local evidence store", "Ring buffer, encrypted clips, SQLite metadata and retention worker.", "gray"),
        ((875, 1035, 1340, 1265), "Edge agent", "Versioned configuration, health checks, offline queue and rollback.", "blue"),
        ((1380, 1035, 1845, 1265), "Observability", "GPU, stream and event metrics; structured logs and drift counters.", "green"),
    ]
    cloud_nodes = [
        ((2010, 335, 2580, 565), "API gateway and RBAC", "Outbound TLS, store identity, least privilege and audit logging.", "purple"),
        ((2010, 685, 2580, 915), "Operations portal", "Fleet health, configuration, event metadata and support diagnostics.", "blue"),
        ((2010, 1035, 2580, 1265), "Model registry and rollout", "Signed releases, staged deployment, quality gates and rollback.", "purple"),
    ]
    for box, title, detail, tone in store_nodes + edge_nodes + cloud_nodes:
        draw_node(draw, box, title, detail, tone)

    draw_arrow(draw, (710, 430), (875, 430), COLORS["teal"])
    draw_arrow_label(draw, (715, 378), "RTSP SUBSTREAM", "teal")
    draw_arrow(draw, (415, 565), (415, 690), COLORS["blue"])
    draw_arrow_label(draw, (425, 606), "MAIN STREAM", "blue")
    draw_arrow(draw, (1340, 430), (1380, 430), COLORS["teal"])
    draw_arrow(draw, (1612, 565), (1108, 685), COLORS["amber"])
    draw_arrow(draw, (1340, 800), (1380, 800), COLORS["gray"])
    draw_elbow_arrow(
        draw,
        [(1380, 800), (1305, 800), (1305, 1145), (1340, 1145)],
        COLORS["green"],
        width=4,
    )
    draw_arrow(draw, (875, 1145), (710, 1145), COLORS["purple"])
    draw_arrow_label(draw, (714, 1090), "EVENT UI", "purple")

    draw_arrow(draw, (1845, 800), (2010, 450), COLORS["purple"])
    draw_arrow(draw, (1845, 1145), (2010, 800), COLORS["purple"])
    draw_elbow_arrow(
        draw,
        [(2010, 1145), (1920, 1325), (1107, 1325), (1107, 1265)],
        COLORS["blue"],
        width=4,
    )
    draw_arrow_label(draw, (1875, 540), "EVENT METADATA", "purple")
    draw_arrow_label(draw, (1460, 1285), "SIGNED RELEASE", "blue")

    add_footer(
        draw,
        width,
        height,
        "Key SLOs: p95 event latency, clip completion, camera uptime, queue age and automatic stream recovery.",
    )
    save(image, "04-system-architecture-en.png")


def render_data_loop() -> None:
    width, height = 2600, 1320
    image = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw_title(
        draw,
        "Data and Model Improvement Loop",
        "Public data starts the model. Local scenes, hard negatives and reviewed events make it operational.",
        width,
    )

    top_nodes = [
        ((75, 265, 435, 490), "Collect", "Normal footage, staged scenarios and adjudicated store events.", "gray", "1"),
        ((500, 265, 860, 490), "Minimize", "Purpose limit, retention policy, access control and privacy masking.", "blue", "2"),
        ((925, 265, 1285, 490), "Label", "Observable actions, track continuity, occlusion and scene context.", "amber", "3"),
        ((1350, 265, 1710, 490), "Adjudicate", "Double review, conflict resolution and label quality metrics.", "amber", "4"),
        ((1775, 265, 2135, 490), "Version and split", "Dataset card; store, person and date-separated test sets.", "purple", "5"),
        ((2200, 265, 2525, 490), "Train and calibrate", "Perception, temporal model, thresholds and uncertainty.", "teal", "6"),
    ]
    for box, title, detail, tone, eyebrow in top_nodes:
        draw_node(draw, box, title, detail, tone, f"STEP {eyebrow}", detail_size=20)
    for left, right in zip(top_nodes, top_nodes[1:]):
        draw_arrow(draw, (left[0][2], 375), (right[0][0], 375))

    draw_node(
        draw,
        (1725, 650, 2225, 900),
        "Offline quality gate",
        "Event recall, false alerts per camera-hour, latency and subgroup checks.",
        "blue",
        "STEP 7",
    )
    draw_node(
        draw,
        (1100, 650, 1600, 900),
        "Shadow deployment",
        "Observe live traffic, suppress notifications and compare reviewer decisions.",
        "red",
        "STEP 8",
    )
    draw_node(
        draw,
        (475, 650, 975, 900),
        "Controlled release",
        "Staged stores, rollback criteria, live SLOs and weekly error review.",
        "green",
        "STEP 9",
    )
    draw_node(
        draw,
        (75, 650, 385, 900),
        "Reviewer feedback",
        "False alarm, miss and workflow reason codes.",
        "purple",
        "STEP 10",
        detail_size=20,
    )
    draw_elbow_arrow(
        draw,
        [(2365, 490), (2365, 590), (1975, 590), (1975, 650)],
        COLORS["teal"],
    )
    draw_arrow(draw, (1725, 775), (1600, 775), COLORS["blue"])
    draw_arrow(draw, (1100, 775), (975, 775), COLORS["green"])
    draw_arrow(draw, (475, 775), (385, 775), COLORS["purple"])
    draw_elbow_arrow(
        draw,
        [(230, 650), (230, 565), (255, 565), (255, 490)],
        COLORS["purple"],
    )

    draw_group(draw, (540, 1010, 2505, 1185), "Training-data portfolio", "gray")
    sources = [
        ("Public pretraining", "Generic people, objects and retail actions"),
        ("Local normal scenes", "Lighting, layout, crowding and camera angles"),
        ("Controlled scenarios", "Rare event coverage with known ground truth"),
        ("Reviewed store events", "Hard negatives and operational corrections"),
    ]
    x = 585
    for index, (title, detail) in enumerate(sources):
        if index:
            draw.line((x - 24, 1058, x - 24, 1145), fill=COLORS["line"], width=2)
        draw.text((x, 1090), title, font=font(22, True), fill=COLORS["ink"])
        detail_font = font(19)
        cursor_y = 1125
        for line in wrap_text(draw, detail, detail_font, 410)[:2]:
            draw.text((x, cursor_y), line, font=detail_font, fill=COLORS["muted"])
            cursor_y += 26
        x += 470

    add_footer(
        draw,
        width,
        height,
        "Split by store, person and time period so adjacent frames and the same event cannot cross evaluation boundaries.",
    )
    save(image, "05-data-loop-en.png")


def main() -> None:
    render_user_workflow()
    render_sidecar()
    render_model_network()
    render_system()
    render_data_loop()
    for path in sorted(OUTPUT.glob("*.png")):
        with Image.open(path) as image:
            print(f"{path.name}: {image.width}x{image.height}")


if __name__ == "__main__":
    main()
