"""Render the English proposal Markdown as a print-ready HTML document."""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/proposal-content-en.md"
OUTPUT = ROOT / "output/proposal-en.html"


def resolve_href(target: str) -> str:
    parsed = urlparse(target)
    if parsed.scheme in {"http", "https", "mailto"}:
        return target
    local = (SOURCE.parent / target).resolve()
    return local.as_uri() if local.exists() else target


def inline(text: str) -> str:
    value = html.escape(text, quote=False)
    value = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: (
            f'<a href="{html.escape(resolve_href(html.unescape(match.group(2))), quote=True)}">'
            f"{match.group(1)}</a>"
        ),
        value,
    )
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", value)
    return value


def parse_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def render_table(lines: list[str], index: int) -> tuple[str, int]:
    rows: list[list[str]] = []
    while index < len(lines) and lines[index].strip().startswith("|"):
        cells = parse_table_row(lines[index])
        if not separator_row(cells):
            rows.append(cells)
        index += 1
    if not rows:
        return "", index
    columns = max(len(row) for row in rows)
    normalized = [row + [""] * (columns - len(row)) for row in rows]
    parts = ['<div class="table-wrap"><table>']
    parts.append("<thead><tr>")
    for cell in normalized[0]:
        parts.append(f"<th>{inline(cell)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in normalized[1:]:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{inline(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts), index


def render_list(lines: list[str], index: int) -> tuple[str, int]:
    first = re.match(r"^(\s*)([-*]|\d+\.)\s+(.+)$", lines[index])
    ordered = bool(first and first.group(2)[0].isdigit())
    tag = "ol" if ordered else "ul"
    parts = [f"<{tag}>"]
    while index < len(lines):
        match = re.match(r"^(\s*)([-*]|\d+\.)\s+(.+)$", lines[index])
        if not match or bool(match.group(2)[0].isdigit()) != ordered:
            break
        parts.append(f"<li>{inline(match.group(3))}</li>")
        index += 1
    parts.append(f"</{tag}>")
    return "".join(parts), index


def is_block_start(line: str) -> bool:
    stripped = line.strip()
    return bool(
        not stripped
        or stripped == "---"
        or stripped.startswith("#")
        or stripped.startswith("|")
        or stripped.startswith(">")
        or stripped.startswith("![")
        or re.match(r"^([-*]|\d+\.)\s+", stripped)
    )


def render_body(lines: list[str]) -> str:
    start = next(index for index, line in enumerate(lines) if line.startswith("## 1."))
    lines = lines[start:]
    parts: list[str] = []
    index = 0
    while index < len(lines):
        raw = lines[index].rstrip()
        stripped = raw.strip()
        if not stripped or stripped == "---":
            index += 1
            continue

        if stripped.startswith("```"):
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index].rstrip())
                index += 1
            if index < len(lines):
                index += 1
            parts.append(f"<pre>{html.escape(chr(10).join(code_lines))}</pre>")
            continue

        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            caption, target = image_match.groups()
            image_path = (SOURCE.parent / target).resolve()
            src = image_path.as_uri()
            parts.append(
                '<figure>'
                f'<img src="{html.escape(src, quote=True)}" alt="{html.escape(caption, quote=True)}">'
                f"<figcaption>{inline(caption)}</figcaption>"
                "</figure>"
            )
            index += 1
            continue

        heading_match = re.match(r"^(#{2,4})\s+(.+)$", stripped)
        if heading_match:
            markdown_level = len(heading_match.group(1))
            level = min(3, markdown_level - 1)
            text = heading_match.group(2)
            section_match = re.match(r"(\d+)\.", text)
            page_class = ""
            if level == 1 and section_match and int(section_match.group(1)) in {4, 9, 13}:
                page_class = ' class="section-break"'
            parts.append(f"<h{level}{page_class}>{inline(text)}</h{level}>")
            index += 1
            continue

        if stripped.startswith("|"):
            block, index = render_table(lines, index)
            parts.append(block)
            continue

        if stripped.startswith(">"):
            quote: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip()[1:].strip())
                index += 1
            parts.append(f'<aside class="callout">{inline(" ".join(quote))}</aside>')
            continue

        if re.match(r"^([-*]|\d+\.)\s+", stripped):
            block, index = render_list(lines, index)
            parts.append(block)
            continue

        paragraph = [stripped]
        index += 1
        while index < len(lines) and not is_block_start(lines[index]):
            paragraph.append(lines[index].strip())
            index += 1
        parts.append(f"<p>{inline(' '.join(paragraph))}</p>")

    return "\n".join(parts)


def build_html(lines: list[str]) -> str:
    headings = [
        line[3:].strip()
        for line in lines
        if line.startswith("## ") and re.match(r"## \d+\.", line)
    ]
    contents = "\n".join(f"<li>{inline(heading)}</li>" for heading in headings)
    body = render_body(lines)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CCTV Retail Loss Prevention - Edcosys</title>
<style>
  @page {{
    size: Letter;
    margin: 0.58in 0.62in 0.64in;
  }}
  * {{ box-sizing: border-box; }}
  html {{ font-family: Calibri, Arial, sans-serif; color: #17242c; }}
  body {{ margin: 0; font-size: 10.25pt; line-height: 1.26; }}
  p {{ margin: 0 0 6pt; }}
  h1 {{ color: #2e74b5; font-size: 16pt; margin: 16pt 0 8pt; break-after: avoid; }}
  h2 {{ color: #2e74b5; font-size: 13pt; margin: 12pt 0 6pt; break-after: avoid; }}
  h3 {{ color: #1f4d78; font-size: 12pt; margin: 8pt 0 4pt; break-after: avoid; }}
  .section-break {{ break-before: page; }}
  strong {{ color: #17242c; }}
  a {{ color: #2e74b5; text-decoration: none; }}
  code {{
    font-family: Consolas, monospace; font-size: 8.8pt;
    background: #f2f4f7; padding: 1px 3px; border-radius: 2px;
  }}
  pre {{
    font-family: Consolas, monospace; font-size: 8.5pt; line-height: 1.35;
    white-space: pre-wrap; overflow-wrap: anywhere;
    background: #f2f4f7; border-left: 4px solid #2e74b5;
    padding: 8pt 10pt; margin: 6pt 0 9pt; break-inside: avoid;
  }}
  ul, ol {{ margin: 0 0 8pt 0.5in; padding-left: 0.18in; }}
  li {{ margin: 0 0 4pt; break-inside: avoid; }}
  .cover {{
    min-height: 9.72in; display: flex; flex-direction: column;
    justify-content: center; text-align: center; break-after: page;
  }}
  .cover .author {{ color: #5d6b75; font-size: 12pt; font-weight: 700; margin-bottom: 8pt; }}
  .cover h1 {{ color: #000; font-size: 24pt; margin: 0 0 4pt; }}
  .cover .subtitle {{ color: #5d6b75; font-size: 14pt; margin-bottom: 8pt; }}
  .cover .descriptor {{ color: #5d6b75; font-size: 10.5pt; font-weight: 700; margin-bottom: 24pt; }}
  .rule {{ border-top: 3px solid #2e74b5; margin: 0 0 16pt; }}
  .meta {{
    display: grid; grid-template-columns: 0.85fr 1.9fr 0.85fr 2.1fr;
    border: 1px solid #cad3da; text-align: left; margin-bottom: 12pt;
  }}
  .meta div {{ padding: 6pt 7pt; border-bottom: 1px solid #e4e8ec; }}
  .meta .label {{ background: #f2f4f7; color: #5d6b75; font-size: 8.3pt; font-weight: 700; }}
  .meta .value {{ font-size: 9pt; }}
  .principle {{
    text-align: left; background: #eaf2f8; border-left: 4px solid #2e74b5;
    padding: 10pt 12pt; margin: 2pt 0 12pt;
  }}
  .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 6pt; }}
  .metric {{ background: #f2f4f7; padding: 9pt; }}
  .metric b {{ display: block; color: #2e74b5; font-size: 8pt; margin-bottom: 3pt; }}
  .metric span {{ font-size: 9.5pt; font-weight: 700; }}
  .contents {{ break-after: page; }}
  .contents ol {{ columns: 2; column-gap: 0.45in; margin: 14pt 0 0; padding: 0; list-style: none; }}
  .contents li {{ break-inside: avoid; padding: 4pt 5pt; margin: 0; }}
  .contents li:nth-child(odd) {{ background: #f2f4f7; }}
  .contents .recommendation {{
    margin-top: 16pt; padding: 10pt 12pt; background: #e7f5f2;
    border-left: 4px solid #138a78;
  }}
  .callout {{
    display: block; background: #eaf2f8; color: #1f4d78;
    border-left: 4px solid #2e74b5; padding: 8pt 10pt; margin: 6pt 0 9pt;
    break-inside: avoid;
  }}
  .table-wrap {{ margin: 6pt 0 10pt; }}
  table {{ border-collapse: collapse; width: 100%; table-layout: fixed; font-size: 8.2pt; }}
  thead {{ display: table-header-group; }}
  tr {{ break-inside: avoid; }}
  th, td {{ border: 1px solid #cad3da; padding: 5pt 6pt; vertical-align: top; overflow-wrap: anywhere; }}
  th {{ background: #f2f4f7; color: #1f4d78; text-align: left; }}
  tbody tr:nth-child(even) {{ background: #fafbfc; }}
  figure {{ margin: 10pt 0 12pt; text-align: center; break-inside: avoid; }}
  figure img {{ max-width: 100%; max-height: 6.45in; object-fit: contain; }}
  figcaption {{ color: #5d6b75; font-size: 8.4pt; margin-top: 4pt; }}
  .footer-note {{ color: #7a8790; font-size: 8pt; margin-top: 12pt; }}
</style>
</head>
<body>
<section class="cover">
  <div class="author">Edcosys</div>
  <h1>CCTV Retail Loss Prevention</h1>
  <div class="subtitle">YOLO26 Feasibility, Dataset Strategy, and Pilot Proposal</div>
  <div class="descriptor">Technical and Product Proposal | Existing CCTV Sidecar Deployment</div>
  <div class="rule"></div>
  <div class="meta">
    <div class="label">Engagement</div><div class="value">Measured retail loss-prevention pilot</div>
    <div class="label">Version</div><div class="value">v1.0 English Proposal</div>
    <div class="label">Scope</div><div class="value">One store / 2–4 cameras</div>
    <div class="label">Date</div><div class="value">27 July 2026</div>
    <div class="label">Integration</div><div class="value">Existing NVR + AI edge sidecar</div>
    <div class="label">Model stack</div><div class="value">YOLO26 + tracking + temporal model</div>
    <div class="label">Operating mode</div><div class="value">Shadow pilot + staff review</div>
    <div class="label">Decision</div><div class="value">Proceed with a controlled pilot</div>
  </div>
  <div class="principle"><strong>Operating principle.</strong> The system creates reviewable alerts from observable actions. Store staff use the evidence clip, live context, and store procedures to decide the response.</div>
  <div class="metrics">
    <div class="metric"><b>PROTOTYPE</b><span>Interactive web UI</span></div>
    <div class="metric"><b>LOCAL VALIDATION</b><span>RTX 4060 real-video run</span></div>
    <div class="metric"><b>FIRST DEPLOYMENT</b><span>10–14 week pilot</span></div>
  </div>
</section>
<section class="contents">
  <h1>Contents</h1>
  <p>The proposal moves from the recommended decision to workflow, model and data design, architecture, prototype evidence, delivery, and acceptance.</p>
  <ol>{contents}</ol>
  <div class="recommendation"><strong>Recommendation.</strong> Approve a one-store, 2–4 camera pilot. YOLO26 supplies the visual perception layer; tracking, temporal modeling, scene rules, store data, and staff review complete the event workflow.</div>
</section>
{body}
</body>
</html>
"""


def main() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_html(lines), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
