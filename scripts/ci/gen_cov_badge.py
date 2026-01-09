# scripts/gen_coverage_badge.py
import xml.etree.ElementTree as ET
from pathlib import Path

COVERAGE_XML = Path("coverage.xml")
BADGE_PATH = Path("assets/coverage.svg")


def read_coverage_percentage() -> float:
    if not COVERAGE_XML.exists():
        raise SystemExit("coverage.xml not found")

    tree = ET.parse(COVERAGE_XML)
    root = tree.getroot()
    # coverage.py XML: line-rate attribute on root
    line_rate = float(root.attrib.get("line-rate", 0.0))
    return round(line_rate * 100, 1)


def coverage_color(pct: float) -> str:
    if pct >= 90:
        return "#4c1"   # bright green
    if pct >= 75:
        return "#97CA00"  # green
    if pct >= 60:
        return "#dfb317"  # yellow
    if pct >= 40:
        return "#fe7d37"  # orange
    return "#e05d44"      # red


def make_badge_svg(pct: float) -> str:
    """Very simple static-width badge SVG."""
    label = "coverage"
    value = f"{pct:.1f}%"
    # widths are rough; good enough for an internal badge
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="120" height="20" role="img" aria-label="{label}: {value}">
  <linearGradient id="smooth" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="round">
    <rect width="120" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#round)">
    <rect width="70" height="20" fill="#555"/>
    <rect x="70" width="50" height="20" fill="{coverage_color(pct)}"/>
    <rect width="120" height="20" fill="url(#smooth)"/>
  </g>
  <g fill="#fff" text-anchor="middle" 
     font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="35" y="14">{label}</text>
    <text x="95" y="14">{value}</text>
  </g>
</svg>"""
    return svg


def main():
    pct = read_coverage_percentage()
    BADGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BADGE_PATH.write_text(make_badge_svg(pct), encoding="utf-8")
    print(f"Coverage: {pct:.1f}% → wrote {BADGE_PATH}")


if __name__ == "__main__":
    main()
