"""Generate SVG and PNG versions of the 5 tables in the first blog post,
matching the styling used in the second blog post (docs/blog/2 DEC/*.svg).

Run after lower_bound_blog_numbers.py; numbers should match its output.
"""

from pathlib import Path

import cairosvg

OUT_DIR = Path("/mnt/c/Users/talru/1cfe/1costingfe/docs/blog/1 Floor")

SVG_STYLE = """<defs><style>
.t-text { fill: #333; font-family: sans-serif; font-size: 14px; }
.t-header { font-weight: 600; }
.t-line { stroke: #b5b5b5; stroke-width: 0.75; }
.t-hline { stroke: #555; stroke-width: 1.5; }
@media (prefers-color-scheme: dark) {
  .t-text { fill: #e0e0e0; }
  .t-line { stroke: #6a6a6a; }
  .t-hline { stroke: #bbb; }
}
</style></defs>"""


def make_svg(headers, rows, col_x_anchors, width=720, subtotal_row=None):
    """Generate an SVG matching the 2nd-blog style.

    headers: list of strings, one per column.
    rows:    list of row tuples; each tuple has len(headers) strings.
    col_x_anchors: list of (x, anchor) for each column. anchor is "start" or
                   "end". The first column is typically ("start", left-x).
    subtotal_row: optional index of a row to bold (for "Subtotal" lines).
    """
    n_rows = len(rows)
    header_y = 26
    header_line_y = 40
    row_h = 36
    height = header_line_y + row_h * n_rows

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        SVG_STYLE,
        "",
        f'<line class="t-hline" x1="0" y1="{header_line_y}" '
        f'x2="{width}" y2="{header_line_y}"/>',
    ]
    for i in range(n_rows - 1):
        y = header_line_y + row_h * (i + 1)
        parts.append(f'<line class="t-line" x1="0" y1="{y}" x2="{width}" y2="{y}"/>')

    parts.append("")
    parts.append('<g class="t-text t-header">')
    for header, (x, anchor) in zip(headers, col_x_anchors):
        attr = ' text-anchor="end"' if anchor == "end" else ""
        parts.append(f'  <text x="{x}" y="{header_y}"{attr}>{header}</text>')
    parts.append("</g>")
    parts.append("")
    parts.append('<g class="t-text">')
    for ri, row in enumerate(rows):
        y = header_line_y + row_h * ri + 23
        bold = ri == subtotal_row
        weight_attr = ' font-weight="600"' if bold else ""
        for cell, (x, anchor) in zip(row, col_x_anchors):
            attr = ' text-anchor="end"' if anchor == "end" else ""
            parts.append(f'  <text x="{x}" y="{y}"{attr}{weight_attr}>{cell}</text>')
        if ri < n_rows - 1:
            parts.append("")
    parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


def write_table(name, svg, output_width=1440):
    svg_path = OUT_DIR / f"{name}.svg"
    png_path = OUT_DIR / f"{name}.png"
    svg_path.write_text(svg)
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(png_path),
        output_width=output_width,
        background_color="white",
    )
    print(f"  wrote {svg_path.name} + {png_path.name}")


# ── Table 1: BOP component breakdown (D-T, 1 GWe, sCO2, free core) ───
print("Generating Table 1: BOP component breakdown")
write_table(
    "BOP component breakdown",
    make_svg(
        headers=["Component (D-T, 1 GWe, sCO2)", "Cost", "What it is"],
        rows=[
            (
                "Turbine and generator",
                "$181M",
                "sCO2 turbomachinery, comparable to gas plant hardware",
            ),
            ("Electrical plant", "$98M", "Switchgear, transformers, grid connection"),
            (
                "Miscellaneous equipment",
                "$60M",
                "Cranes, compressed air, fire protection",
            ),
            ("Heat rejection", "$55M", "Cooling towers and circulating water"),
            ("BOP subtotal", "$393M", ""),
        ],
        col_x_anchors=[(16, "start"), (320, "end"), (344, "start")],
        subtotal_row=4,
        width=1000,
    ),
)

# ── Table 2: D-T floor at different conditions ───────────────────────
print("Generating Table 2: D-T floor scenarios")
write_table(
    "D-T floor scenarios",
    make_svg(
        headers=[
            "Scenario (D-T, free core)",
            "Floor ($/MWh)",
            "Overnight ($/kW)",
            "Core budget ($/MWh)",
        ],
        rows=[
            ("Baseline: 1 GWe, 85%, 7% WACC, 30 yr, 6 yr", "29", "1,723", "-19"),
            ("2 GWe, 85%, 7%, 30 yr, 6 yr", "24", "1,505", "-14"),
            ("2 GWe, 95%, 3%, 50 yr, 3 yr", "14", "1,213", "-3.6"),
            ("3 GWe, 95%, 3%, 50 yr, 3 yr", "12", "1,153", "-1.9"),
            ("5 GWe, 95%, 2%, 50 yr, 3 yr", "9.5", "1,094", "+0.5"),
        ],
        col_x_anchors=[(16, "start"), (470, "end"), (670, "end"), (864, "end")],
        width=880,
    ),
)

# ── Table 3: D-T staffing thresholds at aggressive conditions ────────
print("Generating Table 3: D-T staffing thresholds")
write_table(
    "D-T staffing thresholds",
    make_svg(
        headers=[
            "Scenario (D-T, free core)",
            "Floor ($/MWh)",
            "Capital only ($/MWh)",
            "Staffing threshold for $10/MWh",
        ],
        rows=[
            (
                "2 GWe, 95%, 3% WACC, 50 yr, 3 yr",
                "13.6",
                "5.8",
                "54% of current (42 FTE)",
            ),
            (
                "5 GWe, 95%, 2% WACC, 50 yr, 3 yr",
                "9.5",
                "4.3",
                "108% of current (no cuts)",
            ),
        ],
        col_x_anchors=[(16, "start"), (430, "end"), (610, "end"), (864, "end")],
        width=880,
    ),
)

# ── Table 4: Fuel spectrum ───────────────────────────────────────────
print("Generating Table 4: Fuel spectrum")
write_table(
    "Fuel spectrum",
    make_svg(
        headers=[
            "Fuel (1 GWe, free core)",
            "Buildings",
            "BOP floor ($/MWh)",
            "Staffing",
        ],
        rows=[
            ("D-T", "$574M", "29", "78 FTE"),
            ("D-He3", "$417M", "19", "33 FTE"),
            ("p-B11", "$359M", "17", "30 FTE"),
        ],
        col_x_anchors=[(16, "start"), (364, "end"), (544, "end"), (704, "end")],
    ),
)

# ── Table 5: p-B11 floor at different conditions ─────────────────────
print("Generating Table 5: p-B11 floor scenarios")
write_table(
    "p-B11 floor scenarios",
    make_svg(
        headers=[
            "Scenario (p-B11, free core)",
            "Floor ($/MWh)",
            "Overnight ($/kW)",
            "Core budget ($/MWh)",
        ],
        rows=[
            ("Baseline: 1 GWe, 85%, 7% WACC, 30 yr, 6 yr", "17", "1,221", "-7.3"),
            ("2 GWe, 85%, 7%, 30 yr, 6 yr", "14", "1,058", "-4.4"),
            ("2 GWe, 95%, 3%, 50 yr, 3 yr", "7.1", "846", "+2.9"),
            ("3 GWe, 95%, 3%, 50 yr, 3 yr", "6.3", "802", "+3.7"),
            ("5 GWe, 95%, 2%, 50 yr, 3 yr", "5.0", "759", "+5.0"),
        ],
        col_x_anchors=[(16, "start"), (470, "end"), (670, "end"), (864, "end")],
        width=880,
    ),
)

print("\nAll tables generated in:", OUT_DIR)
