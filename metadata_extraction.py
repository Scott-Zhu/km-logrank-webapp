from __future__ import annotations

import math
import re
from pathlib import Path

from PIL import Image
import pytesseract


def _clean_line(line: str) -> str:
    return " ".join(line.split())


def _looks_like_axis_label(line: str) -> bool:
    lower_line = line.lower()
    axis_keywords = [
        "time",
        "months",
        "years",
        "days",
        "survival",
        "probability",
        "percent",
        "overall survival",
        "progression-free",
    ]
    return any(keyword in lower_line for keyword in axis_keywords)


def _looks_like_legend(line: str) -> bool:
    lower_line = line.lower()
    legend_keywords = [
        "group",
        "arm",
        "treatment",
        "control",
        "placebo",
        "n=",
        "hr",
        "hazard",
        "strata",
    ]
    return any(keyword in lower_line for keyword in legend_keywords)


def _looks_like_number_at_risk(line: str) -> bool:
    lower_line = line.lower()
    return "number at risk" in lower_line or "no. at risk" in lower_line


def _extract_numeric_rows(lines: list[str]) -> list[str]:
    candidate_rows: list[str] = []
    for line in lines:
        if re.search(r"\d", line) and len(re.findall(r"\d+", line)) >= 2:
            candidate_rows.append(line)
    return candidate_rows


def _detect_plot_area(image: Image.Image) -> dict[str, int]:
    """Detect an approximate plot region using a non-white pixel bounding box."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()

    min_x, min_y = width, height
    max_x, max_y = -1, -1

    white_threshold = 242
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r < white_threshold or g < white_threshold or b < white_threshold:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    if max_x < min_x or max_y < min_y:
        return {"x0": 0, "y0": 0, "x1": width - 1, "y1": height - 1}

    # Trim a little margin for titles/footers while keeping axis/curves.
    pad_x = max(2, int(width * 0.01))
    pad_y = max(2, int(height * 0.01))

    x0 = max(0, min_x + pad_x)
    y0 = max(0, min_y + pad_y)
    x1 = min(width - 1, max_x - pad_x)
    y1 = min(height - 1, max_y - pad_y)

    if x1 <= x0 or y1 <= y0:
        return {"x0": min_x, "y0": min_y, "x1": max_x, "y1": max_y}

    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _row_nonwhite_density(image: Image.Image, area: dict[str, int], y: int) -> float:
    rgb = image.convert("RGB")
    pixels = rgb.load()
    x0, x1 = area["x0"], area["x1"]
    total = max(1, x1 - x0 + 1)
    nonwhite = 0
    for x in range(x0, x1 + 1):
        r, g, b = pixels[x, y]
        if min(r, g, b) < 242:
            nonwhite += 1
    return nonwhite / total


def _split_plot_and_risk_table(image: Image.Image, content_area: dict[str, int]) -> tuple[dict[str, int], dict[str, int] | None]:
    """Split the content region into upper KM panel and lower risk-table panel."""
    y0, y1 = content_area["y0"], content_area["y1"]
    height = max(1, y1 - y0 + 1)

    search_start = y0 + int(height * 0.38)
    search_end = y0 + int(height * 0.90)
    if search_end <= search_start:
        return content_area, None

    best_y = None
    best_density = 1.0
    for y in range(search_start, search_end + 1):
        density = _row_nonwhite_density(image, content_area, y)
        if density < best_density:
            best_density = density
            best_y = y

    if best_y is None or best_density > 0.16:
        return content_area, None

    plot_area = {
        "x0": content_area["x0"],
        "y0": content_area["y0"],
        "x1": content_area["x1"],
        "y1": max(content_area["y0"] + 20, best_y - 2),
    }
    if plot_area["y1"] >= content_area["y1"] - 10:
        return content_area, None

    risk_table_area = {
        "x0": content_area["x0"],
        "y0": min(content_area["y1"], best_y + 2),
        "x1": content_area["x1"],
        "y1": content_area["y1"],
    }
    return plot_area, risk_table_area


def _quantize_color(color: tuple[int, int, int], bucket_size: int = 32) -> tuple[int, int, int]:
    return tuple((channel // bucket_size) * bucket_size for channel in color)


def _is_grayish(color: tuple[int, int, int], tolerance: int = 16) -> bool:
    r, g, b = color
    return max(abs(r - g), abs(g - b), abs(r - b)) <= tolerance


def _choose_curve_colors(image: Image.Image, plot_area: dict[str, int]) -> list[tuple[int, int, int]]:
    """Find dominant non-gray colors that could correspond to curve strokes."""
    rgb = image.convert("RGB")
    pixels = rgb.load()

    x0, y0, x1, y1 = plot_area["x0"], plot_area["y0"], plot_area["x1"], plot_area["y1"]
    color_bins: dict[tuple[int, int, int], int] = {}

    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            color = pixels[x, y]
            if min(color) > 245:
                continue
            quantized = _quantize_color(color)
            if _is_grayish(quantized):
                continue
            color_bins[quantized] = color_bins.get(quantized, 0) + 1

    # Keep colors with enough support, sorted by prevalence.
    ranked_colors = sorted(color_bins.items(), key=lambda item: item[1], reverse=True)
    candidates = [color for color, count in ranked_colors if count >= 40]

    # Fallback to dark curve if no colorful lines were found.
    if not candidates:
        candidates = [(35, 35, 35)]

    return candidates[:4]


def _distance_rgb(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _saturation(color: tuple[int, int, int]) -> int:
    return max(color) - min(color)


def _candidate_points_by_color(
    image: Image.Image, plot_area: dict[str, int]
) -> dict[tuple[int, int, int], set[tuple[int, int]]]:
    rgb = image.convert("RGB")
    pixels = rgb.load()
    x0, y0, x1, y1 = plot_area["x0"], plot_area["y0"], plot_area["x1"], plot_area["y1"]
    points_by_color: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            r, g, b = pixels[x, y]
            sat = _saturation((r, g, b))
            is_near_white = min(r, g, b) > 244
            is_axis_like_gray = sat <= 14 and 70 <= max(r, g, b) <= 210
            is_dark_text_like = sat <= 16 and max(r, g, b) < 85
            if is_near_white or is_axis_like_gray:
                continue
            if is_dark_text_like and (x - x0) < 8:
                continue
            quantized = _quantize_color((r, g, b), bucket_size=24)
            points_by_color.setdefault(quantized, set()).add((x, y))
    return points_by_color


def _connected_components(points: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    if not points:
        return []
    points_left = set(points)
    components: list[list[tuple[int, int]]] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while points_left:
        start = points_left.pop()
        stack = [start]
        current_component = [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in neighbors:
                neighbor = (x + dx, y + dy)
                if neighbor in points_left:
                    points_left.remove(neighbor)
                    stack.append(neighbor)
                    current_component.append(neighbor)
        components.append(current_component)
    return components


def _component_bbox(component: list[tuple[int, int]]) -> dict[str, int]:
    xs = [point[0] for point in component]
    ys = [point[1] for point in component]
    return {"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)}


def _component_mean_color(image: Image.Image, component: list[tuple[int, int]]) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    pixels = rgb.load()
    total_r = total_g = total_b = 0
    for x, y in component:
        r, g, b = pixels[x, y]
        total_r += r
        total_g += g
        total_b += b
    size = max(1, len(component))
    return (total_r // size, total_g // size, total_b // size)


def _extract_component_curve_points(
    component: list[tuple[int, int]], plot_area: dict[str, int]
) -> list[dict[str, float]]:
    x0, y0, x1, y1 = plot_area["x0"], plot_area["y0"], plot_area["x1"], plot_area["y1"]
    plot_width = max(1, x1 - x0)
    plot_height = max(1, y1 - y0)
    columns: dict[int, list[int]] = {}
    for x, y in component:
        columns.setdefault(x, []).append(y)

    if not columns:
        return []

    sorted_x = sorted(columns.keys())
    reduced: list[dict[str, float]] = []
    x_min = sorted_x[0]
    x_max = sorted_x[-1]
    span_x = max(1, x_max - x_min)
    for x in sorted_x:
        ys = sorted(columns[x])
        median_y = ys[len(ys) // 2]
        reduced.append(
            {
                # Normalize per-trace span so the visible trace starts near time=0.
                "time": round(max(0.0, min(1.0, (x - x_min) / span_x)), 4),
                "survival_probability": round(
                    max(0.0, min(1.0, 1.0 - ((median_y - y0) / plot_height))),
                    4,
                ),
            }
        )

    # Enforce KM-like monotone non-increasing shape and staircase smoothing.
    monotonic: list[dict[str, float]] = []
    best_so_far = 1.0
    for point in reduced:
        current = float(point["survival_probability"])
        best_so_far = min(best_so_far, current)
        if monotonic:
            previous = monotonic[-1]["survival_probability"]
            if abs(previous - best_so_far) < 0.01:
                best_so_far = previous
        monotonic.append(
            {
                "time": point["time"],
                "survival_probability": round(best_so_far, 4),
            }
        )
    if monotonic:
        start_survival = max(0.01, float(monotonic[0]["survival_probability"]))
        for point in monotonic:
            rescaled = min(1.0, float(point["survival_probability"]) / start_survival)
            point["survival_probability"] = round(rescaled, 4)
        monotonic[0]["time"] = 0.0
        monotonic[0]["survival_probability"] = 1.0
    return monotonic


def _curve_quality(points: list[dict[str, float]]) -> float:
    if len(points) < 8:
        return 0.0
    drops = 0
    flat = 0
    for left, right in zip(points, points[1:]):
        delta = float(left["survival_probability"]) - float(right["survival_probability"])
        if delta > 0.005:
            drops += 1
        else:
            flat += 1
    smoothness = flat / max(1, len(points) - 1)
    step_presence = min(1.0, drops / 4.0)
    return round((0.6 * step_presence) + (0.4 * smoothness), 3)


def _curve_coverage(points: list[dict[str, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return max(0.0, min(1.0, float(points[-1]["time"]) - float(points[0]["time"])))


def _curve_similarity(a: list[dict[str, float]], b: list[dict[str, float]]) -> float:
    """Return similarity score in [0,1] for two monotone traces."""
    if len(a) < 6 or len(b) < 6:
        return 0.0

    a_map = {round(float(point["time"]), 2): float(point["survival_probability"]) for point in a}
    b_map = {round(float(point["time"]), 2): float(point["survival_probability"]) for point in b}
    shared = sorted(set(a_map.keys()) & set(b_map.keys()))
    if len(shared) < 10:
        return 0.0

    avg_abs_diff = sum(abs(a_map[t] - b_map[t]) for t in shared) / len(shared)
    return max(0.0, min(1.0, 1.0 - (avg_abs_diff / 0.12)))


def _deduplicate_curves(curves: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    for curve in curves:
        points = curve.get("points", [])
        if not isinstance(points, list):
            continue
        matched_index = None
        for index, existing in enumerate(deduped):
            similarity = _curve_similarity(
                list(existing.get("points", [])),
                points,
            )
            if similarity >= 0.82:
                matched_index = index
                break
        if matched_index is None:
            deduped.append(curve)
            continue

        existing = deduped[matched_index]
        existing_quality = float(existing.get("quality", 0.0))
        candidate_quality = float(curve.get("quality", 0.0))
        existing_coverage = float(existing.get("coverage", 0.0))
        candidate_coverage = float(curve.get("coverage", 0.0))

        if (candidate_quality, candidate_coverage) > (existing_quality, existing_coverage):
            deduped[matched_index] = curve
    return deduped


def _curve_name(index: int, color: tuple[int, int, int]) -> str:
    return f"curve_{index + 1}_rgb_{color[0]}_{color[1]}_{color[2]}"


def extract_figure_metadata(image_path: Path) -> dict[str, object]:
    """Approximate OCR + curve digitization for Kaplan-Meier figures.

    This intentionally provides heuristic curve-point extraction and metadata only,
    not patient-level event reconstruction.
    """
    base_output: dict[str, object] = {
        "module": "metadata + curve digitization prototype",
        "disclaimer": (
            "Approximate OCR + curve digitization only. This does not reconstruct "
            "patient-level survival data or exact event times."
        ),
        "filename": image_path.name,
        "title": None,
        "legend_text": [],
        "axis_labels": [],
        "number_at_risk_text": [],
        "ocr_reliable": False,
        "curve_extraction": {
            "status": "not_run",
            "content_area": None,
            "plot_area": None,
            "risk_table_area": None,
            "axis_calibration": {"status": "not_run", "time_scale": "normalized_0_1"},
            "curves": [],
            "confidence": 0.0,
            "valid_curve_count": 0,
            "assumptions": [
                "Time axis is left-to-right and survival axis is top-to-bottom.",
                "Curve points are returned in normalized units [0,1].",
                "Extracted points are approximate visual traces, not exact source data.",
            ],
            "limitations": [
                "May fail for low contrast, overlapping curves, heavy grid lines, or anti-aliased markers.",
                "Color-based separation can merge similarly colored curves.",
                "Digitized points should not be interpreted as patient-level event times.",
            ],
            "messages": [],
        },
        "messages": [],
    }

    try:
        with Image.open(image_path) as image:
            content_area = _detect_plot_area(image)
            plot_area, risk_table_area = _split_plot_and_risk_table(image, content_area)
            curve_section = base_output["curve_extraction"]
            curve_section["content_area"] = content_area
            curve_section["plot_area"] = plot_area
            curve_section["risk_table_area"] = risk_table_area

            color_points = _candidate_points_by_color(image, plot_area)
            plot_width = max(1, plot_area["x1"] - plot_area["x0"])
            plot_height = max(1, plot_area["y1"] - plot_area["y0"])

            filtered_components: list[list[tuple[int, int]]] = []
            for color, points in color_points.items():
                if _is_grayish(color, tolerance=18):
                    continue
                for component in _connected_components(points):
                    if len(component) < max(35, int(plot_width * plot_height * 0.0002)):
                        continue
                    bbox = _component_bbox(component)
                    width = bbox["x1"] - bbox["x0"] + 1
                    height = bbox["y1"] - bbox["y0"] + 1
                    touches_left = bbox["x0"] <= plot_area["x0"] + 2
                    touches_bottom = bbox["y1"] >= plot_area["y1"] - 2
                    if width < max(18, int(plot_width * 0.12)):
                        continue
                    if height >= int(plot_height * 0.95):
                        continue
                    if touches_left and touches_bottom:
                        continue
                    filtered_components.append(component)

            # Merge anti-aliased shades/components by similar mean color.
            merged_groups: list[dict[str, object]] = []
            for component in filtered_components:
                mean_color = _component_mean_color(image, component)
                matched = None
                for group in merged_groups:
                    distance = _distance_rgb(mean_color, group["mean_color"])
                    if distance <= 22:
                        matched = group
                        break
                if matched:
                    matched["pixels"].extend(component)
                else:
                    merged_groups.append({"mean_color": mean_color, "pixels": list(component)})

            extracted_curves: list[dict[str, object]] = []
            for index, group in enumerate(merged_groups):
                points = _extract_component_curve_points(group["pixels"], plot_area)
                quality = _curve_quality(points)
                coverage = _curve_coverage(points)
                if len(points) < 14 or quality < 0.45 or coverage < 0.35:
                    continue

                extracted_curves.append(
                    {
                        "name": _curve_name(index, group["mean_color"]),
                        "stroke_color_rgb": list(group["mean_color"]),
                        "quality": quality,
                        "coverage": round(coverage, 3),
                        "points": points,
                    }
                )

            extracted_curves = _deduplicate_curves(extracted_curves)

            if extracted_curves:
                extracted_curves.sort(key=lambda curve: float(curve.get("quality", 0.0)), reverse=True)
                curve_section["status"] = "ok"
                curve_section["curves"] = extracted_curves
                curve_section["valid_curve_count"] = len(extracted_curves)
                confidence = min(0.99, 0.35 + (0.15 * len(extracted_curves)) + sum(
                    float(curve.get("quality", 0.0)) for curve in extracted_curves[:3]
                ) / 6.0)
                curve_section["confidence"] = round(confidence, 3)
                curve_section["axis_calibration"] = {
                    "status": "ok",
                    "time_scale": "normalized_0_1",
                    "details": "Normalized axes calibrated from isolated upper panel bounding box.",
                }
                curve_section["messages"] = [
                    "Heuristic extraction succeeded for one or more visible curves.",
                    "Values are normalized and approximate; lower risk-table region is excluded from curve normalization.",
                ]
            else:
                curve_section["status"] = "fallback"
                curve_section["confidence"] = 0.0
                curve_section["axis_calibration"] = {
                    "status": "failed",
                    "time_scale": "unknown",
                    "details": "Unable to isolate reliable curve traces.",
                }
                curve_section["messages"] = [
                    "No reliable curve traces detected. Falling back to metadata-only output.",
                    "Try a higher-resolution image with stronger line contrast.",
                ]
    except Exception as exc:  # graceful fallback for image processing issues
        base_output["messages"] = [
            "Unable to run image processing for this figure. "
            "Try a higher-resolution image with clearer text and lines.",
            f"Technical detail: {exc}",
        ]
        base_output["curve_extraction"]["status"] = "fallback"
        base_output["curve_extraction"]["confidence"] = 0.0
        base_output["curve_extraction"]["messages"] = [
            "Curve extraction unavailable due to image processing error.",
        ]
        return base_output

    try:
        with Image.open(image_path) as image:
            content_area = base_output["curve_extraction"].get("content_area")
            plot_area = base_output["curve_extraction"].get("plot_area")
            risk_table_area = base_output["curve_extraction"].get("risk_table_area")

            if isinstance(plot_area, dict):
                plot_crop = image.crop(
                    (
                        int(plot_area["x0"]),
                        int(plot_area["y0"]),
                        int(plot_area["x1"]) + 1,
                        int(plot_area["y1"]) + 1,
                    )
                )
                plot_ocr_text = pytesseract.image_to_string(plot_crop)
            elif isinstance(content_area, dict):
                panel_crop = image.crop(
                    (
                        int(content_area["x0"]),
                        int(content_area["y0"]),
                        int(content_area["x1"]) + 1,
                        int(content_area["y1"]) + 1,
                    )
                )
                plot_ocr_text = pytesseract.image_to_string(panel_crop)
            else:
                plot_ocr_text = pytesseract.image_to_string(image)

            risk_ocr_text = ""
            if isinstance(risk_table_area, dict):
                risk_crop = image.crop(
                    (
                        int(risk_table_area["x0"]),
                        int(risk_table_area["y0"]),
                        int(risk_table_area["x1"]) + 1,
                        int(risk_table_area["y1"]) + 1,
                    )
                )
                risk_ocr_text = pytesseract.image_to_string(risk_crop)
    except Exception as exc:
        base_output["messages"] = [
            "OCR text extraction was unavailable for this image.",
            f"Technical detail: {exc}",
        ]
        return base_output

    lines = [_clean_line(line) for line in plot_ocr_text.splitlines() if _clean_line(line)]
    risk_lines = [_clean_line(line) for line in risk_ocr_text.splitlines() if _clean_line(line)]

    if not lines:
        base_output["messages"] = [
            "OCR returned no readable text. Try a clearer image or different file."
        ]
        return base_output

    base_output["ocr_reliable"] = True

    # Heuristic: treat first long-enough non-noise line as figure title.
    title_candidate = next((line for line in lines if len(line) > 12), None)
    if title_candidate:
        base_output["title"] = title_candidate

    axis_labels = [line for line in lines if _looks_like_axis_label(line)]
    legend_text = [line for line in lines if _looks_like_legend(line)]

    number_at_risk_lines = [line for line in risk_lines if _looks_like_number_at_risk(line)]
    number_rows = _extract_numeric_rows(risk_lines or lines)
    if number_rows:
        number_at_risk_lines.extend(number_rows[:8])

    # Deduplicate while preserving order.
    def dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            output.append(item)
        return output

    base_output["axis_labels"] = dedupe(axis_labels)
    base_output["legend_text"] = dedupe(legend_text)
    base_output["number_at_risk_text"] = dedupe(number_at_risk_lines)

    if (
        base_output["title"] is None
        and not base_output["axis_labels"]
        and not base_output["legend_text"]
        and not base_output["number_at_risk_text"]
    ):
        base_output["messages"] = [
            "Metadata extraction completed but no clear labels were detected. "
            "Try a sharper image or one with higher contrast."
        ]

    return base_output
