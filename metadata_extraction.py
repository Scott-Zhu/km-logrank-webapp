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


def _extract_curve_points(
    image: Image.Image,
    plot_area: dict[str, int],
    target_color: tuple[int, int, int],
) -> list[dict[str, float]]:
    """Extract approximate normalized (time, survival) points for one curve color."""
    rgb = image.convert("RGB")
    pixels = rgb.load()

    x0, y0, x1, y1 = plot_area["x0"], plot_area["y0"], plot_area["x1"], plot_area["y1"]
    plot_width = max(1, x1 - x0)
    plot_height = max(1, y1 - y0)

    sampled: list[dict[str, float]] = []
    color_tolerance = 48

    # Sample columns across the full plot region.
    for x in range(x0, x1 + 1):
        matched_y: list[int] = []
        for y in range(y0, y1 + 1):
            current = pixels[x, y]
            if _distance_rgb(current, target_color) <= color_tolerance:
                matched_y.append(y)

        if not matched_y:
            continue

        median_y = matched_y[len(matched_y) // 2]
        normalized_time = (x - x0) / plot_width
        normalized_survival = 1.0 - ((median_y - y0) / plot_height)

        sampled.append(
            {
                "time": round(max(0.0, min(1.0, normalized_time)), 4),
                "survival_probability": round(max(0.0, min(1.0, normalized_survival)), 4),
            }
        )

    if not sampled:
        return []

    # Reduce point count and enforce KM-like non-increasing survival profile.
    stride = max(1, len(sampled) // 45)
    reduced = sampled[::stride]
    monotonic: list[dict[str, float]] = []
    best_so_far = 1.0

    for point in reduced:
        best_so_far = min(best_so_far, point["survival_probability"])
        monotonic.append(
            {
                "time": point["time"],
                "survival_probability": round(best_so_far, 4),
            }
        )

    # Make sure final point is included.
    if monotonic[-1]["time"] < sampled[-1]["time"]:
        tail_survival = min(monotonic[-1]["survival_probability"], sampled[-1]["survival_probability"])
        monotonic.append(
            {
                "time": sampled[-1]["time"],
                "survival_probability": round(tail_survival, 4),
            }
        )

    return monotonic


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
            "plot_area": None,
            "curves": [],
            "assumptions": [
                "Time axis is left-to-right and survival axis is top-to-bottom.",
                "Curve points are returned in normalized units [0,1] unless explicit axis calibration is added.",
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
            plot_area = _detect_plot_area(image)
            curve_section = base_output["curve_extraction"]
            curve_section["plot_area"] = plot_area

            candidate_colors = _choose_curve_colors(image, plot_area)
            extracted_curves: list[dict[str, object]] = []
            for index, color in enumerate(candidate_colors):
                points = _extract_curve_points(image, plot_area, color)
                if len(points) < 5:
                    continue

                extracted_curves.append(
                    {
                        "name": _curve_name(index, color),
                        "stroke_color_rgb": list(color),
                        "points": points,
                    }
                )

            if extracted_curves:
                curve_section["status"] = "ok"
                curve_section["curves"] = extracted_curves
                curve_section["messages"] = [
                    "Heuristic extraction succeeded for one or more visible curves.",
                    "Values are normalized and approximate; axis calibration is not yet applied.",
                ]
            else:
                curve_section["status"] = "fallback"
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
        base_output["curve_extraction"]["messages"] = [
            "Curve extraction unavailable due to image processing error.",
        ]
        return base_output

    try:
        with Image.open(image_path) as image:
            ocr_text = pytesseract.image_to_string(image)
    except Exception as exc:
        base_output["messages"] = [
            "OCR text extraction was unavailable for this image.",
            f"Technical detail: {exc}",
        ]
        return base_output

    lines = [_clean_line(line) for line in ocr_text.splitlines() if _clean_line(line)]

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

    number_at_risk_lines = [line for line in lines if _looks_like_number_at_risk(line)]
    number_rows = _extract_numeric_rows(lines)
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
