from __future__ import annotations

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


def extract_figure_metadata(image_path: Path) -> dict[str, object]:
    """Approximate OCR-based metadata extraction for a Kaplan-Meier figure.

    This intentionally extracts only figure metadata (text labels/annotations),
    not patient-level data reconstruction.
    """
    base_output: dict[str, object] = {
        "module": "metadata extraction",
        "disclaimer": (
            "Approximate OCR metadata extraction only. This does not reconstruct "
            "patient-level survival data."
        ),
        "filename": image_path.name,
        "title": None,
        "legend_text": [],
        "axis_labels": [],
        "number_at_risk_text": [],
        "ocr_reliable": False,
        "messages": [],
    }

    try:
        with Image.open(image_path) as image:
            ocr_text = pytesseract.image_to_string(image)
    except Exception as exc:  # graceful fallback for OCR or image issues
        base_output["messages"] = [
            "Unable to run OCR for this image. "
            "Try a higher-resolution image with clearer text.",
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
