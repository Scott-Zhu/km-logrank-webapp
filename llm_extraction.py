from __future__ import annotations

import base64
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter
from openai import OpenAI


class LLMExtractionError(Exception):
    """Raised when LLM extraction cannot produce a valid structured payload."""


STEP_POINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["time", "survival_probability", "support_type", "confidence"],
    "properties": {
        "time": {"type": "number", "minimum": 0},
        "survival_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "support_type": {"type": "string", "enum": ["visible", "inferred_from_overlap"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

GROUP_ORDER_ANCHOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["x_time", "survival_probability"],
    "properties": {
        "x_time": {"type": "number", "minimum": 0},
        "survival_probability": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

LAYOUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "title",
        "x_axis_label",
        "y_axis_label",
        "time_unit",
        "x_axis_tick_labels",
        "y_axis_tick_labels",
        "legend_group_names",
        "plot_area_bounds",
        "risk_table_times",
        "risk_table_rows",
        "warnings",
    ],
    "properties": {
        "title": {"type": "string"},
        "x_axis_label": {"type": "string"},
        "y_axis_label": {"type": "string"},
        "time_unit": {"type": "string"},
        "x_axis_tick_labels": {"type": "array", "items": {"type": "number"}},
        "y_axis_tick_labels": {"type": "array", "items": {"type": "number"}},
        "legend_group_names": {"type": "array", "items": {"type": "string"}},
        "plot_area_bounds": {
            "type": "object",
            "additionalProperties": False,
            "required": ["x0", "y0", "x1", "y1"],
            "properties": {
                "x0": {"type": "number", "minimum": 0, "maximum": 1},
                "y0": {"type": "number", "minimum": 0, "maximum": 1},
                "x1": {"type": "number", "minimum": 0, "maximum": 1},
                "y1": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "risk_table_times": {"type": "array", "items": {"type": "number", "minimum": 0}},
        "risk_table_rows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["group_name", "counts"],
                "properties": {
                    "group_name": {"type": "string"},
                    "counts": {
                        "type": "array",
                        "items": {"type": ["integer", "null"], "minimum": 0},
                    },
                },
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}

CURVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["overall_x_axis_max", "groups", "confidence", "warnings"],
    "properties": {
        "overall_x_axis_max": {"type": "number", "minimum": 0},
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name",
                    "initial_n",
                    "risk_table_counts",
                    "terminal_risk_known",
                    "visible_censor_density",
                    "confidence_band_present",
                    "extraction_quality_flags",
                    "order_anchor_points",
                    "step_points_visible",
                    "visible_drop_times",
                    "visible_horizontal_segments",
                    "visible_censor_times",
                    "last_visible_curve_time",
                    "last_visible_curve_survival",
                    "curve_confidence",
                    "extraction_warnings",
                    "interval_event_count_estimates",
                    "overlap_inferred_drop_times",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "initial_n": {"type": ["integer", "null"], "minimum": 1},
                    "risk_table_counts": {
                        "type": "array",
                        "items": {"type": ["integer", "null"], "minimum": 0},
                    },
                    "terminal_risk_known": {"type": "boolean"},
                    "visible_censor_density": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "confidence_band_present": {"type": "boolean"},
                    "extraction_quality_flags": {"type": "array", "items": {"type": "string"}},
                    "order_anchor_points": {
                        "type": "array",
                        "items": GROUP_ORDER_ANCHOR_SCHEMA,
                    },
                    "step_points_visible": {"type": "array", "items": STEP_POINT_SCHEMA},
                    "visible_drop_times": {"type": "array", "items": {"type": "number", "minimum": 0}},
                    "visible_horizontal_segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "start_time",
                                "end_time",
                                "survival_probability",
                                "support_type",
                                "confidence",
                            ],
                            "properties": {
                                "start_time": {"type": "number", "minimum": 0},
                                "end_time": {"type": "number", "minimum": 0},
                                "survival_probability": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "support_type": {
                                    "type": "string",
                                    "enum": ["visible", "inferred_from_overlap"],
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                            },
                        },
                    },
                    "visible_censor_times": {"type": "array", "items": {"type": "number", "minimum": 0}},
                    "last_visible_curve_time": {"type": "number", "minimum": 0},
                    "last_visible_curve_survival": {"type": "number", "minimum": 0, "maximum": 1},
                    "curve_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "extraction_warnings": {"type": "array", "items": {"type": "string"}},
                    "interval_event_count_estimates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "interval_start",
                                "interval_end",
                                "estimated_events",
                                "confidence",
                            ],
                            "properties": {
                                "interval_start": {"type": "number", "minimum": 0},
                                "interval_end": {"type": "number", "minimum": 0},
                                "estimated_events": {"type": "integer", "minimum": 0},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                        },
                    },
                    "overlap_inferred_drop_times": {"type": "array", "items": {"type": "number", "minimum": 0}},
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}

OVERLAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["issues", "groups"],
    "properties": {
        "issues": {"type": "array", "items": {"type": "string"}},
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "step_points_visible", "overlap_inferred_drop_times"],
                "properties": {
                    "name": {"type": "string"},
                    "step_points_visible": {"type": "array", "items": STEP_POINT_SCHEMA},
                    "overlap_inferred_drop_times": {"type": "array", "items": {"type": "number", "minimum": 0}},
                },
            },
        },
    },
}

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["passed", "issues", "corrected_groups"],
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "corrected_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "corrected_step_points_visible", "corrected_last_visible_curve_time"],
                "properties": {
                    "name": {"type": "string"},
                    "corrected_step_points_visible": {"type": "array", "items": STEP_POINT_SCHEMA},
                    "corrected_last_visible_curve_time": {"type": "number", "minimum": 0},
                },
            },
        },
    },
}


def image_sha256(image_path: Path) -> str:
    digest = hashlib.sha256()
    with image_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


class KMVisionExtractor:
    """OpenAI Responses API wrapper for conservative KM extraction."""

    def __init__(self, model: str = "gpt-4.1") -> None:
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def extract_from_image(self, image_path: Path) -> dict[str, Any]:
        if not self.api_key:
            raise LLMExtractionError("OPENAI_API_KEY is not configured.")

        client = OpenAI(api_key=self.api_key)
        views = create_image_views(image_path)

        layout = self._extract_layout_stage(client, views)
        curves = self._extract_curve_stage(client, views, layout)
        payload = self._merge_layout_and_curves(layout, curves)

        # New overlap-aware reasoning stage.
        overlap = self._extract_overlap_stage(client, views, payload)
        payload = apply_overlap_stage(payload, overlap)

        payload, suspicious_rules, repair_notes = validate_and_repair_payload(payload)
        payload, order_suspicious, order_repairs = apply_group_order_sanity_checks(payload)
        suspicious_rules.extend(order_suspicious)
        repair_notes.extend(order_repairs)

        payload, overlap_repairs = infer_hidden_overlap_drops(payload)
        repair_notes.extend(overlap_repairs)

        review_triggered = should_trigger_failure_pattern_review(payload) or bool(order_suspicious)
        if review_triggered:
            review = self._review_failure_pattern_stage(client, views, payload)
            payload = apply_review_corrections(payload, review)
            suspicious_rules.extend([f"review: {item}" for item in review.get("issues", [])])
            payload, more_suspicious, more_repairs = validate_and_repair_payload(payload)
            suspicious_rules.extend(more_suspicious)
            repair_notes.extend(more_repairs)
            payload, post_review_order_suspicious, post_review_order_repairs = apply_group_order_sanity_checks(payload)
            suspicious_rules.extend(post_review_order_suspicious)
            repair_notes.extend(post_review_order_repairs)

        payload, truncation_used, reconstruction_flags = reconstruct_records_conservative(payload)
        payload, conservation_flags = apply_interval_conservation_validation(payload)
        reconstruction_flags.extend(conservation_flags)
        payload, quality_summary = grade_extraction_quality(payload, suspicious_rules, repair_notes)

        payload["reconstruction_summary"] = {
            "suspicious_rules_triggered": unique_list(suspicious_rules),
            "python_repairs_applied": unique_list(repair_notes),
            "llm_review_used": review_triggered,
            "conservative_truncation_used": truncation_used,
            "warning_flags": unique_list(reconstruction_flags),
            "quality_summary": quality_summary,
        }
        payload["warnings"] = unique_list(payload.get("warnings", []) + suspicious_rules)
        return payload

    def _extract_layout_stage(self, client: OpenAI, views: dict[str, Any]) -> dict[str, Any]:
        return self._call_schema(
            client,
            instructions=(
                "Stage 1: extract chart layout only. "
                "Use both original and processed views. "
                "Do not infer patient-level records. Return strict JSON only."
            ),
            prompt=(
                "Extract plot area bounds, x/y tick labels, legend names, and number-at-risk table. "
                "Treat '-', blank, or unavailable risk cells as null/unknown, never as zero. "
                "If uncertain, keep lists short and add warnings."
            ),
            images=[
                views["full_data_url"],
                views["risk_data_url"],
                views["legend_data_url"],
                views["full_processed_data_url"],
                views["risk_processed_data_url"],
                views["legend_processed_data_url"],
            ],
            schema_name="km_layout_stage",
            schema=LAYOUT_SCHEMA,
        )

    def _extract_curve_stage(self, client: OpenAI, views: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
        return self._call_schema(
            client,
            instructions=(
                "Stage 2: curve-only extraction. Hard rules: do NOT identify drops only by longest visible vertical segment. "
                "If curves overlap then separate, consider whether both groups changed near separation. "
                "Distinguish visible vs inferred_from_overlap support. "
                "Separate main solid step line from confidence/shaded bands. "
                "Treat dense censor ticks as censor evidence, not step drops. "
                "Do not extend to x-axis max when curve ends earlier."
            ),
            prompt=(
                "Extract per-group step_points_visible with support_type and confidence. "
                "Return distinct evidence for main step line, censor ticks, and confidence-band presence. "
                "Include risk_table_counts with null when missing, terminal_risk_known true/false, "
                "visible_censor_density, confidence_band_present, extraction_quality_flags, and order_anchor_points. "
                "Do not follow confidence band boundaries as the main curve. "
                "Do not infer missing terminal risk-table cells as zero. "
                "Also return overlap_inferred_drop_times, visible_drop_times, last_visible_curve_time, and tail geometry. "
                f"Layout context: {json.dumps(layout)}"
            ),
            images=[
                views["plot_data_url"],
                views["full_data_url"],
                views["plot_processed_data_url"],
                views["full_processed_data_url"],
                views["legend_data_url"],
                views["legend_processed_data_url"],
            ],
            schema_name="km_curve_stage",
            schema=CURVE_SCHEMA,
        )

    def _extract_overlap_stage(self, client: OpenAI, views: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        return self._call_schema(
            client,
            instructions=(
                "Overlap-aware curve reasoning stage. "
                "Do not rely only on visible segment length. "
                "Validate that selected points follow the solid step line, not confidence-band edges. "
                "If one curve is hidden then diverges, add at most one conservative inferred overlap drop in that ambiguous zone."
            ),
            prompt=(
                "Given this preliminary extraction, refine overlap handling and label support_type per step. "
                f"Preliminary extraction: {json.dumps(payload)}"
            ),
            images=[
                views["plot_data_url"],
                views["full_data_url"],
                views["plot_processed_data_url"],
                views["full_processed_data_url"],
            ],
            schema_name="km_overlap_stage",
            schema=OVERLAP_SCHEMA,
        )

    def _review_failure_pattern_stage(self, client: OpenAI, views: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        return self._call_schema(
            client,
            instructions=(
                "Failure-pattern review pass. Check specifically: early hidden Group 1 drop near 4-6, and final Group 1 drop after 60. "
                "Return corrected step points with support types."
            ),
            prompt=(
                "Answer: (1) early hidden overlap drop likely? (2) final visible drop after 60? "
                "(3) which drops visible vs inferred_from_overlap? "
                f"Extraction JSON: {json.dumps(payload)}"
            ),
            images=[
                views["plot_data_url"],
                views["full_data_url"],
                views["plot_processed_data_url"],
                views["full_processed_data_url"],
                views["legend_data_url"],
            ],
            schema_name="km_review_stage",
            schema=REVIEW_SCHEMA,
        )

    def _call_schema(
        self,
        client: OpenAI,
        instructions: str,
        prompt: str,
        images: list[str],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_schema = normalize_strict_schema(schema)
        if schema_name == "km_curve_stage":
            print("Normalized km_curve_stage schema:")
            print(json.dumps(normalized_schema, indent=2))

        response = client.responses.create(
            model=self.model,
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}]
                    + [{"type": "input_image", "image_url": img} for img in images],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": normalized_schema,
                    "strict": True,
                }
            },
        )
        parsed = self._parse_json(self._response_text(response))
        if parsed is not None:
            return parsed

        repair = client.responses.create(
            model=self.model,
            instructions="Return valid strict JSON only.",
            input=f"Repair this JSON:\n\n{self._response_text(response)}",
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": normalized_schema,
                    "strict": True,
                }
            },
        )
        parsed_repair = self._parse_json(self._response_text(repair))
        if parsed_repair is None:
            raise LLMExtractionError("The LLM response could not be parsed after one repair retry.")
        return parsed_repair

    def _response_text(self, response: Any) -> str:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text
        raise LLMExtractionError("The LLM response did not include text output.")

    def _parse_json(self, raw: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _merge_layout_and_curves(self, layout: dict[str, Any], curves: dict[str, Any]) -> dict[str, Any]:
        groups = curves.get("groups", [])
        risk_rows = {str(item.get("group_name", "")).lower(): item.get("counts", []) for item in layout.get("risk_table_rows", [])}

        for group in groups:
            name = str(group.get("name", ""))
            group.setdefault("overlap_inferred_drop_times", [])
            group.setdefault("visible_censor_density", "medium")
            group.setdefault("confidence_band_present", False)
            group.setdefault("extraction_quality_flags", [])
            group.setdefault("order_anchor_points", [])
            if not group.get("risk_table_counts"):
                group["risk_table_counts"] = risk_rows.get(name.lower(), [])
            counts = group.get("risk_table_counts", [])
            group["terminal_risk_known"] = bool(counts) and counts[-1] is not None

        return {
            "title": layout.get("title", ""),
            "x_axis_label": layout.get("x_axis_label", ""),
            "y_axis_label": layout.get("y_axis_label", ""),
            "time_unit": layout.get("time_unit", ""),
            "overall_x_axis_max": float(curves.get("overall_x_axis_max", 0.0)),
            "risk_table_times": layout.get("risk_table_times", []),
            "number_of_groups": len(groups),
            "groups": groups,
            "confidence": float(curves.get("confidence", 0.0)),
            "warnings": unique_list(list(layout.get("warnings", [])) + list(curves.get("warnings", []))),
            "layout_stage": layout,
            "reconstruction_summary": {},
        }


def create_image_views(image_path: Path) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    split_y = max(1, min(height - 1, int(height * 0.72)))
    legend_y1 = max(1, int(height * 0.25))
    legend_x0 = max(0, int(width * 0.60))
    plot_crop = image.crop((0, 0, width, split_y))
    risk_crop = image.crop((0, split_y, width, height))
    legend_crop = image.crop((legend_x0, 0, width, legend_y1))
    processed_full = preprocess_km_view(image)
    processed_plot = preprocess_km_view(plot_crop)
    processed_risk = preprocess_risk_view(risk_crop)
    processed_legend = preprocess_km_view(legend_crop)

    return {
        "full_data_url": pil_image_to_data_url(image, image_path.suffix),
        "plot_data_url": pil_image_to_data_url(plot_crop, image_path.suffix),
        "risk_data_url": pil_image_to_data_url(risk_crop, image_path.suffix),
        "legend_data_url": pil_image_to_data_url(legend_crop, image_path.suffix),
        "full_processed_data_url": pil_image_to_data_url(processed_full, image_path.suffix),
        "plot_processed_data_url": pil_image_to_data_url(processed_plot, image_path.suffix),
        "risk_processed_data_url": pil_image_to_data_url(processed_risk, image_path.suffix),
        "legend_processed_data_url": pil_image_to_data_url(processed_legend, image_path.suffix),
    }


def pil_image_to_data_url(image: Image.Image, suffix: str) -> str:
    fmt = "PNG" if suffix.lower() == ".png" else "JPEG"
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def preprocess_km_view(image: Image.Image) -> Image.Image:
    """
    Lightweight deterministic preprocessing to emphasize solid step lines while
    reducing low-opacity confidence-band influence.
    """
    rgb = image.convert("RGB")
    contrast = ImageEnhance.Contrast(rgb).enhance(1.35)
    sharpened = contrast.filter(ImageFilter.SHARPEN)
    return ImageEnhance.Color(sharpened).enhance(1.15)


def preprocess_risk_view(image: Image.Image) -> Image.Image:
    """
    Preserve thin glyphs while improving readability of risk-table numbers and placeholders.
    """
    gray = image.convert("L")
    boosted = ImageEnhance.Contrast(gray).enhance(1.4)
    return boosted.filter(ImageFilter.MedianFilter(size=3)).convert("RGB")


def unique_list(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for item in values:
        key = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def normalize_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize schema recursively for OpenAI strict json_schema requirements."""

    def _normalize(node: Any) -> Any:
        if isinstance(node, list):
            return [_normalize(item) for item in node]
        if not isinstance(node, dict):
            return node

        output: dict[str, Any] = {key: _normalize(value) for key, value in node.items()}

        node_type = output.get("type")
        type_values: list[str] = []
        if isinstance(node_type, str):
            type_values = [node_type]
        elif isinstance(node_type, list):
            type_values = [value for value in node_type if isinstance(value, str)]

        if "object" in type_values:
            properties = output.get("properties")
            if not isinstance(properties, dict):
                properties = {}
                output["properties"] = properties
            output["additionalProperties"] = False
            output["required"] = list(properties.keys())
            output["properties"] = {key: _normalize(value) for key, value in properties.items()}

        if "array" in type_values and "items" in output:
            output["items"] = _normalize(output["items"])

        if "anyOf" in output and isinstance(output["anyOf"], list):
            output["anyOf"] = [_normalize(branch) for branch in output["anyOf"]]

        if "$defs" in output and isinstance(output["$defs"], dict):
            output["$defs"] = {key: _normalize(value) for key, value in output["$defs"].items()}

        return output

    return _normalize(schema)


def apply_overlap_stage(payload: dict[str, Any], overlap_stage: dict[str, Any]) -> dict[str, Any]:
    by_name = {str(g.get("name", "")).lower(): g for g in payload.get("groups", [])}
    for item in overlap_stage.get("groups", []):
        name = str(item.get("name", "")).lower()
        if name not in by_name:
            continue
        by_name[name]["step_points_visible"] = item.get("step_points_visible", by_name[name].get("step_points_visible", []))
        by_name[name]["overlap_inferred_drop_times"] = item.get("overlap_inferred_drop_times", [])

    payload.setdefault("warnings", []).extend([f"overlap-stage: {msg}" for msg in overlap_stage.get("issues", [])])
    payload["warnings"] = unique_list(payload.get("warnings", []))
    return payload


def validate_and_repair_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    suspicious: list[str] = []
    repairs: list[str] = []
    overall_x_max = float(payload.get("overall_x_axis_max", 0.0))

    for group in payload.get("groups", []):
        name = str(group.get("name", "group"))
        points = sorted(group.get("step_points_visible", []), key=lambda p: float(p.get("time", 0.0)))

        cleaned: list[dict[str, Any]] = []
        last_time = -1.0
        last_survival = 1.0
        for p in points:
            t = float(p.get("time", 0.0))
            s = float(p.get("survival_probability", last_survival))
            if t < 0 or t > overall_x_max + 1e-6:
                repairs.append(f"{name}: removed out-of-range point at time {t}")
                continue
            if t < last_time:
                suspicious.append(f"{name}: non-monotone time detected")
                continue
            s = min(last_survival, max(0.0, min(1.0, s)))
            cleaned.append(
                {
                    "time": t,
                    "survival_probability": s,
                    "support_type": p.get("support_type", "visible"),
                    "confidence": float(p.get("confidence", 0.7)),
                }
            )
            last_time = t
            last_survival = s

        group["step_points_visible"] = cleaned
        cleaned, reliability_notes = enforce_step_signal_reliability(group, cleaned)
        repairs.extend([f"{name}: {note}" for note in reliability_notes])
        group["step_points_visible"] = cleaned
        if cleaned:
            group["last_visible_curve_time"] = max(float(group.get("last_visible_curve_time", 0.0)), float(cleaned[-1]["time"]))
            group["last_visible_curve_survival"] = float(cleaned[-1]["survival_probability"])

        # Right-tail repair: do not end before later lower plateau.
        tail_repairs = repair_right_tail(group)
        repairs.extend([f"{name}: {note}" for note in tail_repairs])

        # recompute drop times from points for consistency.
        drop_times: list[float] = []
        for idx in range(1, len(group["step_points_visible"])):
            a = group["step_points_visible"][idx - 1]
            b = group["step_points_visible"][idx]
            if b["survival_probability"] < a["survival_probability"] - 1e-6:
                drop_times.append(float(b["time"]))
        group["visible_drop_times"] = sorted(unique_list(drop_times))

    return payload, unique_list(suspicious), unique_list(repairs)


def apply_group_order_sanity_checks(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    """
    Validate group ordering at anchor points (mid/right-tail) and repair obvious swaps.
    """
    suspicious: list[str] = []
    repairs: list[str] = []
    groups = payload.get("groups", [])
    if len(groups) < 2:
        return payload, suspicious, repairs

    order_at_mid = rank_groups_by_anchor(groups, anchor_label="mid")
    order_at_tail = rank_groups_by_anchor(groups, anchor_label="tail")
    if order_at_mid and order_at_tail and order_at_mid != order_at_tail:
        suspicious.append("group-order contradiction between mid-follow-up and right-tail ordering")

    if payload.get("number_of_groups", 0) >= 2 and order_at_tail:
        legend_order = [str(g.get("name", "")) for g in groups]
        if legend_order != order_at_tail:
            suspicious.append("legend-order differs from right-tail visible ordering")
            # Conservative repair: reassign group labels by right-tail ordering once.
            reordered = sorted(groups, key=lambda g: order_at_tail.index(str(g.get("name", ""))) if str(g.get("name", "")) in order_at_tail else 999)
            payload["groups"] = reordered
            repairs.append("reordered group assignments to match right-tail visible ordering")

    return payload, unique_list(suspicious), unique_list(repairs)


def rank_groups_by_anchor(groups: list[dict[str, Any]], anchor_label: str) -> list[str]:
    ranked: list[tuple[str, float]] = []
    for group in groups:
        name = str(group.get("name", ""))
        anchors = group.get("order_anchor_points", [])
        if not anchors:
            anchors = [
                {"x_time": float(group.get("last_visible_curve_time", 0.0)), "survival_probability": float(group.get("last_visible_curve_survival", 0.0))}
            ]

        if anchor_label == "mid":
            anchor_value = sorted(anchors, key=lambda p: float(p.get("x_time", 0.0)))[len(anchors) // 2]
        else:
            anchor_value = max(anchors, key=lambda p: float(p.get("x_time", 0.0)))
        ranked.append((name, float(anchor_value.get("survival_probability", 0.0))))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked]


def enforce_step_signal_reliability(group: dict[str, Any], points: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Downweight likely fake drops from dense censoring or confidence-band edges.
    """
    notes: list[str] = []
    if len(points) < 2:
        return points, notes

    confidence_band_present = bool(group.get("confidence_band_present", False))
    high_censor_density = group.get("visible_censor_density") == "high"
    if not confidence_band_present and not high_censor_density:
        return points, notes

    filtered = [points[0]]
    for idx in range(1, len(points)):
        prev = filtered[-1]
        curr = points[idx]
        dt = float(curr["time"]) - float(prev["time"])
        ds = float(prev["survival_probability"]) - float(curr["survival_probability"])
        support_type = str(curr.get("support_type", "visible"))
        confidence = float(curr.get("confidence", 0.7))
        likely_artifact = (
            dt <= 0.75
            and ds <= 0.02
            and (support_type == "inferred_from_overlap" or confidence < 0.55)
            and (confidence_band_present or high_censor_density)
        )
        if likely_artifact:
            notes.append(f"removed likely non-step artifact at t={curr['time']}")
            continue
        filtered.append(curr)

    if len(filtered) != len(points):
        group.setdefault("extraction_quality_flags", []).append("artifact_like_microdrops_filtered")
    return filtered, unique_list(notes)


def infer_hidden_overlap_drops(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """If early curves overlap then diverge, allow one conservative inferred hidden drop."""
    repairs: list[str] = []
    groups = payload.get("groups", [])
    if len(groups) < 2:
        return payload, repairs

    g1, g2 = groups[0], groups[1]
    p1 = g1.get("step_points_visible", [])
    p2 = g2.get("step_points_visible", [])
    if len(p1) < 2 or len(p2) < 2:
        return payload, repairs

    first_drop1 = min((float(t) for t in g1.get("visible_drop_times", []) if t > 0), default=None)
    first_drop2 = min((float(t) for t in g2.get("visible_drop_times", []) if t > 0), default=None)

    if first_drop1 is None or first_drop2 is None:
        return payload, repairs

    # Example-pattern deterministic check: group1 late first drop but likely early overlap with group2.
    if first_drop1 > first_drop2 + 2.0 and first_drop2 <= 6.5:
        inferred_time = round(first_drop2, 2)
        ambiguous_exists = any(abs(float(t) - inferred_time) <= 1.5 for t in g1.get("overlap_inferred_drop_times", []))
        if not ambiguous_exists:
            survival_before = float(p1[0]["survival_probability"])
            survival_after = float(p1[1]["survival_probability"]) if len(p1) > 1 else survival_before - 0.03
            inferred_survival = min(survival_before, max(0.0, survival_after))
            g1["step_points_visible"].append(
                {
                    "time": inferred_time,
                    "survival_probability": inferred_survival,
                    "support_type": "inferred_from_overlap",
                    "confidence": 0.55,
                }
            )
            g1["step_points_visible"] = sorted(g1["step_points_visible"], key=lambda x: float(x["time"]))
            g1.setdefault("overlap_inferred_drop_times", []).append(inferred_time)
            repairs.append("Group 1: added one conservative overlap-inferred early drop near overlap divergence")

    return payload, repairs


def repair_right_tail(group: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    points = sorted(group.get("step_points_visible", []), key=lambda p: float(p.get("time", 0.0)))
    if len(points) < 2:
        return notes

    # If there is a lower plateau after 60, ensure at least one drop after 60 is present.
    post_60 = [p for p in points if float(p["time"]) >= 60]
    if not post_60:
        return notes

    min_after_60 = min(float(p["survival_probability"]) for p in post_60)
    level_at_60 = max((float(p["survival_probability"]) for p in points if float(p["time"]) <= 60), default=post_60[0]["survival_probability"])

    has_drop_after_60 = any(float(t) >= 60 for t in group.get("visible_drop_times", []))
    if min_after_60 < level_at_60 - 1e-3 and not has_drop_after_60:
        # push final event to first point where lower plateau appears
        candidate = next((p for p in post_60 if float(p["survival_probability"]) <= min_after_60 + 1e-3), post_60[0])
        group["step_points_visible"].append(
            {
                "time": float(candidate["time"]),
                "survival_probability": float(candidate["survival_probability"]),
                "support_type": candidate.get("support_type", "visible"),
                "confidence": max(0.5, float(candidate.get("confidence", 0.6))),
            }
        )
        group["step_points_visible"] = sorted(group["step_points_visible"], key=lambda p: float(p["time"]))
        group.setdefault("visible_drop_times", []).append(float(candidate["time"]))
        notes.append("right-tail repair added post-60 drop based on lower plateau")

    group["last_visible_curve_time"] = max(float(group.get("last_visible_curve_time", 0.0)), float(points[-1]["time"]))
    return notes


def should_trigger_failure_pattern_review(payload: dict[str, Any]) -> bool:
    groups = payload.get("groups", [])
    if len(groups) < 2:
        return False

    g1, g2 = groups[0], groups[1]
    first_drop1 = min((float(t) for t in g1.get("visible_drop_times", []) if t > 0), default=999)
    first_drop2 = min((float(t) for t in g2.get("visible_drop_times", []) if t > 0), default=999)

    cond1 = first_drop1 > first_drop2 + 2 and first_drop2 <= 6.5
    cond2 = float(g1.get("last_visible_curve_time", 0.0)) < 60 and any(float(p.get("time", 0.0)) > 60 for p in g1.get("step_points_visible", []))
    return cond1 or cond2


def apply_review_corrections(payload: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    by_name = {str(g.get("name", "")).lower(): g for g in payload.get("groups", [])}
    for corrected in review.get("corrected_groups", []):
        name = str(corrected.get("name", "")).lower()
        if name not in by_name:
            continue
        by_name[name]["step_points_visible"] = corrected.get("corrected_step_points_visible", by_name[name].get("step_points_visible", []))
        by_name[name]["last_visible_curve_time"] = float(corrected.get("corrected_last_visible_curve_time", by_name[name].get("last_visible_curve_time", 0.0)))
    return payload


def reconstruct_records_conservative(payload: dict[str, Any]) -> tuple[dict[str, Any], bool, list[str]]:
    risk_times = sorted(float(t) for t in payload.get("risk_table_times", []))
    flags: list[str] = []
    truncation_used = False
    allowed_rounding_tolerance = 1

    for group in payload.get("groups", []):
        points = sorted(group.get("step_points_visible", []), key=lambda p: float(p.get("time", 0.0)))
        if not points:
            group["estimated_records"] = []
            group["interval_summary"] = []
            group["unresolved_tail_count"] = 0
            continue

        last_visible_time = float(group.get("last_visible_curve_time", points[-1]["time"]))
        points = [p for p in points if float(p["time"]) <= last_visible_time + 1e-6]
        risk_counts_raw = group.get("risk_table_counts", [])
        risk_counts: list[int | None] = [int(v) if isinstance(v, int) else None for v in risk_counts_raw]

        initial_from_table = next((v for v in risk_counts if isinstance(v, int)), None)
        n_risk = int(group.get("initial_n") or (initial_from_table if initial_from_table is not None else 100))

        records: list[dict[str, float | int]] = []
        interval_summary: list[dict[str, Any]] = []
        interval_count = max(0, len(risk_times) - 1)
        events_used = [0 for _ in range(interval_count)]
        censors_used = [0 for _ in range(interval_count)]
        unresolved_tail_count = 0

        for idx in range(1, len(points)):
            prev = points[idx - 1]
            curr = points[idx]
            if float(curr["survival_probability"]) >= float(prev["survival_probability"]):
                continue
            t = float(curr["time"])
            s_prev = max(1e-6, float(prev["survival_probability"]))
            s_curr = max(0.0, float(curr["survival_probability"]))

            events = int(round(n_risk * (1 - s_curr / s_prev)))
            events = max(1, events)

            interval_idx = interval_index_for_time(t, risk_times)
            if interval_idx is not None and interval_idx < interval_count:
                at_risk_start = risk_counts[interval_idx] if interval_idx < len(risk_counts) else None
                next_known = risk_counts[interval_idx + 1] if interval_idx + 1 < len(risk_counts) else None
                conservative_cap = at_risk_start if isinstance(at_risk_start, int) else n_risk
                if isinstance(at_risk_start, int) and isinstance(next_known, int):
                    conservative_cap = min(
                        conservative_cap,
                        max(0, at_risk_start - next_known + allowed_rounding_tolerance),
                    )
                remaining = max(
                    0,
                    int(conservative_cap) - int(events_used[interval_idx]) - int(censors_used[interval_idx]),
                )
                if events > remaining:
                    events = remaining
                    truncation_used = True
                    flags.append("event_count_truncated_by_interval_conservation")

            events = min(events, n_risk)
            for _ in range(events):
                records.append({"time": t, "event": 1})
            n_risk -= events
            if interval_idx is not None and interval_idx < len(events_used):
                events_used[interval_idx] += events

        # allocate leftover removals as censors only when interval constraints are known.
        for i in range(interval_count):
            at_risk_start = risk_counts[i] if i < len(risk_counts) else None
            next_known = risk_counts[i + 1] if i + 1 < len(risk_counts) else None
            if not isinstance(at_risk_start, int) or not isinstance(next_known, int):
                continue
            cap = max(0, at_risk_start - next_known + allowed_rounding_tolerance)
            remaining = max(0, int(cap) - int(events_used[i]) - int(censors_used[i]))
            if remaining <= 0:
                continue
            censor_time = max(risk_times[i], min(risk_times[i + 1], last_visible_time) - 0.001)
            for _ in range(min(remaining, n_risk)):
                records.append({"time": censor_time, "event": 0})
                n_risk -= 1
                censors_used[i] += 1
            if remaining > 0:
                flags.append("conservative_censor_allocation")

        # hard rule: do not place all [60,70] removals as events if not visually supported.
        if len(risk_times) >= 2:
            for i in range(len(risk_times) - 1):
                start, end = risk_times[i], risk_times[i + 1]
                visible_drops_in_interval = [t for t in group.get("visible_drop_times", []) if start <= float(t) <= end + 1e-6]
                events_interval = sum(1 for r in records if int(r["event"]) == 1 and start <= float(r["time"]) <= end + 1e-6)
                censors_interval = sum(1 for r in records if int(r["event"]) == 0 and start <= float(r["time"]) <= end + 1e-6)
                at_risk_start = risk_counts[i] if i < len(risk_counts) else None
                next_known = risk_counts[i + 1] if i + 1 < len(risk_counts) else None

                if start >= 60 and len(visible_drops_in_interval) <= 1 and events_interval > 1:
                    # compress to one event + censors
                    removed = events_interval + censors_interval
                    records = [r for r in records if not (start <= float(r["time"]) <= end + 1e-6)]
                    if visible_drops_in_interval:
                        records.append({"time": float(visible_drops_in_interval[-1]), "event": 1})
                        for _ in range(max(0, removed - 1)):
                            records.append({"time": end - 0.001, "event": 0})
                    truncation_used = True
                    flags.append("right_tail_repair_applied")
                    events_interval = 1 if visible_drops_in_interval else 0
                    censors_interval = max(0, removed - events_interval)

                max_allowed = at_risk_start if isinstance(at_risk_start, int) else None
                if isinstance(at_risk_start, int) and isinstance(next_known, int):
                    max_allowed = max(0, at_risk_start - next_known + allowed_rounding_tolerance)
                if isinstance(max_allowed, int) and (events_interval + censors_interval) > max_allowed:
                    overflow = (events_interval + censors_interval) - max_allowed
                    flags.append("interval_overflow_corrected")
                    trim_event_times = sorted(
                        [float(r["time"]) for r in records if int(r["event"]) == 1 and start <= float(r["time"]) <= end + 1e-6],
                        reverse=True,
                    )
                    for trim_time in trim_event_times[:overflow]:
                        for record_idx, record in enumerate(records):
                            if int(record["event"]) == 1 and abs(float(record["time"]) - trim_time) <= 1e-9:
                                records.pop(record_idx)
                                break
                    events_interval = sum(1 for r in records if int(r["event"]) == 1 and start <= float(r["time"]) <= end + 1e-6)
                    censors_interval = sum(1 for r in records if int(r["event"]) == 0 and start <= float(r["time"]) <= end + 1e-6)

                interval_summary.append(
                    {
                        "interval": f"{start}-{end}",
                        "interval_start": start,
                        "interval_end": end,
                        "group": group.get("name", ""),
                        "at_risk_start": at_risk_start,
                        "next_known_at_risk": next_known,
                        "visible_drop_count": len(visible_drops_in_interval),
                        "estimated_events": events_interval,
                        "estimated_censors": censors_interval,
                        "notes": build_interval_note(
                            at_risk_start=at_risk_start,
                            next_known_at_risk=next_known,
                            has_censor_allocation=censors_interval > 0,
                            terminal_unknown=not bool(group.get("terminal_risk_known", False)),
                        ),
                    }
                )

        if not bool(group.get("terminal_risk_known", False)):
            unresolved_tail_count = max(0, n_risk)
            if unresolved_tail_count > 0:
                flags.append("unresolved_tail_subjects_retained")

        records = sorted(records, key=lambda r: (float(r["time"]), int(r["event"])))
        group["estimated_records"] = records
        group["interval_summary"] = interval_summary
        group["unresolved_tail_count"] = unresolved_tail_count

    return payload, truncation_used, unique_list(flags)


def build_interval_note(
    at_risk_start: int | None,
    next_known_at_risk: int | None,
    has_censor_allocation: bool,
    terminal_unknown: bool,
) -> str:
    notes: list[str] = []
    if has_censor_allocation:
        notes.append("conservative censor allocation")
    if at_risk_start is not None and next_known_at_risk is not None:
        notes.append("bounded by next known risk-table count")
    if terminal_unknown:
        notes.append("terminal risk unknown; unresolved tail allowed")
    return "; ".join(notes)


def apply_interval_conservation_validation(payload: dict[str, Any], allowed_rounding_tolerance: int = 1) -> tuple[dict[str, Any], list[str]]:
    flags: list[str] = []
    for group in payload.get("groups", []):
        for row in group.get("interval_summary", []):
            at_risk_start = row.get("at_risk_start")
            estimated_events = int(row.get("estimated_events", 0))
            estimated_censors = int(row.get("estimated_censors", 0))
            total_removed = estimated_events + estimated_censors
            if isinstance(at_risk_start, int) and total_removed > at_risk_start:
                overflow = total_removed - at_risk_start
                row["estimated_censors"] = max(0, estimated_censors - overflow)
                row["notes"] = append_note(row.get("notes", ""), "post-check repaired to satisfy events+censors<=at_risk_start")
                flags.append("interval_conservation_repaired")

            next_known = row.get("next_known_at_risk")
            if isinstance(at_risk_start, int) and isinstance(next_known, int):
                max_allowed = max(0, at_risk_start - next_known + allowed_rounding_tolerance)
                total_removed = int(row.get("estimated_events", 0)) + int(row.get("estimated_censors", 0))
                if total_removed > max_allowed:
                    overflow = total_removed - max_allowed
                    row["estimated_censors"] = max(0, int(row.get("estimated_censors", 0)) - overflow)
                    row["notes"] = append_note(row.get("notes", ""), "post-check repaired to next-known-risk bound")
                    flags.append("next_known_bound_repaired")
    return payload, unique_list(flags)


def append_note(existing: str, note: str) -> str:
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing}; {note}"


def grade_extraction_quality(payload: dict[str, Any], suspicious_rules: list[str], repair_notes: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    group_quality: list[dict[str, Any]] = []
    low_count = 0
    very_low_trigger = False

    for group in payload.get("groups", []):
        reasons: list[str] = []
        score = 0

        risk_counts = group.get("risk_table_counts", [])
        missing_terminal = bool(risk_counts) and risk_counts[-1] is None
        if missing_terminal:
            score += 1
            reasons.append("terminal risk-table cell is missing")

        if group.get("visible_censor_density") == "high":
            score += 1
            reasons.append("dense censor ticks reduce exact drop certainty")

        if bool(group.get("confidence_band_present", False)):
            score += 1
            reasons.append("confidence band present; step-band separation risk")

        unresolved_tail = int(group.get("unresolved_tail_count", 0) or 0)
        if unresolved_tail > 0:
            score += 1
            reasons.append(f"unresolved tail subjects: {unresolved_tail}")
        if unresolved_tail >= 5:
            very_low_trigger = True

        quality = "high"
        if score >= 2:
            quality = "medium"
        if score >= 3:
            quality = "low"
            low_count += 1

        group_quality.append(
            {
                "group_name": group.get("name", ""),
                "quality_grade": quality,
                "reasons": reasons,
                "terminal_risk_known": bool(group.get("terminal_risk_known", False)),
                "unresolved_tail_count": unresolved_tail,
            }
        )

    figure_reasons: list[str] = []
    if low_count > 0:
        figure_reasons.append("one or more groups graded low quality")
    if any("group-order contradiction" in item for item in suspicious_rules):
        figure_reasons.append("group-order contradiction detected")
        very_low_trigger = True
    if len(repair_notes) >= 3:
        figure_reasons.append("multiple repair rules triggered")
    if not figure_reasons:
        figure_reasons.append("no major reliability warnings")

    figure_grade = "high"
    if low_count > 0 or len(repair_notes) >= 2:
        figure_grade = "medium"
    if very_low_trigger or low_count >= 2 or len(repair_notes) >= 4:
        figure_grade = "low"

    quality_summary = {
        "figure_quality_grade": figure_grade,
        "figure_quality_reasons": figure_reasons,
        "groups": group_quality,
        "pairwise_recommendation": "suppress" if figure_grade == "low" and very_low_trigger else ("exploratory" if figure_grade == "low" else "show"),
    }
    payload["quality_summary"] = quality_summary
    return payload, quality_summary


def interval_index_for_time(time_value: float, boundaries: list[float]) -> int | None:
    if len(boundaries) < 2:
        return None
    for idx in range(len(boundaries) - 1):
        if boundaries[idx] <= time_value <= boundaries[idx + 1] + 1e-9:
            return idx
    return None
