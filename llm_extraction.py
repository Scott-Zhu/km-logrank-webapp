from __future__ import annotations

import base64
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from openai import OpenAI


class LLMExtractionError(Exception):
    """Raised when LLM extraction cannot produce a valid structured payload."""


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
                    "counts": {"type": "array", "items": {"type": "integer", "minimum": 0}},
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
                    "step_points_visible",
                    "visible_drop_times",
                    "visible_horizontal_segments",
                    "visible_censor_times",
                    "last_visible_curve_time",
                    "last_visible_curve_survival",
                    "curve_confidence",
                    "extraction_warnings",
                    "interval_event_count_estimates",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "initial_n": {"type": ["integer", "null"], "minimum": 1},
                    "risk_table_counts": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                    "step_points_visible": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["time", "survival_probability"],
                            "properties": {
                                "time": {"type": "number", "minimum": 0},
                                "survival_probability": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                        },
                    },
                    "visible_drop_times": {"type": "array", "items": {"type": "number", "minimum": 0}},
                    "visible_horizontal_segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["start_time", "end_time", "survival_probability"],
                            "properties": {
                                "start_time": {"type": "number", "minimum": 0},
                                "end_time": {"type": "number", "minimum": 0},
                                "survival_probability": {"type": "number", "minimum": 0, "maximum": 1},
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
                            "required": ["interval_start", "interval_end", "estimated_events"],
                            "properties": {
                                "interval_start": {"type": "number", "minimum": 0},
                                "interval_end": {"type": "number", "minimum": 0},
                                "estimated_events": {"type": "integer", "minimum": 0},
                            },
                        },
                    },
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "warnings": {"type": "array", "items": {"type": "string"}},
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
                    "corrected_step_points_visible": CURVE_SCHEMA["properties"]["groups"]["items"]["properties"]["step_points_visible"],
                    "corrected_last_visible_curve_time": {"type": "number", "minimum": 0},
                },
            },
        },
    },
}


FINAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "title",
        "x_axis_label",
        "y_axis_label",
        "time_unit",
        "overall_x_axis_max",
        "risk_table_times",
        "number_of_groups",
        "groups",
        "confidence",
        "warnings",
        "layout_stage",
        "reconstruction_summary",
    ],
    "properties": {
        "title": {"type": "string"},
        "x_axis_label": {"type": "string"},
        "y_axis_label": {"type": "string"},
        "time_unit": {"type": "string"},
        "overall_x_axis_max": {"type": "number", "minimum": 0},
        "risk_table_times": {"type": "array", "items": {"type": "number", "minimum": 0}},
        "number_of_groups": {"type": "integer", "minimum": 1},
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name",
                    "initial_n",
                    "risk_table_counts",
                    "step_points_visible",
                    "visible_drop_times",
                    "last_visible_curve_time",
                    "last_visible_curve_survival",
                    "curve_confidence",
                    "extraction_warnings",
                    "interval_event_count_estimates",
                    "estimated_records",
                    "interval_summary",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "initial_n": {"type": ["integer", "null"]},
                    "risk_table_counts": {"type": "array", "items": {"type": "integer"}},
                    "step_points_visible": CURVE_SCHEMA["properties"]["groups"]["items"]["properties"]["step_points_visible"],
                    "visible_drop_times": {"type": "array", "items": {"type": "number"}},
                    "last_visible_curve_time": {"type": "number"},
                    "last_visible_curve_survival": {"type": "number"},
                    "curve_confidence": {"type": "number"},
                    "extraction_warnings": {"type": "array", "items": {"type": "string"}},
                    "interval_event_count_estimates": CURVE_SCHEMA["properties"]["groups"]["items"]["properties"]["interval_event_count_estimates"],
                    "estimated_records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["time", "event"],
                            "properties": {
                                "time": {"type": "number", "minimum": 0},
                                "event": {"type": "integer", "enum": [0, 1]},
                            },
                        },
                    },
                    "interval_summary": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["interval_start", "interval_end", "events", "censors", "risk_cap"],
                            "properties": {
                                "interval_start": {"type": "number", "minimum": 0},
                                "interval_end": {"type": "number", "minimum": 0},
                                "events": {"type": "integer", "minimum": 0},
                                "censors": {"type": "integer", "minimum": 0},
                                "risk_cap": {"type": ["integer", "null"], "minimum": 0},
                            },
                        },
                    },
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "layout_stage": LAYOUT_SCHEMA,
        "reconstruction_summary": {
            "type": "object",
            "additionalProperties": False,
            "required": ["suspicious_rules_triggered", "python_repairs_applied", "llm_review_used", "conservative_truncation_used"],
            "properties": {
                "suspicious_rules_triggered": {"type": "array", "items": {"type": "string"}},
                "python_repairs_applied": {"type": "array", "items": {"type": "string"}},
                "llm_review_used": {"type": "boolean"},
                "conservative_truncation_used": {"type": "boolean"},
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

        payload, suspicious_rules, repair_notes = validate_and_repair_payload(payload)

        review_used = False
        if suspicious_rules:
            review_used = True
            review = self._review_stage(client, views, payload)
            payload = apply_review_corrections(payload, review)
            payload, more_suspicious, more_repairs = validate_and_repair_payload(payload)
            suspicious_rules.extend(more_suspicious)
            repair_notes.extend(more_repairs)

        payload, truncation_used = reconstruct_records_conservative(payload)

        payload["reconstruction_summary"] = {
            "suspicious_rules_triggered": unique_list(suspicious_rules),
            "python_repairs_applied": unique_list(repair_notes),
            "llm_review_used": review_used,
            "conservative_truncation_used": truncation_used,
        }

        if not payload.get("warnings"):
            payload["warnings"] = []
        payload["warnings"].extend(unique_list(suspicious_rules))
        payload["warnings"] = unique_list(payload["warnings"])
        return payload

    def _extract_layout_stage(self, client: OpenAI, views: dict[str, Any]) -> dict[str, Any]:
        instructions = (
            "Stage 1: extract chart layout only. "
            "Do not infer patient-level records. Return strict JSON only. "
            "Detect plot area bounds, axis ticks, legend names, and number-at-risk table values."
        )
        prompt = (
            "Extract layout fields only: plot_area_bounds, x/y ticks, legend/group names, risk table times and rows. "
            "If uncertain, leave lists empty and add warnings."
        )
        return self._call_schema(
            client,
            instructions=instructions,
            prompt=prompt,
            images=[views["full_data_url"], views["risk_data_url"]],
            schema_name="km_layout_stage",
            schema=LAYOUT_SCHEMA,
        )

    def _extract_curve_stage(self, client: OpenAI, views: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
        instructions = (
            "Stage 2: curve-only extraction from Kaplan-Meier plot geometry. Hard anti-hallucination rules: "
            "Do not extend a group's curve to the x-axis max if it visibly ends earlier. "
            "Distinguish overall_x_axis_max vs group last visible curve time. "
            "Do not create event times beyond the last visible drop or curve end. "
            "If tail is flat at right edge, do not add drops/events there. "
            "If uncertain, omit points rather than guessing. "
            "Do not infer patient-level records from initial_n alone. "
            "Risk table constrains counts but does not justify invented exact event times."
        )
        prompt = (
            "Extract only visible curve geometry per group. Return: step_points_visible, visible_drop_times, "
            "visible_horizontal_segments, visible_censor_times, last_visible_curve_time, last_visible_curve_survival, "
            "risk_table_counts, initial_n, interval_event_count_estimates. "
            f"Layout context JSON: {json.dumps(layout)}"
        )
        return self._call_schema(
            client,
            instructions=instructions,
            prompt=prompt,
            images=[views["plot_data_url"], views["full_data_url"]],
            schema_name="km_curve_stage",
            schema=CURVE_SCHEMA,
        )

    def _review_stage(self, client: OpenAI, views: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        instructions = (
            "Review pass for suspicious right-edge artifacts. Return strict JSON only. "
            "Check if any group has points beyond visible curve end or fake drops/events at x-axis maximum."
        )
        prompt = (
            "Review this extraction and correct only curve points when needed. "
            "Questions: does any group contain points beyond visible curve end? are there fake right-edge events? "
            "does the rightmost blue curve end before the max tick when visually true? are extracted drops supported? "
            f"Extraction JSON: {json.dumps(payload)}"
        )
        return self._call_schema(
            client,
            instructions=instructions,
            prompt=prompt,
            images=[views["plot_data_url"], views["full_data_url"]],
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
        response = client.responses.create(
            model=self.model,
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}]
                    + [{"type": "input_image", "image_url": image_data_url} for image_data_url in images],
                }
            ],
            text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
        )

        parsed = self._parse_json(self._response_text(response))
        if parsed is not None:
            return parsed

        repair_prompt = (
            "Repair this to valid strict JSON for the same schema. Return JSON only:\n\n"
            f"{self._response_text(response)}"
        )
        second = client.responses.create(
            model=self.model,
            instructions="Return only valid JSON for the required schema.",
            input=repair_prompt,
            text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
        )
        parsed_second = self._parse_json(self._response_text(second))
        if parsed_second is None:
            raise LLMExtractionError("The LLM response could not be parsed after one repair retry.")
        return parsed_second

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
        risk_rows = {item.get("group_name", "").lower(): item.get("counts", []) for item in layout.get("risk_table_rows", [])}

        for group in groups:
            name = str(group.get("name", ""))
            if not group.get("risk_table_counts"):
                group["risk_table_counts"] = risk_rows.get(name.lower(), [])

        payload: dict[str, Any] = {
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
            "reconstruction_summary": {
                "suspicious_rules_triggered": [],
                "python_repairs_applied": [],
                "llm_review_used": False,
                "conservative_truncation_used": False,
            },
        }
        return payload


def create_image_views(image_path: Path) -> dict[str, Any]:
    """Create deterministic image variants to reduce layout-vs-curve confusion."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Simple deterministic split: top panel for plot, bottom panel for risk table.
    split_y = int(height * 0.72)
    split_y = max(1, min(height - 1, split_y))

    plot_crop = image.crop((0, 0, width, split_y))
    risk_crop = image.crop((0, split_y, width, height))

    return {
        "full_data_url": pil_image_to_data_url(image, image_path.suffix),
        "plot_data_url": pil_image_to_data_url(plot_crop, image_path.suffix),
        "risk_data_url": pil_image_to_data_url(risk_crop, image_path.suffix),
    }


def pil_image_to_data_url(image: Image.Image, suffix: str) -> str:
    fmt = "PNG" if suffix.lower() == ".png" else "JPEG"
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def unique_list(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    output: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def validate_and_repair_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    suspicious: list[str] = []
    repairs: list[str] = []

    overall_x_max = float(payload.get("overall_x_axis_max", 0.0))
    groups = payload.get("groups", [])

    for group in groups:
        name = str(group.get("name", "group"))
        points = sorted(group.get("step_points_visible", []), key=lambda p: float(p.get("time", 0.0)))

        # Rule D1 + conservative repair: remove points beyond x-axis max.
        before = len(points)
        points = [p for p in points if 0 <= float(p.get("time", -1)) <= overall_x_max]
        if len(points) < before:
            suspicious.append(f"{name}: removed step points beyond overall_x_axis_max")
            repairs.append(f"{name}: truncated out-of-range step points")

        # Enforce monotonic time/survival (Rule D4).
        repaired_points: list[dict[str, float]] = []
        last_time = -1.0
        last_survival = 1.0
        for point in points:
            time_value = max(0.0, float(point.get("time", 0.0)))
            survival_value = max(0.0, min(1.0, float(point.get("survival_probability", last_survival))))
            if time_value < last_time:
                suspicious.append(f"{name}: non-monotone times detected")
                continue
            if survival_value > last_survival:
                survival_value = last_survival
                repairs.append(f"{name}: clamped increasing survival step")
            repaired_points.append({"time": time_value, "survival_probability": survival_value})
            last_time = time_value
            last_survival = survival_value

        group["step_points_visible"] = repaired_points

        # Ensure last visible curve time is never past axis max.
        last_visible_time = float(group.get("last_visible_curve_time", 0.0))
        if last_visible_time > overall_x_max:
            suspicious.append(f"{name}: last_visible_curve_time exceeded overall_x_axis_max")
            group["last_visible_curve_time"] = overall_x_max
            repairs.append(f"{name}: clamped last visible time to axis max")
            last_visible_time = overall_x_max

        # Rule D5: drops after visible end are removed.
        drop_times = sorted(float(t) for t in group.get("visible_drop_times", []))
        valid_drops = [t for t in drop_times if t <= last_visible_time + 1e-6]
        if len(valid_drops) < len(drop_times):
            suspicious.append(f"{name}: removed visible drops beyond curve end")
            repairs.append(f"{name}: truncated drop times after curve end")

        # Never allow synthetic right-edge drop when no visible geometry supports it.
        has_drop_at_right_edge = any(abs(t - overall_x_max) < 1e-6 for t in valid_drops)
        curve_reaches_right_edge = last_visible_time >= overall_x_max - 1e-6
        if has_drop_at_right_edge and not curve_reaches_right_edge:
            valid_drops = [t for t in valid_drops if abs(t - overall_x_max) >= 1e-6]
            suspicious.append(f"{name}: dropped fake right-edge drop time")
            repairs.append(f"{name}: removed drop at axis max without visible tail")

        group["visible_drop_times"] = valid_drops

        # Rule D6: keep final survival consistent with final step point.
        if repaired_points:
            final_survival = float(repaired_points[-1]["survival_probability"])
            if abs(float(group.get("last_visible_curve_survival", final_survival)) - final_survival) > 0.05:
                suspicious.append(f"{name}: corrected inconsistent last visible survival")
                group["last_visible_curve_survival"] = final_survival
                repairs.append(f"{name}: aligned last visible survival with final step point")

        # Trim points beyond curve end (except same-time anchors).
        trimmed = [p for p in repaired_points if float(p["time"]) <= last_visible_time + 1e-6]
        if len(trimmed) < len(repaired_points):
            suspicious.append(f"{name}: removed points after curve visually ended")
            repairs.append(f"{name}: truncated step points to last visible curve time")
            group["step_points_visible"] = trimmed

    return payload, unique_list(suspicious), unique_list(repairs)


def apply_review_corrections(payload: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    groups_by_name = {str(group.get("name", "")).lower(): group for group in payload.get("groups", [])}
    for corrected in review.get("corrected_groups", []):
        name = str(corrected.get("name", "")).lower()
        if name not in groups_by_name:
            continue
        target = groups_by_name[name]
        corrected_points = corrected.get("corrected_step_points_visible", [])
        if isinstance(corrected_points, list):
            target["step_points_visible"] = corrected_points
        corrected_last_time = corrected.get("corrected_last_visible_curve_time")
        if isinstance(corrected_last_time, (int, float)):
            target["last_visible_curve_time"] = float(corrected_last_time)

    review_issues = review.get("issues", [])
    if review_issues:
        payload.setdefault("warnings", []).extend([f"LLM review: {issue}" for issue in review_issues])
    return payload


def reconstruct_records_conservative(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Stage 3: deterministic reconstruction only, conservative against right-edge hallucination."""
    risk_times = sorted(float(t) for t in payload.get("risk_table_times", []))
    groups = payload.get("groups", [])
    overall_x_max = float(payload.get("overall_x_axis_max", 0.0))

    truncation_used = False

    for group in groups:
        name = str(group.get("name", "group"))
        points = sorted(group.get("step_points_visible", []), key=lambda p: float(p.get("time", 0.0)))
        if not points:
            group["estimated_records"] = []
            group["interval_summary"] = []
            continue

        last_visible_time = float(group.get("last_visible_curve_time", points[-1]["time"]))
        points = [p for p in points if float(p["time"]) <= last_visible_time + 1e-6]

        risk_counts = [int(v) for v in group.get("risk_table_counts", []) if isinstance(v, int)]
        if len(risk_times) >= 2 and len(risk_counts) >= 2:
            risk_caps = [max(0, risk_counts[i] - risk_counts[i + 1]) for i in range(min(len(risk_times), len(risk_counts)) - 1)]
        else:
            risk_caps = []

        initial_n = group.get("initial_n")
        n_risk = int(initial_n) if isinstance(initial_n, int) and initial_n > 0 else (risk_counts[0] if risk_counts else 100)

        records: list[dict[str, float | int]] = []
        interval_events_used = [0 for _ in risk_caps]
        interval_summary: list[dict[str, int | float | None]] = []

        for idx in range(1, len(points)):
            prev = points[idx - 1]
            curr = points[idx]
            t_prev = float(prev["time"])
            t_curr = float(curr["time"])
            s_prev = max(1e-6, float(prev["survival_probability"]))
            s_curr = max(0.0, min(1.0, float(curr["survival_probability"])))

            if t_curr > last_visible_time + 1e-6:
                truncation_used = True
                break

            if s_curr >= s_prev:
                continue

            # Required conservative formula from instruction E1.
            events = int(round(n_risk * (1.0 - (s_curr / s_prev))))
            if events < 0:
                events = 0
            if events == 0:
                events = 1

            interval_idx = interval_index_for_time(t_curr, risk_times)
            if interval_idx is not None and interval_idx < len(risk_caps):
                remaining_cap = max(0, risk_caps[interval_idx] - interval_events_used[interval_idx])
                if events > remaining_cap:
                    events = remaining_cap
                    truncation_used = True

            events = min(events, n_risk)
            for _ in range(events):
                records.append({"time": t_curr, "event": 1})

            if interval_idx is not None and interval_idx < len(interval_events_used):
                interval_events_used[interval_idx] += events

            n_risk -= events
            if n_risk <= 0:
                break

        # Conservative censor allocation: unexplained removals become censors near interval end.
        for interval_idx, cap in enumerate(risk_caps):
            removed_by_events = interval_events_used[interval_idx]
            extra_removals = max(0, cap - removed_by_events)
            if extra_removals <= 0:
                continue

            interval_end = risk_times[interval_idx + 1]
            if interval_end > last_visible_time + 1e-6:
                interval_end = last_visible_time
            censor_time = max(risk_times[interval_idx], interval_end - 0.001)

            censors_to_add = min(extra_removals, n_risk)
            for _ in range(censors_to_add):
                records.append({"time": censor_time, "event": 0})
            n_risk -= censors_to_add

        # Visible censor marks (if present) are safe censored records.
        for censor_time in sorted(float(t) for t in group.get("visible_censor_times", [])):
            if censor_time <= last_visible_time + 1e-6 and n_risk > 0:
                records.append({"time": censor_time, "event": 0})
                n_risk -= 1

        # Rule E4: if tail is flat, no right-edge events.
        if points and abs(points[-1]["time"] - overall_x_max) < 1e-6:
            if len(points) >= 2 and abs(points[-1]["survival_probability"] - points[-2]["survival_probability"]) < 1e-6:
                records = [r for r in records if not (r["event"] == 1 and abs(float(r["time"]) - overall_x_max) < 1e-6)]
                truncation_used = True

        # Rule D3: no repeated fake event spikes at x-axis max without visible drop.
        drop_times = [float(t) for t in group.get("visible_drop_times", [])]
        has_drop_at_max = any(abs(t - overall_x_max) < 1e-6 for t in drop_times)
        if not has_drop_at_max:
            right_edge_events = [r for r in records if int(r["event"]) == 1 and abs(float(r["time"]) - overall_x_max) < 1e-6]
            if len(right_edge_events) > 1:
                records = [r for r in records if not (int(r["event"]) == 1 and abs(float(r["time"]) - overall_x_max) < 1e-6)]
                truncation_used = True

        records = sorted(records, key=lambda record: (float(record["time"]), int(record["event"])))

        # Build interval summary for UI.
        if len(risk_times) >= 2:
            for interval_idx in range(len(risk_times) - 1):
                start = risk_times[interval_idx]
                end = risk_times[interval_idx + 1]
                events_count = sum(1 for r in records if int(r["event"]) == 1 and start <= float(r["time"]) < end + 1e-9)
                censor_count = sum(1 for r in records if int(r["event"]) == 0 and start <= float(r["time"]) < end + 1e-9)
                cap = risk_caps[interval_idx] if interval_idx < len(risk_caps) else None
                interval_summary.append(
                    {
                        "interval_start": start,
                        "interval_end": end,
                        "events": events_count,
                        "censors": censor_count,
                        "risk_cap": cap,
                    }
                )

                if cap is not None and (events_count + censor_count) > cap + 2:
                    truncation_used = True
                    payload.setdefault("warnings", []).append(
                        f"{name}: interval {start}-{end} removals looked too high; conservative truncation was applied."
                    )

        group["estimated_records"] = records
        group["interval_summary"] = interval_summary

    payload["warnings"] = unique_list(payload.get("warnings", []))
    return payload, truncation_used


def interval_index_for_time(time_value: float, boundaries: list[float]) -> int | None:
    if len(boundaries) < 2:
        return None
    for index in range(len(boundaries) - 1):
        if boundaries[index] <= time_value <= boundaries[index + 1] + 1e-9:
            return index
    return None
