from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


class LLMExtractionError(Exception):
    """Raised when LLM extraction cannot produce a valid structured payload."""


KM_JSON_SCHEMA: dict[str, Any] = {
    "name": "km_curve_extraction",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "title",
            "x_axis_label",
            "y_axis_label",
            "time_unit",
            "max_time",
            "number_of_groups",
            "groups",
            "confidence",
            "warnings",
        ],
        "properties": {
            "title": {"type": "string"},
            "x_axis_label": {"type": "string"},
            "y_axis_label": {"type": "string"},
            "time_unit": {"type": "string"},
            "max_time": {"type": "number", "minimum": 0},
            "number_of_groups": {"type": "integer", "minimum": 1},
            "groups": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "name",
                        "initial_n",
                        "step_points",
                        "censor_times",
                        "estimated_records",
                    ],
                    "properties": {
                        "name": {"type": "string"},
                        "initial_n": {"type": ["integer", "null"], "minimum": 1},
                        "step_points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["time", "survival_probability"],
                                "properties": {
                                    "time": {"type": "number", "minimum": 0},
                                    "survival_probability": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                },
                            },
                        },
                        "censor_times": {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0},
                        },
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
                    },
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    },
    "strict": True,
}


def image_sha256(image_path: Path) -> str:
    """Return SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with image_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ExtractionRun:
    payload: dict[str, Any]
    source: str
    image_hash: str


class KMVisionExtractor:
    """OpenAI Responses API wrapper for extracting KM data from images."""

    def __init__(self, model: str = "gpt-4.1") -> None:
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def extract_from_image(self, image_path: Path) -> dict[str, Any]:
        """Call the model and return a strictly-validated payload dict."""
        if not self.api_key:
            raise LLMExtractionError("OPENAI_API_KEY is not configured.")

        client = OpenAI(api_key=self.api_key)
        data_url = self._image_to_data_url(image_path)

        instructions = (
            "You are extracting APPROXIMATE structured information from a Kaplan-Meier survival plot image. "
            "Return strict JSON only. Never claim exact patient-level recovery. "
            "If any field is unclear, provide best estimate and add warnings."
        )

        request_text = (
            "Extract these fields approximately from the Kaplan-Meier figure: "
            "title, x_axis_label, y_axis_label, time_unit, max_time, number_of_groups, "
            "for each group: name, initial_n (if visible, else null), step_points as [{time, survival_probability}], "
            "censor_times (if visible), estimated_records as [{time, event}] where event=1 for event and 0 for censor, "
            "overall confidence score 0..1, and warnings/limitations. "
            "Use empty arrays when unknown."
        )

        first = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": request_text},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            instructions=instructions,
            text={"format": {"type": "json_schema", "name": KM_JSON_SCHEMA["name"], "schema": KM_JSON_SCHEMA["schema"], "strict": True}},
        )

        first_text = self._response_text(first)
        parsed_first = self._parse_json(first_text)
        if parsed_first is not None:
            self._validate_payload(parsed_first)
            return parsed_first

        repair_prompt = (
            "Repair the following invalid JSON so it matches the required schema exactly. "
            "Return only valid JSON, no markdown:\n\n"
            f"{first_text}"
        )
        second = client.responses.create(
            model=self.model,
            input=repair_prompt,
            instructions="Return only strict JSON that matches the schema.",
            text={"format": {"type": "json_schema", "name": KM_JSON_SCHEMA["name"], "schema": KM_JSON_SCHEMA["schema"], "strict": True}},
        )

        second_text = self._response_text(second)
        parsed_second = self._parse_json(second_text)
        if parsed_second is None:
            raise LLMExtractionError("The LLM response could not be parsed as valid JSON after one repair retry.")

        self._validate_payload(parsed_second)
        return parsed_second

    def _image_to_data_url(self, image_path: Path) -> str:
        suffix = image_path.suffix.lower().replace(".", "")
        mime = "image/png" if suffix == "png" else "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

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

    def _validate_payload(self, payload: dict[str, Any]) -> None:
        groups = payload.get("groups")
        if not isinstance(groups, list) or not groups:
            raise LLMExtractionError("Invalid extraction payload: missing groups array.")

        if int(payload.get("number_of_groups", 0)) != len(groups):
            payload["number_of_groups"] = len(groups)

        for group in groups:
            self._validate_group(group)

    def _validate_group(self, group: dict[str, Any]) -> None:
        records = group.get("estimated_records")
        if not isinstance(records, list):
            raise LLMExtractionError("Invalid extraction payload: estimated_records must be an array.")

        for item in records:
            time = float(item.get("time", -1))
            event = int(item.get("event", -1))
            if time < 0:
                raise LLMExtractionError("Invalid extraction payload: record times must be nonnegative.")
            if event not in {0, 1}:
                raise LLMExtractionError("Invalid extraction payload: event indicators must be 0 or 1.")

        ordered_records = sorted(records, key=lambda r: float(r["time"]))
        if ordered_records != records:
            raise LLMExtractionError("Invalid extraction payload: estimated_records must be sorted by time.")

        step_points = group.get("step_points")
        if not isinstance(step_points, list):
            raise LLMExtractionError("Invalid extraction payload: step_points must be an array.")
        ordered_points = sorted(step_points, key=lambda p: float(p.get("time", 0)))
        if ordered_points != step_points:
            raise LLMExtractionError("Invalid extraction payload: step_points must be sorted by time.")

        censor_times = group.get("censor_times")
        if not isinstance(censor_times, list):
            raise LLMExtractionError("Invalid extraction payload: censor_times must be an array.")
        for value in censor_times:
            if float(value) < 0:
                raise LLMExtractionError("Invalid extraction payload: censor times must be nonnegative.")


def reconstruct_missing_records(payload: dict[str, Any]) -> dict[str, Any]:
    """Deterministically reconstruct records when estimated_records are absent."""
    groups = payload.get("groups", [])
    for group in groups:
        records = group.get("estimated_records", [])
        if records:
            continue

        step_points = group.get("step_points", [])
        censor_times = sorted(float(item) for item in group.get("censor_times", []))
        if not step_points and not censor_times:
            continue

        initial_n = group.get("initial_n")
        n_risk = int(initial_n) if isinstance(initial_n, int) and initial_n > 0 else 100

        rebuilt: list[dict[str, int | float]] = []
        ordered_points = sorted(step_points, key=lambda p: float(p["time"]))
        prev_survival = 1.0
        for point in ordered_points:
            current_time = float(point["time"])
            current_survival = max(0.0, min(1.0, float(point["survival_probability"])))
            drop = max(0.0, prev_survival - current_survival)
            events = int(round(drop * n_risk))
            if drop > 0 and events == 0 and n_risk > 0:
                events = 1
            events = min(events, n_risk)
            for _ in range(events):
                rebuilt.append({"time": current_time, "event": 1})
            n_risk -= events
            prev_survival = current_survival

        for ct in censor_times:
            rebuilt.append({"time": float(ct), "event": 0})

        rebuilt = sorted(rebuilt, key=lambda r: float(r["time"]))
        group["estimated_records"] = rebuilt

    return payload
