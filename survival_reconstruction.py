from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ReconstructedGroup:
    name: str
    records: list[dict[str, float | int]]
    initial_n: int
    event_count: int
    censor_count: int
    final_time: float


def _extract_sizes_from_legend(legend_text: list[str]) -> list[int]:
    sizes: list[int] = []
    for line in legend_text:
        matches = re.findall(r"n\s*=\s*(\d+)", line, flags=re.IGNORECASE)
        for match in matches:
            sizes.append(int(match))
    return sizes


def _extract_sizes_from_risk_text(number_at_risk_text: list[str], groups_needed: int) -> list[int]:
    numeric_rows: list[list[int]] = []
    for line in number_at_risk_text:
        numbers = [int(value) for value in re.findall(r"\d+", line)]
        if len(numbers) >= groups_needed:
            numeric_rows.append(numbers)

    if numeric_rows:
        return numeric_rows[0][:groups_needed]

    # Fallback: use first numeric token from each row.
    fallback_sizes: list[int] = []
    for line in number_at_risk_text:
        numbers = re.findall(r"\d+", line)
        if numbers:
            fallback_sizes.append(int(numbers[0]))
        if len(fallback_sizes) >= groups_needed:
            break
    return fallback_sizes


def infer_initial_group_sizes(metadata_output: dict[str, object], groups_needed: int) -> tuple[list[int], str]:
    legend_text = [
        str(item) for item in metadata_output.get("legend_text", []) if isinstance(item, str)
    ]
    number_at_risk_text = [
        str(item)
        for item in metadata_output.get("number_at_risk_text", [])
        if isinstance(item, str)
    ]

    sizes = _extract_sizes_from_legend(legend_text)
    if len(sizes) >= groups_needed:
        return sizes[:groups_needed], "legend n="

    risk_sizes = _extract_sizes_from_risk_text(number_at_risk_text, groups_needed)
    if len(risk_sizes) >= groups_needed:
        return risk_sizes[:groups_needed], "number-at-risk text"

    return [100 for _ in range(groups_needed)], "default assumption (N=100 per group)"


def reconstruct_group_records(
    curve_name: str, points: list[dict[str, float]], initial_n: int
) -> ReconstructedGroup:
    if not points:
        return ReconstructedGroup(
            name=curve_name,
            records=[],
            initial_n=initial_n,
            event_count=0,
            censor_count=0,
            final_time=0.0,
        )

    sorted_points = sorted(points, key=lambda point: float(point.get("time", 0.0)))
    n_risk = max(1, int(initial_n))
    records: list[dict[str, float | int]] = []
    event_count = 0

    previous_survival = max(0.0, min(1.0, float(sorted_points[0].get("survival_probability", 1.0))))
    previous_time = float(sorted_points[0].get("time", 0.0))

    for point in sorted_points[1:]:
        current_time = float(point.get("time", previous_time))
        if current_time < previous_time:
            continue

        current_survival = max(0.0, min(1.0, float(point.get("survival_probability", previous_survival))))
        drop = max(0.0, previous_survival - current_survival)
        events = int(round(drop * n_risk))

        if drop > 0 and events == 0 and n_risk > 0:
            events = 1

        events = min(events, n_risk)
        for _ in range(events):
            records.append({"time": current_time, "event": 1})
        event_count += events
        n_risk -= events

        previous_survival = current_survival
        previous_time = current_time

    censor_count = n_risk
    final_time = float(sorted_points[-1].get("time", previous_time))
    for _ in range(censor_count):
        records.append({"time": final_time, "event": 0})

    return ReconstructedGroup(
        name=curve_name,
        records=records,
        initial_n=initial_n,
        event_count=event_count,
        censor_count=censor_count,
        final_time=final_time,
    )
