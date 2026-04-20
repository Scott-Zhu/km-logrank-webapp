from math import erfc, exp, log, sqrt
from pathlib import Path
import json
import re
from itertools import combinations

from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from llm_extraction import (
    KMVisionExtractor,
    LLMExtractionError,
    cap_confidence_for_quality,
    image_sha256,
)

# Load local .env values if present (safe for local development).
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-key"
app.config["UPLOAD_FOLDER"] = Path("uploads")
app.config["CACHE_FOLDER"] = Path("cache")
app.config["DEMO_CACHE_FOLDER"] = Path("demo_cache")
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
app.config["CACHE_FOLDER"].mkdir(parents=True, exist_ok=True)
app.config["DEMO_CACHE_FOLDER"].mkdir(parents=True, exist_ok=True)

DEMO_EXAMPLES = [
    {
        "slug": "lung-two-arm",
        "title": "Two-arm OS example",
        "description": "Cached two-group Kaplan-Meier reconstruction with a clear separation.",
        "filename": "lung_two_arm.json",
    },
    {
        "slug": "trial-three-arm",
        "title": "Three-arm OS example",
        "description": "Cached multi-group reconstruction with pairwise comparisons.",
        "filename": "trial_three_arm.json",
    },
    {
        "slug": "balanced-two-arm",
        "title": "Balanced two-arm example",
        "description": "Cached two-group reconstruction with similar outcomes.",
        "filename": "balanced_two_arm.json",
    },
]


def is_allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in app.config["ALLOWED_EXTENSIONS"]


def _is_live_extraction_enabled() -> bool:
    return KMVisionExtractor().is_configured()


def _list_demo_examples() -> list[dict[str, str | bool]]:
    folder = app.config["DEMO_CACHE_FOLDER"]
    rows: list[dict[str, str | bool]] = []
    for entry in DEMO_EXAMPLES:
        path = folder / str(entry["filename"])
        rows.append(
            {
                "slug": str(entry["slug"]),
                "title": str(entry["title"]),
                "description": str(entry["description"]),
                "available": path.exists(),
            }
        )
    return rows


def _load_demo_payload(slug: str) -> tuple[dict | None, dict | None]:
    for entry in DEMO_EXAMPLES:
        if entry["slug"] != slug:
            continue
        payload_path = app.config["DEMO_CACHE_FOLDER"] / str(entry["filename"])
        if not payload_path.exists():
            return None, entry
        return json.loads(payload_path.read_text(encoding="utf-8")), entry
    return None, None


def parse_survival_records(raw_text: str, group_label: str) -> list[dict[str, float | int]]:
    lines = raw_text.splitlines()
    records: list[dict[str, float | int]] = []
    comma_pattern = re.compile(r"^\s*([^,\s]+)\s*,\s*([^,\s]+)\s*$")
    whitespace_pattern = re.compile(r"^\s*(\S+)\s+(\S+)\s*$")
    zero_width_pattern = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
    punctuation_translation = str.maketrans({"，": ",", "、": ","})
    whitespace_translation = str.maketrans(
        {
            "\u3000": " ",  # ideographic/full-width space
            "\u00A0": " ",  # non-breaking space
            "\t": " ",
        }
    )

    for line_index, raw_line in enumerate(lines, start=1):
        line = zero_width_pattern.sub("", raw_line)
        line = line.translate(punctuation_translation)
        line = line.translate(whitespace_translation)
        line = line.strip()
        if not line:
            continue

        match = comma_pattern.match(line) or whitespace_pattern.match(line)
        if not match:
            raise ValueError(
                f"{group_label}, line {line_index}: expected 'time,event' or 'time event'. "
                "Full-width punctuation/spaces may need normalization."
            )
        parts = [match.group(1), match.group(2)]

        time_text, event_text = parts
        try:
            time_value = float(time_text)
        except ValueError as exc:
            raise ValueError(f"{group_label}, line {line_index}: time must be a number.") from exc

        if time_value < 0:
            raise ValueError(f"{group_label}, line {line_index}: time must be non-negative.")
        if event_text not in {"0", "1"}:
            raise ValueError(
                f"{group_label}, line {line_index}: event must be 0 (censored) or 1 (event)."
            )

        records.append({"time": time_value, "event": int(event_text)})

    if not records:
        raise ValueError(f"{group_label}: please provide at least one record.")

    return records


def compute_logrank_test(
    group_a_records: list[dict[str, float | int]],
    group_b_records: list[dict[str, float | int]],
) -> dict[str, float]:
    all_records = group_a_records + group_b_records
    event_times = sorted({float(record["time"]) for record in all_records if int(record["event"]) == 1})

    if not event_times:
        raise ValueError("Log-rank test requires at least one event across both groups.")

    observed_group_a = 0.0
    expected_group_a = 0.0
    variance_group_a = 0.0

    for time_point in event_times:
        n_a = sum(1 for record in group_a_records if float(record["time"]) >= time_point)
        n_b = sum(1 for record in group_b_records if float(record["time"]) >= time_point)
        n_total = n_a + n_b

        d_a = sum(
            1
            for record in group_a_records
            if float(record["time"]) == time_point and int(record["event"]) == 1
        )
        d_b = sum(
            1
            for record in group_b_records
            if float(record["time"]) == time_point and int(record["event"]) == 1
        )
        d_total = d_a + d_b

        if n_total == 0 or d_total == 0:
            continue

        expected_at_time = d_total * (n_a / n_total)
        variance_at_time = (
            (n_a * n_b * d_total * (n_total - d_total)) / (n_total**2 * (n_total - 1))
            if n_total > 1
            else 0.0
        )

        observed_group_a += d_a
        expected_group_a += expected_at_time
        variance_group_a += variance_at_time

    if variance_group_a <= 0:
        raise ValueError(
            "Unable to compute log-rank variance from the provided data. "
            "Please check that both groups have valid follow-up records."
        )

    chi_square = ((observed_group_a - expected_group_a) ** 2) / variance_group_a
    p_value = erfc(sqrt(chi_square / 2.0))
    return {"chi_square": chi_square, "p_value": p_value}


def _normalize_group_records(
    groups_records: list[dict[str, list[dict[str, float | int]]]],
) -> tuple[list[dict[str, float | int | str]], list[str]]:
    long_form_records: list[dict[str, float | int | str]] = []
    warnings: list[str] = []

    for group in groups_records:
        group_name = str(group.get("group_name", "Unnamed group"))
        records = group.get("records", [])

        if not isinstance(records, list) or not records:
            warnings.append(f"Skipped '{group_name}' because it has no reconstructed records.")
            continue

        normalized_rows: list[dict[str, float | int | str]] = []
        for row_index, row in enumerate(records, start=1):
            try:
                duration = float(row["time"])
                event = int(row["event"])
            except (KeyError, TypeError, ValueError):
                warnings.append(
                    f"Skipped '{group_name}' because record {row_index} has invalid time/event values."
                )
                normalized_rows = []
                break

            if duration < 0 or event not in {0, 1}:
                warnings.append(
                    f"Skipped '{group_name}' because record {row_index} has out-of-range time/event values."
                )
                normalized_rows = []
                break

            normalized_rows.append({"duration": duration, "event": event, "group": group_name})

        if normalized_rows:
            long_form_records.extend(normalized_rows)

    return long_form_records, warnings


def run_logrank_analysis(
    groups_records: list[dict[str, list[dict[str, float | int]]]],
    include_pairwise: bool = True,
    alpha: float = 0.05,
    correction_method: str = "bonferroni",
    pairwise_policy: str = "show",
) -> dict:
    from lifelines.statistics import logrank_test, multivariate_logrank_test

    long_form_records, validation_warnings = _normalize_group_records(groups_records)
    if not long_form_records:
        return {
            "available": False,
            "message": "No valid reconstructed records were available for log-rank analysis.",
            "warnings": validation_warnings,
        }

    unique_groups = sorted({str(row["group"]) for row in long_form_records})
    if len(unique_groups) < 2:
        return {
            "available": False,
            "message": "Log-rank testing requires at least 2 valid groups.",
            "warnings": validation_warnings,
            "group_count": len(unique_groups),
            "group_names": unique_groups,
        }

    durations = [float(row["duration"]) for row in long_form_records]
    events = [int(row["event"]) for row in long_form_records]
    labels = [str(row["group"]) for row in long_form_records]

    result: dict = {
        "available": True,
        "analysis_type": "two_group" if len(unique_groups) == 2 else "multigroup",
        "group_count": len(unique_groups),
        "group_names": unique_groups,
        "alpha": alpha,
        "correction_method": correction_method,
        "warnings": validation_warnings,
        "pairwise_policy": pairwise_policy,
    }

    if len(unique_groups) == 2:
        group_a, group_b = unique_groups
        durations_a = [float(row["duration"]) for row in long_form_records if row["group"] == group_a]
        events_a = [int(row["event"]) for row in long_form_records if row["group"] == group_a]
        durations_b = [float(row["duration"]) for row in long_form_records if row["group"] == group_b]
        events_b = [int(row["event"]) for row in long_form_records if row["group"] == group_b]

        output = logrank_test(durations_a, durations_b, event_observed_A=events_a, event_observed_B=events_b)
        p_value = float(output.p_value)
        result.update(
            {
                "group_a_name": group_a,
                "group_b_name": group_b,
                "chi_square": float(output.test_statistic),
                "degrees_of_freedom": 1,
                "p_value": p_value,
                "interpretation": "Groups differ (p < 0.05)."
                if p_value < alpha
                else "No clear difference (p >= 0.05).",
            }
        )
        return result

    overall = multivariate_logrank_test(durations, labels, events)
    overall_p = float(overall.p_value)
    result.update(
        {
            "chi_square": float(overall.test_statistic),
            "degrees_of_freedom": len(unique_groups) - 1,
            "p_value": overall_p,
            "interpretation": "The survival curves are not all equal; at least one group differs."
            if overall_p < alpha
            else "No statistically significant overall difference was detected.",
        }
    )

    pairwise_rows: list[dict[str, str | float | bool]] = []
    if include_pairwise and pairwise_policy != "suppress":
        pairs = list(combinations(unique_groups, 2))
        raw_results: list[dict[str, str | float]] = []

        for group_a, group_b in pairs:
            durations_a = [float(row["duration"]) for row in long_form_records if row["group"] == group_a]
            events_a = [int(row["event"]) for row in long_form_records if row["group"] == group_a]
            durations_b = [float(row["duration"]) for row in long_form_records if row["group"] == group_b]
            events_b = [int(row["event"]) for row in long_form_records if row["group"] == group_b]

            pair_result = logrank_test(
                durations_a,
                durations_b,
                event_observed_A=events_a,
                event_observed_B=events_b,
            )
            raw_results.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "chi_square": float(pair_result.test_statistic),
                    "unadjusted_p": float(pair_result.p_value),
                }
            )

        comparisons = len(raw_results)
        for row in raw_results:
            adjusted_p = row["unadjusted_p"]
            if correction_method == "bonferroni" and comparisons > 0:
                adjusted_p = min(float(row["unadjusted_p"]) * comparisons, 1.0)

            pairwise_rows.append(
                {
                    "group_a": str(row["group_a"]),
                    "group_b": str(row["group_b"]),
                    "chi_square": float(row["chi_square"]),
                    "unadjusted_p": float(row["unadjusted_p"]),
                    "adjusted_p": float(adjusted_p),
                    "significant_after_adjustment": float(adjusted_p) < alpha,
                }
            )

    result["pairwise_rows"] = pairwise_rows
    if pairwise_policy == "exploratory" and pairwise_rows:
        result["pairwise_message"] = "Pairwise comparisons are low-confidence exploratory due to extraction quality."
    if pairwise_policy == "suppress":
        result["pairwise_message"] = "Pairwise comparisons were suppressed because extraction quality is very low."
    return result


def _cache_path_for_hash(image_hash: str) -> Path:
    return app.config["CACHE_FOLDER"] / f"{image_hash}.json"


def _load_cached_extraction(image_hash: str) -> dict | None:
    cache_path = _cache_path_for_hash(image_hash)
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _save_cached_extraction(image_hash: str, payload: dict) -> None:
    _cache_path_for_hash(image_hash).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _list_cached_hashes() -> list[str]:
    return sorted(path.stem for path in app.config["CACHE_FOLDER"].glob("*.json"))


def _uploaded_image_url_for_hash(image_hash: str) -> str | None:
    for upload_path in app.config["UPLOAD_FOLDER"].iterdir():
        if not upload_path.is_file():
            continue
        if upload_path.suffix.lower().lstrip(".") not in app.config["ALLOWED_EXTENSIONS"]:
            continue
        try:
            if image_sha256(upload_path) == image_hash:
                return url_for("uploaded_file", filename=upload_path.name)
        except OSError:
            continue
    return None


def _derive_cached_effect_estimate(payload: dict, comparison_label: str) -> dict[str, float | str]:
    from lifelines import CoxPHFitter
    import pandas as pd

    left, right = _parse_comparison_label(comparison_label)
    canonical_groups = payload.get("canonical_reconstruction", {}).get("groups", [])
    if not isinstance(canonical_groups, list) or not canonical_groups:
        raise ValueError("Cached payload is missing canonical reconstructed groups.")

    group_records: dict[str, list[dict]] = {}
    for group in canonical_groups:
        group_name = str(group.get("group_name", "")).strip()
        records = group.get("records", [])
        if group_name:
            group_records[group_name] = records if isinstance(records, list) else []

    if left not in group_records or right not in group_records:
        raise ValueError(
            f"Cached payload does not contain both groups required by '{comparison_label}'. "
            f"Available groups: {', '.join(sorted(group_records.keys())) or 'none'}."
        )

    rows = []
    for row in group_records[left]:
        rows.append({"duration": float(row["time"]), "event": int(row["event"]), "arm_left": 1})
    for row in group_records[right]:
        rows.append({"duration": float(row["time"]), "event": int(row["event"]), "arm_left": 0})

    if not rows:
        raise ValueError("Cached payload has no reconstructed records for the requested comparison.")

    dataframe = pd.DataFrame(rows)
    cox = CoxPHFitter()
    cox.fit(dataframe, duration_col="duration", event_col="event", formula="arm_left")

    hr = float(cox.hazard_ratios_["arm_left"])
    ci_table = cox.confidence_intervals_
    lower_col = next((column for column in ci_table.columns if "lower" in str(column).lower()), None)
    upper_col = next((column for column in ci_table.columns if "upper" in str(column).lower()), None)
    if lower_col is None or upper_col is None:
        raise ValueError("Unable to read confidence interval columns from cached Cox model output.")

    ci_lower = exp(float(ci_table.loc["arm_left", lower_col]))
    ci_upper = exp(float(ci_table.loc["arm_left", upper_col]))
    _compute_log_hr_and_se(hr, ci_lower, ci_upper)
    return {"hr": hr, "ci_lower": ci_lower, "ci_upper": ci_upper, "comparison_label": f"{left} vs {right}"}


def _build_auto_logrank(payload: dict) -> dict:
    canonical = payload.get("canonical_reconstruction", {})
    canonical_groups = canonical.get("groups", [])
    if not isinstance(canonical_groups, list):
        return {"available": False, "message": "Invalid extraction payload: canonical reconstruction is missing."}

    group_summaries = []
    interval_rows = []
    record_previews = []
    groups_records = []
    quality_summary = payload.get("quality_summary", {})
    pairwise_policy = quality_summary.get("pairwise_recommendation", "show")
    reliability_notes: list[str] = []

    for group in canonical_groups:
        group_name = group.get("group_name", "")
        records = group.get("records", [])
        terminal_known = bool(group.get("terminal_risk_known", False))
        unresolved_tail_count = int(group.get("unresolved_tail_count", 0) or 0)
        if not terminal_known:
            reliability_notes.append(
                f"{group_name or 'Unnamed group'} has missing terminal risk-table cells; terminal count kept unknown."
            )
        if unresolved_tail_count > 0:
            reliability_notes.append(
                f"{group_name or 'Unnamed group'} retains {unresolved_tail_count} unresolved tail subjects."
            )

        group_summaries.append(
            {
                "group_name": group_name,
                "initial_n": group.get("initial_n"),
                "last_visible_curve_time": group.get("last_visible_curve_time"),
                "last_visible_survival": group.get("last_visible_curve_survival"),
                "number_of_visible_drops": group.get("visible_drop_count", ""),
                "number_of_inferred_overlap_drops": group.get("overlap_inferred_drop_count", ""),
                "number_of_reconstructed_events": int(group.get("event_total", 0)),
                "number_of_reconstructed_censors": int(group.get("censor_total", 0)),
                "terminal_risk_known": terminal_known,
                "unresolved_tail_count": unresolved_tail_count,
            }
        )

        interval_rows.extend(group.get("interval_rows", []))
        record_previews.append(
            {
                "group_name": group_name or "Unnamed group",
                "records": records,
                "records_preview": records[:10],
                "record_count": len(records),
            }
        )
        groups_records.append({"group_name": group_name or "Unnamed group", "records": records})

    accounting_passed = bool(canonical.get("accounting_identities_passed", False))
    interval_passed = bool(canonical.get("interval_conservation_passed", False))
    severe_ordering_issue = any("group-order contradiction" in reason for reason in quality_summary.get("figure_quality_reasons", []))
    figure_quality = str(quality_summary.get("figure_quality_grade", "high")).lower()
    pairwise_policy = "show"
    if not (accounting_passed and interval_passed) or severe_ordering_issue or figure_quality == "low":
        pairwise_policy = "suppress"

    analysis = run_logrank_analysis(
        groups_records=groups_records,
        include_pairwise=True,
        pairwise_policy=pairwise_policy,
    )
    analysis["group_summaries"] = group_summaries
    analysis["interval_rows"] = interval_rows
    analysis["record_previews"] = record_previews
    analysis["quality_summary"] = quality_summary
    analysis["reliability_notes"] = unique_ordered_strings(reliability_notes)
    analysis["canonical_reconstruction_note"] = "All tables and tests below are derived from one repaired canonical reconstruction."
    analysis["canonical_checks"] = {
        "accounting_identities_passed": accounting_passed,
        "interval_conservation_passed": interval_passed,
    }

    capped_confidence, confidence_note = cap_confidence_for_quality(
        float(payload.get("confidence", 0.0)),
        figure_quality,
    )
    analysis["display_confidence"] = capped_confidence
    analysis["confidence_note"] = confidence_note

    if analysis.get("analysis_type") == "multigroup" and analysis.get("pairwise_rows"):
        if any(row.get("significant_after_adjustment") for row in analysis["pairwise_rows"]):
            analysis["pairwise_interpretation"] = (
                "Pairwise differences with adjusted p-values below 0.05 may indicate specific group-level separation. "
                "Treat pairwise findings as exploratory.")
        else:
            analysis["pairwise_interpretation"] = (
                "No pairwise comparison remained significant after adjustment. "
                "Treat pairwise findings as exploratory.")
        if pairwise_policy == "exploratory":
            analysis["pairwise_interpretation"] = (
                "Pairwise rows are shown as low-confidence exploratory output due to extraction quality limitations."
            )
    elif analysis.get("analysis_type") == "multigroup" and pairwise_policy == "suppress":
        suppression_reason = []
        if not accounting_passed:
            suppression_reason.append("canonical accounting checks did not pass")
        if not interval_passed:
            suppression_reason.append("interval conservation checks did not pass")
        if severe_ordering_issue:
            suppression_reason.append("group-order contradiction remains unresolved")
        if figure_quality == "low":
            suppression_reason.append("overall figure quality is LOW")
        message = "Pairwise comparisons were suppressed"
        if suppression_reason:
            message += ": " + "; ".join(suppression_reason) + "."
        analysis["pairwise_interpretation"] = message
        analysis["pairwise_message"] = message

    if analysis.get("analysis_type") == "multigroup" and analysis.get("p_value", 1.0) < analysis.get("alpha", 0.05):
        analysis["overall_plain_english"] = "At least one survival curve differs from the others."

    return analysis


def unique_ordered_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _parse_comparison_label(raw_label: str) -> tuple[str, str]:
    label = " ".join(raw_label.strip().split())
    if " vs " not in label:
        raise ValueError("Comparison labels must use the format 'Treatment1 vs Treatment2'.")
    left, right = label.split(" vs ", 1)
    if not left or not right:
        raise ValueError("Comparison labels must include both sides around 'vs'.")
    return left.strip(), right.strip()


def _compute_log_hr_and_se(hr: float, ci_lower: float, ci_upper: float) -> tuple[float, float]:
    if hr <= 0 or ci_lower <= 0 or ci_upper <= 0:
        raise ValueError("HR and CI values must be positive numbers.")
    if ci_lower >= ci_upper:
        raise ValueError("Lower CI must be smaller than upper CI.")

    log_hr = log(hr)
    se = (log(ci_upper) - log(ci_lower)) / (2 * 1.96)
    if se <= 0:
        raise ValueError("Unable to compute a positive standard error from the provided CI.")
    return log_hr, se


def _normalize_to_comparator(
    left: str,
    right: str,
    log_hr: float,
    comparator: str,
) -> tuple[str, float, str]:
    if left == comparator:
        return right, -log_hr, f"{left} vs {right} -> {right} vs {left} (inverted)"
    if right == comparator:
        return left, log_hr, f"{left} vs {right} -> {left} vs {right} (already aligned)"
    raise ValueError("Unsupported direction specification: comparison does not include the shared comparator.")


def _run_anchored_indirect_comparison(paper_1: dict, paper_2: dict) -> dict:
    endpoint_1 = paper_1["endpoint_name"].strip()
    endpoint_2 = paper_2["endpoint_name"].strip()
    if endpoint_1.lower() != endpoint_2.lower():
        raise ValueError("Endpoint mismatch between Paper 1 and Paper 2.")

    left_1, right_1 = _parse_comparison_label(paper_1["comparison_label"])
    left_2, right_2 = _parse_comparison_label(paper_2["comparison_label"])

    common_treatments = {left_1, right_1}.intersection({left_2, right_2})
    if len(common_treatments) != 1:
        raise ValueError("No common comparator could be identified across the two comparisons.")
    comparator = common_treatments.pop()

    log_hr_1, se_1 = _compute_log_hr_and_se(paper_1["hr"], paper_1["ci_lower"], paper_1["ci_upper"])
    log_hr_2, se_2 = _compute_log_hr_and_se(paper_2["hr"], paper_2["ci_lower"], paper_2["ci_upper"])

    treatment_a, normalized_log_hr_1, step_1 = _normalize_to_comparator(left_1, right_1, log_hr_1, comparator)
    treatment_c, normalized_log_hr_2, step_2 = _normalize_to_comparator(left_2, right_2, log_hr_2, comparator)

    if treatment_a == treatment_c:
        raise ValueError("Unsupported direction specification: both comparisons map to the same non-comparator treatment.")

    indirect_log_hr = normalized_log_hr_1 - normalized_log_hr_2
    indirect_se = sqrt(se_1**2 + se_2**2)
    if indirect_se <= 0:
        raise ValueError("Unable to compute a positive indirect standard error.")

    z_score = indirect_log_hr / indirect_se
    p_value = erfc(abs(z_score) / sqrt(2))
    lower_log = indirect_log_hr - 1.96 * indirect_se
    upper_log = indirect_log_hr + 1.96 * indirect_se

    indirect_hr = exp(indirect_log_hr)
    ci_lower = exp(lower_log)
    ci_upper = exp(upper_log)

    interpretation = (
        f"Evidence suggests a difference between {treatment_a} and {treatment_c} (p < 0.05)."
        if p_value < 0.05
        else f"No clear difference detected between {treatment_a} and {treatment_c} (p >= 0.05)."
    )

    return {
        "endpoint_name": endpoint_1,
        "comparator": comparator,
        "treatment_a": treatment_a,
        "treatment_c": treatment_c,
        "normalization_steps": [step_1, step_2],
        "paper_1_normalized_label": f"{treatment_a} vs {comparator}",
        "paper_2_normalized_label": f"{treatment_c} vs {comparator}",
        "paper_1_normalized_log_hr": normalized_log_hr_1,
        "paper_2_normalized_log_hr": normalized_log_hr_2,
        "paper_1_se": se_1,
        "paper_2_se": se_2,
        "indirect_log_hr": indirect_log_hr,
        "indirect_se": indirect_se,
        "indirect_hr": indirect_hr,
        "indirect_ci_lower": ci_lower,
        "indirect_ci_upper": ci_upper,
        "z_score": z_score,
        "p_value": p_value,
        "interpretation": interpretation,
    }


def _build_indirect_quality_panel(paper_1: dict, paper_2: dict, effect_source_available: bool) -> dict:
    endpoint_matched = paper_1["endpoint_name"].strip().lower() == paper_2["endpoint_name"].strip().lower()

    comparator_identified = False
    directions_normalized = False
    comparator = ""
    quality_reasons: list[str] = []

    try:
        left_1, right_1 = _parse_comparison_label(paper_1["comparison_label"])
        left_2, right_2 = _parse_comparison_label(paper_2["comparison_label"])
        overlap = {left_1, right_1}.intersection({left_2, right_2})
        comparator_identified = len(overlap) == 1
        if comparator_identified:
            comparator = next(iter(overlap))
            _normalize_to_comparator(left_1, right_1, 0.0, comparator)
            _normalize_to_comparator(left_2, right_2, 0.0, comparator)
            directions_normalized = True
    except ValueError:
        comparator_identified = False
        directions_normalized = False

    if not endpoint_matched:
        quality_reasons.append("Endpoint mismatch between the two direct comparisons.")
    if not comparator_identified:
        quality_reasons.append("Shared comparator is ambiguous or missing.")
    if not directions_normalized:
        quality_reasons.append("Treatment directions could not be normalized to a shared comparator.")
    if not effect_source_available:
        quality_reasons.append("One or both direct effects are unavailable from the selected source.")

    any_cached = paper_1.get("source_mode") == "cached" or paper_2.get("source_mode") == "cached"
    if not comparator_identified or not endpoint_matched:
        quality_label = "LOW"
    elif any_cached:
        quality_label = "MEDIUM"
    else:
        quality_label = "HIGH"

    return {
        "checks": [
            {"label": "Shared comparator identified", "passed": comparator_identified},
            {"label": "Endpoint matched", "passed": endpoint_matched},
            {"label": "Treatment direction normalized", "passed": directions_normalized},
            {"label": "Effect source available", "passed": effect_source_available},
        ],
        "quality_label": quality_label,
        "comparator": comparator or "N/A",
        "reasons": quality_reasons,
        "is_exploratory": quality_label == "MEDIUM",
        "suppress_estimate": quality_label == "LOW",
    }


@app.route("/")
def home():
    return render_template(
        "index.html",
        manual_group_a="",
        manual_group_b="",
        live_extraction_enabled=_is_live_extraction_enabled(),
        demo_examples=_list_demo_examples(),
    )


@app.route("/indirect-comparison", methods=["GET", "POST"])
def indirect_comparison():
    latest_upload = session.get("latest_upload")
    latest_upload_hash = None
    if latest_upload:
        upload_path = app.config["UPLOAD_FOLDER"] / latest_upload["filename"]
        if upload_path.exists():
            latest_upload_hash = image_sha256(upload_path)
    cached_hashes = _list_cached_hashes()

    paper_1 = {
        "title": "Paper 1 title placeholder",
        "authors": "First author et al.",
        "year": "YYYY",
        "journal": "Journal name",
        "comparison_label": "A vs B",
        "endpoint_name": "Overall Survival",
        "figure_status": "No cached KM figure selected for this direct comparison.",
        "image_url": None,
        "hr": 0.82,
        "ci_lower": 0.68,
        "ci_upper": 0.99,
        "source_mode": "reported",
        "cached_hash": latest_upload_hash or "",
        "provenance_note": "Article-reported effect (HR/95% CI entered manually).",
    }
    paper_2 = {
        "title": "Paper 2 title placeholder",
        "authors": "First author et al.",
        "year": "YYYY",
        "journal": "Journal name",
        "comparison_label": "C vs B",
        "endpoint_name": "Overall Survival",
        "figure_status": "No cached KM figure selected for this direct comparison.",
        "image_url": None,
        "hr": 1.10,
        "ci_lower": 0.92,
        "ci_upper": 1.31,
        "source_mode": "reported",
        "cached_hash": "",
        "provenance_note": "Article-reported effect (HR/95% CI entered manually).",
    }

    errors: list[str] = []
    anchored_result = None
    quality_panel = _build_indirect_quality_panel(paper_1, paper_2, effect_source_available=True)

    if request.method == "POST":
        effect_source_available = True
        for paper_label, paper in (("paper_1", paper_1), ("paper_2", paper_2)):
            paper["title"] = request.form.get(f"{paper_label}_title", paper["title"]).strip() or paper["title"]
            paper["authors"] = request.form.get(f"{paper_label}_authors", paper["authors"]).strip() or paper["authors"]
            paper["year"] = request.form.get(f"{paper_label}_year", paper["year"]).strip() or paper["year"]
            paper["journal"] = request.form.get(f"{paper_label}_journal", paper["journal"]).strip() or paper["journal"]
            paper["comparison_label"] = request.form.get(
                f"{paper_label}_comparison_label", paper["comparison_label"]
            ).strip()
            paper["endpoint_name"] = request.form.get(f"{paper_label}_endpoint_name", paper["endpoint_name"]).strip()
            paper["source_mode"] = request.form.get(f"{paper_label}_source_mode", "reported").strip()
            paper["cached_hash"] = request.form.get(f"{paper_label}_cached_hash", "").strip()

            if paper["source_mode"] not in {"reported", "cached"}:
                errors.append(f"{paper_label.replace('_', ' ').title()}: unsupported source mode.")
                effect_source_available = False
                continue

            if paper["source_mode"] == "cached":
                if not paper["cached_hash"]:
                    errors.append(
                        f"{paper_label.replace('_', ' ').title()}: choose a cached extraction hash for KM-derived mode."
                    )
                    effect_source_available = False
                    continue
                cached_payload = _load_cached_extraction(paper["cached_hash"])
                if cached_payload is None:
                    errors.append(
                        f"{paper_label.replace('_', ' ').title()}: no cached extraction found for hash "
                        f"'{paper['cached_hash']}'."
                    )
                    effect_source_available = False
                    continue
                try:
                    cached_effect = _derive_cached_effect_estimate(cached_payload, paper["comparison_label"])
                    paper["hr"] = float(cached_effect["hr"])
                    paper["ci_lower"] = float(cached_effect["ci_lower"])
                    paper["ci_upper"] = float(cached_effect["ci_upper"])
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{paper_label.replace('_', ' ').title()}: {exc}")
                    effect_source_available = False
                    continue
                paper["provenance_note"] = (
                    f"Derived from cached KM reconstruction (cache hash: {paper['cached_hash']}). "
                    "No live API call was used."
                )
                paper["image_url"] = _uploaded_image_url_for_hash(paper["cached_hash"])
                if paper["image_url"]:
                    paper["figure_status"] = f"Preview linked to cached hash {paper['cached_hash']}."
                else:
                    paper["figure_status"] = (
                        f"Cached estimate loaded from hash {paper['cached_hash']}. "
                        "No matching uploaded KM image preview is available."
                    )
                continue

            numeric_fields = (
                ("hr", "HR"),
                ("ci_lower", "95% CI lower"),
                ("ci_upper", "95% CI upper"),
            )
            for field_key, field_label in numeric_fields:
                raw_value = request.form.get(f"{paper_label}_{field_key}", str(paper[field_key])).strip()
                try:
                    paper[field_key] = float(raw_value)
                except ValueError:
                    errors.append(
                        f"{paper_label.replace('_', ' ').title()} {field_label}: please enter a numeric value "
                        "(examples: 0.6, 0.82, 1, 1.25, 0.811001)."
                    )
                    effect_source_available = False
            paper["provenance_note"] = "Article-reported effect (HR/95% CI entered manually)."
            paper["image_url"] = None
            paper["figure_status"] = "No cached KM figure selected for this direct comparison."

        quality_panel = _build_indirect_quality_panel(paper_1, paper_2, effect_source_available)

        if not errors:
            if quality_panel["suppress_estimate"]:
                reasons = quality_panel["reasons"] or [
                    "Indirect estimate suppressed due to low-quality assumptions."
                ]
                errors.append(
                    "Indirect estimate suppressed (quality LOW): "
                    + " ".join(reasons)
                )
            else:
                try:
                    anchored_result = _run_anchored_indirect_comparison(paper_1, paper_2)
                    anchored_result["provenance_summary"] = (
                        f"Paper 1 source: {paper_1['provenance_note']} | "
                        f"Paper 2 source: {paper_2['provenance_note']}"
                    )
                    if quality_panel["is_exploratory"]:
                        anchored_result["quality_note"] = (
                            "Quality is MEDIUM because one or both direct effects are KM-cache derived; "
                            "interpret this indirect estimate as exploratory."
                        )
                except ValueError as exc:
                    errors.append(str(exc))

    return render_template(
        "indirect_comparison.html",
        paper_1=paper_1,
        paper_2=paper_2,
        shared_comparator=anchored_result["comparator"] if anchored_result else "Pending",
        anchored_result=anchored_result,
        indirect_errors=errors,
        cached_hashes=cached_hashes,
        quality_panel=quality_panel,
    )


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image_file" not in request.files:
        flash("No file was provided. Please choose an image file.", "error")
        return redirect(url_for("home"))

    file = request.files["image_file"]
    if not file.filename:
        flash("No file selected. Please choose an image file.", "error")
        return redirect(url_for("home"))
    if not is_allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PNG, JPG, or JPEG file.", "error")
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)
    destination = app.config["UPLOAD_FOLDER"] / filename
    file.save(destination)

    image_hash = image_sha256(destination)
    if not _is_live_extraction_enabled() and _load_cached_extraction(image_hash) is None:
        flash(
            "This public demo runs in cached-output mode only. "
            "No API key is configured, so new live extraction is disabled.",
            "error",
        )
        return redirect(url_for("home"))

    session["latest_upload"] = {"filename": filename, "content_type": file.content_type or "Unknown"}
    session["result_mode"] = "upload"
    session.pop("manual_analysis", None)
    return redirect(url_for("results"))


@app.route("/manual-logrank", methods=["POST"])
def manual_logrank():
    group_a_text = request.form.get("group_a_data", "")
    group_b_text = request.form.get("group_b_data", "")

    try:
        group_a_records = parse_survival_records(group_a_text, "Group A")
        group_b_records = parse_survival_records(group_b_text, "Group B")
        logrank_output = compute_logrank_test(group_a_records, group_b_records)
    except ValueError as exc:
        flash(str(exc), "error")
        return (
            render_template(
                "index.html",
                manual_group_a=group_a_text,
                manual_group_b=group_b_text,
                live_extraction_enabled=_is_live_extraction_enabled(),
                demo_examples=_list_demo_examples(),
            ),
            400,
        )

    session["manual_analysis"] = {
        "group_a_count": len(group_a_records),
        "group_b_count": len(group_b_records),
        "chi_square": logrank_output["chi_square"],
        "p_value": logrank_output["p_value"],
    }
    session["result_mode"] = "manual"
    session.pop("latest_upload", None)
    return redirect(url_for("results"))


@app.route("/results")
def results():
    result_mode = session.get("result_mode")
    manual_analysis = session.get("manual_analysis")
    file_metadata = session.get("latest_upload")

    if result_mode == "manual" and manual_analysis:
        return render_template(
            "results.html",
            mode="manual",
            manual_analysis=manual_analysis,
            file_metadata=None,
            image_url=None,
            metadata_output=None,
            metadata_json=None,
        )

    if result_mode == "upload" and file_metadata:
        upload_path = app.config["UPLOAD_FOLDER"] / file_metadata["filename"]
        image_hash = image_sha256(upload_path)
        extractor = KMVisionExtractor()

        extraction_source = ""
        extraction_error = None
        payload = _load_cached_extraction(image_hash)

        if payload is not None:
            extraction_source = "cached LLM response"
        elif extractor.is_configured():
            try:
                payload = extractor.extract_from_image(upload_path)
                _save_cached_extraction(image_hash, payload)
                extraction_source = "LLM API"
            except LLMExtractionError as exc:
                extraction_error = str(exc)
        else:
            extraction_error = "No OPENAI_API_KEY is configured and no cached extraction exists for this image hash."

        auto_logrank = None
        metadata_json = None
        if payload is not None:
            metadata_json = json.dumps(payload, indent=2)
            try:
                auto_logrank = _build_auto_logrank(payload)
            except (ValueError, ImportError) as exc:
                auto_logrank = {"available": False, "message": str(exc)}

        return render_template(
            "results.html",
            mode="upload",
            file_metadata=file_metadata,
            image_url=url_for("uploaded_file", filename=file_metadata["filename"]),
            metadata_output=payload,
            metadata_json=metadata_json,
            auto_logrank=auto_logrank,
            extraction_source=extraction_source,
            extraction_error=extraction_error,
            demo_mode_notice=not extractor.is_configured(),
            manual_analysis=None,
        )

    flash("Upload an image first or run a manual log-rank test.", "error")
    return redirect(url_for("home"))


@app.route("/demo/<slug>")
def demo_example(slug: str):
    payload, meta = _load_demo_payload(slug)
    if meta is None:
        flash("Unknown cached demo example.", "error")
        return redirect(url_for("home"))
    if payload is None:
        flash("Cached demo payload is missing in this deployment package.", "error")
        return redirect(url_for("home"))

    try:
        auto_logrank = _build_auto_logrank(payload)
    except (ValueError, ImportError) as exc:
        auto_logrank = {"available": False, "message": str(exc)}

    return render_template(
        "results.html",
        mode="upload",
        file_metadata={"filename": f"{meta['slug']} (precomputed demo)"},
        image_url=None,
        metadata_output=payload,
        metadata_json=json.dumps(payload, indent=2),
        auto_logrank=auto_logrank,
        extraction_source="cached demo payload",
        extraction_error=None,
        demo_mode_notice=True,
        manual_analysis=None,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
