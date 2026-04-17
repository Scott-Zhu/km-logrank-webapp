from math import erfc, sqrt
from pathlib import Path
import json

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

from llm_extraction import KMVisionExtractor, LLMExtractionError, image_sha256

# Load local .env values if present (safe for local development).
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-key"
app.config["UPLOAD_FOLDER"] = Path("uploads")
app.config["CACHE_FOLDER"] = Path("cache")
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
app.config["CACHE_FOLDER"].mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in app.config["ALLOWED_EXTENSIONS"]


def parse_survival_records(raw_text: str, group_label: str) -> list[dict[str, float | int]]:
    lines = raw_text.splitlines()
    records: list[dict[str, float | int]] = []

    for line_index, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        normalized = line.replace(",", " ")
        parts = [part for part in normalized.split() if part]
        if len(parts) != 2:
            raise ValueError(
                f"{group_label}, line {line_index}: expected exactly two values (time and event)."
            )

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


def _cache_path_for_hash(image_hash: str) -> Path:
    return app.config["CACHE_FOLDER"] / f"{image_hash}.json"


def _load_cached_extraction(image_hash: str) -> dict | None:
    cache_path = _cache_path_for_hash(image_hash)
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _save_cached_extraction(image_hash: str, payload: dict) -> None:
    _cache_path_for_hash(image_hash).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_auto_logrank(payload: dict) -> dict:
    groups = payload.get("groups", [])
    if not isinstance(groups, list) or len(groups) < 2:
        return {"available": False, "message": "At least two groups are required for a two-group log-rank test."}
    if len(groups) > 2:
        return {"available": False, "message": "More than two groups were extracted; this app runs two-group log-rank only."}

    group_a = groups[0]
    group_b = groups[1]
    records_a = group_a.get("estimated_records", [])
    records_b = group_b.get("estimated_records", [])
    if not records_a or not records_b:
        return {"available": False, "message": "One or both groups have no reconstructed estimated records."}

    output = compute_logrank_test(records_a, records_b)
    return {
        "available": True,
        "group_a_name": group_a.get("name", "Group A"),
        "group_b_name": group_b.get("name", "Group B"),
        "group_a_count": len(records_a),
        "group_b_count": len(records_b),
        "group_a_records": records_a,
        "group_b_records": records_b,
        "group_a_last_visible_curve_time": group_a.get("last_visible_curve_time"),
        "group_b_last_visible_curve_time": group_b.get("last_visible_curve_time"),
        "group_a_interval_summary": group_a.get("interval_summary", []),
        "group_b_interval_summary": group_b.get("interval_summary", []),
        "chi_square": output["chi_square"],
        "p_value": output["p_value"],
    }


@app.route("/")
def home():
    return render_template("index.html", manual_group_a="", manual_group_b="")


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
        return render_template("index.html", manual_group_a=group_a_text, manual_group_b=group_b_text), 400

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
            except ValueError as exc:
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
            manual_analysis=None,
        )

    flash("Upload an image first or run a manual log-rank test.", "error")
    return redirect(url_for("home"))


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
