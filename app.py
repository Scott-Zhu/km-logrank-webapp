from math import erfc, sqrt
from pathlib import Path
import json

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

from metadata_extraction import extract_figure_metadata
from survival_reconstruction import infer_initial_group_sizes, reconstruct_group_records

# Create the Flask application object.
app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-key"
app.config["UPLOAD_FOLDER"] = Path("uploads")
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Ensure upload destination exists.
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename: str) -> bool:
    """Return True when filename has an allowed image extension."""
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in app.config["ALLOWED_EXTENSIONS"]


def parse_survival_records(raw_text: str, group_label: str) -> list[dict[str, float | int]]:
    """Parse manual survival input.

    Expected format:
      - One record per line
      - Two values per line: time and event indicator
      - Allowed separators: comma or whitespace
      - Event indicator must be 0 (censored) or 1 (event)

    Example:
      5,1
      8,0
      12 1
    """
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
                f"{group_label}, line {line_index}: expected exactly two values "
                "(time and event)."
            )

        time_text, event_text = parts

        try:
            time_value = float(time_text)
        except ValueError as exc:
            raise ValueError(
                f"{group_label}, line {line_index}: time must be a number."
            ) from exc

        if time_value < 0:
            raise ValueError(
                f"{group_label}, line {line_index}: time must be non-negative."
            )

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
    """Compute a two-group log-rank test (chi-square with 1 degree of freedom)."""
    all_records = group_a_records + group_b_records
    event_times = sorted(
        {
            float(record["time"])
            for record in all_records
            if int(record["event"]) == 1
        }
    )

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

        if n_total > 1:
            variance_at_time = (
                (n_a * n_b * d_total * (n_total - d_total))
                / (n_total**2 * (n_total - 1))
            )
        else:
            variance_at_time = 0.0

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

    return {
        "chi_square": chi_square,
        "p_value": p_value,
    }


@app.route("/")
def home():
    """Render the homepage."""
    return render_template(
        "index.html",
        manual_group_a="",
        manual_group_b="",
    )


@app.route("/upload", methods=["POST"])
def upload_image():
    """Handle image uploads from the homepage form."""
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

    file_metadata = {
        "filename": filename,
        "content_type": file.content_type or "Unknown",
    }
    # Keep upload and manual analysis workflows separate in session state.
    session["latest_upload"] = file_metadata
    session["result_mode"] = "upload"

    # Clear stale manual output so the upload flow cannot render manual results.
    session.pop("manual_analysis", None)

    return redirect(url_for("results"))


@app.route("/manual-logrank", methods=["POST"])
def manual_logrank():
    """Handle manual survival data input and compute a two-group log-rank test."""
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
            ),
            400,
        )

    # Keep manual and upload flows separate; mark this result as manual.
    session["manual_analysis"] = {
        "group_a_count": len(group_a_records),
        "group_b_count": len(group_b_records),
        "chi_square": logrank_output["chi_square"],
        "p_value": logrank_output["p_value"],
    }
    session["result_mode"] = "manual"

    # Clear stale upload metadata so manual flow cannot render upload output.
    session.pop("latest_upload", None)

    return redirect(url_for("results"))


@app.route("/results")
def results():
    """Show analysis output from upload mode or manual log-rank mode."""
    # Explicitly route based on the last workflow action so the two forms
    # cannot accidentally show each other's output.
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
        metadata_output = extract_figure_metadata(upload_path)
        metadata_json = json.dumps(metadata_output, indent=2)
        image_url = url_for("uploaded_file", filename=file_metadata["filename"])
        auto_logrank = None

        curves = metadata_output.get("curve_extraction", {}).get("curves", [])
        curve_extraction = metadata_output.get("curve_extraction", {})
        confidence = float(curve_extraction.get("confidence", 0.0))
        axis_status = str(curve_extraction.get("axis_calibration", {}).get("status", "failed"))
        valid_curve_count = int(curve_extraction.get("valid_curve_count", len(curves) if isinstance(curves, list) else 0))

        if isinstance(curves, list) and valid_curve_count >= 2:
            initial_sizes, size_source, used_default_sizes = infer_initial_group_sizes(metadata_output, 2)

            curve_a = curves[0]
            curve_b = curves[1]
            group_a = reconstruct_group_records(
                str(curve_a.get("name", "curve_1")),
                list(curve_a.get("points", [])),
                initial_sizes[0],
            )
            group_b = reconstruct_group_records(
                str(curve_b.get("name", "curve_2")),
                list(curve_b.get("points", [])),
                initial_sizes[1],
            )

            sanity_failures: list[str] = []
            if confidence < 0.65:
                sanity_failures.append("extraction confidence is low")
            if axis_status != "ok":
                sanity_failures.append("axis calibration failed")
            if used_default_sizes or len(initial_sizes) < 2:
                sanity_failures.append("reliable group sizes were unavailable")

            if sanity_failures:
                auto_logrank = {
                    "available": False,
                    "warning": (
                        "Approximate automatic mode detected curves but did not run formal "
                        "hypothesis testing because quality checks failed."
                    ),
                    "message": "Log-rank skipped: " + ", ".join(sanity_failures) + ".",
                    "confidence": confidence,
                }
            elif group_a.records and group_b.records:
                try:
                    logrank_output = compute_logrank_test(group_a.records, group_b.records)
                    auto_logrank = {
                        "available": True,
                        "warning": (
                            "Approximate automatic mode: records are reconstructed from "
                            "digitized figure curves, not patient-level source data."
                        ),
                        "group_a_name": group_a.name,
                        "group_b_name": group_b.name,
                        "group_a_count": len(group_a.records),
                        "group_b_count": len(group_b.records),
                        "group_a_events": group_a.event_count,
                        "group_b_events": group_b.event_count,
                        "group_a_censored": group_a.censor_count,
                        "group_b_censored": group_b.censor_count,
                        "initial_size_source": size_source,
                        "group_a_initial_n": group_a.initial_n,
                        "group_b_initial_n": group_b.initial_n,
                        "confidence": confidence,
                        "chi_square": logrank_output["chi_square"],
                        "p_value": logrank_output["p_value"],
                    }
                except ValueError as exc:
                    auto_logrank = {
                        "available": False,
                        "warning": (
                            "Approximate automatic mode failed to produce a stable log-rank "
                            "result from this figure."
                        ),
                        "message": str(exc),
                    }
            else:
                auto_logrank = {
                    "available": False,
                    "warning": (
                        "Approximate automatic mode could not reconstruct enough records "
                        "from extracted curves."
                    ),
                    "message": "Partial extraction detected. Try manual mode for validated inputs.",
                }
        elif isinstance(curves, list):
            single_group_hint = "Single-group figure detected." if valid_curve_count == 1 else "Insufficient curve groups detected."
            auto_logrank = {
                "available": False,
                "warning": (
                    "Approximate automatic mode requires at least two validated curves."
                ),
                "message": f"{single_group_hint} Log-rank analysis was skipped.",
                "confidence": confidence,
            }

        return render_template(
            "results.html",
            mode="upload",
            file_metadata=file_metadata,
            image_url=image_url,
            metadata_output=metadata_output,
            metadata_json=metadata_json,
            metadata_data=metadata_output,
            auto_logrank=auto_logrank,
            manual_analysis=None,
        )

    flash("Upload an image first or run a manual log-rank test.", "error")
    return redirect(url_for("home"))


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    """Serve uploaded files so they can be displayed on the results page."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    # debug=True reloads automatically while you are developing.
    app.run(debug=True)
