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
    session["latest_upload"] = file_metadata

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

    session["manual_analysis"] = {
        "group_a_count": len(group_a_records),
        "group_b_count": len(group_b_records),
        "chi_square": logrank_output["chi_square"],
        "p_value": logrank_output["p_value"],
    }

    return redirect(url_for("results"))


@app.route("/results")
def results():
    """Show analysis output from upload mode or manual log-rank mode."""
    manual_analysis = session.get("manual_analysis")
    file_metadata = session.get("latest_upload")

    if manual_analysis:
        return render_template(
            "results.html",
            mode="manual",
            manual_analysis=manual_analysis,
            file_metadata=None,
            image_url=None,
            analysis_output=None,
        )

    if not file_metadata:
        flash("Upload an image first or run a manual log-rank test.", "error")
        return redirect(url_for("home"))

    upload_path = app.config["UPLOAD_FOLDER"] / file_metadata["filename"]
    metadata_output = extract_figure_metadata(upload_path)
    metadata_json = json.dumps(metadata_output, indent=2)
    image_url = url_for("uploaded_file", filename=file_metadata["filename"])

    return render_template(
        "results.html",
        mode="upload",
        file_metadata=file_metadata,
        image_url=image_url,
        metadata_output=metadata_output,
        metadata_json=metadata_json,
        manual_analysis=None,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    """Serve uploaded files so they can be displayed on the results page."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    # debug=True reloads automatically while you are developing.
    app.run(debug=True)
