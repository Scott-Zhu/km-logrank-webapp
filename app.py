from pathlib import Path

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


def build_placeholder_analysis(file_metadata: dict[str, str]) -> dict[str, str]:
    """Return placeholder analysis data for the results page."""
    return {
        "status": "Placeholder only",
        "summary": (
            "Automatic Kaplan-Meier extraction is not implemented yet. "
            "This panel will show parsed values in a future step."
        ),
        "input_filename": file_metadata["filename"],
    }


@app.route("/")
def home():
    """Render the homepage."""
    return render_template("index.html")


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


@app.route("/results")
def results():
    """Show the uploaded image and placeholder analysis output."""
    file_metadata = session.get("latest_upload")
    if not file_metadata:
        flash("Upload an image first to view results.", "error")
        return redirect(url_for("home"))

    analysis_output = build_placeholder_analysis(file_metadata)
    image_url = url_for("uploaded_file", filename=file_metadata["filename"])

    return render_template(
        "results.html",
        file_metadata=file_metadata,
        image_url=image_url,
        analysis_output=analysis_output,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    """Serve uploaded files so they can be displayed on the results page."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    # debug=True reloads automatically while you are developing.
    app.run(debug=True)
