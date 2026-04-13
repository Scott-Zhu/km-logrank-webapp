from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
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

    flash(f"Upload successful: {filename}", "success")
    return redirect(url_for("home"))


if __name__ == "__main__":
    # debug=True reloads automatically while you are developing.
    app.run(debug=True)
