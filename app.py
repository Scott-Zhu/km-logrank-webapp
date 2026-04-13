from flask import Flask, render_template

# Create the Flask application object.
app = Flask(__name__)


@app.route("/")
def home():
    """Render the homepage."""
    return render_template("index.html")


if __name__ == "__main__":
    # debug=True reloads automatically while you are developing.
    app.run(debug=True)
