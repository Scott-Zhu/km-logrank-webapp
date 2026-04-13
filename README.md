# KM Log-rank Web App

A minimal Flask starter app for a beginner-friendly course project.

## What this project does

Right now, this project only includes a clean homepage at `/`.
Analysis features (like KM extraction or log-rank calculations) are intentionally **not implemented yet**.

## Project structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── static/
│   └── styles.css
└── templates/
    └── index.html
```

## Setup

1. (Optional but recommended) create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the app

```bash
python app.py
```

Then open your browser to:

- <http://127.0.0.1:5000/>

## Notes

- This is a starter template to keep the project simple.
- Add analysis routes and logic later as separate steps.
