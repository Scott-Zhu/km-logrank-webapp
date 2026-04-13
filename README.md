# KM Log-rank Web App

A minimal Flask starter app for a beginner-friendly course project.

## What this project does

Right now, this project includes:
- A clean homepage at `/`
- A manual survival-analysis mode where users can paste two groups of survival records
- Backend parsing and validation for survival text input (`time,event` per row)
- A two-group log-rank test computation in Python
- A results page at `/results` that shows:
  - Manual log-rank test statistic (chi-square)
  - p-value
  - Group record counts
- A simple image upload form (PNG/JPG/JPEG) for future automatic extraction work

## Manual input format

Each group is entered in a text box:
- One record per line
- Two values per line: `time,event` (or `time event`)
- `event=1` means event occurred
- `event=0` means censored

Example:

```text
5,1
8,0
12,1
```

## Project structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── uploads/
├── static/
│   └── styles.css
└── templates/
    ├── index.html
    └── results.html
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

## Route overview

- `GET /` → homepage with manual analysis form and upload form
- `POST /manual-logrank` → parses manual text input, validates records, computes log-rank test
- `POST /upload` → validates file, saves file, stores metadata in session, redirects to results
- `GET /results` → shows manual analysis output (if available) or upload placeholder output
- `GET /uploads/<filename>` → serves uploaded image files

## Notes

- This app is intentionally simple and beginner-focused.
- Uploaded files are saved to the local `uploads/` folder.
- Automatic KM extraction is still a placeholder for a future step.
