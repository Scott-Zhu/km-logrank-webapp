# KM Log-rank Web App

A minimal Flask starter app for a beginner-friendly course project.

## What this project does

This project now includes two workflows:
- **Manual survival-analysis mode** where users can paste two groups of survival records
- **Image upload mode** with an OCR + curve-digitization prototype for Kaplan-Meier figures

### Manual survival-analysis mode
- Backend parsing and validation for survival text input (`time,event` per row)
- Two-group log-rank test computation in Python
- Results page output for:
  - Log-rank test statistic (chi-square)
  - p-value
  - Group record counts

### Image upload mode (prototype)
- Upload a PNG/JPG/JPEG Kaplan-Meier figure
- Run OCR for metadata text (title/axis/legend/number-at-risk-like lines)
- Run a **heuristic curve digitization step** that:
  - Detects an approximate plot area
  - Extracts approximate normalized `(time, survival_probability)` points for visible curves
  - Returns structured JSON and renders extracted curves on the results page

## Important limitations

- The digitization is intentionally approximate and heuristic.
- Output points are in normalized units `[0,1]` unless future calibration is added.
- This tool **does not reconstruct patient-level data or exact event times**.
- Complex layouts (overlapping curves, faint lines, heavy grids, similar colors) can reduce reliability.

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
├── metadata_extraction.py
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

- `GET /` → homepage with manual analysis and upload forms
- `POST /manual-logrank` → parses manual text input, validates records, computes log-rank test
- `POST /upload` → validates file, saves file, stores metadata in session, redirects to results
- `GET /results` → shows manual results or upload extraction output
- `GET /uploads/<filename>` → serves uploaded image files

## Notes

- This app is intentionally simple and beginner-focused.
- Uploaded files are saved to the local `uploads/` folder.
- Curve digitization is a prototype intended for rough visual trace extraction.
