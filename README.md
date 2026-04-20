# KM Log-rank Web App

## Overview
This project is a Flask web app for Kaplan-Meier (KM) survival analysis workflows used in a course project submission. It supports both manual survival input and image-based KM extraction, then presents log-rank testing and related outputs in a grader-friendly interface.

For uploaded figures, extraction is LLM-assisted and reconstruction-based, so image-derived results should be treated as **approximate** (not exact patient-level ground truth).

## Current Features

### 1) Manual Survival Analysis
- Run a two-group log-rank test directly from pasted records.
- Accepted row formats:
  - `time,event`
  - `time event`
- Manual mode is fully independent from upload mode.

### 2) Upload KM Figure Analysis (LLM-assisted)
- Upload PNG/JPG/JPEG Kaplan-Meier figures.
- The app uses LLM vision extraction, reconstructs records, and computes KM/log-rank outputs from the extracted structure.
- Upload analysis supports multi-group reconstruction display; pairwise/group-level analysis is surfaced from reconstructed groups when available.

### 3) Cached-output Public Demo Mode
- Public deployment is configured as a cached-output demo.
- Public users can open precomputed demo outputs without providing an API key.
- Live extraction is disabled in the public deployment.

### 4) Final Demo Examples on the Site
- **Two-group KM example** (cached demo)
- **Three-group KM example** (cached demo)
- **Indirect comparison across related papers** (additional / extra-credit function)

## Public Demo Deliverables
The public site includes final demo deliverables for:
- Two-group KM example
- Three-group KM example
- Indirect comparison example

For the KM examples, demo pages show both:
- the original Kaplan-Meier figure, and
- the cached processed result used for demonstration.

## Local Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables for local live extraction only:
   ```bash
   cp .env.example .env
   ```
   Then set:
   ```env
   OPENAI_API_KEY=your_key_here
   ```

> `OPENAI_API_KEY` is only needed when you want to run **live local extraction**.

## Run Locally
```bash
python app.py
```
Then open: <http://127.0.0.1:5000/>

## Public Demo / No-key Behavior
- The public deployment is a **cached-output demo**.
- **No API key is required** for public viewing of demo results.
- Live extraction is intentionally disabled in the public environment.
- The indirect comparison module remains accessible as part of the final demo.

## Deployment Note (Render)
Deploy as a Flask web service on Render:
- Build command:
  ```bash
  pip install -r requirements.txt
  ```
- Start command:
  ```bash
  gunicorn app:app
  ```
- Expose the app via the Render public URL for grading/demo access.
- Do **not** place API keys in the public demo deployment.

## Project Structure
Only listing items present in this repository:

```text
.
├── app.py
├── llm_extraction.py
├── metadata_extraction.py
├── survival_reconstruction.py
├── requirements.txt
├── Procfile
├── README.md
├── .env.example
├── demo_cache/
├── cache/
├── uploads/
├── static/
│   ├── styles.css
│   └── demo/
├── templates/
│   ├── index.html
│   ├── results.html
│   └── indirect_comparison.html
├── manual_parser_smoke.py
└── post_reconstruction_smoke.py
```

## Notes
- API keys are environment-only and should not be committed.
- Public deployment is cached-output only.
- Manual mode and upload mode are separate workflows.
- Indirect comparison is the additional / extra-credit function in the final demo.
