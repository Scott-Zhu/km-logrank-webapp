# KM Log-rank Web App

A beginner-friendly Flask course project that now uses an **LLM vision API** to extract approximate Kaplan-Meier (KM) information from uploaded figures, then computes a two-group log-rank test from reconstructed records.

## What's new in this upgrade

The previous OCR/heuristic-only prototype has been refactored so upload analysis now uses the OpenAI Python SDK (Responses API with image input).

Current workflows:
- **Manual mode**: paste survival records directly (`time,event`) and run log-rank.
- **Upload mode (LLM API)**: upload a KM image, run LLM-based structured extraction, reconstruct estimated records, then compute log-rank when two groups are available.

## Important disclaimer

All image-derived outputs are **approximate** because they are inferred from plotted curves, not source patient-level time-to-event data.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables (never commit your real key):

   ```bash
   cp .env.example .env
   # then edit .env
   ```

   `.env` should contain:

   ```env
   OPENAI_API_KEY=your_key_here
   ```

   You can also export directly in your shell instead of using `.env`.

## Run the app

```bash
python app.py
```

Open <http://127.0.0.1:5000/>.

Production (Railway/Gunicorn):

```bash
gunicorn -b 0.0.0.0:$PORT app:app
```

## Upload-mode behavior (LLM + cache)

- The app hashes each uploaded image (SHA-256).
- Cache files are stored in `cache/<image_hash>.json` after successful extractions.
- If the same image is uploaded again, cached JSON is reused and the page labels the source as **cached LLM response**.
- If `OPENAI_API_KEY` is missing:
  - cached output is still used when available,
  - otherwise the app shows a clear message that no key is configured and no cached run exists.

## Safe class demo without exposing keys

Recommended assignment demo sequence:

1. Run once locally with `OPENAI_API_KEY` set and upload your KM image.
2. Confirm a new JSON file appears in `cache/`.
3. Stop app, unset key (or remove from environment), restart app.
4. Upload the same image again and show the results page note: **cached LLM response**.
5. Capture screenshots of:
   - home page (manual + upload modes),
   - upload results with extraction source,
   - structured JSON block,
   - reconstructed records + chi-square/p-value,
   - confidence and warnings.

This demonstrates API-based extraction and caching while keeping keys out of screenshots and the UI.

## Railway deployment note (public grading demo)

- Deploy this repository to Railway.
- Use the start command: `gunicorn -b 0.0.0.0:$PORT app:app`.
- Generate a public Railway domain for grader access.
- Do **not** include `OPENAI_API_KEY` in the submitted/public demo deployment.
- Public demo behavior is cached-output only (precomputed cached examples + existing cache hits).

## Project structure

```text
.
├── app.py
├── llm_extraction.py
├── metadata_extraction.py
├── survival_reconstruction.py
├── requirements.txt
├── .env.example
├── cache/
├── uploads/
├── static/
│   └── styles.css
└── templates/
    ├── index.html
    └── results.html
```

## Notes

- API keys are read from environment variables only.
- The website never asks users to paste keys into forms.
- Manual mode remains available and independent from upload mode.
