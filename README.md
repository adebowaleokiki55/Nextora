# Nextora — Experimental AI-Powered Malaria Blood Smear Analysis

Nextora is a machine learning application that classifies blood cell images into parasitized and uninfected classes and provides experimental candidate-region analysis for whole blood-smear images.

The project combines a PyTorch image classification model with a FastAPI backend and a React frontend to make malaria-related image analysis accessible through a web interface.

**Live Demo:** https://nextora-zeta.vercel.app/

> **Important:** Nextora is an experimental research and educational prototype, not a clinically validated diagnostic tool. Its predictions must not be used as a substitute for professional medical assessment or laboratory confirmation.

## Features

- **Single-cell classification:** Upload a blood cell image and receive a model prediction.
- **Whole-smear analysis:** Experimentally identifies candidate regions in a larger blood-smear image and classifies image crops.
- **Visual analysis:** Displays an annotated image showing candidate regions detected by the analysis pipeline.
- **Prediction confidence:** Presents the model's confidence scores where available.
- **Interactive web interface:** Upload images and inspect analysis results through a React-based frontend.
- **REST API:** Uses FastAPI to expose prediction and smear-analysis endpoints.
- **Deep learning:** Uses a custom PyTorch model, `PlasmoSENet`, for two-class image classification.

## How It Works

### 1. Single-cell classification

1. The user uploads a blood cell image.
2. The image is resized to 224 × 224 pixels.
3. The image is converted to a tensor and normalized using ImageNet mean and standard deviation values.
4. The trained `PlasmoSENet` model generates a prediction.
5. The application returns the predicted class and available confidence scores.

### 2. Whole-smear analysis

1. The user uploads a larger blood-smear image.
2. The backend uses OpenCV to find candidate cell regions: adaptive threshold, morphological opening, contours, then a distance-transform watershed that splits touching or overlapping cells. Area and shape filters drop debris.
3. Candidate regions are cropped from the original image.
4. The existing classification model predicts a class for each crop.
5. The backend returns the candidate-region results, counts, and an annotated image.

### 3. Environmental and historical surveillance (Module 1)

Weekly data for the 20 Lagos local government areas, 2016 to the latest full week:

| Data | Origin | Source |
|---|---|---|
| Rainfall, mean/max temperature, relative humidity | **REAL** (reanalysis) | Open-Meteo Historical Weather API, ERA5/ERA5-Land reanalysis, at each LGA's approximate centre point. Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0). Reanalysis is modelled from observations; it is not a rain-gauge record. |
| Weekly confirmed malaria cases | **SYNTHETIC** | Public weekly case data per LGA is not available, so cases are generated from the real weather (rainfall 3–8 weeks earlier, temperature), a slowly drifting transmission intensity, occasional multi-week outbreaks in the history (none in the final 26 weeks), and a scripted demo rise in Ikorodu over the latest 6 weeks. See `backend/app/core/surveillance/synthetic_cases.py`. |
| Verified lab-positive smears | From Module 5 | Only results verified by a qualified reviewer. |

For each location, the surveillance view compares the latest weeks with the same time of year in earlier years: cases against the seasonal baseline, the 4-week case trend against the usual seasonal trend, 8-week rainfall against normal, temperature suitability (context only) and verified lab positives. A signal is raised only when the change is unusual for that series *and* large in relative and absolute terms. These levels describe current conditions; forecasts and risk scores come from the prediction engine (Module 2).

The weather file is committed in `data/real/`. To refresh it (about 3–4 minutes, no API key):

```powershell
python -m backend.scripts.fetch_weather
```

Then reset the demo data (`POST /api/demo/seed`). Without the file, the app falls back to synthetic weather and labels it SYNTHETIC.

### 4. Malaria prediction engine (Module 2)

Forecasts weekly confirmed cases 1–4 weeks ahead for every LGA and turns them into risk scores and early-warning alerts.

- **Model:** one XGBoost regressor per horizon, pooled across LGAs, predicting log(cases + 1) from recent case levels, seasonal references, rainfall lags (0–9 weeks), temperature, humidity and week of year. Every feature for a forecast made in week *t* uses only data up to week *t*.
- **Split (time-based, no shuffling):** train on target weeks up to 2023, validation on 2024 (early stopping, uncertainty, alert cut-off), test on 2025 to November 2026 (reported numbers). The serving model is then refit on all weeks.
- **Alert threshold:** the classic malaria epidemic threshold, mean + 2 SD of weekly cases in the same weeks (±2) of earlier years.
- **Risk score:** the highest estimated probability (×100) that cases reach the threshold in the next 1–4 weeks, from the model's out-of-sample validation errors. An early warning is raised when that probability passes a per-horizon cut-off chosen on the validation year.

**Test-period results** (20 LGAs, 1,940 forecasts per horizon, 73 actual threshold crossings). MAE / RMSE in cases per week; lower is better:

| Weeks ahead | Model | Persistence | Seasonal naive | Historical mean | Model precision / recall | Persistence precision / recall |
|---|---|---|---|---|---|---|
| 1 | **16.4 / 25.4** | 20.0 / 30.0 | 29.9 / 44.8 | 24.3 / 36.3 | 0.43 / 0.27 | 0.39 / 0.37 |
| 2 | **17.1 / 26.5** | 20.7 / 29.8 | 29.9 / 44.8 | 24.3 / 36.3 | 0.30 / 0.26 | 0.37 / 0.37 |
| 3 | **17.9 / 27.8** | 22.5 / 33.0 | 29.9 / 44.8 | 24.3 / 36.3 | 0.25 / 0.19 | 0.30 / 0.32 |
| 4 | **18.6 / 29.4** | 24.5 / 36.1 | 29.9 / 44.8 | 24.3 / 36.3 | 0.24 / 0.22 | 0.22 / 0.26 |

What this shows, honestly:

- The model predicts case numbers more accurately than all three baselines at every horizon.
- For the yes/no question "will cases cross the threshold?", the simple persistence baseline is as good as or better than the model at 1–3 weeks. Detection of crossings is modest for both.
- **These are results on SYNTHETIC case data.** They show how well the pipeline recovers patterns built into the generator, not real-world forecasting accuracy.

Retrain after changing the data (about 30 seconds):

```powershell
python -m ml.train_forecast
```

This writes `ml/models/forecast/` (serving models), `ml/metrics/forecast_metrics.json` and `ml/metrics/forecast_backtest.csv`. Seeding demo data runs the forecasts and raises or resolves `RISK_FORECAST` alerts.

### 5. Medical supply and facility intelligence (Module 4)

Tracks malaria RDT kits, artemether-lumefantrine (ACT) courses and injectable artesunate at every facility. Stock and consumption in the demo are **SYNTHETIC**: consumption is derived from each LGA's cases, and three Ikorodu health centres are scripted short of RDT kits.

All quantities are computed by deterministic code (`backend/app/core/supply/calculations.py`); agents only read and explain them:

- **Expected weekly demand** = average consumption over the last 8 weeks × the location's forecast ratio (next 4 weeks of forecast cases ÷ last 8 weeks of cases), bounded 0.5–3.0. **Precaution:** where the malaria risk score is 50 or more, planned demand never drops below recent consumption.
- **Status** by weeks of stock left: STOCKOUT (none), CRITICAL (< 1 week), LOW (< 2 weeks), OK, SURPLUS (> 10 weeks).
- **Shortage quantity** restores 6 weeks of stock, rounded up to whole packs.
- **Redistribution:** most urgent facility first (worst status, then higher area risk), nearest donor first, within 40 km. Donors keep at least 6 weeks of their own demand and give only whole packs. Anything not covered becomes an order to the central medical store.
- **Alerts:** `SUPPLY_SHORTAGE` alerts are raised, updated and resolved by each scan. Severity is raised one level in areas with a risk score of 50 or more.
- **Nothing moves without approval.** The transfer function re-checks the donor's stock at the moment of execution and refuses if the plan is out of date. It is only called after a human approves a proposal in the command centre.

### 6. AI command centre (Module 3)

Five agents coordinate the response for a selected location:

| Agent | Tools (deterministic Python, scoped to the location) | Can propose |
|---|---|---|
| **Commander** | situation snapshot; `delegate_to_surveillance/supply/outreach/policy` | — |
| **Surveillance** | surveillance signals, risk forecast, verified lab summary, open alerts | — |
| **Supply** | shortages, redistribution plan | stock transfers and central orders (quantities come from the plan, never from the model) |
| **Outreach** | outreach context with approved key messages | one public message draft (40–480 characters) |
| **Policy** | situation snapshot, open alerts | — (writes the situation report) |

- **Human approval:** agents can only create PENDING proposals. A named person approves or rejects each one. Approving a transfer executes it, re-checking the donor's stock first; if the plan has gone stale, the proposal ends FAILED and nothing moves. Approved orders and messages are recorded or queued only: no real order or SMS is sent in the demo.
- **Live feed:** every thought, tool call, tool result, hand-off, final answer and error is written to `agent_messages` and streamed with Server-Sent Events.
- **Modes:** `live` uses the configured AI provider (any provider; see `.env.example`). `offline` is the **OFFLINE DEMO MODE**: a scripted sequence that calls the same real tools, with every message labelled as scripted. It needs no API key or internet. `auto` (the default) picks live when a provider is configured, otherwise offline. A failed live run is recorded as FAILED; it does not silently switch to the script.
- When a run completes, the location's open alerts are marked ACKNOWLEDGED.

### 7. Command-centre dashboard

Open `http://localhost:5173/?view=command` (also linked from the home page). One screen brings the five modules together:

- **Headline numbers:** highest-risk LGA, early warnings, unresolved alerts, supply shortages, proposals awaiting approval, lab results awaiting review.
- **Map (Leaflet + OpenStreetMap):** LGAs coloured by forecast risk and sized by next week's forecast cases; facilities shown red when short of stock. Click a circle to inspect it.
- **Location panel (Plotly):** reported cases, the 1–4 week forecast with its 80% range and alert threshold, past 1-week forecasts from the backtest, surveillance signals, rainfall against cases, and facility stock.
- **AI command centre:** choose a location and mode (Auto, Live AI or Offline demo) and press **Run response**. Agent steps stream in live, followed by the Policy Agent's situation report. Offline runs are clearly badged as scripted.
- **Approval queue:** enter your name, then **Approve** or **Reject** each proposal. Approving a transfer moves the stock; outcomes (executed, failed, rejected) are listed.
- **Alerts** from every module (acknowledge or resolve), **forecast accuracy vs baselines**, **data sources** (REAL or SYNTHETIC), and links to the lab review queue.
- **Seed demo data** resets everything to the synthetic Lagos scenario (about 20 seconds).

SYNTHETIC badges appear on every chart, signal, alert and table that uses synthetic data, and a banner states what is synthetic. The dashboard needs internet for map tiles; everything else is served by the local backend.

### 8. Laboratory review workflow

1. After a whole-smear analysis, the lab worker chooses **Submit for review**, picking the laboratory, sample date and (optionally) a slide number. No patient names or identifiers are collected.
2. The submission is stored as **PENDING_REVIEW**. Uploads are labelled **SYNTHETIC** unless the submitter ticks that it is a real sample from that laboratory.
3. In the **Lab review queue** (`/?view=lab-review`), a qualified laboratory professional examines the slide and either verifies it with their own reading (positive or negative) or rejects it with a reason.
4. Only verified submissions become surveillance observations for that location (`lab_smears_reviewed`, `lab_smears_positive`). A verified positive also raises an alert for the command centre. The reviewer's reading is recorded, not the model's.

**Important limitation:** Candidate regions are not guaranteed to correspond to individual cells. The whole-smear pipeline is experimental, and classifying crops from full-smear images may produce different results from classifying isolated cell images.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Image processing | OpenCV |
| Machine learning utilities | scikit-learn |
| Backend API | FastAPI |
| Database | SQLAlchemy: SQLite locally, PostgreSQL in deployment |
| AI agents | Any provider: Anthropic Claude, OpenAI-compatible APIs, or Ollama |
| Frontend | React, Vite; command centre uses Tailwind CSS, Leaflet (react-leaflet) and Plotly |
| Frontend styling | CSS |
| Frontend hosting | Vercel |
| Backend hosting | Render |

## Architecture

```text
                 User
                  |
                  v
       React + Vite Frontend
                  |
                  | HTTP requests
                  v
           FastAPI Backend
             /         \
            v           v
     Single-cell    Whole-smear
     prediction      analysis
            |           |
            v           v
        PlasmoSENet + OpenCV
                  |
                  v
       Prediction and image results
                  |
                  v
          Frontend display
```

## Model Information

The application uses `PlasmoSENet`, a PyTorch image classification model configured for two classes.

| Class ID | Class label |
|---|---|
| 0 | Parasitized (Infected) |
| 1 | Uninfected |

**Image preprocessing**

- Resize: 224 × 224 pixels
- Convert image to tensor
- Normalize using ImageNet statistics

Mean:

```python
[0.485, 0.456, 0.406]
```

Standard deviation:

```python
[0.229, 0.224, 0.225]
```

Model performance metrics are not included here because independently verified evaluation results have not been provided. The model card in `ml/lab_model_card.json` (served at `/api/lab/model-card` and shown on the review page) is where metrics from the training run go, together with a note on where they came from. Accuracy, precision, recall, F1-score, and confusion-matrix results should be added after evaluation on an appropriately separated test dataset.

## Project Structure

Nextora is growing from a blood-smear analyser into a five-module malaria early-warning and response platform (surveillance, prediction, AI command centre, supply intelligence and the existing laboratory module), sharing one backend and one database. See `PROJECT_SPEC.md`.

```text
Malaria_Detection/
├── backend/
│   ├── main.py                 # entry point: uvicorn backend.main:app
│   ├── requirements.txt
│   ├── requirements-dev.txt    # + pytest
│   ├── app/
│   │   ├── main.py             # FastAPI app, routers, startup
│   │   ├── config.py           # settings from environment / .env
│   │   ├── db.py               # SQLAlchemy engine (SQLite or PostgreSQL)
│   │   ├── models.py           # shared tables: locations, facilities,
│   │   │                       #   observations, alerts, proposals,
│   │   │                       #   agent_runs, agent_messages
│   │   ├── core/               # pure Python, no FastAPI imports
│   │   │   ├── lab/            # PlasmoSENet, inference, smear analysis
│   │   │   ├── llm/            # Anthropic / OpenAI-compatible / Ollama clients
│   │   │   └── treatment_engine.py
│   │   ├── agents/             # tool-calling loop and agent definitions
│   │   └── modules/            # one FastAPI router per module
│   └── tests/                  # pytest
├── frontend/                   # React + Vite
├── data/                       # SQLite database (git-ignored) and datasets
├── model.pth                   # PlasmoSENet weights
├── plasmosenet.py              # compatibility shim -> backend/app/core/lab/
└── .env.example                # all configuration options
```

## Running Locally

### Prerequisites

- Python 3.11 recommended
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/adebowaleokiki55/Nextora.git
cd Nextora
```

### 2. Set up the Python backend

Create and activate a virtual environment from the project root.

**Windows PowerShell**

```powershell
python -m venv backend/.venv
.\backend\.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
pip install -r backend/requirements.txt
```

Ensure `model.pth` and `plasmosenet.py` are available at the paths expected by the backend.

### 3. Configure environment variables

**Backend.** Copy `.env.example` to `.env` in the project root. Everything is optional:

- **AI provider (any API key works).** The explanation agents run on Anthropic Claude, any OpenAI-compatible API (OpenAI, Groq, OpenRouter, Gemini, Together, Mistral, DeepSeek, LM Studio, vLLM...), or Ollama (local or Ollama Cloud). Set `LLM_PROVIDER`, `LLM_API_KEY` and `LLM_MODEL` (plus `LLM_BASE_URL` for OpenAI-compatible providers). If `LLM_PROVIDER` is not set, the first of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `OLLAMA_API_KEY` found is used. Without any key, image analysis still works and the agent endpoints return a clear 503 message. `GET /api/health` shows which provider is active.
- **Database.** SQLite at `data/nextora.db` by default; tables are created on startup. Set `DATABASE_URL` to a PostgreSQL URL in deployment (`postgres://` URLs from Render are accepted).

**Frontend.** Configure the frontend API URL to point to your local FastAPI server.

Create a `.env` file inside `frontend/`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

For production, set `VITE_API_URL` to the deployed backend URL in your hosting provider's environment-variable settings.

### 4. Start the backend

From the project root, run:

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

If your `main.py` uses imports or paths that require running from inside `backend/`, adjust the launch command accordingly.

The local API should be available at:

- API: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs

### 5. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in your terminal.

### 6. Run the tests

```powershell
pip install -r backend/requirements-dev.txt
python -m pytest
```

The tests use a temporary database and a fake AI provider, so they need no API key or network access.

## API Endpoints

The FastAPI backend exposes the following endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Backend, model, database and AI-provider status |
| GET | `/api/diagnostics/memory` | Process memory and inference-slot status |
| POST | `/api/predict` | Classify a single-cell image |
| POST | `/api/analyze-smear` | Perform experimental whole-smear candidate-region analysis |
| POST | `/api/agent/explain` | AI explanation of a single-cell or whole-smear result |
| POST | `/api/lab/submissions` | Analyse a smear and store it as **PENDING_REVIEW** (facility, sample date, submitter, optional slide number; `data_origin` defaults to SYNTHETIC) |
| GET | `/api/lab/submissions` | List submissions; filter by `status`, `location_id`, `facility_id` |
| GET | `/api/lab/submissions/{id}` | One submission with its cell-level results |
| GET | `/api/lab/submissions/{id}/annotated.png` | Annotated smear image |
| POST | `/api/lab/submissions/{id}/review` | Qualified review: `VERIFIED` (with the reviewer's own POSITIVE/NEGATIVE reading) or `REJECTED` (with a reason) |
| GET | `/api/lab/model-card` | Model description, training data, limitations and metrics status |
| GET | `/api/locations`, `/api/facilities` | Monitored Lagos LGAs and participating facilities |
| GET | `/api/surveillance/overview` | Current surveillance level and key signals for every LGA |
| GET | `/api/surveillance/locations/{id}` | Signals plus weekly series (cases, rainfall, temperature, humidity, verified lab results); `?weeks=156` |
| GET | `/api/surveillance/sources` | Every dataset held: real or synthetic, source, coverage |
| GET | `/api/prediction/risk` | Risk score, level, early-warning flag and 1–4 week forecasts for every LGA |
| GET | `/api/prediction/locations/{id}` | Recent cases, latest forecasts and test-period backtest for one LGA |
| GET | `/api/prediction/evaluation` | Model vs baselines: MAE, RMSE, precision and recall per horizon |
| POST | `/api/prediction/run` | Recompute forecasts and update early-warning alerts |
| GET | `/api/supply/items`, `/api/supply/policy` | Tracked commodities and the stock rules in plain language |
| GET | `/api/supply/status` | Stock assessment per facility and item; filter by `item_code`, `location_id`, `status` |
| GET | `/api/supply/shortages` | Facilities at risk of running out, most urgent first |
| GET | `/api/supply/redistribution` | Recommended transfers and central orders (recommendation only) |
| POST | `/api/supply/scan` | Re-assess stock and update shortage alerts |
| GET | `/api/supply/facilities/{id}` | One facility's stock and 12-week ledger |
| POST | `/api/command/runs` | Start the agents for a location: `{"location_id": 12, "mode": "auto" \| "live" \| "offline"}` |
| GET | `/api/command/runs/{id}/events` | Live feed (Server-Sent Events): one `message` event per agent step, then `done` |
| GET | `/api/command/runs`, `/api/command/runs/{id}` | Runs, with messages, summary and situation report |
| GET | `/api/proposals` | Approval queue; filter by `status`, `location_id`, `run_id` |
| POST | `/api/proposals/{id}/approve`, `/reject` | Human decision `{"decided_by": "...", "note": "..."}`; approve executes |
| GET | `/api/alerts` | Unresolved alerts from all modules, most severe first |
| POST | `/api/alerts/{id}/acknowledge`, `/resolve` | Update an alert |
| POST | `/api/demo/seed` | **Deletes all scenario data** and re-seeds the demo Lagos scenario |
| POST | `/api/treatment/recommend` | Rule-based treatment decision support (side page) |
| POST | `/api/agent/treatment-report` | AI explanation of the treatment assessment |

Open `/docs` on the running backend to inspect the request schemas, required image-upload fields, and response formats.

## Deployment

The current application is deployed using:

- **Frontend:** Vercel — https://nextora-zeta.vercel.app/
- **Backend:** Render — https://nextora-npmm.onrender.com

The frontend sends API requests to the backend. The production API URL and CORS configuration must match the deployed frontend domain.

## Validation and Limitations

Nextora is a prototype, and several limitations must be considered:

- A predicted class is not a confirmed medical diagnosis.
- Model confidence is not equivalent to clinical certainty.
- Whole-smear candidate detection uses image-processing heuristics and may miss cells or identify non-cell regions.
- A classifier trained on isolated cell images may not generalize reliably to crops extracted from whole-smear images.
- Clinical use would require rigorous validation, independent testing, expert-annotated data, and assessment across different imaging conditions and patient populations.

Future evaluation should include patient-independent train/test splits to reduce data leakage, per-class precision and recall, F1-score, a confusion matrix, and external validation where appropriate.

## Future Improvements

- Evaluate generalization on unseen patient samples.
- Improve candidate-cell detection and localization.
- Compare predictions against expert-annotated microscopy images.
- Add model evaluation reports and error analysis.
- Improve inference performance and image-upload validation.
- Investigate whether the whole-smear pipeline can support reliable, validated quantitative measurements.

## Disclaimer
Nextora is intended for research, learning, and demonstration purposes only. It has not been established as a clinically validated diagnostic system. Do not use its output to diagnose, treat, or rule out malaria. Medical decisions must be made with qualified healthcare professionals and appropriate laboratory testing.


**Okiki** — Data Science student interested in machine learning, artificial intelligence, and practical applications of AI.

GitHub: https://github.com/adebowaleokiki55

