# Nextora — Malaria Early-Warning, Intelligence and Response Platform

Nextora helps public-health teams anticipate rising malaria risk, spot facilities running out of essential supplies, and coordinate a response, with a human approving every consequential action. It joins five modules in one backend, one database, one alert system and one command-centre dashboard:

| # | Module | What it does |
|---|---|---|
| 1 | **Environmental and historical surveillance** | Weekly rainfall, temperature, humidity and case history per Lagos LGA; flags unusual conditions for the time of year |
| 2 | **Malaria prediction engine** | Forecasts weekly cases 1–4 weeks ahead, compares against simple baselines, produces risk scores and early-warning alerts |
| 3 | **AI command centre** | Commander, Surveillance, Supply, Outreach and Policy agents prepare a coordinated response; consequential actions wait for human approval |
| 4 | **Medical supply and facility intelligence** | Tracks RDT kits and antimalarials, calculates shortages and proposes redistribution from facilities with spare stock |
| 5 | **AI laboratory intelligence** | Classifies cells in uploaded blood-smear images (PlasmoSENet CNN); results count only after a qualified professional verifies them |

> **Important:** Nextora is a decision-support research prototype. It is not a clinically validated diagnostic tool, not a validated parasite counter, and not an official surveillance system. Its outputs must not be used to diagnose, treat or rule out malaria.

**Live demo (laboratory module only):** https://nextora-zeta.vercel.app/ — the full platform runs locally until it is redeployed (see [Deployment](#deployment)).

## How the modules connect: the Ikorodu scenario

1. **Surveillance** sees Ikorodu's weekly cases far above the usual level for this time of year, rising faster than the normal seasonal trend, while real rainfall is close to normal: weather does not explain the rise.
2. **Prediction** forecasts that cases will stay above Ikorodu's alert threshold over the next weeks (risk score 90/100) and raises an early-warning alert.
3. **Supply** finds three Ikorodu health centres short of rapid diagnostic test (RDT) kits — one stocked out — and calculates transfers from nearby facilities with spare stock.
4. In the **command centre**, a coordinator presses *Run response*. The agents investigate, put the transfers and a public SMS draft into the approval queue, and write a situation report.
5. A named person **approves** each transfer; the stock moves and the shortage alerts resolve. Verified **lab** results add evidence for the area.

A step-by-step demo for judges is in [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

## What is real and what is synthetic

Every observation carries `data_origin` (REAL or SYNTHETIC) and a `source`, and the dashboard shows a badge wherever synthetic data appears.

| Data | Origin | Source |
|---|---|---|
| Lagos LGA names | REAL | The 20 local government areas; map points are approximate centres compiled by hand |
| Weekly rainfall, temperature, humidity (2016 → Nov 2026) | **REAL** (reanalysis) | [Open-Meteo](https://open-meteo.com/) Historical Weather API, ERA5/ERA5-Land reanalysis (CC BY 4.0). Modelled from observations; not rain-gauge readings |
| Weekly confirmed malaria cases | **SYNTHETIC** | Public weekly case data per LGA is not available. Generated from the real weather, a drifting transmission intensity, occasional historical outbreaks and a scripted rise in Ikorodu (`backend/app/core/surveillance/synthetic_cases.py`) |
| Facilities, stock and consumption | **SYNTHETIC** | Fictional "Demo" facilities; consumption derived from each LGA's cases; three Ikorodu centres scripted short of RDT kits |
| Lab submissions | As declared | Uploads are SYNTHETIC unless the submitter states they are real samples from that laboratory |
| Blood-smear classifier | Trained on real public data | NIH Malaria Cell Images dataset (single, pre-cropped cells) |

Because case data is synthetic, the forecast accuracy figures below describe how well the pipeline recovers patterns built into the generator. **They are not evidence of real-world forecasting accuracy.**

## Quick start (local)

**Prerequisites:** Python 3.11+, Node.js 20+ and npm, Git.

```powershell
# 1. Backend (from the project root)
python -m venv backend/.venv
.\backend\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
copy .env.example .env            # optional: add an AI provider key (see Configuration)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The first start creates `data/nextora.db` and seeds the demo scenario (about 20 seconds). API docs: http://127.0.0.1:8000/docs

```powershell
# 2. Frontend (second terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 for the home page, or go straight to:

- **Command centre:** http://localhost:5173/?view=command
- **Cell Explorer (lab analysis):** http://localhost:5173/?view=analyzer
- **Lab review queue:** http://localhost:5173/?view=lab-review
- **Treatment support (side page):** http://localhost:5173/?view=treatment

The frontend uses `VITE_API_URL` (default `http://127.0.0.1:8000`); set it in `frontend/.env` if the backend runs elsewhere. Map tiles need internet; everything else is served by the local backend.

**Tests** (temporary database, fake AI provider, no network needed; about 3 minutes):

```powershell
pip install -r backend/requirements-dev.txt
python -m pytest
```

## Configuration

Copy `.env.example` to `.env` in the project root. Everything is optional.

- **AI provider — any API key works.** The agents run on Anthropic Claude, any OpenAI-compatible API (OpenAI, Groq, OpenRouter, Gemini, Together, Mistral, DeepSeek, LM Studio, vLLM…), or Ollama (local or Ollama Cloud). Set `LLM_PROVIDER` (`anthropic`, `openai` or `ollama`), `LLM_API_KEY`, `LLM_MODEL`, and `LLM_BASE_URL` for OpenAI-compatible providers. If `LLM_PROVIDER` is unset, the first of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `OLLAMA_API_KEY` found is used. `GET /api/health` shows the active provider. Without a provider, everything still works and the command centre uses its offline scripted mode.
- **Database.** SQLite at `data/nextora.db` by default. Set `DATABASE_URL` to a PostgreSQL URL in deployment (`postgres://` URLs from Render are accepted).
- **CORS.** `CORS_ORIGINS` (comma-separated) lists the frontend origins allowed to call the API.
- **Lab model.** `LAB_MODEL_PATH` (default `model.pth`).

## The modules in detail

### Module 1 — Environmental and historical surveillance

For each LGA, the latest weeks are compared with the same time of year in earlier years:

- confirmed cases against the seasonal baseline;
- the 4-week case trend against the usual seasonal trend (cases normally rise when the rains start);
- 8-week rainfall against normal;
- temperature suitability (context only: Lagos is almost always in the favourable band);
- reviewer-verified lab positives.

A signal is raised only when the change is unusual for that series (z-score) **and** large in relative and absolute terms, which keeps false alarms to about 1% of ordinary weeks per signal. These levels describe current conditions; forecasts and risk scores come from Module 2.

Refresh the real weather (about 3–4 minutes, no API key; only LGA coordinates are sent), then reset the demo data:

```powershell
python -m backend.scripts.fetch_weather
```

Without `data/real/lagos_weather_weekly.csv`, the app falls back to synthetic weather and labels it SYNTHETIC.

### Module 2 — Malaria prediction engine

- **Model:** one XGBoost regressor per horizon (1–4 weeks), pooled across LGAs, predicting log(cases + 1) from recent case levels, seasonal references, rainfall lags (0–9 weeks), temperature, humidity and week of year. Every feature for a forecast made in week *t* uses only data up to week *t*.
- **Time-based split, no shuffling:** train on target weeks up to 2023; validate on 2024 (early stopping, uncertainty, alert cut-off); test on January 2025 to November 2026. The serving model is then refit on all weeks.
- **Alert threshold:** the classic malaria epidemic threshold, mean + 2 SD of weekly cases in the same weeks (±2) of earlier years.
- **Risk score:** the highest estimated probability (×100) that cases reach the threshold in the next 4 weeks, from the model's out-of-sample validation errors. An early warning is raised when that probability passes a per-horizon cut-off chosen on the validation year.

**Test-period results** (20 LGAs, 1,940 forecasts per horizon, 73 actual threshold crossings). MAE / RMSE in cases per week; lower is better:

| Weeks ahead | Model | Persistence (last week) | Seasonal naive (same week last year) | Historical weekly mean | Model precision / recall | Persistence precision / recall |
|---|---|---|---|---|---|---|
| 1 | **16.4 / 25.4** | 20.0 / 30.0 | 29.9 / 44.8 | 24.3 / 36.3 | 0.43 / 0.27 | 0.39 / 0.37 |
| 2 | **17.1 / 26.5** | 20.7 / 29.8 | 29.9 / 44.8 | 24.3 / 36.3 | 0.30 / 0.26 | 0.37 / 0.37 |
| 3 | **17.9 / 27.8** | 22.5 / 33.0 | 29.9 / 44.8 | 24.3 / 36.3 | 0.25 / 0.19 | 0.30 / 0.32 |
| 4 | **18.6 / 29.4** | 24.5 / 36.1 | 29.9 / 44.8 | 24.3 / 36.3 | 0.24 / 0.22 | 0.22 / 0.26 |

- The model predicts case numbers more accurately than all three baselines at every horizon.
- For the yes/no question "will cases cross the threshold?", persistence is as good as or better than the model at 1–3 weeks; detection of crossings is modest for both.
- These results are on **SYNTHETIC** case data (see above).

Retrain after changing the data (about 30 seconds); this writes `ml/models/forecast/`, `ml/metrics/forecast_metrics.json` and `ml/metrics/forecast_backtest.csv`:

```powershell
python -m ml.train_forecast
```

### Module 3 — AI command centre

| Agent | Tools (deterministic Python, scoped to the selected location) | Can propose |
|---|---|---|
| **Commander** | situation snapshot; delegates to the four specialists | — |
| **Surveillance** | surveillance signals, risk forecast, verified lab summary, open alerts | — |
| **Supply** | shortages, redistribution plan | stock transfers and central orders (quantities from the plan, never from the model) |
| **Outreach** | outreach context with approved key prevention messages | one public message draft (40–480 characters) |
| **Policy** | situation snapshot, open alerts | — (writes the situation report) |

- **Math lives in code.** Shortages, plans, risk scores and forecasts are computed by Python functions; agents call them as tools and explain the results.
- **Human approval.** Agents only create PENDING proposals. A named person approves or rejects each one. Approving a transfer executes it after re-checking the donor's stock; if the plan has gone stale the proposal ends FAILED and nothing moves. Approved orders and messages are recorded or queued only — no real order or SMS is sent in the demo.
- **Live feed.** Every thought, tool call, tool result, hand-off, final answer and error is written to `agent_messages` and streamed with Server-Sent Events.
- **Modes.** `live` uses the configured AI provider. `offline` is the **OFFLINE DEMO MODE**: a scripted sequence calling the same real tools, every message labelled as scripted, no key or internet needed. `auto` picks live when a provider is configured, otherwise offline. A failed live run is recorded as FAILED rather than silently switching to the script.

### Module 4 — Medical supply and facility intelligence

Tracks malaria RDT kits, artemether-lumefantrine (ACT) courses and injectable artesunate. All quantities come from `backend/app/core/supply/calculations.py`:

- **Expected weekly demand** = average consumption over the last 8 weeks × the location's forecast ratio, bounded 0.5–3.0. **Precaution:** where the risk score is 50 or more, planned demand never drops below recent consumption.
- **Status** by weeks of stock left: STOCKOUT, CRITICAL (< 1 week), LOW (< 2), OK, SURPLUS (> 10).
- **Shortage quantity** restores 6 weeks of stock, rounded up to whole packs.
- **Redistribution:** most urgent facility first (worst status, then higher area risk), nearest donor first within 40 km; donors keep at least 6 weeks of their own demand and give whole packs only. Anything uncovered becomes a central order.
- **Alerts:** `SUPPLY_SHORTAGE` alerts are raised, updated and resolved by each scan, one severity level higher in high-risk areas.

### Module 5 — AI laboratory intelligence

**Single-cell classification:** an uploaded cell image is resized to 224 × 224, normalised with ImageNet statistics and classified by PlasmoSENet (class 0 *Parasitized*, class 1 *Uninfected*).

**Whole-smear analysis:** NIH training images are single pre-cropped cells, so a smear photo is first segmented with OpenCV — adaptive threshold, morphological opening, contours, then a distance-transform watershed that splits touching or overlapping cells, with area and shape filters to drop debris. Each crop is classified and the image is returned with green (uninfected) and red (possibly parasitized) boxes.

**Review workflow:**

1. After a whole-smear analysis, the lab worker chooses **Submit for review**, giving the laboratory, sample date and optionally a slide number. No patient names or identifiers are collected.
2. The submission is stored as **PENDING_REVIEW**, labelled SYNTHETIC unless the submitter states it is a real sample from that laboratory.
3. In the **Lab review queue**, a qualified professional examines the slide and verifies it with their own reading (positive or negative) or rejects it with a reason.
4. Only verified submissions become surveillance observations for that location; a verified positive also raises an alert. The reviewer's reading is recorded, not the model's.

**Model information:** `PlasmoSENet`, a custom PyTorch CNN (multi-scale stem, residual and inverted-residual blocks with squeeze-and-excitation), trained by the developer on the NIH Malaria Cell Images dataset. Held-out test metrics are not reported because the training run's train/test split was not recorded; the model card in `ml/lab_model_card.json` (served at `/api/lab/model-card`) is where metrics go, together with where they came from.

### Command-centre dashboard

One screen (`/?view=command`) brings the modules together: headline numbers; a Leaflet map of LGAs coloured by forecast risk with facilities flagged when short of stock; a location panel with the forecast chart (80% range, alert threshold, past forecasts), surveillance signals, rainfall against cases and facility stock; **Run response** with the live agent feed and situation report; the **approval queue**; alerts from every module; forecast accuracy against baselines; data sources; and a **Seed demo data** button that resets the synthetic Lagos scenario.

## Architecture

```text
                React + Vite frontend
   Home · Cell Explorer · Lab review · Command centre · Treatment
                          |
                     HTTP + SSE
                          |
                   FastAPI backend
  ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
  │ Surveillance│ Prediction  │ Command     │ Supply      │ Lab         │
  │ (Module 1)  │ (Module 2)  │ centre (3)  │ (Module 4)  │ (Module 5)  │
  │ signals     │ XGBoost     │ 5 agents,   │ demand,     │ PlasmoSENet │
  │ weather     │ baselines   │ approvals,  │ shortages,  │ + OpenCV    │
  │             │ risk scores │ live feed   │ transfers   │ review      │
  └──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
         └─────────────┴──── shared database ──────┴─────────────┘
   locations · facilities · observations · alerts · proposals · forecasts
   stock · lab submissions · agent runs and messages   (SQLite / PostgreSQL)
                          |
             Any AI provider (agents only)
```

## Project structure

```text
Malaria_Detection/
├── backend/
│   ├── main.py                 # entry point: uvicorn backend.main:app
│   ├── requirements.txt / requirements-dev.txt
│   ├── app/
│   │   ├── main.py             # FastAPI app, routers, startup seeding
│   │   ├── config.py           # settings from environment / .env
│   │   ├── db.py, models.py    # shared database and tables
│   │   ├── core/               # pure Python logic, unit-tested
│   │   │   ├── surveillance/   # weather, synthetic cases, signals
│   │   │   ├── prediction/     # features, training, forecasts, risk
│   │   │   ├── supply/         # stock calculations, plans, transfers
│   │   │   ├── command/        # agent tool data, approval queue
│   │   │   ├── lab/            # PlasmoSENet, inference, segmentation, review
│   │   │   ├── llm/            # Anthropic / OpenAI-compatible / Ollama clients
│   │   │   ├── demo_seed.py    # the synthetic Lagos scenario
│   │   │   └── treatment_engine.py
│   │   ├── agents/             # tool loop, command centre, offline script
│   │   └── modules/            # one FastAPI router per module
│   ├── scripts/fetch_weather.py
│   └── tests/                  # pytest (136 tests)
├── ml/                         # forecast training, models, metrics, lab model card
├── frontend/src/               # React app; dashboard/ is the command centre
├── data/real/                  # real weather (committed); local DB (git-ignored)
├── docs/DEMO_SCRIPT.md         # step-by-step demo for judges
├── model.pth                   # PlasmoSENet weights
└── .env.example                # all configuration options
```

## API endpoints

Open `/docs` on the running backend for request and response schemas.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Backend, model, database and AI-provider status |
| GET | `/api/locations`, `/api/facilities` | Monitored LGAs and participating facilities |
| GET | `/api/surveillance/overview` | Current surveillance level and key signals for every LGA |
| GET | `/api/surveillance/locations/{id}` | Signals and weekly series for one LGA (`?weeks=156`) |
| GET | `/api/surveillance/sources` | Every dataset held: real or synthetic, source, coverage |
| GET | `/api/prediction/risk` | Risk score, level, early warning and 1–4 week forecasts for every LGA |
| GET | `/api/prediction/locations/{id}` | Recent cases, latest forecasts and test-period backtest |
| GET | `/api/prediction/evaluation` | Model vs baselines: MAE, RMSE, precision and recall per horizon |
| POST | `/api/prediction/run` | Recompute forecasts and update early-warning alerts |
| GET | `/api/supply/items`, `/api/supply/policy` | Tracked commodities and the stock rules |
| GET | `/api/supply/status`, `/api/supply/shortages` | Stock assessments; shortages, most urgent first |
| GET | `/api/supply/redistribution` | Recommended transfers and central orders (recommendation only) |
| POST | `/api/supply/scan` | Re-assess stock and update shortage alerts |
| GET | `/api/supply/facilities/{id}` | One facility's stock and 12-week ledger |
| POST | `/api/command/runs` | Start the agents: `{"location_id": 12, "mode": "auto" \| "live" \| "offline"}` |
| GET | `/api/command/runs/{id}/events` | Live feed (Server-Sent Events) |
| GET | `/api/command/runs`, `/api/command/runs/{id}` | Runs with messages, summary and situation report |
| GET | `/api/proposals` | Approval queue; filter by `status`, `location_id`, `run_id` |
| POST | `/api/proposals/{id}/approve`, `/reject` | Human decision `{"decided_by": "...", "note": "..."}`; approve executes |
| GET | `/api/alerts` | Unresolved alerts from all modules, most severe first |
| POST | `/api/alerts/{id}/acknowledge`, `/resolve` | Update an alert |
| POST | `/api/predict` | Classify a single-cell image |
| POST | `/api/analyze-smear` | Whole-smear analysis (nothing stored) |
| POST | `/api/agent/explain` | AI explanation of a lab result |
| POST | `/api/lab/submissions` | Analyse a smear and store it as PENDING_REVIEW |
| GET | `/api/lab/submissions`, `/api/lab/submissions/{id}` | Submissions and cell-level results |
| GET | `/api/lab/submissions/{id}/annotated.png` | Annotated smear image |
| POST | `/api/lab/submissions/{id}/review` | Qualified review: VERIFIED (with reading) or REJECTED (with reason) |
| GET | `/api/lab/model-card` | Lab model description, training data, limitations, metrics status |
| POST | `/api/demo/seed` | **Deletes all scenario data** and re-seeds the demo Lagos scenario |
| POST | `/api/treatment/recommend`, `/api/agent/treatment-report` | Treatment decision support (side page) |
| GET | `/api/diagnostics/memory` | Process memory and inference-slot status |

## Deployment

The laboratory module is currently deployed (frontend on Vercel, backend on Render). To deploy the full platform:

**Backend (Render web service)**

- Build: `pip install -r backend/requirements.txt` · Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` · Python 3.11.
- Use a **PostgreSQL** database and set `DATABASE_URL`. Render's free disk is wiped on every redeploy, so SQLite would lose approvals and lab reviews. The PostgreSQL path is configured but has only been tested on SQLite so far.
- Set `CORS_ORIGINS` to your Vercel domain(s) and, for live agents, the `LLM_*` variables.
- **Memory:** measured locally (Windows), the backend uses about 380 MB after imports and 530–580 MB once seeded and in use (PyTorch, XGBoost and pandas). That is at or above the 512 MB of Render's free and Starter instances; plan for a larger instance or trim dependencies.
- **First start** seeds the database (about 20 seconds plus model loading); allow for this in the health-check timeout.

**Frontend (Vercel):** set `VITE_API_URL` to the backend URL.

## Limitations

- **Synthetic data.** Case counts, facilities and stock are synthetic, so forecast metrics and the scenario do not demonstrate real-world performance.
- **Not a diagnostic tool.** A predicted class is not a diagnosis, model confidence is not a calibrated clinical probability, and candidate counts are not a validated parasitemia measurement.
- **Smear segmentation is heuristic.** It may miss cells or include non-cell regions; it was tuned on drawn test images, not on real smear photos. A classifier trained on isolated cells may not generalise to crops from whole-smear photos.
- **Lab model metrics unavailable.** The training split was not recorded, so no held-out accuracy, sensitivity, specificity or AUC is reported.
- **Agent outputs** depend on the AI provider and can contain mistakes; every consequential action needs human approval, and the situation report should be read as a draft.
- **No authentication.** Anyone who can reach the app can approve or reject proposals. Real use would need accounts, roles and audit review.
- Clinical or operational use would require rigorous validation, independent testing, expert-annotated data, real surveillance data and assessment across different settings.

## Future improvements

- Replace synthetic cases with real surveillance data (for example DHIS2 extracts) and re-evaluate the forecasts.
- Add satellite vegetation (NDVI) and other environmental layers to Module 1.
- Evaluate the lab model on patient-independent splits and real, expert-annotated smear photos; report sensitivity, specificity and AUC.
- Improve cell detection and localisation on real microscopy images.
- Add authentication and role-based approval, and database migrations.

## Disclaimer

Nextora is intended for research, learning and demonstration purposes only. It has not been established as a clinically validated diagnostic system. Do not use its output to diagnose, treat or rule out malaria. Medical decisions must be made with qualified healthcare professionals and appropriate laboratory testing.

**Okiki** — Data Science student interested in machine learning, artificial intelligence and practical applications of AI.

GitHub: https://github.com/adebowaleokiki55
