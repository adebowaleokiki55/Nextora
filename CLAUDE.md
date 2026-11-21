# Malaria Mission Control: instructions for Claude Code

Read `PROJECT_SPEC.md` first. It is the full project description. This file adds build rules.

## What we are building
One integrated web app with five modules that share one backend, one database, one alert system and one command-centre dashboard:
1. Environmental and historical surveillance
2. Malaria prediction engine (ML forecasting)
3. AI command centre (multi-agent coordination with human approval)
4. Medical supply and facility intelligence
5. AI laboratory intelligence (blood-smear CNN)

This is a hackathon project. Build to a working demo, not to production.

## Tech stack
- Frontend: React + Vite + Tailwind CSS, Leaflet (react-leaflet) for maps, Plotly for charts
- Backend: Python 3.11+, FastAPI, SQLAlchemy
- Database: SQLite for local development, PostgreSQL in deployment, switched with `DATABASE_URL`. Store latitude/longitude as columns; do not add PostGIS unless a real geospatial query needs it.
- ML: pandas, numpy, scikit-learn, XGBoost or LightGBM
- Computer vision: PyTorch + OpenCV
- Agents: Anthropic Python SDK (Claude API) with tool use. Model name comes from the env var `CLAUDE_MODEL`.

## Folder layout
```
backend/app/core/       pure-Python logic (no FastAPI imports), unit-tested
backend/app/modules/    one FastAPI router per module
backend/app/agents/     agent definitions, tools, runner
backend/tests/          pytest tests
ml/                     training scripts and saved weights
frontend/               React app
data/                   real datasets (data/real) and generated demo data
```

## Non-negotiable rules
- **Data honesty:** every observation row has `data_origin` (REAL or SYNTHETIC) and `source`. The UI shows a visible SYNTHETIC badge wherever synthetic data appears. Never present simulated results as real-world achievements.
- **Verification status:** lab results start as PENDING_REVIEW. Only VERIFIED lab observations may feed surveillance or prediction.
- **Human approval:** agents may only *propose* consequential actions (stock transfers, public messages). Proposals go into an approval queue. Nothing is executed until a human approves it in the UI.
- **Math lives in code, not in the LLM:** shortage quantities, redistribution plans, risk scores and forecasts are computed by deterministic Python functions. Agents call these as tools and reason about the results.
- **Lab module wording:** never describe it as a diagnostic tool or a validated parasite counter. It is a decision-support prototype trained on the NIH Malaria Cell Images dataset.

## Module-specific notes
**Module 2 (prediction)**
- Forecast 1 to 4 weeks ahead per location.
- Always compare against baselines: persistence (last value), seasonal naive (same week last year), and the historical weekly mean.
- Use a time-based train/test split (no random shuffling). Report MAE and RMSE per horizon, plus precision and recall for "crossed the alert threshold."
- Public weekly case data per local government area may not be available. If it is not, generate clearly labelled synthetic case data driven by real or realistic rainfall and temperature, and say so in the UI.

**Module 5 (lab)**
- NIH images are single pre-cropped cells. A real smear photo has many cells, so build an OpenCV segmentation step first (threshold, morphology, distance transform/watershed, area filtering). Then classify each crop with the CNN and draw green/red boxes on the uploaded image.
- Report held-out test accuracy, sensitivity, specificity and AUC from training.

**Module 3 (agents)**
- Agents: Commander, Surveillance, Supply, Outreach, Policy. The Commander delegates to the others through tools.
- Log every agent step (thought, tool call, tool result, handoff) to an `agent_messages` table so the dashboard can show a live feed.
- Include an OFFLINE DEMO MODE that runs a scripted sequence calling the same real tools when no API key or internet is available, clearly labelled as scripted. This is the stage backup.

## Demo requirements
- A "Seed demo data" action that resets the database with the synthetic Lagos scenario (Ikorodu rising risk, three facilities short of RDT kits).
- A "Run response" button in the command centre that triggers the agents for a selected location and streams their messages live.
- An approval queue where a judge can click Approve or Reject.

## Working style
- Work in phases. At the end of each phase, run the tests and make sure the app starts before moving on.
- Keep the README up to date with setup and run commands.