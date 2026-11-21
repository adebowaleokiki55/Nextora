"""Train and evaluate the malaria case forecast models (Module 2).

    python -m ml.train_forecast

Reads weekly cases and weather from the platform database (seeding the
demo scenario first if the database is empty), trains one model per
horizon (1-4 weeks) with a time-based split, compares against the three
baselines and writes:

    ml/models/forecast/        serving models + bundle.json
    ml/metrics/forecast_metrics.json
    ml/metrics/forecast_backtest.csv   test-period predictions vs actuals

IMPORTANT: the case data is SYNTHETIC, so these metrics show how well the
pipeline recovers the patterns built into the synthetic generator. They
are not evidence of real-world forecasting accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from sqlalchemy.orm import Session

from backend.app.config import PROJECT_ROOT
from backend.app.core.demo_seed import seed_if_empty
from backend.app.core.prediction.model import Splits, save_bundle, train_and_evaluate
from backend.app.core.prediction.training_data import load_training_wides
from backend.app.db import SessionLocal, init_db

MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "forecast"
METRICS_DIR = PROJECT_ROOT / "ml" / "metrics"

DEFAULT_SPLITS = Splits(train_end=date(2023, 12, 31), validation_end=date(2024, 12, 29))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-end", type=date.fromisoformat, default=DEFAULT_SPLITS.train_end)
    parser.add_argument("--validation-end", type=date.fromisoformat, default=DEFAULT_SPLITS.validation_end)
    args = parser.parse_args(argv)
    splits = Splits(args.train_end, args.validation_end)

    init_db()
    with SessionLocal() as session:
        seed_if_empty(session)
        wides, data_origin = load_training_wides(session)

    print(f"Training on {len(wides)} locations; case data origin: {data_origin}", flush=True)
    bundle, report, backtest = train_and_evaluate(wides, splits)
    bundle.metadata["case_data_origin"] = data_origin
    version = save_bundle(bundle, MODEL_DIR)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model_version": version,
        "case_data_origin": data_origin,
        "honesty_note": (
            "Evaluated on SYNTHETIC case data generated from real weather. These numbers show how "
            "well the pipeline recovers patterns built into the synthetic generator; they are NOT "
            "evidence of real-world forecasting accuracy."
            if data_origin != "REAL"
            else "Evaluated on real case data."
        ),
        **{k: v for k, v in bundle.metadata.items() if k not in ("features", "params")},
        "horizons": report,
    }
    (METRICS_DIR / "forecast_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    backtest.to_csv(METRICS_DIR / "forecast_backtest.csv", index=False)

    print(f"Saved model {version}")
    print(f"{'h':>2} {'model MAE':>10} {'persist':>8} {'seasonal':>9} {'hist mean':>10} | "
          f"{'precision':>9} {'recall':>6} {'crossings':>9}")
    for key, result in report.items():
        crossing = result["model"]["crossing"]
        print(
            f"{result['horizon_weeks']:>2} {result['model']['mae']:>10} "
            f"{result['baseline_persistence']['mae']:>8} {result['baseline_seasonal_naive']['mae']:>9} "
            f"{result['baseline_hist_mean']['mae']:>10} | {crossing['precision']!s:>9} "
            f"{crossing['recall']!s:>6} {crossing['actual_crossings']:>9}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
