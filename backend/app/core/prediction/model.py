"""Train, evaluate, save and load the 1-4 week forecast models.

One XGBoost regressor per horizon, pooled across locations, predicting
log1p(cases). Time-based split by target week (no shuffling):

    train:      target week <= train_end
    validation: train_end < target week <= validation_end  (early stopping,
                residuals for intervals and exceedance probabilities)
    test:       target week >  validation_end              (reported metrics)

After evaluation, each horizon is refit on all weeks with a known target,
using the tree count chosen on validation; that refit model serves live
forecasts. Validation residuals are kept, so live uncertainty estimates
come from out-of-sample errors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.prediction import HORIZONS
from backend.app.core.prediction.evaluation import crossing_metrics, evaluate_horizon
from backend.app.core.prediction.features import FEATURES, build_rows

DEFAULT_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
}
EARLY_STOPPING_ROUNDS = 60
DEFAULT_ALERT_PROBABILITY = 0.5  # used if validation has no crossings
CUTOFF_GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.75, 0.05))
INTERVAL = (0.10, 0.90)


@dataclass
class Splits:
    train_end: date
    validation_end: date

    def masks(self, rows: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        target = pd.to_datetime(rows["target_week"])
        train = target <= pd.Timestamp(self.train_end)
        validation = (target > pd.Timestamp(self.train_end)) & (target <= pd.Timestamp(self.validation_end))
        test = target > pd.Timestamp(self.validation_end)
        return train, validation, test


@dataclass
class HorizonModel:
    horizon: int
    booster: Any
    residuals: np.ndarray  # validation residuals in log space
    best_iteration: int
    alert_cutoff: float = DEFAULT_ALERT_PROBABILITY  # chosen on validation

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        return self.booster.predict(xgb.DMatrix(X[FEATURES], feature_names=FEATURES))

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per-feature contributions (last column is the bias)."""
        import xgboost as xgb

        return self.booster.predict(xgb.DMatrix(X[FEATURES], feature_names=FEATURES), pred_contribs=True)


@dataclass
class ForecastBundle:
    models: dict[int, HorizonModel]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return self.metadata.get("version", "unversioned")


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------

def exceedance_probability(pred_log: np.ndarray, residuals: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """P(cases >= threshold), from the empirical validation residuals."""
    pred_log = np.asarray(pred_log, dtype=float)
    threshold_log = np.log1p(np.asarray(threshold, dtype=float))
    simulated = pred_log[:, None] + residuals[None, :]
    probability = np.mean(simulated >= threshold_log[:, None], axis=1)
    probability[np.isnan(threshold_log)] = np.nan
    return probability


def choose_alert_cutoff(probability: np.ndarray, actual_cross: np.ndarray) -> float:
    """Probability cut-off that maximises F1 on the validation period."""
    if not actual_cross.any():
        return DEFAULT_ALERT_PROBABILITY
    best_cutoff, best_f1 = DEFAULT_ALERT_PROBABILITY, -1.0
    for cutoff in CUTOFF_GRID:
        f1 = crossing_metrics(actual_cross, probability >= cutoff)["f1"] or 0.0
        if f1 > best_f1:  # ties keep the lower cut-off (fewer missed epidemics)
            best_cutoff, best_f1 = cutoff, f1
    return best_cutoff


def interval(pred_log: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = np.quantile(residuals, INTERVAL)
    return np.expm1(pred_log + low).clip(min=0), np.expm1(pred_log + high).clip(min=0)


# --------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------

def _regressor(params: dict, **overrides):
    import xgboost as xgb

    return xgb.XGBRegressor(**{**params, **overrides})


def train_and_evaluate(
    wides: dict[int, pd.DataFrame],
    splits: Splits,
    params: dict | None = None,
    horizons=HORIZONS,
) -> tuple[ForecastBundle, dict[str, Any], pd.DataFrame]:
    """Returns (serving bundle, metrics report, test-period backtest rows)."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    models: dict[int, HorizonModel] = {}
    report: dict[str, Any] = {}
    backtests = []
    latest_origin = pd.Timestamp.min

    for horizon in horizons:
        rows = build_rows(wides, horizon)
        rows = rows[rows["target_log"].notna()].reset_index(drop=True)
        train, validation, test = splits.masks(rows)
        latest_origin = max(latest_origin, pd.Timestamp(rows["origin_week"].max()))

        model = _regressor(params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(
            rows.loc[train, FEATURES],
            rows.loc[train, "target_log"],
            eval_set=[(rows.loc[validation, FEATURES], rows.loc[validation, "target_log"])],
            verbose=False,
        )
        best = int(model.best_iteration) + 1

        validation_pred = model.predict(rows.loc[validation, FEATURES], iteration_range=(0, best))
        residuals = rows.loc[validation, "target_log"].to_numpy() - validation_pred

        has_threshold = rows.loc[validation, "threshold"].notna().to_numpy()
        validation_threshold = rows.loc[validation, "threshold"].to_numpy()[has_threshold]
        validation_probability = exceedance_probability(
            validation_pred[has_threshold], residuals, validation_threshold
        )
        validation_actual = rows.loc[validation, "target_cases"].to_numpy()[has_threshold] >= validation_threshold
        cutoff = choose_alert_cutoff(validation_probability, validation_actual)

        evaluation_rows = rows[test & rows["threshold"].notna()].reset_index(drop=True)
        test_pred_log = model.predict(evaluation_rows[FEATURES], iteration_range=(0, best))
        test_pred = np.expm1(test_pred_log).clip(min=0)
        probability = exceedance_probability(test_pred_log, residuals, evaluation_rows["threshold"].to_numpy())
        report[f"h{horizon}"] = {
            "horizon_weeks": horizon,
            "train_rows": int(train.sum()),
            "validation_rows": int(validation.sum()),
            "test_rows": int(len(evaluation_rows)),
            "best_iteration": best,
            "alert_cutoff": cutoff,
            "validation_crossings": int(validation_actual.sum()),
            **evaluate_horizon(evaluation_rows, test_pred, probability >= cutoff),
        }

        low, high = interval(test_pred_log, residuals)
        backtests.append(
            evaluation_rows[["location_id", "origin_week", "target_week", "horizon", "target_cases", "threshold",
                             "baseline_persistence", "baseline_seasonal_naive", "baseline_hist_mean"]]
            .assign(predicted=test_pred.round(1), lower=low.round(1), upper=high.round(1),
                    exceedance_probability=probability.round(3))
        )

        # Refit on every week with a known target for live forecasting.
        final = _regressor(params, n_estimators=best)
        final.fit(rows[FEATURES], rows["target_log"], verbose=False)
        models[horizon] = HorizonModel(horizon, final.get_booster(), np.sort(residuals), best, cutoff)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "XGBoost regressor on log1p(weekly cases), one model per horizon, pooled across locations",
        "features": FEATURES,
        "horizons": list(horizons),
        "split": {
            "train_target_weeks": f"up to {splits.train_end.isoformat()}",
            "validation_target_weeks": f"{splits.train_end.isoformat()} to {splits.validation_end.isoformat()} (exclusive start)",
            "test_target_weeks": f"after {splits.validation_end.isoformat()}",
            "method": "time-based, no shuffling",
        },
        "data_through_week": str((latest_origin + pd.Timedelta(weeks=1)).date()),
        "params": params,
        "alert_rule": {
            "threshold": "mean + 2 SD of weekly cases in the same week (+/- 2 weeks) of earlier years",
            "crossing_forecast": (
                "P(cases >= threshold) >= a per-horizon cut-off chosen to maximise F1 on the "
                "validation period (never on the test period)"
            ),
            "probability_method": "empirical distribution of validation-period errors (log scale)",
            "interval": f"{int(INTERVAL[0] * 100)}th-{int(INTERVAL[1] * 100)}th percentile of validation errors",
        },
    }
    backtest = pd.concat(backtests, ignore_index=True)
    return ForecastBundle(models, metadata), report, backtest


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_bundle(bundle: ForecastBundle, directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    horizons = {}
    for horizon, model in sorted(bundle.models.items()):
        path = directory / f"h{horizon}.json"
        model.booster.save_model(str(path))
        digest.update(path.read_bytes())
        horizons[str(horizon)] = {
            "model_file": path.name,
            "best_iteration": model.best_iteration,
            "alert_cutoff": model.alert_cutoff,
            "validation_residuals": [round(float(r), 5) for r in model.residuals],
        }
    version = f"xgb-{digest.hexdigest()[:10]}"
    bundle.metadata["version"] = version
    (directory / "bundle.json").write_text(
        json.dumps({**bundle.metadata, "horizon_models": horizons}, indent=1), encoding="utf-8"
    )
    return version


def load_bundle(directory: Path) -> ForecastBundle | None:
    manifest = directory / "bundle.json"
    if not manifest.exists():
        return None
    import xgboost as xgb

    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    models = {}
    for key, info in metadata.pop("horizon_models").items():
        booster = xgb.Booster()
        booster.load_model(str(directory / info["model_file"]))
        models[int(key)] = HorizonModel(
            int(key),
            booster,
            np.asarray(info["validation_residuals"], dtype=float),
            info["best_iteration"],
            info.get("alert_cutoff", DEFAULT_ALERT_PROBABILITY),
        )
    return ForecastBundle(models, metadata)
