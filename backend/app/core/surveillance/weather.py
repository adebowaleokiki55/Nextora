"""Weather for monitored locations: fetch, weekly aggregation, cached CSV,
and a clearly-labelled synthetic fallback.

Real source: Open-Meteo Historical Weather API (ERA5 / ERA5-Land
reanalysis, CC BY 4.0). Reanalysis is modelled from observations; it is
not a rain-gauge or weather-station record.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.core.surveillance.metrics import HUMIDITY, RAINFALL, TEMP_MAX, TEMP_MEAN

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_DAILY = {
    "precipitation_sum": RAINFALL,
    "temperature_2m_mean": TEMP_MEAN,
    "temperature_2m_max": TEMP_MAX,
    "relative_humidity_2m_mean": HUMIDITY,
}
REAL_SOURCE = (
    "Open-Meteo Historical Weather API (ERA5/ERA5-Land reanalysis), CC BY 4.0; "
    "daily values at the LGA centre point, aggregated to ISO weeks"
)
SYNTHETIC_SOURCE = (
    "Synthetic weather generated from an approximate Lagos monthly climatology "
    "(used because real weather data has not been downloaded)"
)

WEEKLY_COLUMNS = ["location", "week_start", RAINFALL, TEMP_MEAN, TEMP_MAX, HUMIDITY]


def week_start(day: date) -> date:
    """Monday of the ISO week containing `day`."""
    return day - timedelta(days=day.weekday())


def last_complete_week_start(today: date) -> date:
    """Monday of the most recent week that has fully ended before `today`."""
    return week_start(today) - timedelta(days=7)


# --------------------------------------------------------------------------
# Weekly aggregation
# --------------------------------------------------------------------------

def aggregate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily rows (location, date, metrics...) -> ISO-week rows.

    Rainfall is summed, temperature and humidity averaged. Weeks with fewer
    than 7 days of data are dropped so partial weeks never look dry.
    """
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["week_start"] = (frame["date"] - pd.to_timedelta(frame["date"].dt.weekday, unit="D")).dt.date

    aggregations = {"date": "count"}
    if RAINFALL in frame:
        aggregations[RAINFALL] = lambda s: s.sum(min_count=7)
    for column in (TEMP_MEAN, TEMP_MAX, HUMIDITY):
        if column in frame:
            aggregations[column] = "mean"

    weekly = frame.groupby(["location", "week_start"], as_index=False).agg(aggregations)
    weekly = weekly[weekly["date"] == 7].drop(columns="date")

    for column in (RAINFALL, TEMP_MEAN, TEMP_MAX, HUMIDITY):
        if column in weekly:
            weekly[column] = weekly[column].round(2)

    return weekly.reset_index(drop=True)


# --------------------------------------------------------------------------
# Cached real data
# --------------------------------------------------------------------------

def load_weekly_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["week_start"] = pd.to_datetime(frame["week_start"]).dt.date
    return frame


def save_weekly_csv(weekly: pd.DataFrame, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weekly.sort_values(["location", "week_start"]).to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_metadata(path: Path) -> dict | None:
    meta = path.with_suffix(".json")
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None


# --------------------------------------------------------------------------
# Open-Meteo download (used by backend/scripts/fetch_weather.py)
# --------------------------------------------------------------------------

def fetch_open_meteo_daily(
    name: str,
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    session=None,
    max_attempts: int = 6,
) -> pd.DataFrame:
    """Daily weather for one point. Retries politely on rate limits."""
    import requests

    http = session or requests.Session()
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(OPEN_METEO_DAILY),
        "timezone": "Africa/Lagos",
    }

    for attempt in range(1, max_attempts + 1):
        response = http.get(OPEN_METEO_URL, params=params, timeout=120)
        if response.status_code == 429 and attempt < max_attempts:
            time.sleep(min(90, 15 * attempt))
            continue
        response.raise_for_status()
        break

    daily = response.json()["daily"]
    frame = pd.DataFrame({"date": daily["time"]})
    for api_name, metric in OPEN_METEO_DAILY.items():
        frame[metric] = daily.get(api_name)
    frame.insert(0, "location", name)
    return frame


# --------------------------------------------------------------------------
# Synthetic fallback
# --------------------------------------------------------------------------

# Approximate Lagos climatology by month (rainfall mm/month, mean temp °C).
# Two rainy seasons: a major one peaking in June, a minor one in Sep-Oct.
_MONTHLY_RAIN = [15, 40, 85, 145, 215, 320, 200, 70, 160, 170, 60, 15]
_MONTHLY_TEMP = [27.6, 28.5, 28.6, 28.1, 27.3, 26.1, 25.4, 25.3, 25.9, 26.6, 27.5, 27.6]


def synthetic_weekly_weather(
    locations: dict[str, tuple[float, float]],
    first_week: date,
    last_week: date,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(first_week, last_week, freq="7D")
    rows = []

    for index, name in enumerate(sorted(locations)):
        wetness = 0.9 + 0.2 * ((index * 7) % 10) / 10  # some places a bit wetter
        # Keyed by the year of each week's midpoint, which can be the next year.
        last_mid = weeks[-1] + pd.Timedelta(days=3)
        year_factor = {year: rng.gamma(30, 1 / 30) for year in range(weeks[0].year, last_mid.year + 1)}

        for week in weeks:
            mid = week + pd.Timedelta(days=3)
            month = mid.month - 1
            expected = _MONTHLY_RAIN[month] / 4.35 * wetness * year_factor[mid.year]
            rain = expected * rng.gamma(1.6, 1 / 1.6) if expected > 0 else 0.0
            temp = _MONTHLY_TEMP[month] + rng.normal(0, 0.6)
            rows.append(
                {
                    "location": name,
                    "week_start": week.date(),
                    RAINFALL: round(float(rain), 2),
                    TEMP_MEAN: round(float(temp), 2),
                    TEMP_MAX: round(float(temp + 4.2 + rng.normal(0, 0.5)), 2),
                    HUMIDITY: round(float(np.clip(78 + (rain - expected) * 0.05 + rng.normal(0, 3), 55, 97)), 2),
                }
            )

    return pd.DataFrame(rows, columns=WEEKLY_COLUMNS)
