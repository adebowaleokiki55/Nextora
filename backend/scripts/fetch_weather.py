"""Download real weekly weather for the 20 Lagos LGAs from Open-Meteo.

    python -m backend.scripts.fetch_weather [--start 2016-01-04] [--end YYYY-MM-DD]

Writes data/real/lagos_weather_weekly.csv and a .json file with provenance.
After it finishes, use "Seed demo data" (POST /api/demo/seed) to load it.

Source: Open-Meteo Historical Weather API (ERA5 / ERA5-Land reanalysis),
licensed CC BY 4.0: "Weather data by Open-Meteo.com". No API key needed.
Only the LGA centre coordinates are sent.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

from backend.app.core.demo_seed import HISTORY_START, LAGOS_LGAS, WEATHER_CSV
from backend.app.core.surveillance import weather


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, default=HISTORY_START)
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        # The archive lags real time by a few days: stop at the last full week.
        default=weather.last_complete_week_start(date.today() - timedelta(days=6)) + timedelta(days=6),
    )
    parser.add_argument("--pause", type=float, default=8.0, help="Seconds between requests.")
    args = parser.parse_args(argv)

    http = requests.Session()
    http.headers["User-Agent"] = "nextora-weather-fetch/1.0"
    frames = []

    for index, (name, (lat, lon)) in enumerate(sorted(LAGOS_LGAS.items()), start=1):
        print(f"[{index:2d}/{len(LAGOS_LGAS)}] {name} ({lat}, {lon}) {args.start} -> {args.end}", flush=True)
        frames.append(weather.fetch_open_meteo_daily(name, lat, lon, args.start, args.end, session=http))
        if index < len(LAGOS_LGAS):
            time.sleep(args.pause)

    daily = pd.concat(frames, ignore_index=True)
    weekly = weather.aggregate_weekly(daily)

    metadata = {
        "source": weather.REAL_SOURCE,
        "provider": "Open-Meteo Historical Weather API",
        "url": weather.OPEN_METEO_URL,
        "licence": "CC BY 4.0",
        "attribution": "Weather data by Open-Meteo.com",
        "underlying_data": "ERA5 / ERA5-Land reanalysis (Copernicus Climate Change Service)",
        "points": "Approximate LGA centre points (see backend/app/core/demo_seed.py)",
        "daily_variables": list(weather.OPEN_METEO_DAILY),
        "aggregation": "ISO weeks (Mon-Sun); rainfall summed, temperature and humidity averaged; partial weeks dropped",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "retrieved_on": date.today().isoformat(),
        "weekly_rows": int(len(weekly)),
    }
    weather.save_weekly_csv(weekly, WEATHER_CSV, metadata)
    print(f"Saved {len(weekly)} weekly rows to {WEATHER_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
