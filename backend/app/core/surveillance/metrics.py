"""Metric names shared by all modules (stored in observations.metric)."""

CASES = "malaria_cases_confirmed"
RAINFALL = "rainfall_mm"
TEMP_MEAN = "temp_mean_c"
TEMP_MAX = "temp_max_c"
HUMIDITY = "rel_humidity_pct"

# Daily, from verified lab submissions (Module 5).
LAB_REVIEWED = "lab_smears_reviewed"
LAB_POSITIVE = "lab_smears_positive"

WEATHER_METRICS = (RAINFALL, TEMP_MEAN, TEMP_MAX, HUMIDITY)

UNITS = {
    CASES: "cases",
    RAINFALL: "mm",
    TEMP_MEAN: "°C",
    TEMP_MAX: "°C",
    HUMIDITY: "%",
    LAB_REVIEWED: "smears",
    LAB_POSITIVE: "smears",
}

LABELS = {
    CASES: "Confirmed malaria cases",
    RAINFALL: "Rainfall",
    TEMP_MEAN: "Mean temperature",
    TEMP_MAX: "Maximum temperature",
    HUMIDITY: "Relative humidity",
    LAB_REVIEWED: "Verified lab smears reviewed",
    LAB_POSITIVE: "Verified lab-positive smears",
}
