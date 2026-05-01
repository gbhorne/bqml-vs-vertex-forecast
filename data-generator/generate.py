"""
data-generator/generate.py
Generates 5 years of daily store-SKU sales facts on top of catalog.py output.

Demand model is multiplicative. Each factor is documented with the intent
so that downstream models (ARIMA_PLUS, TFT, TimesFM) have something
recoverable to fit.

Run:
    python data-generator/generate.py

Outputs (under data/sales/):
    year=YYYY/sales.parquet  - one Hive-style partition per year
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Different seed than catalog.py so noise is independent of catalog choice
RNG = np.random.default_rng(seed=2026)

# ─────────────────────────────────────────────────────────────────────
# US holidays we'll model. The format is (month, day, lift_factor_by_category).
# Lift factor 1.0 = no effect; 1.5 = 50% above baseline; 0.5 = halved.
# These are intentionally rough — the goal is recoverable seasonal lift,
# not a competitive forecast benchmark.
# ─────────────────────────────────────────────────────────────────────
HOLIDAYS = {
    # New Year's Day
    (1, 1):   {"default": 0.55, "Beverages_Hot": 0.80},
    # Valentine's Day
    (2, 14):  {"default": 1.20, "Health_Beauty": 1.80, "Snacks": 1.40},
    # Easter (approximation: April 10)
    (4, 10):  {"default": 1.15, "Snacks": 1.45},
    # Memorial Day weekend (last Monday of May, approx May 27)
    (5, 27):  {"default": 1.25, "Beverages_Cold": 1.60},
    # Independence Day
    (7, 4):   {"default": 1.30, "Beverages_Cold": 1.85, "Snacks": 1.50},
    # Labor Day (first Monday in Sept, approx Sept 2)
    (9, 2):   {"default": 1.20, "Beverages_Cold": 1.40},
    # Halloween
    (10, 31): {"default": 1.10, "Snacks": 2.00},
    # Black Friday (day after Thanksgiving, approx Nov 28)
    (11, 28): {"default": 1.85, "Electronics": 3.50, "Apparel": 2.80, "Home_Goods": 2.20},
    # Cyber Monday (approx Dec 1)
    (12, 1):  {"default": 1.50, "Electronics": 2.80},
    # Christmas Eve / Christmas
    (12, 24): {"default": 1.55, "Electronics": 1.80, "Apparel": 1.70},
    (12, 25): {"default": 0.30},  # most stores closed or thin
    # New Year's Eve
    (12, 31): {"default": 1.35, "Beverages_Cold": 1.60, "Snacks": 1.50},
}


def holiday_lift(d: pd.Timestamp, category: str) -> float:
    """Return holiday lift multiplier for a date and category."""
    key = (d.month, d.day)
    if key not in HOLIDAYS:
        return 1.0
    rule = HOLIDAYS[key]
    return rule.get(category, rule.get("default", 1.0))


# ─────────────────────────────────────────────────────────────────────
# Day-of-week pattern. Saturday and Sunday are ~30-40% above baseline,
# Monday is the slowest day. This is the dominant weekly pattern in retail.
# ─────────────────────────────────────────────────────────────────────
DOW_MULTIPLIERS = np.array([
    0.85,  # Mon
    0.90,  # Tue
    0.95,  # Wed
    1.00,  # Thu
    1.15,  # Fri
    1.35,  # Sat
    1.30,  # Sun
])


def dow_lift(weekday: int) -> float:
    return float(DOW_MULTIPLIERS[weekday])


# ─────────────────────────────────────────────────────────────────────
# Synthetic daily weather per (region, date). We don't need real weather;
# we need *plausible* weather so the temperature coefficient on SKUs
# produces visible signal.
# ─────────────────────────────────────────────────────────────────────
REGION_BASE_TEMP = {
    "Northeast": 52,  # average annual F
    "Midwest":   50,
    "South":     68,
    "West":      62,
}
REGION_AMPLITUDE = {
    "Northeast": 22,  # peak-to-trough swing / 2
    "Midwest":   25,
    "South":     15,
    "West":      14,
}


def daily_temp(d: pd.Timestamp, region: str) -> float:
    """Simulated daily mean temperature in F for a region."""
    base = REGION_BASE_TEMP[region]
    amp = REGION_AMPLITUDE[region]
    # Day of year, 0-365
    doy = d.day_of_year
    # Sinusoid: hottest ~ July 22 (doy 203), coldest ~ Jan 22 (doy 22)
    seasonal = amp * np.sin(2 * np.pi * (doy - 113) / 365)
    # Daily noise
    noise = RNG.normal(0, 4)
    return float(base + seasonal + noise)


# ─────────────────────────────────────────────────────────────────────
# Promo flag: probability of being on promotion any given day.
# Roughly 12% of category-days have a promo; promos give 1.5-2.2x demand.
# ─────────────────────────────────────────────────────────────────────
def promo_for_day(category: str) -> tuple[bool, float]:
    """Return (is_on_promo, multiplier)."""
    is_promo = RNG.random() < 0.12
    if not is_promo:
        return False, 1.0
    # Promo lift varies by category
    lift = {
        "Electronics":     2.20,
        "Apparel":         1.95,
        "Home_Goods":      1.80,
        "Beverages_Hot":   1.50,
        "Beverages_Cold":  1.55,
        "Health_Beauty":   1.65,
        "Snacks":          1.60,
    }.get(category, 1.7)
    # Add some noise around the lift
    return True, float(lift * RNG.uniform(0.9, 1.1))


# ─────────────────────────────────────────────────────────────────────
# Year generator
# ─────────────────────────────────────────────────────────────────────
def generate_year(stores_df: pd.DataFrame, skus_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Generate one full year of daily store-SKU sales.
    Vectorized over (store, sku) cross product per day.
    """
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    n_days = (end - start).days + 1

    # Pre-compute date attributes
    dates = pd.date_range(start, end, freq="D")
    day_of_year = dates.dayofyear.values
    weekday = dates.weekday.values
    months = dates.month.values
    days = dates.day.values

    # Pre-compute weather per (region, date) — much cheaper than per row
    weather = {
        region: np.array([daily_temp(d, region) for d in dates])
        for region in REGION_BASE_TEMP
    }

    # Cross product (store_id, sku_id) once
    pairs = stores_df.merge(skus_df, how="cross", suffixes=("_store", "_sku"))
    n_pairs = len(pairs)

    # Pre-compute per-pair static factors
    pair_base = pairs["base_demand"].values * pairs["traffic_factor"].values
    pair_peak_month = pairs["peak_month"].values
    pair_amp = pairs["seasonality_amplitude"].values
    pair_temp_coef = pairs["weather_temp_coef"].values
    pair_region = pairs["region"].values
    pair_category = pairs["category"].values
    pair_lifecycle = pairs["lifecycle"].values

    # Lifecycle masks per pair (boolean array over the year)
    lifecycle_active = np.ones((n_pairs, n_days), dtype=bool)
    launch_date = np.array([
        date(2022, 6, 1) if lc == "launched" else date(1900, 1, 1)
        for lc in pair_lifecycle
    ])
    discontinue_date = np.array([
        date(2024, 6, 1) if lc == "discontinued" else date(2999, 12, 31)
        for lc in pair_lifecycle
    ])
    for i in range(n_days):
        d = start + timedelta(days=i)
        # Active if d >= launch and d < discontinue
        active = (launch_date <= d) & (d < discontinue_date)
        lifecycle_active[:, i] = active

    # Build day-by-day. We iterate over days (365) and vectorize over pairs (25,000).
    rows = []
    for i in tqdm(range(n_days), desc=f"  {year}", leave=False):
        d = dates[i]

        # Seasonal factor per pair: cosine peak at peak_month
        # Peak month converted to day-of-year (mid-month = day 15 of that month)
        peak_doy = (pair_peak_month - 1) * 30 + 15
        # Cosine: 1 at peak, -1 opposite; we want amp-scaled multiplier centered at 1
        seasonal = 1.0 + pair_amp * np.cos(2 * np.pi * (day_of_year[i] - peak_doy) / 365)

        # Day-of-week factor
        dow = dow_lift(weekday[i])

        # Holiday factor per pair (depends on category)
        h_lift = np.array([holiday_lift(d, c) for c in pair_category])

        # Weather effect per pair (depends on region)
        temp_per_pair = np.array([weather[r][i] for r in pair_region])
        # Effect: 1 + coef * (temp - 60). At 60F effect is 1.0; coef is per-degree.
        weather_factor = 1.0 + pair_temp_coef * (temp_per_pair - 60)
        weather_factor = np.clip(weather_factor, 0.3, 2.5)  # cap extremes

        # Promo per pair (independent draws)
        promo_flags = RNG.random(n_pairs) < 0.12
        promo_lift_arr = np.where(promo_flags,
                                  RNG.uniform(1.4, 2.1, size=n_pairs),
                                  1.0)

        # Combine multiplicatively
        mu = (
            pair_base
            * seasonal
            * dow
            * h_lift
            * weather_factor
            * promo_lift_arr
        )

        # Lifecycle mask: zero out inactive pairs
        active_today = lifecycle_active[:, i]
        mu = np.where(active_today, mu, 0.0)

        # Draw units sold from a Poisson with rate mu
        # (Negative binomial would be more realistic but Poisson is fast,
        # and for mu > 5 the difference is small.)
        units = RNG.poisson(np.maximum(mu, 0.0))

        # Skip rows with zero units only if SKU is inactive that day.
        # Active SKUs that randomly sold zero units stay in the dataset
        # (zero is a valid observation).
        keep = active_today
        if not keep.any():
            continue

        chunk = pd.DataFrame({
            "sale_date":   d.date(),
            "store_id":    pairs["store_id"].values[keep],
            "sku_id":      pairs["sku_id"].values[keep],
            "units_sold":  units[keep].astype(np.int32),
            "promo_flag":  promo_flags[keep],
            "weather_temp_f": np.round(temp_per_pair[keep], 1).astype(np.float32),
            "year":        np.int16(year),
        })
        # Compute net_revenue using SKU regular price minus a promo discount
        sku_price_lookup = pairs.set_index(pairs.index)["regular_price"].values
        chunk_prices = sku_price_lookup[keep]
        # Promo discount: 15-30% off when on promo
        discount = np.where(promo_flags[keep], RNG.uniform(0.15, 0.30, size=keep.sum()), 0.0)
        effective_price = chunk_prices * (1 - discount)
        chunk["net_revenue"] = np.round(units[keep] * effective_price, 2).astype(np.float32)
        chunk["price"] = np.round(effective_price, 2).astype(np.float32)

        rows.append(chunk)

    return pd.concat(rows, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    start_date = pd.Timestamp(os.environ.get("DATA_START_DATE", "2021-01-01"))
    end_date = pd.Timestamp(os.environ.get("DATA_END_DATE", "2025-12-31"))

    catalog_dir = Path("data/catalog")
    out_dir = Path("data/sales")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading catalog...")
    stores_df = pd.read_parquet(catalog_dir / "stores.parquet")
    skus_df = pd.read_parquet(catalog_dir / "skus.parquet")
    print(f"  {len(stores_df)} stores, {len(skus_df)} SKUs")
    print(f"  Cross product: {len(stores_df) * len(skus_df):,} pairs")

    total_rows = 0
    for year in range(start_date.year, end_date.year + 1):
        print(f"\nGenerating {year}...")
        year_df = generate_year(stores_df, skus_df, year)

        partition_dir = out_dir / f"year={year}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        out_path = partition_dir / "sales.parquet"

        # Compression: snappy is fast and BigQuery handles it natively
        year_df.to_parquet(out_path, index=False, compression="snappy")

        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  Wrote {out_path}: {len(year_df):,} rows, {size_mb:.1f} MB")
        total_rows += len(year_df)

    print()
    print("=" * 63)
    print(f"Done. Total rows: {total_rows:,}")
    print(f"Output: {out_dir}/")
    print("=" * 63)


if __name__ == "__main__":
    main()