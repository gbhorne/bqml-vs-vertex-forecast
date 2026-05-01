"""
data-generator/catalog.py
Generates the store and SKU dimension tables.

Stores are placed in real US metro areas with realistic coordinates.
Cycles through 25 metros: first pass = flagship, second pass = flagship,
third pass and beyond = satellite. With 75 stores total, that's 50 flagships
(2 per metro) and 25 satellites (1 per metro).

SKUs span seven retail categories with category-specific seasonality,
price ranges, and weather sensitivity.

Run independently:
    python data-generator/catalog.py

Outputs (under data/catalog/):
    stores.parquet  - one row per store
    skus.parquet    - one row per SKU
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Reproducibility: same seed = same catalog every time.
# We'll use a *different* seed in generate.py for the random sales noise.
RNG = np.random.default_rng(seed=42)

# ─────────────────────────────────────────────────────────────────────
# Stores: cycled across 25 US metros.
# (city, state, region, lat, lon, base_traffic_factor)
# base_traffic_factor: multiplier on baseline sales; flagship stores get 1.3+,
# satellite stores 0.7-0.9, average ~ 1.0
# ─────────────────────────────────────────────────────────────────────
METRO_AREAS = [
    ("New York",      "NY", "Northeast", 40.7128, -74.0060, 1.45),
    ("Los Angeles",   "CA", "West",      34.0522, -118.2437, 1.40),
    ("Chicago",       "IL", "Midwest",   41.8781, -87.6298, 1.25),
    ("Houston",       "TX", "South",     29.7604, -95.3698, 1.20),
    ("Phoenix",       "AZ", "West",      33.4484, -112.0740, 1.05),
    ("Philadelphia",  "PA", "Northeast", 39.9526, -75.1652, 1.10),
    ("San Antonio",   "TX", "South",     29.4241, -98.4936, 0.95),
    ("San Diego",     "CA", "West",      32.7157, -117.1611, 1.10),
    ("Dallas",        "TX", "South",     32.7767, -96.7970, 1.20),
    ("Austin",        "TX", "South",     30.2672, -97.7431, 1.05),
    ("Jacksonville",  "FL", "South",     30.3322, -81.6557, 0.90),
    ("Columbus",      "OH", "Midwest",   39.9612, -82.9988, 0.95),
    ("Charlotte",     "NC", "South",     35.2271, -80.8431, 1.00),
    ("Indianapolis",  "IN", "Midwest",   39.7684, -86.1581, 0.90),
    ("Seattle",       "WA", "West",      47.6062, -122.3321, 1.20),
    ("Denver",        "CO", "West",      39.7392, -104.9903, 1.05),
    ("Boston",        "MA", "Northeast", 42.3601, -71.0589, 1.25),
    ("Atlanta",       "GA", "South",     33.7490, -84.3880, 1.15),
    ("Miami",         "FL", "South",     25.7617, -80.1918, 1.15),
    ("Minneapolis",   "MN", "Midwest",   44.9778, -93.2650, 1.00),
    ("Detroit",       "MI", "Midwest",   42.3314, -83.0458, 0.85),
    ("Portland",      "OR", "West",      45.5152, -122.6784, 1.00),
    ("Nashville",     "TN", "South",     36.1627, -86.7816, 1.00),
    ("Baltimore",     "MD", "Northeast", 39.2904, -76.6122, 0.95),
    ("St. Louis",     "MO", "Midwest",   38.6270, -90.1994, 0.85),
]


def generate_stores(num_stores: int = 75) -> pd.DataFrame:
    """Build the store dimension. Cycles through metros; first two passes are
    flagship, third pass and beyond are satellites."""
    rows = []
    for store_idx in range(num_stores):
        metro_idx = store_idx % len(METRO_AREAS)
        # 0..24 = first flagship, 25..49 = second flagship, 50+ = satellite
        flagship_pass = store_idx // len(METRO_AREAS)
        is_flagship = flagship_pass < 2
        city, state, region, lat, lon, traffic_base = METRO_AREAS[metro_idx]

        # Within a metro, stagger stores slightly (different neighborhoods)
        traffic_jitter = RNG.normal(0, 0.05)
        traffic_factor = traffic_base * (1.0 + traffic_jitter)
        if not is_flagship:
            traffic_factor *= 0.85  # satellite stores ~ 15% smaller

        # Open dates: 60% opened pre-2019, 30% opened 2019-2022, 10% in 2023-2024
        opened_year = RNG.choice(
            [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            p=[0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.08, 0.07]
        )

        rows.append({
            "store_id": f"S{store_idx + 1:04d}",
            "store_name": f"{city} {'Flagship' if is_flagship else 'Satellite'}",
            "city": city,
            "state": state,
            "region": region,
            "latitude": round(lat + RNG.normal(0, 0.05), 6),
            "longitude": round(lon + RNG.normal(0, 0.05), 6),
            "store_format": "Flagship" if is_flagship else "Satellite",
            "square_feet": int(RNG.normal(35000 if is_flagship else 18000, 3000)),
            "opened_year": int(opened_year),
            "traffic_factor": round(traffic_factor, 4),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# SKUs: 700 SKUs across 7 categories.
# Each category has different seasonality, price range, and weather sensitivity.
# ─────────────────────────────────────────────────────────────────────
# Seasonality is encoded as (peak_month, trough_month, amplitude).
# amplitude 0.4 = swings between 60% and 140% of baseline across the year.
CATEGORIES = {
    # category, peak_month, trough_month, amplitude, weather_temp_coef
    # weather_temp_coef: per-degree-F change in demand (negative = cold-weather product)
    "Beverages_Hot":    {"peak": 1,  "trough": 7,  "amp": 0.45, "temp_coef": -0.012,
                         "price_min": 3, "price_max": 8, "share": 0.15},
    "Beverages_Cold":   {"peak": 7,  "trough": 1,  "amp": 0.50, "temp_coef": 0.018,
                         "price_min": 2, "price_max": 6, "share": 0.15},
    "Apparel":          {"peak": 11, "trough": 2,  "amp": 0.35, "temp_coef": 0.000,
                         "price_min": 15, "price_max": 120, "share": 0.20},
    "Electronics":      {"peak": 12, "trough": 6,  "amp": 0.55, "temp_coef": 0.000,
                         "price_min": 30, "price_max": 800, "share": 0.10},
    "Home_Goods":       {"peak": 11, "trough": 8,  "amp": 0.25, "temp_coef": 0.000,
                         "price_min": 8, "price_max": 90, "share": 0.20},
    "Health_Beauty":    {"peak": 12, "trough": 7,  "amp": 0.20, "temp_coef": 0.000,
                         "price_min": 4, "price_max": 45, "share": 0.10},
    "Snacks":           {"peak": 10, "trough": 3,  "amp": 0.30, "temp_coef": 0.005,
                         "price_min": 1.5, "price_max": 12, "share": 0.10},
}


def generate_skus(num_skus: int = 700) -> pd.DataFrame:
    """Build the SKU dimension with category-aware attributes."""
    rows = []
    sku_idx = 0

    for category, attrs in CATEGORIES.items():
        n_in_category = int(round(num_skus * attrs["share"]))

        for _ in range(n_in_category):
            sku_idx += 1

            # Base demand: log-normal so a few SKUs sell a lot, most sell modestly
            base_demand = float(RNG.lognormal(mean=2.0, sigma=0.7))

            # Price within category range, log-uniform
            price = float(np.exp(RNG.uniform(
                np.log(attrs["price_min"]),
                np.log(attrs["price_max"])
            )))

            # Lifecycle: 70% are continuously available, 20% launched mid-window,
            # 10% discontinued mid-window
            lifecycle = RNG.choice(
                ["continuous", "launched", "discontinued"],
                p=[0.70, 0.20, 0.10]
            )

            # Brand: 1 of 12 made-up brands, weighted toward a few popular ones
            brand_idx = int(RNG.choice(range(12), p=[
                0.18, 0.14, 0.12, 0.10, 0.09, 0.08,
                0.07, 0.06, 0.05, 0.04, 0.04, 0.03
            ]))

            rows.append({
                "sku_id": f"SKU{sku_idx:05d}",
                "sku_name": f"{category}_Item_{sku_idx}",
                "category": category,
                "subcategory": f"{category}_Sub_{sku_idx % 5 + 1}",
                "brand": f"Brand_{brand_idx + 1:02d}",
                "regular_price": round(price, 2),
                "base_demand": round(base_demand, 3),
                "peak_month": attrs["peak"],
                "trough_month": attrs["trough"],
                "seasonality_amplitude": attrs["amp"],
                "weather_temp_coef": attrs["temp_coef"],
                "lifecycle": lifecycle,
                "launch_date": "2022-06-01" if lifecycle == "launched" else None,
                "discontinue_date": "2024-06-01" if lifecycle == "discontinued" else None,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────
def main():
    num_stores = int(os.environ.get("NUM_STORES", 75))
    num_skus = int(os.environ.get("NUM_SKUS", 700))

    out_dir = Path("data/catalog")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_stores} stores...")
    stores_df = generate_stores(num_stores)
    stores_df.to_parquet(out_dir / "stores.parquet", index=False)
    print(f"  Wrote {out_dir / 'stores.parquet'} ({len(stores_df)} rows)")

    print(f"Generating {num_skus} SKUs...")
    skus_df = generate_skus(num_skus)
    skus_df.to_parquet(out_dir / "skus.parquet", index=False)
    print(f"  Wrote {out_dir / 'skus.parquet'} ({len(skus_df)} rows)")

    print()
    print("Sample stores:")
    print(stores_df.head(5).to_string(index=False))
    print()
    print("Store format distribution:")
    print(stores_df["store_format"].value_counts().to_string())
    print()
    print("Sample SKUs:")
    print(skus_df[["sku_id", "category", "regular_price", "base_demand", "lifecycle"]].head(5).to_string(index=False))
    print()
    print("Category distribution:")
    print(skus_df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()