import numpy as np
import pandas as pd
import time

# ----------------------------
# CONFIGURATION
# ----------------------------
NUM_ROWS = 200_000          # 2 Lakh rows
NUM_APIARIES = 1_000        # More apiaries for diversity
START_DATE = "2010-01-01"   # Extended date range for more temporal coverage
END_DATE = "2024-12-01"

np.random.seed(42)

print(f"Generating {NUM_ROWS:,} rows of synthetic honey production data...")
start_time = time.time()

# ----------------------------
# TIME INDEX (Monthly)
# ----------------------------
dates = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
date_sample = np.random.choice(dates, NUM_ROWS)

df = pd.DataFrame({"date": date_sample})
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter   # NEW: quarter feature

# --- BUG FIX: Vectorized season assignment (avoid slow .apply on 200k rows) ---
season_map = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring",  4: "Spring",  5: "Spring",
              6: "Summer",  7: "Summer",  8: "Summer",
              9: "Autumn",  10: "Autumn", 11: "Autumn"}
df["season"] = df["month"].map(season_map)

# ----------------------------
# LOCATION
# ----------------------------
countries = ["USA", "Canada", "Brazil", "Germany", "India",
             "Australia", "France", "Argentina", "China", "Kenya"]   # Expanded
regions   = ["North", "South", "East", "West", "Central"]

df["country"]      = np.random.choice(countries, NUM_ROWS)
df["state_region"] = np.random.choice(regions, NUM_ROWS)
df["latitude"]     = np.random.uniform(-45, 55, NUM_ROWS).round(4)
df["longitude"]    = np.random.uniform(-120, 120, NUM_ROWS).round(4)
df["altitude_m"]   = np.random.uniform(0, 2000, NUM_ROWS).round(1)

# Country-based climate offset (NEW: realistic regional bias)
country_temp_offsets = {
    "USA": 0, "Canada": -5, "Brazil": 8, "Germany": -3,
    "India": 7, "Australia": 5, "France": -1, "Argentina": 2,
    "China": 1, "Kenya": 6
}
temp_offset = df["country"].map(country_temp_offsets).values

# ----------------------------
# WEATHER (Seasonal Pattern)
# ----------------------------
month_angle = 2 * np.pi * (df["month"] / 12)

base_temp           = 20 + 10 * np.sin(month_angle) + temp_offset
df["avg_temperature"] = (base_temp + np.random.normal(0, 5, NUM_ROWS)).round(2)
df["min_temperature"] = (df["avg_temperature"] - np.random.uniform(3, 10, NUM_ROWS)).round(2)
df["max_temperature"] = (df["avg_temperature"] + np.random.uniform(3, 10, NUM_ROWS)).round(2)

df["rainfall_mm"] = np.abs(
    100 + 80 * np.sin(month_angle + np.pi / 3) + np.random.normal(0, 40, NUM_ROWS)
).round(2)

df["humidity_percent"] = np.clip(
    60 + 20 * np.sin(month_angle) + np.random.normal(0, 10, NUM_ROWS),
    20, 100
).round(2)

df["sunshine_hours"] = np.clip(
    200 + 80 * np.sin(month_angle - np.pi / 4) + np.random.normal(0, 30, NUM_ROWS),
    50, 350
).round(2)

df["wind_speed_kmh"] = np.random.uniform(0, 35, NUM_ROWS).round(2)

# NEW: UV Index
df["uv_index"] = np.clip(
    5 + 4 * np.sin(month_angle) + np.random.normal(0, 1.5, NUM_ROWS),
    0, 11
).round(1)

df["drought_index"] = np.clip(1 - (df["rainfall_mm"] / 400), 0, 1).round(4)

# ----------------------------
# ENVIRONMENTAL FACTORS
# ----------------------------
temp_in_range = df["avg_temperature"].between(20, 35).astype(float)

df["flower_density_index"] = np.clip(
    0.3 +
    0.4 * temp_in_range +
    0.3 * (1 - df["drought_index"]) +
    np.random.normal(0, 0.1, NUM_ROWS),
    0, 1
).round(4)

df["pesticide_exposure_index"] = np.clip(
    np.random.beta(2, 5, NUM_ROWS), 0, 1
).round(4)

df["land_use_type"]        = np.random.choice(["Agricultural", "Forest", "Mixed", "Urban"], NUM_ROWS)
df["dominant_flower_type"] = np.random.choice(["Clover", "Wildflower", "Sunflower", "Acacia", "Mixed", "Lavender"], NUM_ROWS)
df["nearby_crop_type"]     = np.random.choice(["Alfalfa", "Canola", "Sunflower", "Orchard", "None", "Wheat"], NUM_ROWS)

# NEW: Pollution index
df["air_quality_index"] = np.clip(
    np.random.gamma(2, 25, NUM_ROWS), 10, 300
).round(1)

# ----------------------------
# APIARY CHARACTERISTICS
# ----------------------------
df["apiary_id"]      = np.random.randint(1, NUM_APIARIES + 1, NUM_ROWS)
df["number_of_hives"] = np.random.randint(10, 500, NUM_ROWS)

df["disease_incidence_rate"] = np.clip(
    np.random.beta(2, 6, NUM_ROWS), 0, 1
).round(4)

# --- BUG FIX: varroa_mite_index was not clipped correctly in edge cases ---
df["varroa_mite_index"] = np.clip(
    df["disease_incidence_rate"] + np.random.normal(0, 0.1, NUM_ROWS),
    0, 1
).round(4)

df["avg_colony_strength"] = np.clip(
    8 -
    4 * df["disease_incidence_rate"] -
    2 * df["pesticide_exposure_index"] +
    np.random.normal(0, 1, NUM_ROWS),
    1, 10
).round(2)

df["queen_age_months"]            = np.random.randint(1, 36, NUM_ROWS)
df["queen_replaced_this_month"]   = (np.random.rand(NUM_ROWS) < 0.05).astype(int)  # NEW: 5% chance

# --- BUG FIX: feeding_supplement_used was only True for Winter, now includes drought stress ---
df["feeding_supplement_used"] = (
    (df["season"] == "Winter") | (df["drought_index"] > 0.7)
).astype(int)

df["beekeeping_experience_years"] = np.random.randint(0, 40, NUM_ROWS)
df["beekeeper_certification"]     = np.random.choice(["None", "Basic", "Advanced"], NUM_ROWS)  # NEW

# NEW: Hive inspection score (1-10)
df["hive_inspection_score"] = np.clip(
    df["avg_colony_strength"] - 2 * df["disease_incidence_rate"] + np.random.normal(0, 0.5, NUM_ROWS),
    1, 10
).round(2)

# ----------------------------
# ECONOMIC FACTORS
# ----------------------------
year_min = df["year"].min()
df["honey_price_per_kg"] = (
    5 +
    0.2 * (df["year"] - year_min) +
    np.random.normal(0, 0.5, NUM_ROWS)
).round(2)

df["labor_hours"]      = (df["number_of_hives"] * np.random.uniform(0.2, 0.5, NUM_ROWS)).round(2)
df["maintenance_cost"] = (df["number_of_hives"] * np.random.uniform(5, 15, NUM_ROWS)).round(2)

df["equipment_type"] = np.random.choice(["Traditional", "Modern", "Semi-Modern"], NUM_ROWS)  # NEW: added type

# ----------------------------
# EXTREME WEATHER FLAG
# ----------------------------
df["anomaly_weather_flag"] = (
    (df["rainfall_mm"] > 300) |
    (df["avg_temperature"] > 38) |
    (df["avg_temperature"] < 5)
).astype(int)

# NEW: Frost flag
df["frost_risk_flag"] = (df["min_temperature"] < 0).astype(int)

# NEW: Heat stress flag
df["heat_stress_flag"] = (df["avg_temperature"] > 36).astype(int)

# ----------------------------
# HONEY PRODUCTION TARGET
# ----------------------------
temp_factor  = np.where(df["avg_temperature"].between(20, 35), 1.2, 0.7)
rain_penalty = np.where(df["rainfall_mm"] > 300, 0.6, 1.0)

# NEW: Experience bonus (more experienced beekeepers produce more)
experience_bonus = 1 + 0.005 * df["beekeeping_experience_years"]

# NEW: Queen penalty (old queens reduce production)
queen_penalty = np.where(df["queen_age_months"] > 24, 0.85, 1.0)

# NEW: Equipment bonus
equipment_bonus = np.where(df["equipment_type"] == "Modern", 1.1,
                  np.where(df["equipment_type"] == "Semi-Modern", 1.05, 1.0))

production = (
    df["number_of_hives"] *
    df["avg_colony_strength"] *
    df["flower_density_index"] *
    temp_factor *
    rain_penalty *
    (1 - df["pesticide_exposure_index"]) *
    (1 - df["disease_incidence_rate"]) *
    (1 - df["drought_index"]) *
    experience_bonus *
    queen_penalty *
    equipment_bonus
)

noise = np.random.normal(1, 0.1, NUM_ROWS)

df["honey_production_kg"] = np.clip(
    production * noise,
    50,
    5000
).round(2)

# NEW: Derived economic target
df["revenue"] = (df["honey_production_kg"] * df["honey_price_per_kg"]).round(2)
df["profit"]  = (df["revenue"] - df["maintenance_cost"]).round(2)

# ----------------------------
# LAG FEATURES (sorted by apiary + date for correctness)
# ----------------------------
df = df.sort_values(["apiary_id", "date"]).reset_index(drop=True)

global_mean_prod = df["honey_production_kg"].mean()

df["previous_month_production"] = (
    df.groupby("apiary_id")["honey_production_kg"]
    .shift(1)
    .fillna(global_mean_prod)
    .round(2)
)

df["rolling_3_month_avg_production"] = (
    df.groupby("apiary_id")["honey_production_kg"]
    .transform(lambda x: x.rolling(3, min_periods=1).mean())
    .round(2)
)

# NEW: 6-month rolling average
df["rolling_6_month_avg_production"] = (
    df.groupby("apiary_id")["honey_production_kg"]
    .transform(lambda x: x.rolling(6, min_periods=1).mean())
    .round(2)
)

# NEW: Year-over-year production change
df["yoy_production_change"] = (
    df["honey_production_kg"] - df["previous_month_production"]
).round(2)

df["colony_loss_rate"] = np.clip(
    1 - (df["avg_colony_strength"] / 10) + np.random.normal(0, 0.05, NUM_ROWS),
    0, 0.5
).round(4)

# ----------------------------
# DATA QUALITY CHECKS
# ----------------------------
assert df["honey_production_kg"].between(50, 5000).all(), "Production out of bounds!"
assert df["humidity_percent"].between(20, 100).all(), "Humidity out of bounds!"
assert (df["min_temperature"] <= df["avg_temperature"]).all(), "Min temp > Avg temp!"
assert (df["avg_temperature"] <= df["max_temperature"]).all(), "Avg temp > Max temp!"
assert df.shape[0] >= 200_000, f"Expected 200k rows, got {df.shape[0]}"
print("✅ All data quality checks passed.")

# ----------------------------
# SAVE DATASET
# ----------------------------
output_path = "/mnt/user-data/outputs/synthetic_honey_production_dataset.csv"
df.to_csv(output_path, index=False)

elapsed = time.time() - start_time
print(f"\n✅ Dataset generated successfully in {elapsed:.2f}s")
print(f"   Rows    : {df.shape[0]:,}")
print(f"   Columns : {df.shape[1]}")
print(f"   File    : {output_path}")
print(f"\nColumn list:\n{list(df.columns)}")
print(f"\nSample:\n{df.head(3).to_string()}")
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
