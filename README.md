# Climate-Driven-Honey-Production-Forecasting-Synthetic-Dataset
Python script to generate a large-scale, realistic synthetic honey production dataset with 200,000+ rows and 51 feature columns. Includes climate, apiary management, environmental, and economic factors for machine learning, data science practice, predictive modeling, and climate-driven honey yield research.
## 📋 Table of Contents

- [Overview](#overview)
- [Dataset Summary](#dataset-summary)
- [Getting Started](#getting-started)
- [Column Reference](#column-reference)
- [Production Formula](#production-formula)
- [Configuration](#configuration)
- [Data Quality](#data-quality)
- [Use Cases](#use-cases)
- [Requirements](#requirements)
- [License](#license)

---

## Overview

This project generates a fully synthetic dataset simulating monthly honey production records from 1,000 apiaries across 10 countries (2010–2024). All values are statistically modeled with realistic correlations — temperature affects flower density, disease reduces colony strength, experienced beekeepers produce more, and so on — making it suitable for building and benchmarking predictive models.

The script is vectorized with NumPy for fast generation (~10 seconds for 200k rows) and includes built-in data quality assertions.

---

## Dataset Summary

| Property | Value |
|---|---|
| Rows | 200,000 |
| Columns | 51 |
| Date Range | January 2010 – December 2024 |
| Frequency | Monthly |
| Apiaries | 1,000 |
| Countries | 10 |
| Random Seed | 42 (reproducible) |
| Output Format | CSV |
| File Size | ~155 MB |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/honey-production-dataset.git
cd honey-production-dataset
```

### 2. Install dependencies

```bash
pip install numpy pandas
```

### 3. Run the generator

```bash
python generate_honey_data.py
```

The CSV file `synthetic_honey_production_dataset.csv` will be created in the same directory.

### 4. Load in Python

```python
import pandas as pd

df = pd.read_csv("synthetic_honey_production_dataset.csv", parse_dates=["date"])
print(df.shape)        # (200000, 51)
print(df.head())
```

---

## Column Reference

### 🕐 Temporal Features

| Column | Type | Description |
|---|---|---|
| `date` | datetime | First day of the month (monthly record) |
| `year` | int | Year (2010–2024) |
| `month` | int | Month number (1–12) |
| `quarter` | int | Calendar quarter (1–4) |
| `season` | str | Season: Winter, Spring, Summer, Autumn |

### 📍 Location Features

| Column | Type | Description |
|---|---|---|
| `country` | str | Country: USA, Canada, Brazil, Germany, India, Australia, France, Argentina, China, Kenya |
| `state_region` | str | Region: North, South, East, West, Central |
| `latitude` | float | Latitude (−45 to 55) |
| `longitude` | float | Longitude (−120 to 120) |
| `altitude_m` | float | Elevation above sea level in metres (0–2000) |

### 🌦️ Weather Features

| Column | Type | Description |
|---|---|---|
| `avg_temperature` | float | Monthly average temperature (°C), seasonally modelled with country offsets |
| `min_temperature` | float | Monthly minimum temperature (°C) |
| `max_temperature` | float | Monthly maximum temperature (°C) |
| `rainfall_mm` | float | Monthly rainfall in millimetres |
| `humidity_percent` | float | Average relative humidity (20–100%) |
| `sunshine_hours` | float | Monthly sunshine hours (50–350) |
| `wind_speed_kmh` | float | Average wind speed in km/h (0–35) |
| `uv_index` | float | UV index (0–11), seasonally modelled |
| `drought_index` | float | Drought severity (0 = no drought, 1 = extreme drought) |

### 🌿 Environmental Features

| Column | Type | Description |
|---|---|---|
| `flower_density_index` | float | Abundance of flowering plants near the apiary (0–1) |
| `pesticide_exposure_index` | float | Pesticide contamination level (0–1), Beta-distributed |
| `land_use_type` | str | Agricultural, Forest, Mixed, Urban |
| `dominant_flower_type` | str | Clover, Wildflower, Sunflower, Acacia, Mixed, Lavender |
| `nearby_crop_type` | str | Alfalfa, Canola, Sunflower, Orchard, None, Wheat |
| `air_quality_index` | float | Air quality index (10–300), Gamma-distributed |

### 🐝 Apiary & Colony Features

| Column | Type | Description |
|---|---|---|
| `apiary_id` | int | Unique apiary identifier (1–1000) |
| `number_of_hives` | int | Number of hives in the apiary (10–500) |
| `disease_incidence_rate` | float | Proportion of hives affected by disease (0–1) |
| `varroa_mite_index` | float | Varroa mite infestation severity (0–1) |
| `avg_colony_strength` | float | Colony health score (1–10); penalised by disease and pesticides |
| `queen_age_months` | int | Age of the queen bee in months (1–36) |
| `queen_replaced_this_month` | int | Binary flag: queen replacement event (5% probability) |
| `feeding_supplement_used` | int | Binary flag: 1 if supplemental feeding was used (Winter or drought) |
| `beekeeping_experience_years` | int | Years of experience of the beekeeper (0–40) |
| `beekeeper_certification` | str | Certification level: None, Basic, Advanced |
| `hive_inspection_score` | float | Overall hive health score from inspection (1–10) |

### 💰 Economic Features

| Column | Type | Description |
|---|---|---|
| `honey_price_per_kg` | float | Market price of honey in USD/kg; increases ~$0.20/year |
| `labor_hours` | float | Total labour hours spent on the apiary that month |
| `maintenance_cost` | float | Total maintenance cost in USD |
| `equipment_type` | str | Hive equipment type: Traditional, Semi-Modern, Modern |

### ⚠️ Flag Features

| Column | Type | Description |
|---|---|---|
| `anomaly_weather_flag` | int | 1 if extreme weather occurred (heavy rain, heat, or cold) |
| `frost_risk_flag` | int | 1 if minimum temperature dropped below 0°C |
| `heat_stress_flag` | int | 1 if average temperature exceeded 36°C |

### 🎯 Target & Derived Features

| Column | Type | Description |
|---|---|---|
| `honey_production_kg` | float | **Primary target** — total honey produced in kg (50–5000) |
| `revenue` | float | Gross revenue in USD (production × price) |
| `profit` | float | Net profit in USD (revenue − maintenance cost) |

### 📈 Lag & Rolling Features

| Column | Type | Description |
|---|---|---|
| `previous_month_production` | float | Honey production from the previous month (per apiary) |
| `rolling_3_month_avg_production` | float | 3-month rolling average production (per apiary) |
| `rolling_6_month_avg_production` | float | 6-month rolling average production (per apiary) |
| `yoy_production_change` | float | Difference vs. previous month production |
| `colony_loss_rate` | float | Estimated proportion of colony losses (0–0.5) |

---

## Production Formula

Honey production is modelled as a multiplicative function of key factors:

```
production = number_of_hives
           × avg_colony_strength
           × flower_density_index
           × temperature_factor       # 1.2 if 20–35°C, else 0.7
           × rain_penalty             # 0.6 if rainfall > 300mm, else 1.0
           × (1 − pesticide_exposure_index)
           × (1 − disease_incidence_rate)
           × (1 − drought_index)
           × experience_bonus         # 1 + 0.005 × years_experience
           × queen_penalty            # 0.85 if queen_age > 24 months
           × equipment_bonus          # Modern=1.1, Semi-Modern=1.05, Traditional=1.0
           × noise                    # Normal(1, 0.1)
```

Final values are clipped to **[50, 5000] kg**.

---

## Configuration

Edit the constants at the top of `generate_honey_data.py` to customise the output:

```python
NUM_ROWS      = 200_000        # Number of records to generate
NUM_APIARIES  = 1_000          # Number of unique apiaries
START_DATE    = "2010-01-01"   # Start of date range
END_DATE      = "2024-12-01"   # End of date range
np.random.seed(42)             # Change for different random data
```

---

## Data Quality

The script runs 5 built-in assertions before saving:

- `honey_production_kg` is within [50, 5000]
- `humidity_percent` is within [20, 100]
- `min_temperature` ≤ `avg_temperature` for every row
- `avg_temperature` ≤ `max_temperature` for every row
- Row count is at least 200,000

Any failure raises an `AssertionError` with a descriptive message.

---

## Use Cases

- **Regression / Forecasting** — predict `honey_production_kg` from weather, colony, and environmental features
- **Time Series Analysis** — use lag/rolling features and the monthly structure for ARIMA, LSTM, or Prophet models
- **Feature Importance Studies** — rich multi-domain feature set for SHAP or permutation importance analysis
- **Clustering** — group apiaries by production patterns, geography, or environmental conditions
- **Data Engineering Practice** — ETL pipelines, partitioning by country/year, aggregation workflows
- **Dashboard / BI** — visualise production trends, seasonal patterns, and profitability by region

---

## Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.8 |
| NumPy | ≥ 1.21 |
| Pandas | ≥ 1.3 |

Install with:

```bash
pip install numpy pandas
```

---

## Project Structure

```
honey-production-dataset/
│
├── generate_honey_data.py              # Main data generation script
├── synthetic_honey_production_dataset.csv  # Generated dataset (run script first)
└── README.md                           # This file
```

---

## License

This project is released under the MIT License. The dataset is entirely synthetic and does not contain any real or personally identifiable information.

---

*Generated with NumPy seed 42 — fully reproducible. Run the script again with the same seed to get identical output.*
