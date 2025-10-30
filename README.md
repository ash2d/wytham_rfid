# Wytham RFID Analysis

Analysis tools for bird movement data from RFID feeders in Wytham Woods.

## Overview

This repository provides tools for analyzing bird visitation patterns and movement between RFID-tagged feeders, with a focus on spatial gravity models.

## Features

### Gravity Model Analysis (`gravity_model.py`)

A complete implementation of spatial gravity models for analyzing bird movement patterns between feeders:

- **Data Processing**
  - Load bout data from CSV (bird visits at feeders)
  - Build transitions from consecutive bouts
  - Aggregate Origin-Destination (OD) flows with exposure calculation
  - Compute distance matrices (Euclidean from OSGB coordinates)

- **Statistical Modeling**
  - Poisson GLM with exposure offset
  - Negative-Binomial GLM for overdispersed data
  - Automatic model selection via AIC/BIC
  - Overdispersion testing
  - k-fold cross-validation

- **Covariates**
  - Distance-decay parameter (γ)
  - Origin and destination attractiveness (total visits)
  - Time of day stratification (morning/afternoon/evening/night)
  - Species-specific effects
  - Seasonal trends (day of year)

- **Outputs**
  - Model coefficients (CSV)
  - Predicted flows and residuals (CSV)
  - Diagnostic plots (observed vs predicted, residual distribution, Q-Q plots)
  - Distance-decay curves with confidence intervals
  - Flow maps showing residual edges
  - Cross-validation metrics (RMSE, MAE)

## Installation

```bash
# Clone the repository
git clone https://github.com/ash2d/wytham_rfid.git
cd wytham_rfid

# Install dependencies (requires Python 3.12+)
pip install -e .
```

## Usage

### Command Line Interface

The gravity model can be run from the command line:

```bash
# With your own data
python gravity_model.py --bouts bouts.csv --feeders feeders.csv --output results/

# With sample data (for testing)
python gravity_model.py --use-sample-data --output results/

# Fit per-species models
python gravity_model.py --bouts bouts.csv --feeders feeders.csv --per-species --output results/

# Without time-of-day stratification
python gravity_model.py --bouts bouts.csv --feeders feeders.csv --time-block none --output results/

# Custom cross-validation folds
python gravity_model.py --use-sample-data --cv-folds 10 --output results/
```

**Command-line options:**

- `--bouts PATH`: Path to bouts CSV file (required unless --use-sample-data)
- `--feeders PATH`: Path to feeders CSV file (required unless --use-sample-data)
- `--output DIR`: Output directory (default: `output`)
- `--use-sample-data`: Generate and use sample data for testing
- `--per-species`: Fit separate models for each species
- `--time-block {time_of_day,none}`: Time stratification (default: time_of_day)
- `--cv-folds N`: Number of cross-validation folds (default: 5)

### Jupyter Notebook

For interactive analysis, see `gravity_model_example.ipynb`:

```python
from gravity_model import *

# Load data
bouts_df, feeders_df = load_data('bouts.csv', 'feeders.csv')

# Or generate sample data
bouts_df, feeders_df = generate_sample_data(n_birds=50, n_feeders=10)

# Build transitions
transitions_df = build_transitions(bouts_df)

# Compute distances
distance_df = compute_distance_matrix(feeders_df)

# Aggregate OD flows
od_df = aggregate_od(transitions_df, bouts_df, distance_df, time_block='time_of_day')

# Fit models
poisson_result = fit_poisson_model(od_df)
nb_result = fit_negative_binomial_model(od_df)

# Select best
best_model, model_name = select_best_model(poisson_result, nb_result)

# Evaluate
residuals_df = compute_residuals(od_df, best_model)
cv_results = cross_validate_model(od_df, formula, n_folds=5, model_type=model_name)

# Visualize
plot_diagnostics(residuals_df, output_dir, model_name)
plot_distance_decay(residuals_df, best_model, output_dir, model_name)
plot_flow_map(residuals_df, feeders_df, output_dir, model_name)
```

## Data Format

### Input Files

**bouts.csv** - Bird visit records:
```csv
bird_id,feeder_id,bout_start,bout_end,species
B0001,F001,2024-03-01 06:15:23,2024-03-01 06:18:45,GRETI
B0001,F003,2024-03-01 06:22:10,2024-03-01 06:25:33,GRETI
...
```

Columns:
- `bird_id`: Unique bird identifier
- `feeder_id`: Feeder identifier
- `bout_start`: ISO datetime of visit start
- `bout_end`: ISO datetime of visit end
- `species`: Bird species code (e.g., GRETI, BLUTI, COATI, MARTI)

**feeders.csv** - Feeder locations:
```csv
feeder_id,x,y
F001,450123.45,207234.56
F002,450234.67,207345.78
...
```

Columns:
- `feeder_id`: Feeder identifier (must match bouts.csv)
- `x`: Easting coordinate in OSGB metres (EPSG:27700)
- `y`: Northing coordinate in OSGB metres (EPSG:27700)

### Output Files

The script generates several output files in the specified output directory:

**Model coefficients:**
- `{model}_coefficients.csv`: Parameter estimates, standard errors, z-values, p-values

**Predictions and residuals:**
- `{model}_predicted_flows.csv`: Observed vs predicted flows for each OD pair
- `{model}_top_residuals.csv`: Top 20 edges by residual magnitude

**Diagnostic plots:**
- `{model}_diagnostics.png`: 4-panel diagnostic plot (observed vs predicted, residual histogram, residuals vs fitted, Q-Q plot)
- `{model}_distance_decay.png`: Distance-decay relationship with fitted curve
- `{model}_flow_map.png`: Spatial map of feeders with residual flow arrows

**Sample data (if --use-sample-data):**
- `sample_bouts.csv`: Generated sample bout data
- `sample_feeders.csv`: Generated sample feeder locations

## Gravity Model Specification

The gravity model is specified as a Generalized Linear Model (GLM) with log link:

```
log(E[n_ij]) = log(exposure_i) + α + O_i + D_j - γ·log(distance_ij + 1) + β_time + β_season + β_species + ε
```

Where:
- `n_ij`: Count of transitions from origin i to destination j
- `exposure_i`: Total time birds spent at origin i (offset)
- `O_i`: Origin attractiveness (log of total visits at i)
- `D_j`: Destination attractiveness (log of total visits at j)
- `γ`: Distance-decay parameter (higher = stronger decay)
- `distance_ij`: Euclidean distance between feeders i and j
- `β_time`: Time-of-day fixed effects (morning, afternoon, evening, night)
- `β_season`: Day of year (linear trend)
- `β_species`: Species fixed effects

The model is fit using:
1. **Poisson GLM**: Assumes mean = variance
2. **Negative-Binomial GLM**: Allows overdispersion (variance > mean)

Model selection is based on:
- AIC/BIC comparison (lower is better)
- Overdispersion test (Pearson χ² / df > 1.5 suggests overdispersion)

## Model Interpretation

### Distance-Decay Parameter (γ)

The coefficient on `log_distance` represents the distance-decay effect:

- **γ < 0**: Negative decay (transitions decrease with distance) - **expected**
- **γ = -0.5**: ~40% reduction in transition rate per log-unit distance
- **γ = -1.0**: ~63% reduction per log-unit distance
- **γ ≈ 0**: No distance effect
- **γ > 0**: Positive effect (unusual, may indicate spatial confounding)

### Attractiveness

- **log_origin_visits > 0**: Birds are more likely to leave busy feeders
- **log_origin_visits < 0**: Birds are less likely to leave busy feeders
- **log_dest_visits > 0**: Birds are attracted to busy feeders
- **log_dest_visits < 0**: Birds avoid busy feeders

### Time of Day

Positive coefficients indicate higher transition rates during that time period relative to the reference category (typically night).

### Cross-Validation

- **RMSE (Root Mean Squared Error)**: Average prediction error in transition counts
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- Lower values indicate better out-of-sample predictions

## Dependencies

- Python >= 3.12
- pandas >= 2.3.3
- numpy >= 2.3.3
- scipy >= 1.16.2
- matplotlib >= 3.10.7
- statsmodels >= 0.14.0
- scikit-learn >= 1.3.0
- geopandas >= 0.14.0 (optional, for spatial mapping)

## Examples

See `gravity_model_example.ipynb` for a complete worked example with sample data.

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```

## License

[Add license information here]

## Contact

For questions or issues, please contact [add contact information] or open an issue on GitHub.
