# Gravity Model Implementation Guide

## Quick Start

The gravity model script is ready to use! Here's how to get started:

### 1. Test with Sample Data

```bash
python gravity_model.py --use-sample-data --output demo/
```

This generates:
- Sample bout data (50 birds, 10 feeders, 1000 bouts)
- Feeder coordinates
- Fits gravity model
- Produces all outputs

### 2. Run with Your Data

```bash
python gravity_model.py \
  --bouts your_bouts.csv \
  --feeders your_feeders.csv \
  --output results/
```

### 3. Explore Interactively

Open `gravity_model_example.ipynb` in Jupyter for step-by-step tutorial.

## What the Script Does

### Input Data

**bouts.csv**: Bird visit records
```
bird_id, feeder_id, bout_start, bout_end, species
B0001, F001, 2024-03-01 06:15:23, 2024-03-01 06:18:45, GRETI
B0001, F003, 2024-03-01 06:22:10, 2024-03-01 06:25:33, GRETI
```

**feeders.csv**: Feeder locations (OSGB metres)
```
feeder_id, x, y
F001, 450123.45, 207234.56
F002, 450234.67, 207345.78
```

### Processing Steps

1. **Build Transitions**: Create consecutive movements between feeders for each bird
2. **Compute Distances**: Calculate Euclidean distances between all feeder pairs
3. **Aggregate Flows**: Count transitions by origin-destination-species-time
4. **Calculate Exposure**: Total time birds spent at each origin feeder
5. **Fit Models**: Both Poisson and Negative-Binomial GLMs
6. **Select Best**: Choose model based on AIC/BIC and overdispersion
7. **Cross-Validate**: k-fold CV for out-of-sample performance
8. **Diagnose**: Residual plots and statistical tests
9. **Visualize**: Distance-decay curves and flow maps
10. **Export**: Save coefficients, predictions, and plots

### Model Specification

The gravity model estimates transition counts as:

```
log(E[n_ij]) = log(exposure_i) + α + O_i + D_j - γ·log(distance_ij + 1) 
               + β_time + β_season + β_species
```

Where:
- `n_ij`: Transitions from feeder i to j
- `exposure_i`: Time at origin (offset)
- `O_i`, `D_j`: Origin/destination attractiveness
- `γ`: Distance-decay parameter
- `β`: Fixed effects for time, season, species

### Key Parameters

**Distance-decay (γ)**: How transitions decrease with distance
- γ = -0.5 → 40% reduction per log-unit distance
- γ = -1.0 → 63% reduction per log-unit distance

**Attractiveness**:
- Positive `log_dest_visits`: Birds attracted to busy feeders
- Positive `log_origin_visits`: More departures from busy feeders

**Time effects**: Relative transition rates by time of day

## Output Files

All files saved to specified output directory:

### CSV Files

1. **{model}_coefficients.csv**
   - Parameter estimates
   - Standard errors
   - z-values and p-values

2. **{model}_predicted_flows.csv**
   - Observed vs predicted for each OD pair
   - Raw, Pearson, and deviance residuals
   - Distance and exposure values

3. **{model}_top_residuals.csv**
   - Top 20 edges by residual magnitude
   - Identifies poorly predicted flows

4. **sample_bouts.csv** (if --use-sample-data)
   - Generated sample bout data

5. **sample_feeders.csv** (if --use-sample-data)
   - Generated feeder locations

### Plot Files

1. **{model}_diagnostics.png**
   - 4-panel diagnostic plot
   - Observed vs predicted scatter
   - Residual histogram
   - Residuals vs fitted values
   - Q-Q plot

2. **{model}_distance_decay.png**
   - Distance-decay relationship
   - Observed vs fitted curve
   - 95% confidence interval
   - Estimated γ parameter

3. **{model}_flow_map.png**
   - Spatial map of feeders
   - Arrows for top residual edges
   - Red = over-predicted
   - Blue = under-predicted

## Command-Line Options

```bash
python gravity_model.py [OPTIONS]

Required (unless --use-sample-data):
  --bouts PATH          Path to bouts CSV
  --feeders PATH        Path to feeders CSV

Optional:
  --output DIR          Output directory (default: output)
  --use-sample-data     Generate test data
  --per-species         Fit separate models per species
  --time-block BLOCK    Time stratification (time_of_day|none)
  --cv-folds N          Number of CV folds (default: 5)
```

## Advanced Usage

### Per-Species Models

Fit separate models for each species to capture species-specific movement patterns:

```bash
python gravity_model.py \
  --bouts bouts.csv \
  --feeders feeders.csv \
  --per-species \
  --output species_models/
```

This creates subdirectories for each species with separate outputs.

### Custom Cross-Validation

Increase CV folds for more robust validation:

```bash
python gravity_model.py \
  --use-sample-data \
  --cv-folds 10 \
  --output results/
```

### No Time Stratification

Fit model without time-of-day effects:

```bash
python gravity_model.py \
  --bouts bouts.csv \
  --feeders feeders.csv \
  --time-block none \
  --output results/
```

## Interpretation Guide

### Reading the Output

When the script runs, it prints:

1. **Data Summary**: Number of bouts, birds, transitions
2. **Model Fitting**: AIC, BIC, dispersion statistics
3. **Model Selection**: Which model was chosen and why
4. **Cross-Validation**: RMSE and MAE scores
5. **Coefficients**: Parameter estimates with significance

### Understanding Coefficients

**Significant Time Effects** (e.g., morning coefficient = 0.57, p < 0.001):
- Birds make 57% more transitions in morning vs night (reference)
- exp(0.57) = 1.77 → 77% increase

**Distance-Decay** (e.g., log_distance = -0.5, SE = 0.1):
- Strong negative distance effect
- Doubling distance reduces transitions by ~30%

**Species Effects** (e.g., COATI = 0.15, p = 0.10):
- COATI makes slightly more transitions than reference species
- Not statistically significant (p > 0.05)

### Model Quality

**Good Model**:
- Residuals normally distributed
- No patterns in residual plots
- Low cross-validation RMSE
- Reasonable parameter estimates

**Issues to Watch**:
- Very high dispersion (>5): Consider zero-inflated models
- Poor Q-Q plot fit: Check for outliers
- Large residuals for specific edges: Spatial confounding?
- Positive distance coefficient: Model misspecification

## Example Workflow

### Complete Analysis

```python
# In Python or Jupyter

from gravity_model import *

# 1. Load data
bouts_df, feeders_df = load_data('bouts.csv', 'feeders.csv')

# 2. Process
transitions_df = build_transitions(bouts_df)
distance_df = compute_distance_matrix(feeders_df)
od_df = aggregate_od(transitions_df, bouts_df, distance_df, 
                     time_block='time_of_day')

# 3. Model
poisson_result = fit_poisson_model(od_df)
nb_result = fit_negative_binomial_model(od_df)
best_model, model_name = select_best_model(poisson_result, nb_result)

# 4. Evaluate
residuals_df = compute_residuals(od_df, best_model)
cv_results = cross_validate_model(od_df, formula, 5, model_name)

# 5. Visualize
from pathlib import Path
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

plot_diagnostics(residuals_df, output_dir, model_name)
plot_distance_decay(residuals_df, best_model, output_dir, model_name)
plot_flow_map(residuals_df, feeders_df, output_dir, model_name)

# 6. Export
save_model_outputs(best_model, residuals_df, output_dir, model_name)
print_model_summary(best_model, model_name, cv_results)
```

## Troubleshooting

### Common Issues

**Error: "NaN or inf detected"**
- Check for zero exposure values
- Ensure positive bout durations
- Verify feeder coordinates are valid

**Warning: "Significant overdispersion"**
- Normal if Negative-Binomial is selected
- If extreme (>10), consider data quality

**Low prediction accuracy**
- Try per-species models
- Add more covariates
- Check for spatial confounding
- Consider non-linear distance effects

**Empty transitions**
- Check bout data is sorted by time
- Verify bird_id consistency
- Ensure multiple bouts per bird

### Performance Tips

- For large datasets (>100k bouts): Use time stratification
- For many species: Use per-species option
- For many feeders (>50): Consider regularization
- Monitor memory usage with very large OD matrices

## Citation

If you use this gravity model implementation, please cite:

```
[Add publication details when available]
```

## Further Reading

### Gravity Models in Movement Ecology

- Wilson et al. (2003). "Spatial Patterns of Foraging Behavior"
- Zipf (1946). "The P1P2/D Hypothesis"
- Anderson (2011). "The Gravity Model"

### Statistical Methods

- Cameron & Trivedi (2013). "Regression Analysis of Count Data"
- Zeileis et al. (2008). "Regression Models for Count Data in R"

### Applications to Birds

- Farine et al. (2015). "The role of social and ecological processes in bird social networks"
- [Add relevant citations]

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Contact: [add contact information]
- Documentation: See README.md and code comments

## Version History

- v1.0.0 (2024-10): Initial release
  - Complete gravity model implementation
  - Poisson and Negative-Binomial GLMs
  - Cross-validation
  - Diagnostic plots
  - Sample data generation
  - Comprehensive documentation
