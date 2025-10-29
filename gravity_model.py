#!/usr/bin/env python3
"""
Gravity Model for Bird Transition Data between RFID Feeders

This script builds and fits gravity models to analyze bird movement patterns
between RFID feeders using Origin-Destination (OD) flow data.

Usage:
    python gravity_model.py --bouts bouts.csv --feeders feeders.csv --output output_dir/

Inputs:
    - bouts.csv: Bird visit data with columns [bird_id, feeder_id, bout_start, bout_end, species]
    - feeders.csv: Feeder locations with columns [feeder_id, x, y] in OSGB metres (EPSG:27700)

Outputs:
    - model_coefficients.csv: Fitted parameters
    - predicted_flows.csv: Expected flows and residuals per OD pair
    - Various diagnostic plots (PNG)
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson, NegativeBinomial
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore', category=FutureWarning)


def generate_sample_data(n_birds: int = 50, n_feeders: int = 10, 
                         n_bouts: int = 1000, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate sample bird bout and feeder data for testing.
    
    Parameters:
    -----------
    n_birds : int
        Number of unique birds
    n_feeders : int
        Number of feeders
    n_bouts : int
        Number of bout records
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    bouts_df : pd.DataFrame
        Bout data with columns [bird_id, feeder_id, bout_start, bout_end, species]
    feeders_df : pd.DataFrame
        Feeder locations with columns [feeder_id, x, y]
    """
    np.random.seed(seed)
    
    # Create feeder locations (OSGB grid, roughly 1km x 1km area)
    feeders_df = pd.DataFrame({
        'feeder_id': [f'F{i:03d}' for i in range(n_feeders)],
        'x': 450000 + np.random.uniform(0, 1000, n_feeders),
        'y': 207000 + np.random.uniform(0, 1000, n_feeders)
    })
    
    # Create bird bouts
    species_list = ['GRETI', 'BLUTI', 'COATI', 'MARTI']
    bird_ids = [f'B{i:04d}' for i in range(n_birds)]
    
    bouts_data = []
    base_time = pd.Timestamp('2024-03-01 06:00:00')
    
    for i in range(n_bouts):
        bird_id = np.random.choice(bird_ids)
        species = np.random.choice(species_list)
        feeder_id = np.random.choice(feeders_df['feeder_id'].values)
        
        # Random bout duration between 1 and 60 minutes
        bout_duration = np.random.exponential(10) + 1
        bout_start = base_time + pd.Timedelta(minutes=np.random.uniform(0, 10080))  # week
        bout_end = bout_start + pd.Timedelta(minutes=bout_duration)
        
        bouts_data.append({
            'bird_id': bird_id,
            'feeder_id': feeder_id,
            'bout_start': bout_start,
            'bout_end': bout_end,
            'species': species
        })
    
    bouts_df = pd.DataFrame(bouts_data)
    bouts_df = bouts_df.sort_values(['bird_id', 'bout_start']).reset_index(drop=True)
    
    # Calculate bout duration in minutes
    bouts_df['bout_duration'] = (bouts_df['bout_end'] - bouts_df['bout_start']).dt.total_seconds() / 60
    
    return bouts_df, feeders_df


def load_data(bouts_path: str, feeders_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load bout and feeder data from CSV files.
    
    Parameters:
    -----------
    bouts_path : str
        Path to bouts CSV file
    feeders_path : str
        Path to feeders CSV file
        
    Returns:
    --------
    bouts_df : pd.DataFrame
        Bout data
    feeders_df : pd.DataFrame
        Feeder locations
    """
    print(f"Loading data from {bouts_path} and {feeders_path}...")
    
    bouts_df = pd.read_csv(bouts_path)
    feeders_df = pd.read_csv(feeders_path)
    
    # Convert datetime columns
    bouts_df['bout_start'] = pd.to_datetime(bouts_df['bout_start'])
    bouts_df['bout_end'] = pd.to_datetime(bouts_df['bout_end'])
    
    # Calculate bout duration in minutes
    bouts_df['bout_duration'] = (bouts_df['bout_end'] - bouts_df['bout_start']).dt.total_seconds() / 60
    
    print(f"Loaded {len(bouts_df)} bouts for {bouts_df['bird_id'].nunique()} birds")
    print(f"Loaded {len(feeders_df)} feeders")
    print(f"Species: {bouts_df['species'].unique()}")
    
    return bouts_df, feeders_df


def build_transitions(bouts_df: pd.DataFrame, exclude_self_loops: bool = True) -> pd.DataFrame:
    """
    Build transitions from consecutive bouts for each bird.
    
    Parameters:
    -----------
    bouts_df : pd.DataFrame
        Bout data with bird_id, feeder_id, bout_start, bout_end, species
    exclude_self_loops : bool
        Whether to exclude transitions where origin == destination
        
    Returns:
    --------
    transitions_df : pd.DataFrame
        Transition data with origin, destination, species, time info
    """
    print("Building transitions from consecutive bouts...")
    
    transitions = []
    
    # Group by bird and create consecutive transitions
    for bird_id, bird_bouts in bouts_df.groupby('bird_id'):
        bird_bouts = bird_bouts.sort_values('bout_start').reset_index(drop=True)
        
        for i in range(len(bird_bouts) - 1):
            origin_feeder = bird_bouts.loc[i, 'feeder_id']
            dest_feeder = bird_bouts.loc[i + 1, 'feeder_id']
            
            # Skip self-loops if requested
            if exclude_self_loops and origin_feeder == dest_feeder:
                continue
            
            transitions.append({
                'bird_id': bird_id,
                'origin': origin_feeder,
                'destination': dest_feeder,
                'origin_end': bird_bouts.loc[i, 'bout_end'],
                'dest_start': bird_bouts.loc[i + 1, 'bout_start'],
                'origin_duration': bird_bouts.loc[i, 'bout_duration'],
                'species': bird_bouts.loc[i, 'species']
            })
    
    transitions_df = pd.DataFrame(transitions)
    
    if len(transitions_df) > 0:
        # Add time features
        transitions_df['hour'] = transitions_df['origin_end'].dt.hour
        transitions_df['day_of_year'] = transitions_df['origin_end'].dt.dayofyear
        
        # Time of day bins
        transitions_df['time_of_day'] = pd.cut(
            transitions_df['hour'],
            bins=[0, 9, 15, 20, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        print(f"Created {len(transitions_df)} transitions")
        print(f"Transitions by species:\n{transitions_df['species'].value_counts()}")
    else:
        print("Warning: No transitions created!")
    
    return transitions_df


def compute_distance_matrix(feeders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Euclidean distance matrix between all feeders.
    
    Parameters:
    -----------
    feeders_df : pd.DataFrame
        Feeder locations with feeder_id, x, y
        
    Returns:
    --------
    distance_df : pd.DataFrame
        Distance matrix with origin, destination, distance_m
    """
    print("Computing distance matrix...")
    
    coords = feeders_df[['x', 'y']].values
    distances = cdist(coords, coords, metric='euclidean')
    
    distance_data = []
    for i, origin in enumerate(feeders_df['feeder_id']):
        for j, dest in enumerate(feeders_df['feeder_id']):
            if i != j:  # Exclude self-distances
                distance_data.append({
                    'origin': origin,
                    'destination': dest,
                    'distance_m': distances[i, j]
                })
    
    distance_df = pd.DataFrame(distance_data)
    print(f"Distance range: {distance_df['distance_m'].min():.1f} - {distance_df['distance_m'].max():.1f} metres")
    
    return distance_df


def aggregate_od(transitions_df: pd.DataFrame, bouts_df: pd.DataFrame,
                 distance_df: pd.DataFrame, time_block: Optional[str] = None) -> pd.DataFrame:
    """
    Aggregate transitions into OD flow table with counts, exposure, and covariates.
    
    Parameters:
    -----------
    transitions_df : pd.DataFrame
        Individual transitions
    bouts_df : pd.DataFrame
        Original bout data for computing exposure
    distance_df : pd.DataFrame
        Distance matrix
    time_block : str, optional
        Grouping variable for time blocks (e.g., 'time_of_day')
        
    Returns:
    --------
    od_df : pd.DataFrame
        Aggregated OD flow table
    """
    print("Aggregating transitions into OD flow table...")
    
    # Group variables
    group_vars = ['origin', 'destination', 'species']
    if time_block and time_block in transitions_df.columns:
        group_vars.append(time_block)
    
    # Count transitions
    od_counts = transitions_df.groupby(group_vars).size().reset_index(name='n_transitions')
    
    # Compute exposure (total time at origin) from bouts
    exposure_vars = ['feeder_id', 'species']
    if time_block:
        # Add time_of_day to bouts if not present
        if 'hour' not in bouts_df.columns:
            bouts_df['hour'] = bouts_df['bout_start'].dt.hour
        if time_block == 'time_of_day' and 'time_of_day' not in bouts_df.columns:
            bouts_df['time_of_day'] = pd.cut(
                bouts_df['hour'],
                bins=[0, 9, 15, 20, 24],
                labels=['night', 'morning', 'afternoon', 'evening'],
                include_lowest=True
            )
        if time_block in bouts_df.columns:
            exposure_vars.append(time_block)
    
    exposure = bouts_df.groupby(exposure_vars)['bout_duration'].sum().reset_index()
    # Rename columns properly
    rename_dict = {'feeder_id': 'origin', 'bout_duration': 'exposure_minutes'}
    exposure = exposure.rename(columns=rename_dict)
    
    # Merge counts with exposure
    od_df = od_counts.merge(exposure, on=['origin', 'species'] + ([time_block] if time_block else []), how='left')
    # Avoid zero or missing exposure
    od_df['exposure_minutes'] = od_df['exposure_minutes'].fillna(0.1)
    od_df.loc[od_df['exposure_minutes'] <= 0, 'exposure_minutes'] = 0.1
    
    # Add distances
    od_df = od_df.merge(distance_df, on=['origin', 'destination'], how='left')
    
    # Compute attractiveness terms (total visits at each feeder)
    origin_visits = bouts_df.groupby('feeder_id').size().reset_index(name='origin_visits')
    dest_visits = bouts_df.groupby('feeder_id').size().reset_index(name='dest_visits')
    
    od_df = od_df.merge(origin_visits.rename(columns={'feeder_id': 'origin'}), on='origin', how='left')
    od_df = od_df.merge(dest_visits.rename(columns={'feeder_id': 'destination'}), on='destination', how='left')
    
    # Add day of year (mean for the block)
    if 'day_of_year' in transitions_df.columns:
        day_means = transitions_df.groupby(group_vars)['day_of_year'].mean().reset_index()
        od_df = od_df.merge(day_means, on=group_vars, how='left')
    
    print(f"Created {len(od_df)} OD flow records")
    print(f"Total transitions: {od_df['n_transitions'].sum()}")
    
    return od_df


def fit_poisson_model(od_df: pd.DataFrame, formula: str = None) -> sm.GLM:
    """
    Fit Poisson GLM to OD flow data.
    
    Parameters:
    -----------
    od_df : pd.DataFrame
        OD flow table with response and predictors
    formula : str, optional
        Model formula (if None, uses default)
        
    Returns:
    --------
    model_result : statsmodels GLM results
    """
    print("\nFitting Poisson GLM...")
    
    # Prepare data - keep only records with observed transitions
    model_df = od_df[od_df['n_transitions'] > 0].copy()
    model_df['log_distance'] = np.log(model_df['distance_m'] + 1)
    model_df['log_origin_visits'] = np.log(model_df['origin_visits'] + 1)
    model_df['log_dest_visits'] = np.log(model_df['dest_visits'] + 1)
    model_df['log_exposure'] = np.log(model_df['exposure_minutes'])
    
    print(f"Using {len(model_df)} records with observed transitions")
    
    # Default formula
    if formula is None:
        formula = 'n_transitions ~ log_distance + log_origin_visits + log_dest_visits'
        if 'day_of_year' in model_df.columns:
            formula += ' + day_of_year'
        if 'time_of_day' in model_df.columns:
            formula += ' + C(time_of_day)'
        if 'species' in model_df.columns and model_df['species'].nunique() > 1:
            formula += ' + C(species)'
    
    print(f"Formula: {formula}")
    
    # Fit model with exposure offset
    try:
        model = smf.glm(formula=formula, data=model_df, family=Poisson(),
                       offset=model_df['log_exposure'])
        result = model.fit()
        
        print(f"AIC: {result.aic:.2f}")
        print(f"BIC: {result.bic:.2f}")
        print(f"Pearson chi2: {result.pearson_chi2:.2f}")
        print(f"Deviance: {result.deviance:.2f}")
        
        return result
    except Exception as e:
        print(f"Error fitting Poisson model: {e}")
        return None


def fit_negative_binomial_model(od_df: pd.DataFrame, formula: str = None) -> sm.GLM:
    """
    Fit Negative Binomial GLM to OD flow data.
    
    Parameters:
    -----------
    od_df : pd.DataFrame
        OD flow table with response and predictors
    formula : str, optional
        Model formula (if None, uses default)
        
    Returns:
    --------
    model_result : statsmodels GLM results
    """
    print("\nFitting Negative Binomial GLM...")
    
    # Prepare data - keep only records with observed transitions
    model_df = od_df[od_df['n_transitions'] > 0].copy()
    model_df['log_distance'] = np.log(model_df['distance_m'] + 1)
    model_df['log_origin_visits'] = np.log(model_df['origin_visits'] + 1)
    model_df['log_dest_visits'] = np.log(model_df['dest_visits'] + 1)
    model_df['log_exposure'] = np.log(model_df['exposure_minutes'])
    
    print(f"Using {len(model_df)} records with observed transitions")
    
    # Default formula
    if formula is None:
        formula = 'n_transitions ~ log_distance + log_origin_visits + log_dest_visits'
        if 'day_of_year' in model_df.columns:
            formula += ' + day_of_year'
        if 'time_of_day' in model_df.columns:
            formula += ' + C(time_of_day)'
        if 'species' in model_df.columns and model_df['species'].nunique() > 1:
            formula += ' + C(species)'
    
    print(f"Formula: {formula}")
    
    # Fit model with exposure offset
    try:
        # Use NegativeBinomial family with initial alpha
        model = smf.glm(formula=formula, data=model_df, 
                       family=NegativeBinomial(alpha=1.0),
                       offset=model_df['log_exposure'])
        result = model.fit()
        
        print(f"AIC: {result.aic:.2f}")
        print(f"BIC: {result.bic:.2f}")
        print(f"Alpha (dispersion): {result.scale:.4f}")
        
        return result
    except Exception as e:
        print(f"Error fitting Negative Binomial model: {e}")
        print("Falling back to Poisson model...")
        return fit_poisson_model(od_df, formula)


def select_best_model(poisson_result, nb_result) -> Tuple[sm.GLM, str]:
    """
    Select the best model based on AIC/BIC and overdispersion test.
    
    Parameters:
    -----------
    poisson_result : statsmodels results
        Fitted Poisson model
    nb_result : statsmodels results
        Fitted Negative Binomial model
        
    Returns:
    --------
    best_model : statsmodels results
        Selected model
    model_name : str
        'Poisson' or 'NegativeBinomial'
    """
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    
    # Overdispersion test
    if poisson_result is not None:
        dispersion = poisson_result.pearson_chi2 / poisson_result.df_resid
        print(f"\nPoisson model dispersion: {dispersion:.3f}")
        if dispersion > 1.5:
            print("  → Significant overdispersion detected (> 1.5)")
        else:
            print("  → No strong overdispersion")
    
    # AIC comparison
    if poisson_result is not None and nb_result is not None:
        print(f"\nPoisson AIC: {poisson_result.aic:.2f}, BIC: {poisson_result.bic:.2f}")
        print(f"NegBin AIC:  {nb_result.aic:.2f}, BIC: {nb_result.bic:.2f}")
        
        aic_diff = poisson_result.aic - nb_result.aic
        if aic_diff > 2:
            print(f"\n✓ Negative Binomial preferred (ΔAIC = {aic_diff:.2f})")
            return nb_result, 'NegativeBinomial'
        else:
            print(f"\n✓ Poisson preferred or equivalent (ΔAIC = {aic_diff:.2f})")
            return poisson_result, 'Poisson'
    
    elif poisson_result is not None:
        print("\n✓ Using Poisson (NegBin fit failed)")
        return poisson_result, 'Poisson'
    else:
        print("\n✗ Both models failed to fit")
        return None, None


def compute_residuals(od_df: pd.DataFrame, model_result) -> pd.DataFrame:
    """
    Compute residuals: raw, Pearson, and deviance.
    
    Parameters:
    -----------
    od_df : pd.DataFrame
        Original OD data (only records with transitions > 0)
    model_result : statsmodels results
        Fitted model
        
    Returns:
    --------
    residuals_df : pd.DataFrame
        OD data with predicted values and residuals
    """
    # Use only the data that was in the model (n_transitions > 0)
    result_df = od_df[od_df['n_transitions'] > 0].copy().reset_index(drop=True)
    
    # Get predictions
    result_df['predicted'] = model_result.fittedvalues.values
    result_df['observed'] = model_result.model.endog
    
    # Raw residuals
    result_df['raw_residual'] = result_df['observed'] - result_df['predicted']
    
    # Pearson residuals
    result_df['pearson_residual'] = ((result_df['observed'] - result_df['predicted']) / 
                                     np.sqrt(result_df['predicted'] + 1e-10))
    
    # Deviance residuals
    result_df['deviance_residual'] = model_result.resid_deviance.values
    
    return result_df


def cross_validate_model(od_df: pd.DataFrame, formula: str, 
                         n_folds: int = 5, model_type: str = 'Poisson') -> Dict:
    """
    Perform k-fold cross-validation.
    
    Parameters:
    -----------
    od_df : pd.DataFrame
        OD flow data
    formula : str
        Model formula
    n_folds : int
        Number of CV folds
    model_type : str
        'Poisson' or 'NegativeBinomial'
        
    Returns:
    --------
    cv_results : dict
        Cross-validation metrics
    """
    print(f"\nPerforming {n_folds}-fold cross-validation...")
    
    # Use only records with observed transitions
    model_df = od_df[od_df['n_transitions'] > 0].copy()
    model_df['log_distance'] = np.log(model_df['distance_m'] + 1)
    model_df['log_origin_visits'] = np.log(model_df['origin_visits'] + 1)
    model_df['log_dest_visits'] = np.log(model_df['dest_visits'] + 1)
    model_df['log_exposure'] = np.log(model_df['exposure_minutes'])
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    rmse_scores = []
    mae_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(model_df)):
        train_df = model_df.iloc[train_idx]
        test_df = model_df.iloc[test_idx]
        
        try:
            if model_type == 'Poisson':
                model = smf.glm(formula=formula, data=train_df, family=Poisson(),
                               offset=train_df['log_exposure'])
            else:
                model = smf.glm(formula=formula, data=train_df,
                               family=NegativeBinomial(alpha=1.0),
                               offset=train_df['log_exposure'])
            
            result = model.fit()
            
            # Predict on test set
            predictions = result.predict(test_df, offset=test_df['log_exposure'])
            
            rmse = np.sqrt(mean_squared_error(test_df['n_transitions'], predictions))
            mae = np.mean(np.abs(test_df['n_transitions'] - predictions))
            
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            
        except Exception as e:
            print(f"  Fold {fold+1} failed: {e}")
            continue
    
    cv_results = {
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'n_folds_completed': len(rmse_scores)
    }
    
    print(f"  RMSE: {cv_results['rmse_mean']:.3f} ± {cv_results['rmse_std']:.3f}")
    print(f"  MAE:  {cv_results['mae_mean']:.3f} ± {cv_results['mae_std']:.3f}")
    
    return cv_results


def plot_diagnostics(residuals_df: pd.DataFrame, output_dir: Path, 
                     model_name: str = 'model'):
    """
    Create diagnostic plots.
    
    Parameters:
    -----------
    residuals_df : pd.DataFrame
        OD data with residuals
    output_dir : Path
        Output directory for plots
    model_name : str
        Model identifier for filename
    """
    print("\nCreating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Observed vs Predicted
    ax = axes[0, 0]
    ax.scatter(residuals_df['predicted'], residuals_df['observed'], 
               alpha=0.5, s=20)
    max_val = max(residuals_df['predicted'].max(), residuals_df['observed'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='1:1 line')
    ax.set_xlabel('Predicted transitions')
    ax.set_ylabel('Observed transitions')
    ax.set_title('Observed vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Residuals histogram
    ax = axes[0, 1]
    ax.hist(residuals_df['pearson_residual'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('Pearson residuals')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual distribution')
    ax.grid(True, alpha=0.3)
    
    # 3. Residuals vs Predicted
    ax = axes[1, 0]
    ax.scatter(residuals_df['predicted'], residuals_df['pearson_residual'],
               alpha=0.5, s=20)
    ax.axhline(0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('Predicted transitions')
    ax.set_ylabel('Pearson residuals')
    ax.set_title('Residuals vs Fitted')
    ax.grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    ax = axes[1, 1]
    stats.probplot(residuals_df['pearson_residual'], dist="norm", plot=ax)
    ax.set_title('Q-Q Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'{model_name}_diagnostics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_distance_decay(residuals_df: pd.DataFrame, model_result,
                       output_dir: Path, model_name: str = 'model'):
    """
    Plot distance-decay relationship.
    
    Parameters:
    -----------
    residuals_df : pd.DataFrame
        OD data with predictions
    model_result : statsmodels results
        Fitted model for coefficient extraction
    output_dir : Path
        Output directory
    model_name : str
        Model identifier
    """
    print("Creating distance-decay plot...")
    
    # Extract distance coefficient
    param_names = model_result.params.index
    dist_coef = None
    for name in param_names:
        if 'log_distance' in name:
            dist_coef = model_result.params[name]
            dist_se = model_result.bse[name]
            break
    
    if dist_coef is None:
        print("  Warning: No distance coefficient found")
        return
    
    # Create distance bins
    residuals_df['distance_bin'] = pd.cut(residuals_df['distance_m'], bins=10)
    binned = residuals_df.groupby('distance_bin').agg({
        'distance_m': 'mean',
        'n_transitions': 'sum',
        'exposure_minutes': 'sum'
    }).reset_index()
    binned['rate'] = binned['n_transitions'] / (binned['exposure_minutes'] + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Observed rates
    ax.scatter(binned['distance_m'], binned['rate'], s=100, alpha=0.6,
               label='Observed', color='blue', edgecolors='black', linewidths=1)
    
    # Fitted curve
    dist_range = np.linspace(residuals_df['distance_m'].min(),
                            residuals_df['distance_m'].max(), 100)
    # Rate ∝ exp(-gamma * log(d+1)) = (d+1)^(-gamma)
    fitted_rate = np.exp(dist_coef * np.log(dist_range + 1))
    # Normalize to match scale
    fitted_rate = fitted_rate * (binned['rate'].mean() / fitted_rate.mean())
    
    ax.plot(dist_range, fitted_rate, 'r-', lw=2, 
            label=f'Fitted (γ={-dist_coef:.3f} ± {dist_se:.3f})')
    
    # Confidence interval
    fitted_upper = np.exp((dist_coef + 1.96*dist_se) * np.log(dist_range + 1))
    fitted_lower = np.exp((dist_coef - 1.96*dist_se) * np.log(dist_range + 1))
    fitted_upper = fitted_upper * (binned['rate'].mean() / fitted_upper.mean())
    fitted_lower = fitted_lower * (binned['rate'].mean() / fitted_lower.mean())
    ax.fill_between(dist_range, fitted_lower, fitted_upper, alpha=0.2, color='red')
    
    ax.set_xlabel('Distance (metres)')
    ax.set_ylabel('Transition rate (per minute)')
    ax.set_title('Distance-Decay Relationship')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / f'{model_name}_distance_decay.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_flow_map(residuals_df: pd.DataFrame, feeders_df: pd.DataFrame,
                 output_dir: Path, model_name: str = 'model', top_n: int = 20):
    """
    Create map of feeders with arrows showing top residual flows.
    
    Parameters:
    -----------
    residuals_df : pd.DataFrame
        OD data with residuals
    feeders_df : pd.DataFrame
        Feeder coordinates
    output_dir : Path
        Output directory
    model_name : str
        Model identifier
    top_n : int
        Number of top residuals to show
    """
    print(f"Creating flow map with top {top_n} residual edges...")
    
    # Get top residuals (both positive and negative)
    residuals_df['abs_residual'] = np.abs(residuals_df['raw_residual'])
    top_residuals = residuals_df.nlargest(top_n, 'abs_residual')
    
    # Create map
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot feeders
    ax.scatter(feeders_df['x'], feeders_df['y'], s=100, c='lightblue',
               edgecolors='black', linewidths=2, zorder=3, label='Feeders')
    
    # Add feeder labels
    for _, row in feeders_df.iterrows():
        ax.annotate(row['feeder_id'], (row['x'], row['y']), 
                   fontsize=8, ha='center', va='center')
    
    # Plot top residual flows as arrows
    for _, flow in top_residuals.iterrows():
        origin_coords = feeders_df[feeders_df['feeder_id'] == flow['origin']][['x', 'y']].values[0]
        dest_coords = feeders_df[feeders_df['feeder_id'] == flow['destination']][['x', 'y']].values[0]
        
        # Color by sign of residual
        color = 'red' if flow['raw_residual'] > 0 else 'blue'
        width = min(np.abs(flow['raw_residual']) / 10, 3)
        
        ax.annotate('', xy=dest_coords, xytext=origin_coords,
                   arrowprops=dict(arrowstyle='->', lw=width, color=color, alpha=0.6))
    
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title('Top Residual Flows\n(Red = Over-predicted, Blue = Under-predicted)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    output_path = output_dir / f'{model_name}_flow_map.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # Save top residuals table
    top_table = top_residuals[['origin', 'destination', 'observed', 'predicted', 
                                'raw_residual', 'pearson_residual', 'distance_m']].copy()
    top_table = top_table.sort_values('raw_residual', ascending=False)
    
    table_path = output_dir / f'{model_name}_top_residuals.csv'
    top_table.to_csv(table_path, index=False)
    print(f"  Saved: {table_path}")


def save_model_outputs(model_result, residuals_df: pd.DataFrame,
                      output_dir: Path, model_name: str = 'model'):
    """
    Save model coefficients and predictions to CSV.
    
    Parameters:
    -----------
    model_result : statsmodels results
        Fitted model
    residuals_df : pd.DataFrame
        OD data with predictions and residuals
    output_dir : Path
        Output directory
    model_name : str
        Model identifier
    """
    print("\nSaving model outputs...")
    
    # Save coefficients
    coef_df = pd.DataFrame({
        'parameter': model_result.params.index,
        'coefficient': model_result.params.values,
        'std_error': model_result.bse.values,
        'z_value': model_result.tvalues.values,
        'p_value': model_result.pvalues.values
    })
    
    coef_path = output_dir / f'{model_name}_coefficients.csv'
    coef_df.to_csv(coef_path, index=False)
    print(f"  Saved: {coef_path}")
    
    # Save predictions and residuals
    output_cols = ['origin', 'destination', 'species', 'observed', 'predicted',
                   'raw_residual', 'pearson_residual', 'deviance_residual',
                   'distance_m', 'exposure_minutes']
    if 'time_of_day' in residuals_df.columns:
        output_cols.insert(3, 'time_of_day')
    
    flows_df = residuals_df[output_cols].copy()
    flows_path = output_dir / f'{model_name}_predicted_flows.csv'
    flows_df.to_csv(flows_path, index=False)
    print(f"  Saved: {flows_path}")


def print_model_summary(model_result, model_name: str, cv_results: Optional[Dict] = None):
    """
    Print comprehensive model summary.
    
    Parameters:
    -----------
    model_result : statsmodels results
        Fitted model
    model_name : str
        Model type name
    cv_results : dict, optional
        Cross-validation results
    """
    print("\n" + "="*60)
    print(f"MODEL SUMMARY: {model_name}")
    print("="*60)
    
    print(f"\nModel fit statistics:")
    print(f"  AIC: {model_result.aic:.2f}")
    print(f"  BIC: {model_result.bic:.2f}")
    print(f"  Log-likelihood: {model_result.llf:.2f}")
    print(f"  Pearson chi2: {model_result.pearson_chi2:.2f}")
    print(f"  Deviance: {model_result.deviance:.2f}")
    print(f"  Dispersion: {model_result.pearson_chi2 / model_result.df_resid:.3f}")
    
    if cv_results:
        print(f"\nCross-validation ({cv_results['n_folds_completed']} folds):")
        print(f"  RMSE: {cv_results['rmse_mean']:.3f} ± {cv_results['rmse_std']:.3f}")
        print(f"  MAE:  {cv_results['mae_mean']:.3f} ± {cv_results['mae_std']:.3f}")
    
    print(f"\nKey coefficients:")
    for param in model_result.params.index:
        coef = model_result.params[param]
        se = model_result.bse[param]
        pval = model_result.pvalues[param]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"  {param:30s}: {coef:8.4f} (SE: {se:.4f}) {sig}")
    
    # Extract distance-decay parameter
    for param in model_result.params.index:
        if 'log_distance' in param:
            gamma = -model_result.params[param]
            gamma_se = model_result.bse[param]
            print(f"\n  Distance-decay parameter (γ): {gamma:.4f} ± {gamma_se:.4f}")
            print(f"  → Transitions decrease by {(1-np.exp(-gamma))*100:.1f}% per log-unit distance")
            break


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Fit gravity model to bird transition data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--bouts', type=str, help='Path to bouts CSV file')
    parser.add_argument('--feeders', type=str, help='Path to feeders CSV file')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--use-sample-data', action='store_true',
                       help='Generate and use sample data for testing')
    parser.add_argument('--per-species', action='store_true',
                       help='Fit separate models per species')
    parser.add_argument('--time-block', type=str, default='time_of_day',
                       choices=['time_of_day', 'none'],
                       help='Time stratification (default: time_of_day)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load or generate data
    if args.use_sample_data:
        print("\n" + "="*60)
        print("GENERATING SAMPLE DATA")
        print("="*60)
        bouts_df, feeders_df = generate_sample_data()
        
        # Save sample data
        bouts_df.to_csv(output_dir / 'sample_bouts.csv', index=False)
        feeders_df.to_csv(output_dir / 'sample_feeders.csv', index=False)
        print("Sample data saved to output directory")
    else:
        if not args.bouts or not args.feeders:
            print("Error: Must provide --bouts and --feeders, or use --use-sample-data")
            sys.exit(1)
        bouts_df, feeders_df = load_data(args.bouts, args.feeders)
    
    # Build transitions
    print("\n" + "="*60)
    print("BUILDING TRANSITIONS")
    print("="*60)
    transitions_df = build_transitions(bouts_df, exclude_self_loops=True)
    
    if len(transitions_df) == 0:
        print("Error: No transitions created. Check input data.")
        sys.exit(1)
    
    # Compute distances
    print("\n" + "="*60)
    print("COMPUTING DISTANCES")
    print("="*60)
    distance_df = compute_distance_matrix(feeders_df)
    
    # Aggregate OD flows
    print("\n" + "="*60)
    print("AGGREGATING OD FLOWS")
    print("="*60)
    time_block = args.time_block if args.time_block != 'none' else None
    od_df = aggregate_od(transitions_df, bouts_df, distance_df, time_block=time_block)
    
    # Fit models
    print("\n" + "="*60)
    print("FITTING MODELS")
    print("="*60)
    
    if args.per_species and od_df['species'].nunique() > 1:
        # Fit separate models per species
        for species in od_df['species'].unique():
            print(f"\n{'='*60}")
            print(f"SPECIES: {species}")
            print(f"{'='*60}")
            
            species_df = od_df[od_df['species'] == species].copy()
            
            # Fit both models
            poisson_result = fit_poisson_model(species_df)
            nb_result = fit_negative_binomial_model(species_df)
            
            # Select best model
            best_model, model_name = select_best_model(poisson_result, nb_result)
            
            if best_model is None:
                print(f"Error: Model fitting failed for {species}")
                continue
            
            # Compute residuals
            residuals_df = compute_residuals(species_df, best_model)
            
            # Cross-validation
            formula = 'n_transitions ~ log_distance + log_origin_visits + log_dest_visits'
            if 'day_of_year' in species_df.columns:
                formula += ' + day_of_year'
            if time_block and time_block in species_df.columns:
                formula += f' + C({time_block})'
            
            cv_results = cross_validate_model(species_df, formula, 
                                             args.cv_folds, model_name)
            
            # Print summary
            print_model_summary(best_model, f"{model_name} ({species})", cv_results)
            
            # Create plots
            model_prefix = f"{species}_{model_name}"
            plot_diagnostics(residuals_df, output_dir, model_prefix)
            plot_distance_decay(residuals_df, best_model, output_dir, model_prefix)
            plot_flow_map(residuals_df, feeders_df, output_dir, model_prefix)
            
            # Save outputs
            save_model_outputs(best_model, residuals_df, output_dir, model_prefix)
    
    else:
        # Fit pooled model
        poisson_result = fit_poisson_model(od_df)
        nb_result = fit_negative_binomial_model(od_df)
        
        # Select best model
        best_model, model_name = select_best_model(poisson_result, nb_result)
        
        if best_model is None:
            print("Error: Model fitting failed")
            sys.exit(1)
        
        # Compute residuals
        residuals_df = compute_residuals(od_df, best_model)
        
        # Cross-validation
        formula = 'n_transitions ~ log_distance + log_origin_visits + log_dest_visits'
        if 'day_of_year' in od_df.columns:
            formula += ' + day_of_year'
        if time_block and time_block in od_df.columns:
            formula += f' + C({time_block})'
        if 'species' in od_df.columns and od_df['species'].nunique() > 1:
            formula += ' + C(species)'
        
        cv_results = cross_validate_model(od_df, formula, args.cv_folds, model_name)
        
        # Print summary
        print_model_summary(best_model, model_name, cv_results)
        
        # Create plots
        plot_diagnostics(residuals_df, output_dir, model_name)
        plot_distance_decay(residuals_df, best_model, output_dir, model_name)
        plot_flow_map(residuals_df, feeders_df, output_dir, model_name)
        
        # Save outputs
        save_model_outputs(best_model, residuals_df, output_dir, model_name)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"All outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
