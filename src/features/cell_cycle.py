# -*- coding: utf-8 -*-
"""
Cell cycle state inference from chromatin features.

This module provides methods to infer cell cycle stage from nuclear
morphology and chromatin texture features.

Biological relevance:
- DZ cells are actively proliferating (more S/G2 cells)
- LZ cells are quiescent (more G0/G1 cells)
- Cell cycle state confounds DZ/LZ classification
- Should be included as covariate or stratification factor

Features used:
- Nuclear area (larger in S/G2)
- Chromatin texture (more heterogeneous in S phase)
- Intensity distribution (replication foci in S phase)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def infer_cell_cycle_state(
    features: pd.DataFrame,
    method: str = 'threshold',
    area_col: str = 'area',
    intensity_cols: Optional[List[str]] = None,
    texture_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Infer cell cycle stage from chromatin features.
    
    Features used:
    - Nuclear area (larger in S/G2)
    - Chromatin texture (more heterogeneous in S phase)
    - Intensity distribution (replication foci in S phase)
    
    Args:
        features: DataFrame with nuclear features
        method: 'threshold' (rule-based) or 'clustering' (data-driven)
        area_col: Column name for nuclear area
        intensity_cols: List of intensity feature columns
        texture_cols: List of texture feature columns
        
    Returns:
        DataFrame with cell cycle predictions
    """
    if features.empty:
        return pd.DataFrame()
    
    if intensity_cols is None:
        intensity_cols = [c for c in features.columns if 'int_' in c.lower() or 'intensity' in c.lower()]
    
    if texture_cols is None:
        texture_cols = [c for c in features.columns if any(kw in c.lower() for kw in 
                       ['entropy', 'contrast', 'homogeneity', 'variance'])]
    
    if method == 'threshold':
        return _threshold_based_inference(features, area_col, intensity_cols, texture_cols)
    elif method == 'clustering':
        return _clustering_based_inference(features, area_col, intensity_cols, texture_cols)
    else:
        raise ValueError(f"Unknown method: {method}")


def _threshold_based_inference(
    features: pd.DataFrame,
    area_col: str,
    intensity_cols: List[str],
    texture_cols: List[str]
) -> pd.DataFrame:
    """
    Rule-based cell cycle inference using feature thresholds.
    
    Heuristics:
    - G0/G1: Small-medium area, low texture heterogeneity, compact chromatin
    - S: Medium-large area, high texture heterogeneity (replication foci)
    - G2/M: Large area, intermediate heterogeneity, condensed chromatin
    """
    results = pd.DataFrame()
    n_cells = len(features)
    
    # Compute cell cycle score based on multiple features
    scores = np.zeros((n_cells, 3))  # [G0/G1, S, G2/M]
    
    # Area-based scoring
    if area_col in features.columns:
        areas = features[area_col].values
        area_percentiles = _compute_percentiles(areas)
        
        # Small area suggests G0/G1
        scores[:, 0] += (area_percentiles < 40).astype(float)
        # Medium area could be S
        scores[:, 1] += ((area_percentiles >= 30) & (area_percentiles <= 70)).astype(float)
        # Large area suggests G2/M
        scores[:, 2] += (area_percentiles > 60).astype(float)
    
    # Texture heterogeneity scoring
    if texture_cols:
        texture_features = []
        for col in texture_cols:
            if col in features.columns:
                texture_features.append(features[col].values)
        
        if texture_features:
            # Average normalized texture scores
            texture_matrix = np.column_stack(texture_features)
            texture_score = np.nanmean(texture_matrix, axis=1)
            texture_percentiles = _compute_percentiles(texture_score)
            
            # High heterogeneity suggests S phase (replication foci)
            scores[:, 0] += (texture_percentiles < 35).astype(float)
            scores[:, 1] += (texture_percentiles > 50).astype(float) * 1.5  # Weight S phase
            scores[:, 2] += ((texture_percentiles >= 25) & (texture_percentiles <= 55)).astype(float)
    
    # Intensity variance scoring (S phase has distinct foci)
    intensity_var_cols = [c for c in intensity_cols if 'var' in c.lower() or 'std' in c.lower()]
    if intensity_var_cols:
        var_features = []
        for col in intensity_var_cols:
            if col in features.columns:
                var_features.append(features[col].values)
        
        if var_features:
            var_score = np.nanmean(np.column_stack(var_features), axis=1)
            var_percentiles = _compute_percentiles(var_score)
            
            # High variance suggests S phase
            scores[:, 1] += (var_percentiles > 60).astype(float)
    
    # Normalize scores to probabilities
    score_sums = np.sum(scores, axis=1, keepdims=True)
    score_sums[score_sums == 0] = 1
    probabilities = scores / score_sums
    
    # Assign phase based on highest probability
    phase_labels = ['G0/G1', 'S', 'G2/M']
    predicted_phases = [phase_labels[np.argmax(p)] for p in probabilities]
    
    results['cell_cycle_phase'] = predicted_phases
    results['cell_cycle_prob_G0G1'] = probabilities[:, 0]
    results['cell_cycle_prob_S'] = probabilities[:, 1]
    results['cell_cycle_prob_G2M'] = probabilities[:, 2]
    results['cell_cycle_confidence'] = np.max(probabilities, axis=1)
    
    # Add binary proliferating flag (S or G2/M)
    results['is_proliferating'] = (probabilities[:, 1] + probabilities[:, 2]) > probabilities[:, 0]
    
    return results


def _clustering_based_inference(
    features: pd.DataFrame,
    area_col: str,
    intensity_cols: List[str],
    texture_cols: List[str]
) -> pd.DataFrame:
    """
    Data-driven cell cycle inference using clustering.
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture
    except ImportError:
        logger.warning("scikit-learn not installed. Falling back to threshold method.")
        return _threshold_based_inference(features, area_col, intensity_cols, texture_cols)
    
    # Select features for clustering
    feature_cols = []
    if area_col in features.columns:
        feature_cols.append(area_col)
    
    for col in intensity_cols + texture_cols:
        if col in features.columns:
            feature_cols.append(col)
    
    if len(feature_cols) < 2:
        logger.warning("Not enough features for clustering. Falling back to threshold method.")
        return _threshold_based_inference(features, area_col, intensity_cols, texture_cols)
    
    # Prepare feature matrix
    X = features[feature_cols].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=np.nanmean(X))
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster into 3 groups (G0/G1, S, G2/M)
    n_clusters = 3
    
    try:
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(X_scaled)
        probabilities = gmm.predict_proba(X_scaled)
    except Exception:
        # Fall back to KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        # Create pseudo-probabilities from distances
        distances = kmeans.transform(X_scaled)
        inv_distances = 1 / (distances + 1e-6)
        probabilities = inv_distances / inv_distances.sum(axis=1, keepdims=True)
    
    # Order clusters by area (G0/G1 = smallest, G2/M = largest)
    if area_col in features.columns:
        cluster_areas = [features.loc[cluster_labels == c, area_col].median() 
                        for c in range(n_clusters)]
        cluster_order = np.argsort(cluster_areas)
    else:
        cluster_order = np.arange(n_clusters)
    
    # Map clusters to phases
    phase_map = {cluster_order[0]: 'G0/G1', 
                 cluster_order[1]: 'S', 
                 cluster_order[2]: 'G2/M'}
    
    results = pd.DataFrame()
    results['cell_cycle_phase'] = [phase_map[c] for c in cluster_labels]
    
    # Reorder probabilities to match G0/G1, S, G2/M
    reordered_probs = probabilities[:, cluster_order]
    results['cell_cycle_prob_G0G1'] = reordered_probs[:, 0]
    results['cell_cycle_prob_S'] = reordered_probs[:, 1]
    results['cell_cycle_prob_G2M'] = reordered_probs[:, 2]
    results['cell_cycle_confidence'] = np.max(probabilities, axis=1)
    results['is_proliferating'] = (reordered_probs[:, 1] + reordered_probs[:, 2]) > reordered_probs[:, 0]
    
    return results


def _compute_percentiles(values: np.ndarray) -> np.ndarray:
    """Compute percentile rank for each value."""
    valid_mask = ~np.isnan(values)
    percentiles = np.zeros_like(values)
    
    if np.sum(valid_mask) > 0:
        percentiles[valid_mask] = stats.rankdata(values[valid_mask]) / np.sum(valid_mask) * 100
    
    return percentiles


def compute_cell_cycle_features(
    intensity_image: np.ndarray,
    mask: np.ndarray
) -> pd.DataFrame:
    """
    Extract features specifically indicative of cell cycle state.
    
    These features complement standard morphology features and are
    designed to capture cell cycle-specific patterns.
    
    Args:
        intensity_image: Grayscale intensity image
        mask: Binary mask for the nucleus
        
    Returns:
        DataFrame with cell cycle-indicative features
    """
    if mask is None or intensity_image is None:
        return _empty_cell_cycle_features()
    
    if np.sum(mask) == 0:
        return _empty_cell_cycle_features()
    
    try:
        from scipy import ndimage
        from skimage.filters import threshold_otsu
        
        # Apply mask
        masked = intensity_image.astype(float) * mask
        valid_pixels = masked[mask > 0]
        
        if len(valid_pixels) == 0:
            return _empty_cell_cycle_features()
        
        features = {}
        
        # 1. Replication foci detection (S phase marker)
        try:
            threshold = threshold_otsu(valid_pixels)
        except ValueError:
            threshold = np.mean(valid_pixels)
        
        high_intensity = masked > (threshold + np.std(valid_pixels))
        labeled_foci, n_foci = ndimage.label(high_intensity)
        
        features['n_bright_foci'] = n_foci
        features['bright_foci_fraction'] = np.sum(high_intensity) / np.sum(mask)
        
        if n_foci > 0:
            foci_sizes = ndimage.sum(high_intensity, labeled_foci, range(1, n_foci + 1))
            features['mean_foci_size'] = np.mean(foci_sizes)
            features['foci_size_std'] = np.std(foci_sizes) if len(foci_sizes) > 1 else 0
        else:
            features['mean_foci_size'] = 0
            features['foci_size_std'] = 0
        
        # 2. Chromatin condensation (G2/M marker)
        # High local variance indicates condensed chromatin
        from scipy.ndimage import generic_filter
        
        def local_variance(values):
            return np.var(values)
        
        # Compute local variance in 5x5 windows
        local_var = generic_filter(masked, local_variance, size=5)
        features['chromatin_condensation_score'] = np.mean(local_var[mask > 0])
        
        # 3. Nuclear roundness (changes during mitosis)
        from skimage import measure as skmeasure
        props = skmeasure.regionprops(mask.astype(int))[0]
        
        perimeter = props.perimeter
        area = props.area
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        features['nuclear_circularity'] = circularity
        
        # Solidity (convex hull filled-ness)
        features['nuclear_solidity'] = props.solidity
        
        # 4. Intensity distribution metrics
        features['intensity_kurtosis'] = stats.kurtosis(valid_pixels)
        features['intensity_skewness'] = stats.skew(valid_pixels)
        
        # Bimodality coefficient (S phase has bimodal distribution)
        n = len(valid_pixels)
        skew = features['intensity_skewness']
        kurt = features['intensity_kurtosis']
        bimodality = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        features['intensity_bimodality'] = bimodality
        
        return pd.DataFrame([features])
        
    except Exception as e:
        logger.debug(f"Cell cycle feature extraction failed: {e}")
        return _empty_cell_cycle_features()


def _empty_cell_cycle_features() -> pd.DataFrame:
    """Return empty cell cycle features DataFrame."""
    features = {
        'n_bright_foci': np.nan,
        'bright_foci_fraction': np.nan,
        'mean_foci_size': np.nan,
        'foci_size_std': np.nan,
        'chromatin_condensation_score': np.nan,
        'nuclear_circularity': np.nan,
        'nuclear_solidity': np.nan,
        'intensity_kurtosis': np.nan,
        'intensity_skewness': np.nan,
        'intensity_bimodality': np.nan
    }
    return pd.DataFrame([features])


def stratify_by_cell_cycle(
    features: pd.DataFrame,
    cell_cycle_predictions: pd.DataFrame,
    target_col: str = 'cell_type'
) -> Dict[str, pd.DataFrame]:
    """
    Stratify analysis by cell cycle phase.
    
    This enables separate analysis of chromatin features while
    controlling for cell cycle confounding.
    
    Args:
        features: DataFrame with all features
        cell_cycle_predictions: DataFrame with cell cycle predictions
        target_col: Target variable column name
        
    Returns:
        Dict mapping cell cycle phase to stratified features
    """
    if 'cell_cycle_phase' not in cell_cycle_predictions.columns:
        return {'all': features}
    
    # Combine features with cell cycle
    combined = pd.concat([features.reset_index(drop=True), 
                         cell_cycle_predictions.reset_index(drop=True)], axis=1)
    
    stratified = {}
    for phase in ['G0/G1', 'S', 'G2/M']:
        phase_mask = combined['cell_cycle_phase'] == phase
        if phase_mask.sum() > 0:
            stratified[phase] = combined[phase_mask].copy()
    
    return stratified


def compute_cell_cycle_adjusted_features(
    features: pd.DataFrame,
    cell_cycle_predictions: pd.DataFrame,
    adjust_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute cell cycle-adjusted features by regressing out cell cycle effects.
    
    Args:
        features: DataFrame with all features
        cell_cycle_predictions: DataFrame with cell cycle predictions
        adjust_cols: Columns to adjust (default: all numeric)
        
    Returns:
        DataFrame with adjusted features
    """
    if 'cell_cycle_prob_G0G1' not in cell_cycle_predictions.columns:
        return features
    
    if adjust_cols is None:
        adjust_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    
    adjusted = features.copy()
    
    # Create cell cycle design matrix
    cc_probs = cell_cycle_predictions[['cell_cycle_prob_G0G1', 'cell_cycle_prob_S', 'cell_cycle_prob_G2M']].values
    
    for col in adjust_cols:
        if col not in features.columns:
            continue
        
        y = features[col].values
        valid_mask = ~np.isnan(y)
        
        if np.sum(valid_mask) < 10:
            continue
        
        try:
            # Regress out cell cycle effects
            X = cc_probs[valid_mask]
            y_valid = y[valid_mask]
            
            # Add intercept
            X_design = np.column_stack([np.ones(len(y_valid)), X])
            
            # OLS regression
            beta = np.linalg.lstsq(X_design, y_valid, rcond=None)[0]
            
            # Residuals are adjusted values
            residuals = y_valid - X_design @ beta
            
            # Add back mean
            adjusted_values = residuals + np.mean(y_valid)
            
            adjusted.loc[valid_mask, f'{col}_cc_adjusted'] = adjusted_values
            
        except Exception as e:
            logger.debug(f"Could not adjust {col}: {e}")
    
    return adjusted

