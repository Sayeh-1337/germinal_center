# -*- coding: utf-8 -*-
"""
Relative and interaction features for germinal center analysis.

This module computes features relative to the local neighborhood,
capturing cell-to-cell relationships and microenvironmental context.

Key feature types:
1. Relative features: Cell properties normalized to local neighbors
2. Interaction features: Properties capturing cell-cell relationships
3. Gradient features: Spatial derivatives of features

Biological rationale:
- Absolute features may vary with imaging conditions
- Relative features capture cell state within local context
- May improve robustness to batch effects
- Captures microenvironmental influence on chromatin
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def compute_relative_features(
    features: pd.DataFrame,
    spatial_coords: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    k_neighbors: int = 10,
    coord_cols: Tuple[str, str] = ('centroid-0', 'centroid-1')
) -> pd.DataFrame:
    """
    Compute features relative to local neighborhood.
    
    Examples:
    - area_relative = cell_area / mean(neighbor_areas)
    - intensity_relative = cell_intensity / mean(neighbor_intensities)
    
    Biological rationale:
    - Absolute features may vary with imaging conditions
    - Relative features capture cell state within local context
    - May improve robustness to batch effects
    
    Args:
        features: DataFrame with cell features
        spatial_coords: DataFrame with spatial coordinates
        feature_cols: Columns to compute relative features for
        k_neighbors: Number of neighbors to consider
        coord_cols: Column names for x,y coordinates
        
    Returns:
        DataFrame with relative features
    """
    coords = spatial_coords[[coord_cols[0], coord_cols[1]]].values
    n_cells = len(coords)
    
    if n_cells < k_neighbors + 1:
        logger.warning(f"Not enough cells for k={k_neighbors} neighbors")
        k_neighbors = max(1, n_cells - 1)
    
    if feature_cols is None:
        # Select numeric columns, excluding identifiers
        feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c not in ['label', 'nuc_id']]
    
    # Build KD-tree
    tree = KDTree(coords)
    _, indices = tree.query(coords, k=k_neighbors + 1)  # +1 for self
    
    relative_features = {}
    
    for col in feature_cols:
        if col not in features.columns:
            continue
        
        values = features[col].values
        
        # Compute neighbor means (excluding self)
        neighbor_means = np.array([
            np.nanmean(values[indices[i, 1:]]) for i in range(n_cells)
        ])
        
        # Relative feature: value / neighbor_mean
        with np.errstate(divide='ignore', invalid='ignore'):
            relative = values / neighbor_means
            relative[~np.isfinite(relative)] = np.nan
        
        relative_features[f'{col}_relative'] = relative
        
        # Difference feature: value - neighbor_mean
        relative_features[f'{col}_diff_from_neighbors'] = values - neighbor_means
        
        # Z-score relative to neighbors
        neighbor_stds = np.array([
            np.nanstd(values[indices[i, 1:]]) for i in range(n_cells)
        ])
        with np.errstate(divide='ignore', invalid='ignore'):
            z_score = (values - neighbor_means) / neighbor_stds
            z_score[~np.isfinite(z_score)] = np.nan
        
        relative_features[f'{col}_zscore_local'] = z_score
    
    return pd.DataFrame(relative_features)


def compute_interaction_features(
    features: pd.DataFrame,
    spatial_coords: pd.DataFrame,
    cell_types: Optional[pd.Series] = None,
    feature_cols: Optional[List[str]] = None,
    k_neighbors: int = 10,
    coord_cols: Tuple[str, str] = ('centroid-0', 'centroid-1')
) -> pd.DataFrame:
    """
    Compute features capturing cell-cell interactions.
    
    Examples:
    - Mean chromatin feature of T-cell neighbors
    - Fraction of neighbors that are DZ vs LZ
    - Chromatin feature gradient (spatial derivative)
    
    Captures: Microenvironmental influence on chromatin
    
    Args:
        features: DataFrame with cell features
        spatial_coords: DataFrame with spatial coordinates
        cell_types: Optional series with cell type labels
        feature_cols: Columns to compute interaction features for
        k_neighbors: Number of neighbors to consider
        coord_cols: Column names for x,y coordinates
        
    Returns:
        DataFrame with interaction features
    """
    coords = spatial_coords[[coord_cols[0], coord_cols[1]]].values
    n_cells = len(coords)
    
    if n_cells < 2:
        return pd.DataFrame()
    
    if feature_cols is None:
        feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c not in ['label', 'nuc_id']][:20]  # Limit
    
    # Build KD-tree
    tree = KDTree(coords)
    k_query = min(k_neighbors + 1, n_cells)
    distances, indices = tree.query(coords, k=k_query)
    
    interaction_features = {}
    
    # Distance to nearest neighbors
    if k_query > 1:
        interaction_features['dist_nearest_neighbor'] = distances[:, 1]
        interaction_features['dist_mean_neighbors'] = np.mean(distances[:, 1:], axis=1)
    
    # Cell type-specific features
    if cell_types is not None and len(cell_types) == n_cells:
        unique_types = cell_types.unique()
        
        for cell_type in unique_types:
            type_mask = (cell_types == cell_type).values
            
            # Distance to nearest cell of each type
            type_coords = coords[type_mask]
            if len(type_coords) > 0:
                type_tree = KDTree(type_coords)
                dist_to_type, _ = type_tree.query(coords, k=1)
                interaction_features[f'dist_nearest_{cell_type}'] = dist_to_type
            
            # Fraction of neighbors that are this type
            neighbor_type_fractions = []
            for i in range(n_cells):
                neighbor_idx = indices[i, 1:]
                n_type = np.sum(type_mask[neighbor_idx])
                neighbor_type_fractions.append(n_type / len(neighbor_idx))
            
            interaction_features[f'neighbor_fraction_{cell_type}'] = neighbor_type_fractions
            
            # Mean features of neighbors of this type
            for col in feature_cols[:5]:  # Limit to key features
                if col not in features.columns:
                    continue
                
                values = features[col].values
                type_neighbor_means = []
                
                for i in range(n_cells):
                    neighbor_idx = indices[i, 1:]
                    type_neighbors = neighbor_idx[type_mask[neighbor_idx]]
                    
                    if len(type_neighbors) > 0:
                        type_neighbor_means.append(np.nanmean(values[type_neighbors]))
                    else:
                        type_neighbor_means.append(np.nan)
                
                interaction_features[f'{col}_mean_{cell_type}_neighbors'] = type_neighbor_means
    
    # Feature heterogeneity in neighborhood
    for col in feature_cols[:10]:  # Limit
        if col not in features.columns:
            continue
        
        values = features[col].values
        
        # Variance among neighbors (local heterogeneity)
        neighbor_vars = []
        for i in range(n_cells):
            neighbor_idx = indices[i, 1:]
            neighbor_vars.append(np.nanvar(values[neighbor_idx]))
        
        interaction_features[f'{col}_neighbor_variance'] = neighbor_vars
        
        # Range among neighbors
        neighbor_ranges = []
        for i in range(n_cells):
            neighbor_idx = indices[i, 1:]
            neighbor_vals = values[neighbor_idx]
            valid_vals = neighbor_vals[~np.isnan(neighbor_vals)]
            if len(valid_vals) > 0:
                neighbor_ranges.append(np.max(valid_vals) - np.min(valid_vals))
            else:
                neighbor_ranges.append(np.nan)
        
        interaction_features[f'{col}_neighbor_range'] = neighbor_ranges
    
    return pd.DataFrame(interaction_features)


def compute_spatial_gradients(
    features: pd.DataFrame,
    spatial_coords: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    k_neighbors: int = 10,
    coord_cols: Tuple[str, str] = ('centroid-0', 'centroid-1')
) -> pd.DataFrame:
    """
    Compute spatial gradients of features.
    
    Gradients indicate how features change across space, which can
    reveal microenvironmental influences.
    
    Args:
        features: DataFrame with cell features
        spatial_coords: DataFrame with spatial coordinates
        feature_cols: Columns to compute gradients for
        k_neighbors: Number of neighbors for gradient estimation
        coord_cols: Column names for x,y coordinates
        
    Returns:
        DataFrame with gradient features
    """
    coords = spatial_coords[[coord_cols[0], coord_cols[1]]].values
    n_cells = len(coords)
    
    if n_cells < 3:
        return pd.DataFrame()
    
    if feature_cols is None:
        feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c not in ['label', 'nuc_id']][:10]
    
    # Build KD-tree
    tree = KDTree(coords)
    k_query = min(k_neighbors + 1, n_cells)
    distances, indices = tree.query(coords, k=k_query)
    
    gradient_features = {}
    
    for col in feature_cols:
        if col not in features.columns:
            continue
        
        values = features[col].values
        
        # Estimate gradient using weighted differences
        grad_x = np.zeros(n_cells)
        grad_y = np.zeros(n_cells)
        grad_magnitude = np.zeros(n_cells)
        
        for i in range(n_cells):
            neighbor_idx = indices[i, 1:]
            neighbor_dists = distances[i, 1:]
            
            # Skip if no valid neighbors
            if len(neighbor_idx) == 0:
                continue
            
            # Compute weighted differences
            dx = coords[neighbor_idx, 0] - coords[i, 0]
            dy = coords[neighbor_idx, 1] - coords[i, 1]
            dv = values[neighbor_idx] - values[i]
            
            # Handle missing values
            valid = ~np.isnan(dv)
            if np.sum(valid) < 2:
                continue
            
            dx, dy, dv = dx[valid], dy[valid], dv[valid]
            weights = 1 / (neighbor_dists[valid] + 1e-6)
            
            # Weighted least squares for gradient
            try:
                A = np.column_stack([dx, dy])
                W = np.diag(weights)
                gradient = np.linalg.lstsq(W @ A, W @ dv, rcond=None)[0]
                grad_x[i] = gradient[0]
                grad_y[i] = gradient[1]
                grad_magnitude[i] = np.sqrt(gradient[0]**2 + gradient[1]**2)
            except Exception:
                pass
        
        gradient_features[f'{col}_gradient_x'] = grad_x
        gradient_features[f'{col}_gradient_y'] = grad_y
        gradient_features[f'{col}_gradient_magnitude'] = grad_magnitude
        
        # Direction of gradient
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_direction = np.arctan2(grad_y, grad_x)
            grad_direction[grad_magnitude < 1e-6] = np.nan
        gradient_features[f'{col}_gradient_direction'] = grad_direction
    
    return pd.DataFrame(gradient_features)


def compute_neighborhood_composition(
    spatial_coords: pd.DataFrame,
    cell_types: pd.Series,
    radii: List[float] = [25, 50, 100],
    coord_cols: Tuple[str, str] = ('centroid-0', 'centroid-1')
) -> pd.DataFrame:
    """
    Compute cell type composition at multiple neighborhood radii.
    
    Args:
        spatial_coords: DataFrame with spatial coordinates
        cell_types: Series with cell type labels
        radii: List of radii to consider
        coord_cols: Column names for x,y coordinates
        
    Returns:
        DataFrame with neighborhood composition features
    """
    coords = spatial_coords[[coord_cols[0], coord_cols[1]]].values
    n_cells = len(coords)
    
    if n_cells < 2:
        return pd.DataFrame()
    
    tree = KDTree(coords)
    unique_types = cell_types.unique()
    
    composition_features = {}
    
    for radius in radii:
        # Query all neighbors within radius
        neighbors_lists = tree.query_ball_point(coords, radius)
        
        for cell_type in unique_types:
            type_mask = (cell_types == cell_type).values
            
            fractions = []
            counts = []
            
            for i in range(n_cells):
                neighbors = neighbors_lists[i]
                # Exclude self
                neighbors = [n for n in neighbors if n != i]
                
                if len(neighbors) > 0:
                    n_type = np.sum(type_mask[neighbors])
                    fractions.append(n_type / len(neighbors))
                    counts.append(n_type)
                else:
                    fractions.append(np.nan)
                    counts.append(0)
            
            composition_features[f'{cell_type}_fraction_r{int(radius)}'] = fractions
            composition_features[f'{cell_type}_count_r{int(radius)}'] = counts
        
        # Total neighbor count
        total_counts = [len([n for n in neighbors_lists[i] if n != i]) for i in range(n_cells)]
        composition_features[f'total_neighbors_r{int(radius)}'] = total_counts
    
    return pd.DataFrame(composition_features)


def compute_boundary_proximity(
    spatial_coords: pd.DataFrame,
    zone_labels: pd.Series,
    coord_cols: Tuple[str, str] = ('centroid-0', 'centroid-1'),
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Compute proximity to DZ/LZ boundary.
    
    Biological relevance:
    - Cells near boundary may be transitioning
    - Boundary region has distinct microenvironment
    - May explain intermediate phenotypes
    
    Args:
        spatial_coords: DataFrame with spatial coordinates
        zone_labels: Series with DZ/LZ zone labels
        coord_cols: Column names for x,y coordinates
        n_bins: Number of distance bins
        
    Returns:
        DataFrame with boundary proximity features
    """
    coords = spatial_coords[[coord_cols[0], coord_cols[1]]].values
    n_cells = len(coords)
    
    if n_cells < 2:
        return pd.DataFrame()
    
    # Identify unique zones
    unique_zones = zone_labels.unique()
    if len(unique_zones) < 2:
        return pd.DataFrame({'boundary_distance': [np.nan] * n_cells})
    
    features = {}
    
    for zone in unique_zones:
        zone_mask = (zone_labels == zone).values
        zone_coords = coords[zone_mask]
        
        if len(zone_coords) == 0:
            features[f'dist_to_{zone}'] = [np.nan] * n_cells
            continue
        
        zone_tree = KDTree(zone_coords)
        distances, _ = zone_tree.query(coords, k=1)
        features[f'dist_to_{zone}'] = distances
    
    # Distance to nearest cell of opposite zone
    boundary_distances = np.zeros(n_cells)
    for i in range(n_cells):
        my_zone = zone_labels.iloc[i]
        other_zone_mask = zone_labels != my_zone
        if np.sum(other_zone_mask) > 0:
            other_coords = coords[other_zone_mask.values]
            other_tree = KDTree(other_coords)
            dist, _ = other_tree.query(coords[i:i+1], k=1)
            boundary_distances[i] = dist[0]
        else:
            boundary_distances[i] = np.nan
    
    features['boundary_distance'] = boundary_distances
    
    # Classify as boundary region (within certain distance)
    threshold = np.nanpercentile(boundary_distances, 25)
    features['is_boundary_region'] = (boundary_distances <= threshold).astype(int)
    
    return pd.DataFrame(features)


def extract_all_relative_features(
    features: pd.DataFrame,
    spatial_coords: pd.DataFrame,
    cell_types: Optional[pd.Series] = None,
    zone_labels: Optional[pd.Series] = None,
    feature_cols: Optional[List[str]] = None,
    k_neighbors: int = 10,
    coord_cols: Tuple[str, str] = ('centroid-0', 'centroid-1'),
    compute_gradients: bool = True
) -> pd.DataFrame:
    """
    Extract all relative and interaction features.
    
    This is the main entry point for relative feature analysis.
    
    Args:
        features: DataFrame with cell features
        spatial_coords: DataFrame with spatial coordinates
        cell_types: Optional series with cell type labels
        zone_labels: Optional series with DZ/LZ zone labels
        feature_cols: Columns to compute features for
        k_neighbors: Number of neighbors
        coord_cols: Column names for x,y coordinates
        compute_gradients: Whether to compute spatial gradients
        
    Returns:
        DataFrame with all relative features
    """
    all_features = pd.DataFrame()
    
    # Relative features
    logger.info("Computing relative features...")
    relative_feat = compute_relative_features(
        features, spatial_coords, feature_cols, k_neighbors, coord_cols
    )
    if not relative_feat.empty:
        all_features = pd.concat([all_features, relative_feat], axis=1)
    
    # Interaction features
    logger.info("Computing interaction features...")
    interaction_feat = compute_interaction_features(
        features, spatial_coords, cell_types, feature_cols, k_neighbors, coord_cols
    )
    if not interaction_feat.empty:
        all_features = pd.concat([all_features, interaction_feat], axis=1)
    
    # Spatial gradients
    if compute_gradients:
        logger.info("Computing spatial gradients...")
        gradient_feat = compute_spatial_gradients(
            features, spatial_coords, feature_cols, k_neighbors, coord_cols
        )
        if not gradient_feat.empty:
            all_features = pd.concat([all_features, gradient_feat], axis=1)
    
    # Neighborhood composition (if cell types provided)
    if cell_types is not None:
        logger.info("Computing neighborhood composition...")
        composition_feat = compute_neighborhood_composition(
            spatial_coords, cell_types, [25, 50, 100], coord_cols
        )
        if not composition_feat.empty:
            all_features = pd.concat([all_features, composition_feat], axis=1)
    
    # Boundary proximity (if zone labels provided)
    if zone_labels is not None:
        logger.info("Computing boundary proximity...")
        boundary_feat = compute_boundary_proximity(
            spatial_coords, zone_labels, coord_cols
        )
        if not boundary_feat.empty:
            all_features = pd.concat([all_features, boundary_feat], axis=1)
    
    return all_features

