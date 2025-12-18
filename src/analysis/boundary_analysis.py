"""DZ/LZ boundary distance and proximity analysis"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def get_distances_to_dz_lz_border(
    data: pd.DataFrame,
    spatial_coords: pd.DataFrame,
    alpha: float = 0.02,
    dz_label: str = 'DZ B-cells',
    lz_label: str = 'LZ B-cells',
    cell_type_col: str = 'cell_type',
    id_col: str = 'nuc_id',
    n_neighbors: int = 50
) -> pd.DataFrame:
    """Compute distance measures to the DZ/LZ boundary
    
    Two measures are computed:
    1. scaled_distance_to_border: Average distance to k closest cells of opposite type
    2. frequency_based_distance_to_border: Fraction of same-type cells among n nearest neighbors
    
    Args:
        data: DataFrame with cell data and cell types
        spatial_coords: DataFrame with spatial coordinates
        alpha: Fraction of cells to use as k for average distance (default 0.02 = 2%)
        dz_label: Label for dark zone B-cells
        lz_label: Label for light zone B-cells
        cell_type_col: Column with cell type labels
        id_col: Column with cell identifiers
        n_neighbors: Number of neighbors for frequency-based distance
        
    Returns:
        DataFrame with boundary distance measures
    """
    # Merge coordinates - handle different column naming conventions
    centroid_cols = [c for c in spatial_coords.columns if 'centroid' in c.lower()]
    if 'centroid-0' in spatial_coords.columns and 'centroid-1' in spatial_coords.columns:
        merge_cols = ['nuc_id', 'centroid-0', 'centroid-1']
        y_col, x_col = 'centroid-0', 'centroid-1'
    elif 'centroid_0' in spatial_coords.columns and 'centroid_1' in spatial_coords.columns:
        merge_cols = ['nuc_id', 'centroid_0', 'centroid_1']
        y_col, x_col = 'centroid_0', 'centroid_1'
    else:
        merge_cols = ['nuc_id'] + centroid_cols[:2]
        y_col, x_col = centroid_cols[0], centroid_cols[1] if len(centroid_cols) > 1 else centroid_cols[0]
    
    # Check if data already has these columns (will cause conflict)
    merged = data.copy()
    has_conflict = y_col in merged.columns or x_col in merged.columns
    if has_conflict:
        # Drop conflicting columns before merge to avoid suffixes
        merged = merged.drop(columns=[c for c in [y_col, x_col] if c in merged.columns])
    
    merged = merged.merge(
        spatial_coords[merge_cols],
        on='nuc_id',
        how='left'
    )
    
    results = []
    images = merged['image'].unique()
    
    for img in tqdm(images, desc="Computing boundary distances"):
        img_data = merged[merged['image'] == img].copy()
        
        # Get B-cells only
        bcells = img_data[img_data[cell_type_col].isin([dz_label, lz_label])]
        
        if len(bcells) < 10:
            continue
        
        coords = bcells[[y_col, x_col]].values
        ids = bcells[id_col].values
        cell_types = bcells[cell_type_col].values
        
        # Compute distance matrix
        dist_matrix = squareform(pdist(coords))
        max_dist = np.max(dist_matrix) if dist_matrix.size > 0 else 1.0
        
        # Determine k for average distance
        k = max(1, int(alpha * len(bcells)))
        
        # Precompute masks for efficiency
        dz_mask = cell_types == dz_label
        lz_mask = cell_types == lz_label
        
        for i, (cell_id, cell_type) in enumerate(zip(ids, cell_types)):
            # Get opposite type
            opposite_mask = lz_mask if cell_type == dz_label else dz_mask
            
            if not np.any(opposite_mask):
                continue
            
            # Method 1: Average distance to k closest opposite-type cells
            opposite_distances = dist_matrix[i, opposite_mask]
            k_closest = min(k, len(opposite_distances))
            avg_dist = np.mean(np.partition(opposite_distances, k_closest-1)[:k_closest])
            
            # Normalize by max distance in image
            scaled_dist = avg_dist / max_dist if max_dist > 0 else 0
            
            # Method 2: Frequency-based (fraction of same type among n nearest neighbors)
            all_distances = dist_matrix[i, :].copy()
            # Exclude self (distance 0)
            all_distances[i] = np.inf
            nearest_indices = np.argpartition(all_distances, n_neighbors-1)[:n_neighbors]
            nearest_types = cell_types[nearest_indices]
            same_type_freq = np.mean(nearest_types == cell_type)
            
            # Convert to "distance to border" - cells at border have ~0.5 frequency
            freq_based_dist = abs(same_type_freq - 0.5)
            
            results.append({
                id_col: cell_id,
                'image': img,
                'cell_type': cell_type,
                'centroid-0': coords[i, 0],  # Keep consistent output column names
                'centroid-1': coords[i, 1],
                'scaled_distance_to_border': scaled_dist,
                'frequency_based_distance_to_border': freq_based_dist,
                'same_type_neighbor_frequency': same_type_freq
            })
    
    return pd.DataFrame(results)


def assign_border_proximity(
    data: pd.DataFrame,
    threshold: float = 0.4,
    distance_col: str = 'frequency_based_distance_to_border'
) -> pd.DataFrame:
    """Assign border proximity status based on distance measure
    
    Args:
        data: DataFrame with boundary distance measures
        threshold: Distance threshold (cells below are "close" to border)
        distance_col: Column with distance measure
        
    Returns:
        DataFrame with border_proximity column added
    """
    result = data.copy()
    
    result['border_proximity'] = np.where(
        result[distance_col] < threshold,
        'close',
        'distant'
    )
    
    # Log summary
    prox_counts = result['border_proximity'].value_counts()
    logger.info("Border proximity distribution:")
    for status, count in prox_counts.items():
        logger.info(f"  {status}: {count}")
    
    return result


def analyze_boundary_differences(
    data: pd.DataFrame,
    features: pd.DataFrame,
    cell_type_col: str = 'cell_type',
    proximity_col: str = 'border_proximity'
) -> Dict:
    """Analyze chrometric differences between cells close/distant to border
    
    Args:
        data: DataFrame with cell data and proximity labels
        features: DataFrame with chrometric features
        cell_type_col: Column with cell type labels
        proximity_col: Column with proximity status
        
    Returns:
        Dictionary with analysis results
    """
    from scipy.stats import ttest_ind
    
    results = {'cell_types': {}}
    
    # Merge data with features
    common_idx = data.index.intersection(features.index)
    merged = data.loc[common_idx].copy()
    merged_features = features.loc[common_idx].copy()
    
    for cell_type in merged[cell_type_col].unique():
        if cell_type == 'n/a':
            continue
            
        ct_mask = merged[cell_type_col] == cell_type
        ct_data = merged[ct_mask]
        ct_features = merged_features[ct_mask]
        
        close_mask = ct_data[proximity_col] == 'close'
        distant_mask = ct_data[proximity_col] == 'distant'
        
        if close_mask.sum() < 10 or distant_mask.sum() < 10:
            continue
        
        # Test each feature
        feature_results = []
        for col in ct_features.columns:
            close_vals = ct_features.loc[close_mask, col].dropna()
            distant_vals = ct_features.loc[distant_mask, col].dropna()
            
            if len(close_vals) > 1 and len(distant_vals) > 1:
                stat, pval = ttest_ind(close_vals, distant_vals, equal_var=False)
                
                feature_results.append({
                    'feature': col,
                    't_statistic': stat,
                    'p_value': pval,
                    'mean_close': close_vals.mean(),
                    'mean_distant': distant_vals.mean(),
                    'effect_size': (close_vals.mean() - distant_vals.mean()) / np.sqrt(
                        (close_vals.std()**2 + distant_vals.std()**2) / 2
                    )
                })
        
        results['cell_types'][cell_type] = pd.DataFrame(feature_results)
    
    return results

