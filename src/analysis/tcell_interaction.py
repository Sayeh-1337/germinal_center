"""T-cell interaction and spatial proximity analysis"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial import cKDTree
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def compute_distance_matrix(
    data: pd.DataFrame,
    x_col: str = 'spat_centroid_x',
    y_col: str = 'spat_centroid_y',
    id_col: str = 'nuc_id'
) -> pd.DataFrame:
    """Compute pairwise Euclidean distance matrix
    
    Args:
        data: DataFrame with spatial coordinates
        x_col: Column name for x coordinate
        y_col: Column name for y coordinate
        id_col: Column name for cell identifiers
        
    Returns:
        DataFrame with pairwise distances
    """
    coords = data[[y_col, x_col]].values
    ids = data[id_col].values
    
    distances = squareform(pdist(coords))
    dist_df = pd.DataFrame(distances, index=ids, columns=ids)
    
    return dist_df


def tcell_radius_neighbors(
    dataset: pd.DataFrame,
    radius: float,
    tcell_label: str = 'T-cells',
    cell_type_col: str = 'cell_type',
    id_col: str = 'nuc_id',
    x_col: str = 'spat_centroid_x',
    y_col: str = 'spat_centroid_y',
    pixel_size: float = 1.0,
    show_progress: bool = True
) -> List[str]:
    """Find cells within a given radius of T-cells
    
    Uses KD-tree for efficient spatial queries (O(n log n) instead of O(n*m)).
    
    Args:
        dataset: DataFrame with cell data and spatial coordinates
        radius: Radius in pixels (or microns if pixel_size is set)
        tcell_label: Label for T-cells in cell_type_col
        cell_type_col: Column with cell type labels
        id_col: Column with cell identifiers
        x_col: Column for x coordinate
        y_col: Column for y coordinate
        pixel_size: Pixel size in microns (for radius conversion)
        show_progress: Whether to show progress bar
        
    Returns:
        List of cell IDs within radius of T-cells
    """
    # Convert radius to pixels if pixel_size provided
    radius_pixels = radius / pixel_size if pixel_size != 1.0 else radius
    
    neighbors = []
    images = dataset['image'].unique()
    
    img_iter = tqdm(images, desc=f"Finding neighbors (r={radius:.0f}μm)", disable=not show_progress)
    
    for img in img_iter:
        img_data = dataset[dataset['image'] == img]
        
        # Get T-cells and non-T-cells
        tcells = img_data[img_data[cell_type_col] == tcell_label]
        non_tcells = img_data[img_data[cell_type_col] != tcell_label]
        
        if len(tcells) == 0 or len(non_tcells) == 0:
            continue
        
        # Build KD-tree from T-cell coordinates for fast queries
        tcell_coords = tcells[[y_col, x_col]].values
        non_tcell_coords = non_tcells[[y_col, x_col]].values
        non_tcell_ids = non_tcells[id_col].values
        
        # Build tree and query for neighbors within radius
        tree = cKDTree(tcell_coords)
        
        # Query each non-T-cell's distance to nearest T-cell
        distances, _ = tree.query(non_tcell_coords, k=1)
        
        # Find non-T-cells within radius
        within_radius = distances <= radius_pixels
        neighbors.extend(non_tcell_ids[within_radius])
    
    return neighbors


def assign_tcell_influence(
    data: pd.DataFrame,
    spatial_coords: pd.DataFrame,
    contact_radius: float = 15.0,  # microns
    signaling_radius: float = 30.0,  # microns
    pixel_size: float = 0.3225,  # microns per pixel
    tcell_label: str = 'T-cells',
    cell_type_col: str = 'cell_type'
) -> pd.DataFrame:
    """Assign T-cell influence status to cells
    
    Categories:
    - T-cell interactors: within contact_radius
    - potential T-cell interactors: between contact and signaling radius
    - Non-T-cell interactors: beyond signaling_radius
    
    Args:
        data: DataFrame with cell data
        spatial_coords: DataFrame with spatial coordinates
        contact_radius: Physical contact radius in microns
        signaling_radius: Cell signaling radius in microns
        pixel_size: Pixel size in microns
        tcell_label: Label for T-cells
        cell_type_col: Column with cell type labels
        
    Returns:
        DataFrame with tcell_influence column added
    """
    result = data.copy()
    
    # Merge spatial coordinates
    if 'spat_centroid_x' not in result.columns:
        # Check what centroid columns exist in spatial_coords
        centroid_cols = [c for c in spatial_coords.columns if 'centroid' in c.lower()]
        if 'centroid-0' in spatial_coords.columns and 'centroid-1' in spatial_coords.columns:
            merge_cols = ['nuc_id', 'centroid-0', 'centroid-1']
        elif 'centroid_0' in spatial_coords.columns and 'centroid_1' in spatial_coords.columns:
            merge_cols = ['nuc_id', 'centroid_0', 'centroid_1']
        else:
            # Use whatever centroid columns exist
            merge_cols = ['nuc_id'] + centroid_cols[:2]
        
        # Check if result already has these columns (will cause conflict)
        has_conflict = merge_cols[1] in result.columns or merge_cols[2] in result.columns
        
        if has_conflict:
            # Drop conflicting columns before merge to avoid suffixes
            result = result.drop(columns=[c for c in merge_cols[1:] if c in result.columns])
        
        result = result.merge(
            spatial_coords[merge_cols],
            on='nuc_id',
            how='left'
        )
        
        # Use the actual column names that were merged
        y_col = merge_cols[1]  # centroid-0 or centroid_0
        x_col = merge_cols[2]  # centroid-1 or centroid_1
        result['spat_centroid_y'] = result[y_col]
        result['spat_centroid_x'] = result[x_col]
    
    # Convert radii to pixels
    contact_radius_px = contact_radius / pixel_size
    signaling_radius_px = signaling_radius / pixel_size
    
    logger.info(f"Computing T-cell influence zones...")
    logger.info(f"  Contact radius: {contact_radius} μm ({contact_radius_px:.1f} px)")
    logger.info(f"  Signaling radius: {signaling_radius} μm ({signaling_radius_px:.1f} px)")
    
    # Find cells in each zone using optimized KD-tree queries
    contact_neighbors = tcell_radius_neighbors(
        result, contact_radius_px, tcell_label, cell_type_col,
        show_progress=True
    )
    
    signaling_neighbors = tcell_radius_neighbors(
        result, signaling_radius_px, tcell_label, cell_type_col,
        show_progress=True
    )
    
    # Assign influence status
    result['tcell_influence'] = 'Non-T-cell interactors'
    
    # Potential interactors (between contact and signaling radius)
    potential_ids = set(signaling_neighbors) - set(contact_neighbors)
    result.loc[result['nuc_id'].isin(potential_ids), 'tcell_influence'] = 'potential T-cell interactors'
    
    # Direct interactors (within contact radius)
    result.loc[result['nuc_id'].isin(contact_neighbors), 'tcell_influence'] = 'T-cell interactors'
    
    # Log summary
    influence_counts = result['tcell_influence'].value_counts()
    logger.info("T-cell influence distribution:")
    for status, count in influence_counts.items():
        logger.info(f"  {status}: {count}")
    
    return result


def get_distances_to_tcells(
    dataset: pd.DataFrame,
    spatial_coords: pd.DataFrame,
    tcell_label: str = 'T-cells',
    cell_type_col: str = 'cell_type',
    id_col: str = 'nuc_id',
    range_normalize: bool = True,
    show_progress: bool = True
) -> pd.DataFrame:
    """Compute distance metrics from each cell to T-cells
    
    Uses vectorized operations for efficient computation.
    
    Args:
        dataset: DataFrame with cell data
        spatial_coords: DataFrame with spatial coordinates
        tcell_label: Label for T-cells
        cell_type_col: Column with cell type labels
        id_col: Column with cell identifiers
        range_normalize: Whether to normalize distances per image
        show_progress: Whether to show progress bar
        
    Returns:
        DataFrame with distance metrics (mean, median, min to T-cells)
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
    
    # Check if dataset already has these columns (will cause conflict)
    data = dataset.copy()
    has_conflict = y_col in data.columns or x_col in data.columns
    if has_conflict:
        # Drop conflicting columns before merge to avoid suffixes
        data = data.drop(columns=[c for c in [y_col, x_col] if c in data.columns])
    
    data = data.merge(
        spatial_coords[merge_cols],
        on='nuc_id',
        how='left'
    )
    data['spat_centroid_y'] = data[y_col]
    data['spat_centroid_x'] = data[x_col]
    
    results = []
    images = data['image'].unique()
    
    img_iter = tqdm(images, desc="Computing T-cell distances", disable=not show_progress)
    
    for img in img_iter:
        img_data = data[data['image'] == img]
        
        coords = img_data[['spat_centroid_y', 'spat_centroid_x']].values
        ids = img_data[id_col].values
        cell_types = img_data[cell_type_col].values
        
        # Get T-cell indices
        tcell_mask = cell_types == tcell_label
        
        if not np.any(tcell_mask):
            continue
        
        tcell_coords = coords[tcell_mask]
        
        # Use cdist for efficient distance computation (all cells to T-cells)
        dist_to_tcells = cdist(coords, tcell_coords)
        
        # Range normalize if requested
        if range_normalize and dist_to_tcells.max() > 0:
            dist_to_tcells = (dist_to_tcells - dist_to_tcells.min()) / (dist_to_tcells.max() - dist_to_tcells.min())
        
        # Vectorized computation of statistics
        mean_dists = np.mean(dist_to_tcells, axis=1)
        median_dists = np.median(dist_to_tcells, axis=1)
        min_dists = np.min(dist_to_tcells, axis=1)
        
        # Create results for this image
        img_results = pd.DataFrame({
            id_col: ids,
            'image': img,
            'tcell_mean_distance': mean_dists,
            'tcell_median_distance': median_dists,
            'tcell_min_distance': min_dists
        })
        results.append(img_results)
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

