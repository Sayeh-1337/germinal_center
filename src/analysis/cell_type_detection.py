"""Cell type detection using Gaussian Mixture Model thresholding"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def fit_gmm_threshold(intensities: np.ndarray, n_components: int = 2) -> float:
    """Fit a Gaussian Mixture Model and return threshold between components
    
    Args:
        intensities: Array of intensity values
        n_components: Number of GMM components (default 2 for positive/negative)
        
    Returns:
        Threshold value (mean of the two component means)
    """
    from sklearn.mixture import GaussianMixture
    
    # Remove NaN and reshape
    valid_intensities = intensities[~np.isnan(intensities)].reshape(-1, 1)
    
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(valid_intensities)
    
    # Get means and sort them
    means = gmm.means_.flatten()
    means_sorted = np.sort(means)
    
    # Threshold is the midpoint between the two means
    threshold = np.mean(means_sorted)
    
    return threshold


def get_positive_cells(
    intensity_df: pd.DataFrame,
    intensity_col: str = 'int_mean',
    id_col: str = 'nuc_id',
    image_col: str = 'image',
    per_image: bool = True
) -> List[str]:
    """Identify positive cells using GMM thresholding
    
    Args:
        intensity_df: DataFrame with intensity measurements
        intensity_col: Column containing intensity values
        id_col: Column with cell identifiers
        image_col: Column with image identifiers
        per_image: Whether to fit GMM per image or globally
        
    Returns:
        List of positive cell IDs
    """
    positive_cells = []
    
    if per_image:
        # Fit GMM separately for each image
        images = intensity_df[image_col].unique()
        for img in images:
            img_data = intensity_df[intensity_df[image_col] == img]
            intensities = img_data[intensity_col].values
            
            try:
                threshold = fit_gmm_threshold(intensities)
                positive_ids = img_data[img_data[intensity_col] > threshold][id_col].tolist()
                positive_cells.extend(positive_ids)
            except Exception as e:
                logger.warning(f"GMM fitting failed for image {img}: {e}")
    else:
        # Fit GMM globally
        intensities = intensity_df[intensity_col].values
        threshold = fit_gmm_threshold(intensities)
        positive_cells = intensity_df[intensity_df[intensity_col] > threshold][id_col].tolist()
    
    return positive_cells


def assign_cell_types(
    nuc_features: pd.DataFrame,
    aicda_levels: pd.DataFrame,
    cd3_levels: pd.DataFrame,
    gc_levels: Optional[pd.DataFrame] = None,
    gc_threshold: float = 0
) -> pd.DataFrame:
    """Assign cell types based on marker expression
    
    Cell type assignment:
    - CD3+ AICDA- -> T-cells
    - CD3- AICDA+ -> DZ B-cells (Dark Zone)
    - CD3- AICDA- -> LZ B-cells (Light Zone)
    - CD3+ AICDA+ -> n/a (unusual)
    
    Args:
        nuc_features: Nuclear features DataFrame
        aicda_levels: AICDA intensity measurements
        cd3_levels: CD3 intensity measurements
        gc_levels: Germinal center mask levels (optional)
        gc_threshold: Threshold for germinal center positivity
        
    Returns:
        DataFrame with cell type annotations
    """
    result = nuc_features.copy()
    
    # Get positive cells for each marker
    logger.info("Detecting AICDA+ cells...")
    aicda_positive = get_positive_cells(aicda_levels)
    logger.info(f"  Found {len(aicda_positive)} AICDA+ cells")
    
    logger.info("Detecting CD3+ cells...")
    cd3_positive = get_positive_cells(cd3_levels)
    logger.info(f"  Found {len(cd3_positive)} CD3+ cells")
    
    # Initialize status columns
    result['aicda_status'] = 'negative'
    result['cd3_status'] = 'negative'
    result['gc_status'] = 'negative'
    
    # Set positive status
    result.loc[result['nuc_id'].isin(aicda_positive), 'aicda_status'] = 'positive'
    result.loc[result['nuc_id'].isin(cd3_positive), 'cd3_status'] = 'positive'
    
    # Set germinal center status if available
    if gc_levels is not None:
        gc_positive = gc_levels[gc_levels['int_mean'] > gc_threshold]['nuc_id'].tolist()
        result.loc[result['nuc_id'].isin(gc_positive), 'gc_status'] = 'positive'
        logger.info(f"  Found {len(gc_positive)} cells within germinal center")
    
    # Assign cell types
    result['cell_type'] = 'n/a'
    
    # DZ B-cells: AICDA+ CD3-
    dz_mask = (result['aicda_status'] == 'positive') & (result['cd3_status'] == 'negative')
    result.loc[dz_mask, 'cell_type'] = 'DZ B-cells'
    
    # LZ B-cells: AICDA- CD3-
    lz_mask = (result['aicda_status'] == 'negative') & (result['cd3_status'] == 'negative')
    result.loc[lz_mask, 'cell_type'] = 'LZ B-cells'
    
    # T-cells: CD3+ (regardless of AICDA)
    t_mask = result['cd3_status'] == 'positive'
    result.loc[t_mask, 'cell_type'] = 'T-cells'
    
    # Log summary
    cell_type_counts = result['cell_type'].value_counts()
    logger.info("Cell type distribution:")
    for ct, count in cell_type_counts.items():
        logger.info(f"  {ct}: {count}")
    
    return result


def filter_gc_cells(data: pd.DataFrame, gc_status_col: str = 'gc_status') -> pd.DataFrame:
    """Filter to cells within germinal center
    
    Args:
        data: DataFrame with gc_status column
        gc_status_col: Name of germinal center status column
        
    Returns:
        Filtered DataFrame with only GC-positive cells
    """
    if gc_status_col not in data.columns:
        logger.warning(f"Column '{gc_status_col}' not found. Returning all cells.")
        return data
    
    gc_data = data[data[gc_status_col] == 'positive'].copy()
    logger.info(f"Filtered to {len(gc_data)} cells within germinal center")
    
    return gc_data

