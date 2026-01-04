# -*- coding: utf-8 -*-
"""
Multi-scale chromatin texture features.

This module provides advanced texture analysis at multiple scales:
- Wavelet-based decomposition
- Fractal dimension analysis
- Chromatin domain segmentation

Biological insight:
- Different scales capture different biological processes:
  * Fine scale (1-2 pixels): Local heterochromatin/euchromatin boundaries
  * Medium scale (5-10 pixels): Topologically associating domains (TADs)
  * Coarse scale (20-50 pixels): Nuclear compartmentalization
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

logger = logging.getLogger(__name__)


def extract_wavelet_chromatin_features(
    intensity_image: np.ndarray,
    mask: np.ndarray,
    wavelet: str = 'db4',
    levels: int = 3
) -> pd.DataFrame:
    """
    Multi-scale wavelet decomposition of chromatin texture.
    
    Scales:
    - Fine scale (level 1): Local chromatin compaction
    - Medium scale (level 2): Chromatin domain organization
    - Coarse scale (level 3+): Nuclear-wide patterns
    
    Biological insight:
    - Different scales capture different biological processes:
      * Fine: Heterochromatin/euchromatin boundaries
      * Medium: Topologically associating domains (TADs)
      * Coarse: Nuclear compartmentalization
      
    Args:
        intensity_image: Grayscale intensity image
        mask: Binary mask for the nucleus
        wavelet: Wavelet family to use (default: 'db4')
        levels: Number of decomposition levels
        
    Returns:
        DataFrame with wavelet features
    """
    try:
        import pywt
    except ImportError:
        logger.warning("PyWavelets not installed. Install with: pip install PyWavelets")
        return _empty_wavelet_features(levels)
    
    if mask is None or intensity_image is None:
        return _empty_wavelet_features(levels)
    
    if np.sum(mask) == 0:
        return _empty_wavelet_features(levels)
    
    try:
        # Apply mask to image
        masked_image = intensity_image.astype(float) * mask
        
        # Pad to power of 2 for efficient wavelet transform
        shape = masked_image.shape
        new_shape = tuple(2 ** int(np.ceil(np.log2(s))) for s in shape)
        padded = np.zeros(new_shape)
        padded[:shape[0], :shape[1]] = masked_image
        
        # Perform 2D discrete wavelet transform
        coeffs = pywt.wavedec2(padded, wavelet, level=levels)
        
        features = {}
        
        # Approximation coefficients (low frequency - overall structure)
        approx = coeffs[0]
        features['wavelet_approx_energy'] = np.sum(approx ** 2) / np.sum(mask)
        features['wavelet_approx_entropy'] = _compute_entropy(approx[approx != 0])
        
        # Detail coefficients at each level
        for level, (cH, cV, cD) in enumerate(coeffs[1:], 1):
            # Horizontal details
            h_energy = np.sum(cH ** 2) / np.sum(mask)
            features[f'wavelet_h_energy_l{level}'] = h_energy
            
            # Vertical details
            v_energy = np.sum(cV ** 2) / np.sum(mask)
            features[f'wavelet_v_energy_l{level}'] = v_energy
            
            # Diagonal details
            d_energy = np.sum(cD ** 2) / np.sum(mask)
            features[f'wavelet_d_energy_l{level}'] = d_energy
            
            # Total detail energy at this level
            total_detail = h_energy + v_energy + d_energy
            features[f'wavelet_total_energy_l{level}'] = total_detail
            
            # Anisotropy (horizontal vs vertical dominance)
            if h_energy + v_energy > 0:
                features[f'wavelet_anisotropy_l{level}'] = (h_energy - v_energy) / (h_energy + v_energy)
            else:
                features[f'wavelet_anisotropy_l{level}'] = 0
        
        # Cross-scale relationships
        if levels >= 2:
            # Ratio of fine to coarse detail (texture complexity)
            fine_energy = features.get('wavelet_total_energy_l1', 0)
            coarse_energy = features.get(f'wavelet_total_energy_l{levels}', 0)
            if coarse_energy > 0:
                features['wavelet_fine_coarse_ratio'] = fine_energy / coarse_energy
            else:
                features['wavelet_fine_coarse_ratio'] = np.nan
        
        return pd.DataFrame([features])
        
    except Exception as e:
        logger.debug(f"Wavelet feature extraction failed: {e}")
        return _empty_wavelet_features(levels)


def _compute_entropy(values: np.ndarray) -> float:
    """Compute entropy of a value distribution."""
    if len(values) == 0:
        return 0
    
    # Normalize to probability distribution
    values = np.abs(values)
    total = np.sum(values)
    if total == 0:
        return 0
    
    p = values / total
    p = p[p > 0]  # Remove zeros for log
    return -np.sum(p * np.log2(p))


def _empty_wavelet_features(levels: int) -> pd.DataFrame:
    """Return empty wavelet features DataFrame."""
    features = {
        'wavelet_approx_energy': np.nan,
        'wavelet_approx_entropy': np.nan,
        'wavelet_fine_coarse_ratio': np.nan
    }
    for level in range(1, levels + 1):
        features[f'wavelet_h_energy_l{level}'] = np.nan
        features[f'wavelet_v_energy_l{level}'] = np.nan
        features[f'wavelet_d_energy_l{level}'] = np.nan
        features[f'wavelet_total_energy_l{level}'] = np.nan
        features[f'wavelet_anisotropy_l{level}'] = np.nan
    return pd.DataFrame([features])


def compute_fractal_dimension(
    intensity_image: np.ndarray,
    mask: np.ndarray,
    method: str = 'box_counting',
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Measure fractal dimension of chromatin distribution.
    
    Biological interpretation:
    - DZ cells (active transcription): Higher fractal dimension
      (more complex, less self-similar structure)
    - LZ cells (quiescent): Lower fractal dimension
      (more ordered, condensed structure)
    
    Range: 1.0 (smooth) to 2.0 (space-filling)
    
    Args:
        intensity_image: Grayscale intensity image
        mask: Binary mask for the nucleus
        method: 'box_counting' or 'differential_box_counting'
        threshold: Intensity threshold for binarization (default: Otsu)
        
    Returns:
        Dict with fractal dimension and lacunarity
    """
    if mask is None or intensity_image is None:
        return {'fractal_dimension': np.nan, 'lacunarity': np.nan}
    
    if np.sum(mask) == 0:
        return {'fractal_dimension': np.nan, 'lacunarity': np.nan}
    
    try:
        # Apply mask
        masked = intensity_image.astype(float) * mask
        
        if threshold is None:
            # Use Otsu's threshold
            from skimage.filters import threshold_otsu
            try:
                threshold = threshold_otsu(masked[mask > 0])
            except ValueError:
                threshold = np.mean(masked[mask > 0])
        
        # Binarize for box counting
        binary = (masked > threshold).astype(int)
        
        if method == 'box_counting':
            fd, lacunarity = _box_counting_dimension(binary)
        else:
            fd, lacunarity = _differential_box_counting(masked, mask)
        
        return {
            'fractal_dimension': fd,
            'lacunarity': lacunarity
        }
        
    except Exception as e:
        logger.debug(f"Fractal dimension calculation failed: {e}")
        return {'fractal_dimension': np.nan, 'lacunarity': np.nan}


def _box_counting_dimension(binary_image: np.ndarray) -> Tuple[float, float]:
    """Compute fractal dimension using box-counting method."""
    # Get dimensions
    shape = binary_image.shape
    min_dim = min(shape)
    
    # Box sizes (powers of 2)
    box_sizes = []
    size = 2
    while size < min_dim // 2:
        box_sizes.append(size)
        size *= 2
    
    if len(box_sizes) < 3:
        return np.nan, np.nan
    
    counts = []
    for box_size in box_sizes:
        # Count boxes needed to cover the pattern
        n_boxes_x = int(np.ceil(shape[1] / box_size))
        n_boxes_y = int(np.ceil(shape[0] / box_size))
        
        box_count = 0
        box_counts_per_box = []
        
        for i in range(n_boxes_y):
            for j in range(n_boxes_x):
                # Extract box
                y_start = i * box_size
                y_end = min((i + 1) * box_size, shape[0])
                x_start = j * box_size
                x_end = min((j + 1) * box_size, shape[1])
                
                box = binary_image[y_start:y_end, x_start:x_end]
                pixels_in_box = np.sum(box)
                
                if pixels_in_box > 0:
                    box_count += 1
                    box_counts_per_box.append(pixels_in_box)
        
        counts.append(box_count)
    
    # Linear regression of log(count) vs log(1/size)
    log_sizes = np.log(1.0 / np.array(box_sizes))
    log_counts = np.log(np.array(counts) + 1)  # +1 to avoid log(0)
    
    # Fit line
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dim = coeffs[0]
    
    # Lacunarity (measure of gappiness)
    # Using simplified approach
    if len(box_counts_per_box) > 0:
        mean_count = np.mean(box_counts_per_box)
        var_count = np.var(box_counts_per_box)
        lacunarity = var_count / (mean_count ** 2) if mean_count > 0 else np.nan
    else:
        lacunarity = np.nan
    
    return fractal_dim, lacunarity


def _differential_box_counting(
    intensity_image: np.ndarray,
    mask: np.ndarray
) -> Tuple[float, float]:
    """Compute fractal dimension using differential box counting for grayscale."""
    shape = intensity_image.shape
    min_dim = min(shape)
    
    # Normalize intensity to 0-255 range
    masked = intensity_image * mask
    if np.max(masked) == 0:
        return np.nan, np.nan
    
    normalized = (masked / np.max(masked) * 255).astype(int)
    
    # Box sizes
    box_sizes = []
    size = 2
    while size < min_dim // 4:
        box_sizes.append(size)
        size *= 2
    
    if len(box_sizes) < 3:
        return np.nan, np.nan
    
    counts = []
    for box_size in box_sizes:
        n_boxes_x = int(np.ceil(shape[1] / box_size))
        n_boxes_y = int(np.ceil(shape[0] / box_size))
        
        total_nr = 0
        for i in range(n_boxes_y):
            for j in range(n_boxes_x):
                y_start = i * box_size
                y_end = min((i + 1) * box_size, shape[0])
                x_start = j * box_size
                x_end = min((j + 1) * box_size, shape[1])
                
                box = normalized[y_start:y_end, x_start:x_end]
                mask_box = mask[y_start:y_end, x_start:x_end]
                
                if np.sum(mask_box) > 0:
                    # Differential box counting uses intensity range
                    min_val = np.min(box[mask_box > 0])
                    max_val = np.max(box[mask_box > 0])
                    nr = max(1, (max_val - min_val) // box_size + 1)
                    total_nr += nr
        
        counts.append(total_nr)
    
    # Linear regression
    log_sizes = np.log(1.0 / np.array(box_sizes))
    log_counts = np.log(np.array(counts) + 1)
    
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dim = coeffs[0]
    
    return fractal_dim, np.nan


def analyze_chromatin_domain_sizes(
    intensity_image: np.ndarray,
    mask: np.ndarray,
    threshold_method: str = 'otsu',
    min_domain_size: int = 10
) -> pd.DataFrame:
    """
    Segment and analyze individual chromatin domains.
    
    Features:
    - Number of domains
    - Mean/median/max domain size
    - Domain size distribution (power law exponent)
    - Domain circularity
    
    Biological relevance:
    - DZ: Smaller, more numerous domains (active transcription)
    - LZ: Larger, fewer domains (condensed chromatin)
    
    Args:
        intensity_image: Grayscale intensity image
        mask: Binary mask for the nucleus
        threshold_method: 'otsu', 'mean', or 'percentile'
        min_domain_size: Minimum domain size in pixels
        
    Returns:
        DataFrame with domain statistics
    """
    if mask is None or intensity_image is None:
        return _empty_domain_features()
    
    if np.sum(mask) == 0:
        return _empty_domain_features()
    
    try:
        # Apply mask
        masked = intensity_image.astype(float) * mask
        valid_pixels = masked[mask > 0]
        
        if len(valid_pixels) == 0:
            return _empty_domain_features()
        
        # Threshold to identify heterochromatin domains
        if threshold_method == 'otsu':
            from skimage.filters import threshold_otsu
            try:
                threshold = threshold_otsu(valid_pixels)
            except ValueError:
                threshold = np.mean(valid_pixels)
        elif threshold_method == 'mean':
            threshold = np.mean(valid_pixels)
        else:  # percentile
            threshold = np.percentile(valid_pixels, 75)
        
        # Binarize (high intensity = heterochromatin)
        binary = (masked > threshold).astype(int)
        
        # Label connected components
        labeled, n_domains = ndimage.label(binary)
        
        if n_domains == 0:
            return _empty_domain_features()
        
        # Measure domain properties
        props = measure.regionprops(labeled)
        
        domain_areas = []
        domain_circularities = []
        
        for prop in props:
            if prop.area >= min_domain_size:
                domain_areas.append(prop.area)
                # Circularity = 4π × area / perimeter²
                perimeter = prop.perimeter
                if perimeter > 0:
                    circularity = 4 * np.pi * prop.area / (perimeter ** 2)
                else:
                    circularity = 0
                domain_circularities.append(circularity)
        
        if len(domain_areas) == 0:
            return _empty_domain_features()
        
        features = {
            'n_chromatin_domains': len(domain_areas),
            'domain_area_mean': np.mean(domain_areas),
            'domain_area_median': np.median(domain_areas),
            'domain_area_max': np.max(domain_areas),
            'domain_area_std': np.std(domain_areas),
            'domain_area_total': np.sum(domain_areas),
            'domain_area_fraction': np.sum(domain_areas) / np.sum(mask),
            'domain_circularity_mean': np.mean(domain_circularities),
            'domain_circularity_std': np.std(domain_circularities) if len(domain_circularities) > 1 else 0
        }
        
        # Fit power law to domain size distribution (if enough domains)
        if len(domain_areas) >= 5:
            try:
                # Simple power law fit using log-log linear regression
                sorted_areas = np.sort(domain_areas)[::-1]
                ranks = np.arange(1, len(sorted_areas) + 1)
                
                log_ranks = np.log(ranks)
                log_areas = np.log(sorted_areas)
                
                coeffs = np.polyfit(log_ranks, log_areas, 1)
                features['domain_size_power_exponent'] = -coeffs[0]
            except Exception:
                features['domain_size_power_exponent'] = np.nan
        else:
            features['domain_size_power_exponent'] = np.nan
        
        return pd.DataFrame([features])
        
    except Exception as e:
        logger.debug(f"Domain analysis failed: {e}")
        return _empty_domain_features()


def _empty_domain_features() -> pd.DataFrame:
    """Return empty domain features DataFrame."""
    features = {
        'n_chromatin_domains': np.nan,
        'domain_area_mean': np.nan,
        'domain_area_median': np.nan,
        'domain_area_max': np.nan,
        'domain_area_std': np.nan,
        'domain_area_total': np.nan,
        'domain_area_fraction': np.nan,
        'domain_circularity_mean': np.nan,
        'domain_circularity_std': np.nan,
        'domain_size_power_exponent': np.nan
    }
    return pd.DataFrame([features])


def compute_radial_intensity_profile(
    intensity_image: np.ndarray,
    mask: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Compute radial intensity profile from nuclear center to periphery.
    
    Biological insight:
    - Heterochromatin tends to localize at nuclear periphery
    - Active chromatin in nuclear interior
    - Changes in radial organization during activation
    
    Args:
        intensity_image: Grayscale intensity image
        mask: Binary mask for the nucleus
        n_bins: Number of radial bins
        
    Returns:
        DataFrame with radial profile features
    """
    if mask is None or intensity_image is None:
        return _empty_radial_features(n_bins)
    
    if np.sum(mask) == 0:
        return _empty_radial_features(n_bins)
    
    try:
        # Find centroid
        props = measure.regionprops(mask.astype(int))
        if len(props) == 0:
            return _empty_radial_features(n_bins)
        
        centroid = props[0].centroid
        
        # Compute distance transform
        dist = ndimage.distance_transform_edt(mask)
        max_dist = np.max(dist)
        
        if max_dist == 0:
            return _empty_radial_features(n_bins)
        
        # Normalize distances to 0-1 (center to periphery)
        normalized_dist = dist / max_dist
        
        # Bin intensities by radial distance
        features = {}
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            bin_mask = (normalized_dist >= bin_edges[i]) & (normalized_dist < bin_edges[i + 1]) & (mask > 0)
            if np.sum(bin_mask) > 0:
                bin_intensity = np.mean(intensity_image[bin_mask])
                features[f'radial_intensity_bin{i + 1}'] = bin_intensity
            else:
                features[f'radial_intensity_bin{i + 1}'] = np.nan
        
        # Compute radial gradient (periphery minus center)
        center_intensity = features.get('radial_intensity_bin1', np.nan)
        periphery_intensity = features.get(f'radial_intensity_bin{n_bins}', np.nan)
        
        if not np.isnan(center_intensity) and not np.isnan(periphery_intensity):
            features['radial_intensity_gradient'] = periphery_intensity - center_intensity
            if center_intensity > 0:
                features['radial_intensity_ratio'] = periphery_intensity / center_intensity
            else:
                features['radial_intensity_ratio'] = np.nan
        else:
            features['radial_intensity_gradient'] = np.nan
            features['radial_intensity_ratio'] = np.nan
        
        return pd.DataFrame([features])
        
    except Exception as e:
        logger.debug(f"Radial profile computation failed: {e}")
        return _empty_radial_features(n_bins)


def _empty_radial_features(n_bins: int) -> pd.DataFrame:
    """Return empty radial profile features DataFrame."""
    features = {}
    for i in range(1, n_bins + 1):
        features[f'radial_intensity_bin{i}'] = np.nan
    features['radial_intensity_gradient'] = np.nan
    features['radial_intensity_ratio'] = np.nan
    return pd.DataFrame([features])


def extract_all_multiscale_features(
    intensity_image: np.ndarray,
    mask: np.ndarray,
    wavelet_levels: int = 3,
    radial_bins: int = 5
) -> pd.DataFrame:
    """
    Extract all multi-scale chromatin features.
    
    This is the main entry point for multi-scale analysis.
    
    Args:
        intensity_image: Grayscale intensity image
        mask: Binary mask for the nucleus
        wavelet_levels: Number of wavelet decomposition levels
        radial_bins: Number of radial profile bins
        
    Returns:
        DataFrame with all multi-scale features
    """
    all_features = pd.DataFrame()
    
    # Wavelet features
    wavelet_feat = extract_wavelet_chromatin_features(
        intensity_image, mask, levels=wavelet_levels
    )
    all_features = pd.concat([all_features, wavelet_feat.reset_index(drop=True)], axis=1)
    
    # Fractal dimension
    fractal_feat = compute_fractal_dimension(intensity_image, mask)
    all_features = pd.concat([
        all_features, 
        pd.DataFrame([fractal_feat]).reset_index(drop=True)
    ], axis=1)
    
    # Chromatin domain analysis
    domain_feat = analyze_chromatin_domain_sizes(intensity_image, mask)
    all_features = pd.concat([all_features, domain_feat.reset_index(drop=True)], axis=1)
    
    # Radial profile
    radial_feat = compute_radial_intensity_profile(intensity_image, mask, radial_bins)
    all_features = pd.concat([all_features, radial_feat.reset_index(drop=True)], axis=1)
    
    return all_features

