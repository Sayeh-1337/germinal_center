# -*- coding: utf-8 -*-
"""
Intensity distribution features.
Adapted from nmco library with fixes for empty arrays and scipy compatibility.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kurtosis, skew
from skimage.measure import shannon_entropy


def safe_mode(arr):
    """Compute mode with scipy 1.11+ compatibility and empty array handling."""
    if len(arr) == 0:
        return np.nan
    result = stats.mode(arr, axis=None, keepdims=False)
    if hasattr(result, 'mode'):
        mode_val = result.mode
        if hasattr(mode_val, '__len__') and len(mode_val) > 0:
            return mode_val[0]
        return mode_val
    return result[0]


def hetero_euchro_measures(regionmask: np.ndarray, intensity: np.ndarray, alpha: float = 1.0):
    """Compute heterochromatin to euchromatin features.
    
    Args:
        regionmask: Binary background mask
        intensity: Intensity image
        alpha: Threshold for calculating heterochromatin intensity
        
    Returns:
        Dictionary of HC/EC features
    """
    # Validate inputs
    if regionmask is None or intensity is None:
        return _empty_hc_ec_features()
    
    masked_intensity = intensity[regionmask > 0]
    if len(masked_intensity) == 0:
        return _empty_hc_ec_features()
    
    try:
        high, low = np.percentile(masked_intensity, q=(80, 20))
        mean_int = np.mean(masked_intensity)
        std_int = np.std(masked_intensity)
        hc = mean_int + (alpha * std_int)
        
        n_high = np.sum(masked_intensity >= high)
        n_low = np.sum(masked_intensity <= low)
        n_hc = np.sum(masked_intensity >= hc)
        n_ec = np.sum(masked_intensity < hc)
        n_total = np.sum(masked_intensity > 0)
        
        hc_content = np.sum(np.where(masked_intensity >= hc, masked_intensity, 0))
        ec_content = np.sum(np.where(masked_intensity < hc, masked_intensity, 0))
        total_content = np.sum(np.where(masked_intensity > 0, masked_intensity, 0))
        
        return {
            "i80_i20": high / low if low > 0 else np.nan,
            "nhigh_nlow": n_high / n_low if n_low > 0 else np.nan,
            "hc_area_ec_area": n_hc / n_ec if n_ec > 0 else np.nan,
            "hc_area_nuc_area": n_hc / n_total if n_total > 0 else np.nan,
            "hc_content_ec_content": hc_content / ec_content if ec_content > 0 else np.nan,
            "hc_content_dna_content": hc_content / total_content if total_content > 0 else np.nan
        }
    except Exception:
        return _empty_hc_ec_features()


def _empty_hc_ec_features():
    """Return empty HC/EC features."""
    return {
        "i80_i20": np.nan, "nhigh_nlow": np.nan, "hc_area_ec_area": np.nan,
        "hc_area_nuc_area": np.nan, "hc_content_ec_content": np.nan,
        "hc_content_dna_content": np.nan
    }


def intensity_histogram_measures(regionmask: np.ndarray, intensity: np.ndarray):
    """Compute intensity distribution features.
    
    Args:
        regionmask: Binary background mask
        intensity: Intensity image
        
    Returns:
        Dictionary of intensity features
    """
    if regionmask is None or intensity is None:
        return _empty_intensity_features()
    
    masked_intensity = intensity[regionmask > 0]
    if len(masked_intensity) == 0:
        return _empty_intensity_features()
    
    try:
        return {
            "int_min": np.percentile(masked_intensity, 0),
            "int_d25": np.percentile(masked_intensity, 25),
            "int_media": np.percentile(masked_intensity, 50),  # Median (matches notebook naming)
            "int_d75": np.percentile(masked_intensity, 75),
            "int_max": np.percentile(masked_intensity, 100),
            "int_mean": np.mean(masked_intensity),
            "int_mode": safe_mode(masked_intensity),
            "int_sd": np.std(masked_intensity),
            "kurtosis": float(kurtosis(masked_intensity.ravel())),
            "skewness": float(skew(masked_intensity.ravel())),
            "entropy": shannon_entropy(intensity * regionmask)
        }
    except Exception:
        return _empty_intensity_features()


def _empty_intensity_features():
    """Return empty intensity features."""
    return {
        "int_min": np.nan, "int_d25": np.nan, "int_media": np.nan,  # Median (matches notebook naming)
        "int_d75": np.nan, "int_max": np.nan, "int_mean": np.nan,
        "int_mode": np.nan, "int_sd": np.nan, "kurtosis": np.nan,
        "skewness": np.nan, "entropy": np.nan
    }


def measure_intensity_features(
    regionmask: np.ndarray,
    intensity: np.ndarray,
    measure_int_dist: bool = True,
    measure_hc_ec_ratios: bool = True,
    hc_alpha: float = 1.0
):
    """Compute all intensity distribution features.
    
    Args:
        regionmask: Binary background mask
        intensity: Intensity image
        measure_int_dist: Compute intensity distribution features
        measure_hc_ec_ratios: Compute HC/EC ratio features
        hc_alpha: Threshold for heterochromatin calculation
        
    Returns:
        DataFrame with all intensity features
    """
    feat = {}
    
    if measure_int_dist:
        feat.update(intensity_histogram_measures(regionmask, intensity))
    
    if measure_hc_ec_ratios:
        feat.update(hetero_euchro_measures(regionmask, intensity, hc_alpha))
    
    return pd.DataFrame([feat])

