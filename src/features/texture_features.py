# -*- coding: utf-8 -*-
"""
Texture features using GLCM and image moments.
Adapted from nmco library with fixes for empty arrays.
"""

import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from skimage import measure


def gclm_textures(regionmask: np.ndarray, intensity: np.ndarray, lengths=[1, 5, 20]):
    """Compute GLCM texture features at given length scales.
    
    Args:
        regionmask: Binary background mask
        intensity: Intensity image
        lengths: Length scales for GLCM computation
        
    Returns:
        DataFrame with GLCM features
    """
    if regionmask is None or intensity is None:
        return _empty_gclm_features(lengths)
    
    masked = intensity * regionmask
    if np.max(masked) == 0:
        return _empty_gclm_features(lengths)
    
    try:
        # Normalize and convert to uint8
        normalized = masked / max(np.max(masked), 1) * 255
        img_ubyte = img_as_ubyte(normalized / 255)
        
        glcm = graycomatrix(
            img_ubyte,
            distances=lengths,
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        )
        
        contrast = pd.DataFrame(np.mean(graycoprops(glcm, "contrast"), axis=1).tolist()).T
        contrast.columns = ["contrast_" + str(col) for col in lengths]
        
        dissimilarity = pd.DataFrame(np.mean(graycoprops(glcm, "dissimilarity"), axis=1).tolist()).T
        dissimilarity.columns = ["dissimilarity_" + str(col) for col in lengths]
        
        homogeneity = pd.DataFrame(np.mean(graycoprops(glcm, "homogeneity"), axis=1).tolist()).T
        homogeneity.columns = ["homogeneity_" + str(col) for col in lengths]
        
        asm = pd.DataFrame(np.mean(graycoprops(glcm, "ASM"), axis=1).tolist()).T
        asm.columns = ["asm_" + str(col) for col in lengths]
        
        energy = pd.DataFrame(np.mean(graycoprops(glcm, "energy"), axis=1).tolist()).T
        energy.columns = ["energy_" + str(col) for col in lengths]
        
        correlation = pd.DataFrame(np.mean(graycoprops(glcm, "correlation"), axis=1).tolist()).T
        correlation.columns = ["correlation_" + str(col) for col in lengths]
        
        feat = pd.concat([
            contrast.reset_index(drop=True),
            dissimilarity.reset_index(drop=True),
            homogeneity.reset_index(drop=True),
            asm.reset_index(drop=True),
            energy.reset_index(drop=True),
            correlation.reset_index(drop=True),
        ], axis=1)
        
        return feat
    except Exception:
        return _empty_gclm_features(lengths)


def _empty_gclm_features(lengths):
    """Return empty GLCM features."""
    feat = {}
    for l in lengths:
        feat[f"contrast_{l}"] = np.nan
        feat[f"dissimilarity_{l}"] = np.nan
        feat[f"homogeneity_{l}"] = np.nan
        feat[f"asm_{l}"] = np.nan
        feat[f"energy_{l}"] = np.nan
        feat[f"correlation_{l}"] = np.nan
    return pd.DataFrame([feat])


def image_moments(regionmask: np.ndarray, intensity: np.ndarray):
    """Compute image moments features.
    
    Args:
        regionmask: Binary background mask
        intensity: Intensity image
        
    Returns:
        DataFrame with moments features
    """
    if regionmask is None or intensity is None:
        return _empty_moments_features()
    
    if np.sum(regionmask > 0) == 0:
        return _empty_moments_features()
    
    try:
        moments_features = [
            'weighted_centroid', 'weighted_moments', 'weighted_moments_normalized',
            'weighted_moments_central', 'weighted_moments_hu',
            'moments', 'moments_normalized', 'moments_central', 'moments_hu'
        ]
        
        regionmask_uint8 = regionmask.astype('uint8')
        feat = pd.DataFrame(measure.regionprops_table(
            regionmask_uint8, intensity, properties=moments_features
        ))
        
        if len(feat) == 0:
            return _empty_moments_features()
        
        return feat
    except Exception:
        return _empty_moments_features()


def _empty_moments_features():
    """Return empty moments features."""
    # Return a minimal set of moment features as NaN
    return pd.DataFrame([{
        "weighted_centroid-0": np.nan, "weighted_centroid-1": np.nan,
        "moments-0-0": np.nan, "moments-0-1": np.nan,
        "moments_hu-0": np.nan, "moments_hu-1": np.nan
    }])


def measure_texture_features(
    regionmask: np.ndarray,
    intensity: np.ndarray,
    lengths=[1, 5, 20],
    measure_gclm: bool = True,
    measure_moments: bool = True
):
    """Compute all texture features.
    
    Args:
        regionmask: Binary background mask
        intensity: Intensity image
        lengths: Length scales for GLCM
        measure_gclm: Compute GLCM features
        measure_moments: Compute moments features
        
    Returns:
        DataFrame with all texture features
    """
    all_features = pd.DataFrame()
    
    if measure_gclm:
        gclm_feat = gclm_textures(regionmask, intensity, lengths)
        all_features = pd.concat([all_features, gclm_feat.reset_index(drop=True)], axis=1)
    
    if measure_moments:
        moments_feat = image_moments(regionmask, intensity)
        all_features = pd.concat([all_features, moments_feat.reset_index(drop=True)], axis=1)
    
    return all_features

