# -*- coding: utf-8 -*-
"""
Global boundary morphology features.
Adapted from nmco library with fixes for empty arrays and scipy compatibility.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from skimage import measure
from skimage.morphology import erosion
from skimage.transform import rotate


def safe_mode(arr):
    """Compute mode with scipy 1.11+ compatibility and empty array handling."""
    if len(arr) == 0:
        return np.nan
    result = stats.mode(arr, axis=None, keepdims=False)
    # Handle both old and new scipy versions
    if hasattr(result, 'mode'):
        mode_val = result.mode
        # In newer scipy, mode is a scalar; in older, it's an array
        if hasattr(mode_val, '__len__') and len(mode_val) > 0:
            return mode_val[0]
        return mode_val
    return result[0]


def radii_features(binary_mask: np.ndarray):
    """Compute centroid to boundary distance (radii) features.
    
    Args:
        binary_mask: Binary image with object (non-zero pixels)
        
    Returns:
        Dictionary of radii features
    """
    # Validate input
    if binary_mask is None or binary_mask.size == 0 or np.sum(binary_mask > 0) == 0:
        return {
            "min_radius": np.nan, "max_radius": np.nan, "med_radius": np.nan,
            "avg_radius": np.nan, "mode_radius": np.nan, "d25_radius": np.nan,
            "d75_radius": np.nan, "std_radius": np.nan, "feret_max": np.nan
        }
    
    try:
        props = measure.regionprops_table(binary_mask.astype(int), properties=["centroid"])
        
        # Get centroid
        cenx = float(props['centroid-0'])
        ceny = float(props['centroid-1'])
        
        # Get edge pixels
        bw = binary_mask > 0
        edge = np.subtract(bw.astype(int), erosion(bw).astype(int))
        boundary_coords = np.where(edge > 0)
        boundary_x, boundary_y = boundary_coords[0], boundary_coords[1]
        
        if len(boundary_x) < 3:
            return {
                "min_radius": np.nan, "max_radius": np.nan, "med_radius": np.nan,
                "avg_radius": np.nan, "mode_radius": np.nan, "d25_radius": np.nan,
                "d75_radius": np.nan, "std_radius": np.nan, "feret_max": np.nan
            }
        
        # Calculate radii
        dist_b_c = np.sqrt(np.square(boundary_x - cenx) + np.square(boundary_y - ceny))
        
        # Compute feret distances
        coords = np.column_stack((boundary_x, boundary_y))
        dist_matrix = distance.squareform(distance.pdist(coords, "euclidean"))
        feret = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]
        
        return {
            "min_radius": np.min(dist_b_c),
            "max_radius": np.max(dist_b_c),
            "med_radius": np.median(dist_b_c),
            "avg_radius": np.mean(dist_b_c),
            "mode_radius": safe_mode(dist_b_c),
            "d25_radius": np.percentile(dist_b_c, 25),
            "d75_radius": np.percentile(dist_b_c, 75),
            "std_radius": np.std(dist_b_c),
            "feret_max": np.max(feret) if len(feret) > 0 else np.nan
        }
    except Exception:
        return {
            "min_radius": np.nan, "max_radius": np.nan, "med_radius": np.nan,
            "avg_radius": np.nan, "mode_radius": np.nan, "d25_radius": np.nan,
            "d75_radius": np.nan, "std_radius": np.nan, "feret_max": np.nan
        }


def calliper_sizes(binary_mask: np.ndarray, angular_resolution: int = 10):
    """Compute min and max calliper distances by rotating the image.
    
    Args:
        binary_mask: Binary image
        angular_resolution: Rotation step in degrees (1-359)
        
    Returns:
        Dictionary of calliper features
    """
    if binary_mask is None or binary_mask.size == 0 or np.sum(binary_mask > 0) == 0:
        return {"min_calliper": np.nan, "max_calliper": np.nan, "smallest_largest_calliper": np.nan}
    
    try:
        img = binary_mask > 0
        callipers = []
        
        for angle in range(1, 360, angular_resolution):
            rot_img = rotate(img, angle, resize=True)
            col_sums = np.sum(rot_img, axis=0)
            if len(col_sums) > 0:
                callipers.append(np.max(col_sums))
        
        if len(callipers) == 0:
            return {"min_calliper": np.nan, "max_calliper": np.nan, "smallest_largest_calliper": np.nan}
        
        min_cal = min(callipers)
        max_cal = max(callipers)
        
        return {
            "min_calliper": min_cal,
            "max_calliper": max_cal,
            "smallest_largest_calliper": min_cal / max_cal if max_cal > 0 else np.nan
        }
    except Exception:
        return {"min_calliper": np.nan, "max_calliper": np.nan, "smallest_largest_calliper": np.nan}


def simple_morphology(regionmask: np.ndarray):
    """Compute simple morphology features.
    
    Args:
        regionmask: Binary region mask
        
    Returns:
        DataFrame with morphology features
    """
    if regionmask is None or regionmask.size == 0 or np.sum(regionmask > 0) == 0:
        return pd.DataFrame([{
            "centroid-0": np.nan, "centroid-1": np.nan, "area": np.nan,
            "perimeter": np.nan, "bbox_area": np.nan, "convex_area": np.nan,
            "equivalent_diameter": np.nan, "major_axis_length": np.nan,
            "minor_axis_length": np.nan, "eccentricity": np.nan,
            "orientation": np.nan, "concavity": np.nan, "solidity": np.nan,
            "a_r": np.nan, "shape_factor": np.nan, "area_bbarea": np.nan
        }])
    
    try:
        morphology_features = [
            'centroid', 'area', 'perimeter', 'bbox_area', 'convex_area',
            'equivalent_diameter', 'major_axis_length', 'minor_axis_length',
            'eccentricity', 'orientation'
        ]
        
        regionmask = regionmask.astype('uint8')
        feat = pd.DataFrame(measure.regionprops_table(regionmask, properties=morphology_features))
        
        if len(feat) == 0:
            return pd.DataFrame([{k: np.nan for k in morphology_features}])
        
        # Compute derived features safely
        convex_area = feat["convex_area"].values[0]
        area = feat["area"].values[0]
        bbox_area = feat["bbox_area"].values[0]
        major = feat["major_axis_length"].values[0]
        perimeter = feat["perimeter"].values[0]
        
        feat["concavity"] = (convex_area - area) / convex_area if convex_area > 0 else np.nan
        feat["solidity"] = area / convex_area if convex_area > 0 else np.nan
        feat["a_r"] = feat["minor_axis_length"] / major if major > 0 else np.nan
        feat["shape_factor"] = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else np.nan
        feat["area_bbarea"] = area / bbox_area if bbox_area > 0 else np.nan
        
        return feat
    except Exception:
        return pd.DataFrame([{
            "centroid-0": np.nan, "centroid-1": np.nan, "area": np.nan,
            "perimeter": np.nan, "bbox_area": np.nan, "convex_area": np.nan,
            "equivalent_diameter": np.nan, "major_axis_length": np.nan,
            "minor_axis_length": np.nan, "eccentricity": np.nan,
            "orientation": np.nan, "concavity": np.nan, "solidity": np.nan,
            "a_r": np.nan, "shape_factor": np.nan, "area_bbarea": np.nan
        }])


def measure_global_morphometrics(
    binary_image: np.ndarray,
    angular_resolution: int = 10,
    measure_simple: bool = True,
    measure_calliper: bool = True,
    measure_radii: bool = True
):
    """Compute all global morphology features.
    
    Args:
        binary_image: Binary image of nucleus
        angular_resolution: Rotation step for calliper calculation
        measure_simple: Compute simple morphology features
        measure_calliper: Compute calliper features
        measure_radii: Compute radii features
        
    Returns:
        DataFrame with all morphology features
    """
    feat = {}
    
    if measure_calliper:
        feat.update(calliper_sizes(binary_image, angular_resolution))
    
    if measure_radii:
        feat.update(radii_features(binary_image))
    
    if measure_simple:
        simple_feat = simple_morphology(binary_image)
        feat = pd.concat([pd.DataFrame([feat]), simple_feat], axis=1)
    else:
        feat = pd.DataFrame([feat])
    
    return feat

