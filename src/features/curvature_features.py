# -*- coding: utf-8 -*-
"""
Local boundary curvature features.
Adapted from nmco library with fixes for empty arrays.
"""

import numpy as np
import pandas as pd
from math import sqrt
from skimage.morphology import erosion
from skimage import measure
from scipy import signal


def circumradius(T, binary_mask: np.ndarray):
    """Find the radius of a circumcircle.
    
    Args:
        T: Tuple of cartesian coordinates of three points
        binary_mask: Binary image
        
    Returns:
        Radius of circumcircle, or False if cannot be calculated
    """
    (x1, y1), (x2, y2), (x3, y3) = T
    
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if D == 0:
        return False
    
    # Centroid of circumcircle
    Ux = (
        ((x1 ** 2 + y1 ** 2) * (y2 - y3))
        + ((x2 ** 2 + y2 ** 2) * (y3 - y1))
        + ((x3 ** 2 + y3 ** 2) * (y1 - y2))
    ) / D
    Uy = (
        ((x1 ** 2 + y1 ** 2) * (x3 - x2))
        + ((x2 ** 2 + y2 ** 2) * (x1 - x3))
        + ((x3 ** 2 + y3 ** 2) * (x2 - x1))
    ) / D
    
    # Radius
    r = sqrt((Ux - x2) ** 2 + (Uy - y2) ** 2) + 1
    
    # Determine sign
    x = int(np.floor(Ux))
    y = int(np.floor(Uy))
    
    if x >= binary_mask.shape[0] or y >= binary_mask.shape[1]:
        r = -r
    elif x < 0 or y < 0:
        r = -r
    elif binary_mask[x, y]:
        r = r
    else:
        r = -r
    
    return r


def local_radius_curvature(binary_image: np.ndarray, step: int = 2):
    """Compute local radius of curvature.
    
    Args:
        binary_image: Binary region image
        step: Step size for obtaining vertices
        
    Returns:
        Array of local curvature values, or empty array on failure
    """
    if binary_image is None or binary_image.size == 0:
        return np.array([])
    
    if np.sum(binary_image > 0) == 0:
        return np.array([])
    
    try:
        # Get edge of binary image
        bw = binary_image > 0
        bw = np.pad(bw, pad_width=5, mode="constant", constant_values=0)
        edge = np.subtract(bw.astype(int), erosion(bw).astype(int))
        
        boundary_coords = np.where(edge > 0)
        boundary_x, boundary_y = boundary_coords[0], boundary_coords[1]
        
        if len(boundary_x) < step * 2 + 1:
            return np.array([])
        
        # Sort boundary points by angle from centroid
        cenx, ceny = np.mean(boundary_x), np.mean(boundary_y)
        arr1inds = np.arctan2(boundary_x - cenx, boundary_y - ceny).argsort()
        boundary_x, boundary_y = boundary_x[arr1inds[::-1]], boundary_y[arr1inds[::-1]]
        
        # Compute local radii of curvature
        coords = np.column_stack((boundary_x, boundary_y))
        coords_circ = np.vstack((coords[-step:], coords, coords[:step]))
        
        r_c = []
        for i in range(step, coords.shape[0] + step):
            r = circumradius(
                (tuple(coords_circ[i - step]), tuple(coords_circ[i]), tuple(coords_circ[i + step])),
                bw
            )
            r_c.append(r if r is not False else 0)
        
        return np.array(r_c)
    except Exception:
        return np.array([])


def global_curvature_features(local_curvatures: np.ndarray):
    """Compute features describing local curvature distributions.
    
    Args:
        local_curvatures: Array of ordered local curvatures
        
    Returns:
        Dictionary of curvature features
    """
    if len(local_curvatures) == 0:
        return _empty_global_curvature_features()
    
    try:
        pos_curvature = local_curvatures[local_curvatures > 0]
        neg_curvature = np.abs(local_curvatures[local_curvatures < 0])
        
        feat = {
            "avg_curvature": np.mean(local_curvatures),
            "std_curvature": np.std(local_curvatures),
            "npolarity_changes": np.where(np.diff(np.sign(local_curvatures)))[0].shape[0]
        }
        
        if len(pos_curvature) > 0:
            feat.update({
                "max_posi_curv": np.max(pos_curvature),
                "avg_posi_curv": np.mean(pos_curvature),
                "med_posi_curv": np.median(pos_curvature),
                "std_posi_curv": np.std(pos_curvature),
                "sum_posi_curv": np.sum(pos_curvature),
                "len_posi_curv": len(pos_curvature)
            })
        else:
            feat.update({
                "max_posi_curv": np.nan, "avg_posi_curv": np.nan,
                "med_posi_curv": np.nan, "std_posi_curv": np.nan,
                "sum_posi_curv": np.nan, "len_posi_curv": np.nan
            })
        
        if len(neg_curvature) > 0:
            feat.update({
                "max_neg_curv": np.max(neg_curvature),
                "avg_neg_curv": np.mean(neg_curvature),
                "med_neg_curv": np.median(neg_curvature),
                "std_neg_curv": np.std(neg_curvature),
                "sum_neg_curv": np.sum(neg_curvature),
                "len_neg_curv": len(neg_curvature)
            })
        else:
            feat.update({
                "max_neg_curv": np.nan, "avg_neg_curv": np.nan,
                "med_neg_curv": np.nan, "std_neg_curv": np.nan,
                "sum_neg_curv": np.nan, "len_neg_curv": np.nan
            })
        
        return feat
    except Exception:
        return _empty_global_curvature_features()


def _empty_global_curvature_features():
    """Return empty global curvature features."""
    return {
        "avg_curvature": np.nan, "std_curvature": np.nan, "npolarity_changes": np.nan,
        "max_posi_curv": np.nan, "avg_posi_curv": np.nan, "med_posi_curv": np.nan,
        "std_posi_curv": np.nan, "sum_posi_curv": np.nan, "len_posi_curv": np.nan,
        "max_neg_curv": np.nan, "avg_neg_curv": np.nan, "med_neg_curv": np.nan,
        "std_neg_curv": np.nan, "sum_neg_curv": np.nan, "len_neg_curv": np.nan
    }


def prominant_curvature_features(
    local_curvatures: np.ndarray,
    min_prominance: float = 0.1,
    min_width: int = 5,
    dist_bwt_peaks: int = 10
):
    """Find prominent peaks in local curvature.
    
    Args:
        local_curvatures: Array of local curvatures
        min_prominance: Minimum peak prominence
        min_width: Minimum peak width
        dist_bwt_peaks: Minimum distance between peaks
        
    Returns:
        Dictionary of prominent curvature features
    """
    if len(local_curvatures) == 0:
        return _empty_prominent_features()
    
    try:
        # Find positive peaks
        pos_peaks, pos_prop = signal.find_peaks(
            local_curvatures,
            prominence=min_prominance,
            distance=dist_bwt_peaks,
            width=min_width,
        )
        
        # Find negative peaks
        neg_peaks, neg_prop = signal.find_peaks(
            -local_curvatures,
            prominence=min_prominance,
            distance=dist_bwt_peaks,
            width=min_width,
        )
        
        feat = {}
        
        # Positive peak features
        feat["num_prominant_pos_curv"] = len(pos_peaks)
        if len(pos_peaks) > 0:
            feat["prominance_prominant_pos_curv"] = np.mean(pos_prop["prominences"])
            feat["width_prominant_pos_curv"] = np.mean(pos_prop["widths"])
            feat["prominant_pos_curv"] = np.mean(local_curvatures[pos_peaks])
        else:
            feat["prominance_prominant_pos_curv"] = np.nan
            feat["width_prominant_pos_curv"] = np.nan
            feat["prominant_pos_curv"] = np.nan
        
        # Negative peak features
        feat["num_prominant_neg_curv"] = len(neg_peaks)
        if len(neg_peaks) > 0:
            feat["prominance_prominant_neg_curv"] = np.mean(neg_prop["prominences"])
            feat["width_prominant_neg_curv"] = np.mean(neg_prop["widths"])
            feat["prominant_neg_curv"] = np.mean(local_curvatures[neg_peaks])
        else:
            feat["prominance_prominant_neg_curv"] = np.nan
            feat["width_prominant_neg_curv"] = np.nan
            feat["prominant_neg_curv"] = np.nan
        
        return feat
    except Exception:
        return _empty_prominent_features()


def _empty_prominent_features():
    """Return empty prominent curvature features."""
    return {
        "num_prominant_pos_curv": np.nan, "prominance_prominant_pos_curv": np.nan,
        "width_prominant_pos_curv": np.nan, "prominant_pos_curv": np.nan,
        "num_prominant_neg_curv": np.nan, "prominance_prominant_neg_curv": np.nan,
        "width_prominant_neg_curv": np.nan, "prominant_neg_curv": np.nan
    }


def measure_curvature_features(
    binary_image: np.ndarray,
    step: int = 2,
    prominance: float = 0.1,
    width: int = 5,
    dist_bt_peaks: int = 10
):
    """Compute all curvature features.
    
    Args:
        binary_image: Binary image
        step: Step size for curvature calculation
        prominance: Minimum peak prominence
        width: Minimum peak width
        dist_bt_peaks: Minimum distance between peaks
        
    Returns:
        DataFrame with all curvature features
    """
    r_c = local_radius_curvature(binary_image, step)
    
    if len(r_c) == 0:
        feat = {}
        feat.update(_empty_global_curvature_features())
        feat.update(_empty_prominent_features())
        feat["frac_peri_w_posi_curvature"] = np.nan
        feat["frac_peri_w_neg_curvature"] = np.nan
        feat["frac_peri_w_polarity_changes"] = np.nan
        return pd.DataFrame([feat])
    
    try:
        # Calculate local curvature (1/radius)
        local_curvature = np.array([
            1.0 / r if r != 0 else 0 for r in r_c
        ])
        
        feat = {}
        feat.update(global_curvature_features(local_curvature))
        feat.update(prominant_curvature_features(
            local_curvature,
            min_prominance=prominance,
            min_width=width,
            dist_bwt_peaks=dist_bt_peaks
        ))
        
        feat = pd.DataFrame([feat])
        
        # Compute perimeter-normalized features
        try:
            props = measure.regionprops_table(binary_image.astype(int), properties=["perimeter"])
            perimeter = float(props["perimeter"][0]) if len(props["perimeter"]) > 0 else 1
        except Exception:
            perimeter = 1
        
        len_posi = feat["len_posi_curv"].values[0]
        len_neg = feat["len_neg_curv"].values[0]
        npol = feat["npolarity_changes"].values[0]
        
        feat["frac_peri_w_posi_curvature"] = len_posi / perimeter if not np.isnan(len_posi) else np.nan
        feat["frac_peri_w_neg_curvature"] = len_neg / perimeter if not np.isnan(len_neg) else np.nan
        feat["frac_peri_w_polarity_changes"] = npol / perimeter if not np.isnan(npol) else np.nan
        
        return feat
    except Exception:
        feat = {}
        feat.update(_empty_global_curvature_features())
        feat.update(_empty_prominent_features())
        feat["frac_peri_w_posi_curvature"] = np.nan
        feat["frac_peri_w_neg_curvature"] = np.nan
        feat["frac_peri_w_polarity_changes"] = np.nan
        return pd.DataFrame([feat])

