# -*- coding: utf-8 -*-
"""
Main feature extraction function.
Adapted from nmco library with pandas 2.0+ compatibility and robust error handling.
"""

import logging
import os
import cv2 as cv
import numpy as np
import pandas as pd
from tifffile import imread
from skimage import measure
from tqdm import tqdm

from src.features.global_morphology import measure_global_morphometrics
from src.features.intensity_features import measure_intensity_features
from src.features.texture_features import measure_texture_features
from src.features.curvature_features import measure_curvature_features

logger = logging.getLogger(__name__)


def run_nuclear_chromatin_feat_ext(
    raw_image_path: str,
    labelled_image_path: str,
    output_dir: str,
    calliper_angular_resolution: int = 10,
    measure_simple_geometry: bool = True,
    measure_calliper_distances: bool = True,
    measure_radii_features: bool = True,
    step_size_curvature: int = 2,
    prominance_curvature: float = 0.1,
    width_prominent_curvature: int = 5,
    dist_bt_peaks_curvature: int = 10,
    measure_int_dist_features: bool = True,
    measure_hc_ec_ratios_features: bool = True,
    hc_threshold: float = 1,
    gclm_lengths: list = None,
    measure_gclm_features: bool = True,
    measure_moments_features: bool = True,
    normalize: bool = False,
    save_output: bool = False
):
    """Extract nuclear chromatin features from raw and labelled images.
    
    This function reads in raw and segmented/labelled images and computes
    comprehensive nuclear features including morphology, intensity distribution,
    texture, and boundary curvature.
    
    Args:
        raw_image_path: Path to raw image file
        labelled_image_path: Path to segmented label image
        output_dir: Output directory for results
        calliper_angular_resolution: Angular resolution for calliper calculation
        measure_simple_geometry: Compute simple morphology features
        measure_calliper_distances: Compute calliper features
        measure_radii_features: Compute radii features
        step_size_curvature: Step size for curvature calculation
        prominance_curvature: Minimum prominence for peak detection
        width_prominent_curvature: Minimum width for peak detection
        dist_bt_peaks_curvature: Minimum distance between peaks
        measure_int_dist_features: Compute intensity distribution features
        measure_hc_ec_ratios_features: Compute HC/EC ratio features
        hc_threshold: Threshold for heterochromatin calculation
        gclm_lengths: Length scales for GLCM features
        measure_gclm_features: Compute GLCM texture features
        measure_moments_features: Compute moments features
        normalize: Normalize raw image before processing
        save_output: Save features to CSV file
        
    Returns:
        DataFrame with features for all nuclei in the image
    """
    if gclm_lengths is None:
        gclm_lengths = [1, 5, 20]
    
    # Read images
    labelled_image = imread(labelled_image_path).astype(int)
    raw_image = imread(raw_image_path).astype(int)
    
    # Normalize if requested
    if normalize:
        raw_image = cv.normalize(
            raw_image, None, alpha=0, beta=255,
            norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F
        )
        raw_image = np.clip(raw_image, 0, 255)
    
    # Get region properties
    props = measure.regionprops(labelled_image, raw_image)
    
    if len(props) == 0:
        logger.warning(f"No regions found in {labelled_image_path}")
        return pd.DataFrame()
    
    # Extract features for each nucleus
    all_features_list = []
    failed_nuclei = 0
    
    for i in tqdm(range(len(props)), desc="Nuclei", leave=False):
        try:
            # Get nucleus mask and intensity
            nucleus_mask = props[i].image
            intensity_image = props[i].intensity_image
            
            # Validate nucleus has pixels
            if nucleus_mask is None or np.sum(nucleus_mask) == 0:
                failed_nuclei += 1
                continue
            
            # Extract all feature types
            label_df = pd.DataFrame([{"label": props[i].label}])
            
            morphology_df = measure_global_morphometrics(
                nucleus_mask,
                angular_resolution=calliper_angular_resolution,
                measure_simple=measure_simple_geometry,
                measure_calliper=measure_calliper_distances,
                measure_radii=measure_radii_features
            ).reset_index(drop=True)
            
            curvature_df = measure_curvature_features(
                nucleus_mask,
                step=step_size_curvature,
                prominance=prominance_curvature,
                width=width_prominent_curvature,
                dist_bt_peaks=dist_bt_peaks_curvature
            ).reset_index(drop=True)
            
            intensity_df = measure_intensity_features(
                nucleus_mask,
                intensity_image,
                measure_int_dist=measure_int_dist_features,
                measure_hc_ec_ratios=measure_hc_ec_ratios_features,
                hc_alpha=hc_threshold
            ).reset_index(drop=True)
            
            texture_df = measure_texture_features(
                nucleus_mask,
                intensity_image,
                lengths=gclm_lengths,
                measure_gclm=measure_gclm_features,
                measure_moments=measure_moments_features
            ).reset_index(drop=True)
            
            # Combine all features
            nucleus_features = pd.concat([
                label_df,
                morphology_df,
                curvature_df,
                intensity_df,
                texture_df
            ], axis=1)
            
            all_features_list.append(nucleus_features)
            
        except Exception as e:
            failed_nuclei += 1
            logger.debug(f"Failed to extract features for nucleus {i}: {str(e)}")
            continue
    
    if failed_nuclei > 0:
        logger.debug(f"Skipped {failed_nuclei}/{len(props)} nuclei due to extraction errors")
    
    if len(all_features_list) == 0:
        logger.warning(f"No valid nuclei features extracted from {labelled_image_path}")
        return pd.DataFrame()
    
    # Combine all nuclei features
    all_features = pd.concat(all_features_list, ignore_index=True)
    
    # Save output if requested
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(labelled_image_path))[0]
        output_path = os.path.join(output_dir, f"{filename}.csv")
        all_features.to_csv(output_path, index=False)
    
    return all_features

